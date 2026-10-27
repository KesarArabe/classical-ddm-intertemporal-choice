"""Diagnostic investigation of the Lempert_2015 RT misfit.

This script is deliberately diagnostic. It does not change the model, the
cleaning rules, or the fitted parameters. It asks four questions:

1. Is the misfit confined to the upper tail, or is it present across the
   distribution?
2. Is there evidence of an unrecorded response deadline or a pile-up?
3. Is the misfit concentrated in a few participants?
4. Can the boundary separation alone explain the misfit (profile likelihood)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.rt_cdf import (  # noqa: E402
    conditional_pit,
    model_quantile_table,
    participant_quantile_deviations,
    pit_calibration,
)
from src.fitting.optimize import fit_classical_ddm  # noqa: E402
from src.models.classical_ddm import (  # noqa: E402
    DDMParameters,
    PARAMETER_NAMES,
    log_likelihood,
)

STUDY = "Lempert_2015"
PREFIX = "classical_ddm_itc"
REPORT_PATH = (
    ROOT / "results" / "reports" / f"{PREFIX}_lempert_2015_diagnostic_report.md"
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _survival(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.full(grid.size, np.nan)
    survival = 1.0 - np.searchsorted(values, grid, side="right") / values.size
    return survival


def _tail_table(
    observed_rt: np.ndarray,
    simulated_rt: np.ndarray,
    thresholds: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
) -> list[dict[str, float]]:
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold_seconds": float(threshold),
                "observed_proportion": float(np.mean(observed_rt > threshold)),
                "predicted_proportion": float(np.mean(simulated_rt > threshold)),
                "difference": float(
                    np.mean(simulated_rt > threshold)
                    - np.mean(observed_rt > threshold)
                ),
            }
        )
    for q in (0.5, 0.9, 0.95, 0.99):
        rows.append(
            {
                "quantile": float(q),
                "observed_seconds": float(np.quantile(observed_rt, q)),
                "predicted_seconds": float(np.quantile(simulated_rt, q)),
                "difference": float(
                    np.quantile(simulated_rt, q) - np.quantile(observed_rt, q)
                ),
            }
        )
    return rows


def _deadline_checks(df: pd.DataFrame) -> dict[str, Any]:
    rt = df["rt"].to_numpy(dtype=float)
    rounded = np.round(rt, 2)
    counts = pd.Series(rounded).value_counts()
    top = counts.head(10)
    n = rt.size
    return {
        "n_trials": int(n),
        "min_rt": float(np.min(rt)),
        "max_rt": float(np.max(rt)),
        "n_above_4s": int(np.sum(rt > 4.0)),
        "n_above_5s": int(np.sum(rt > 5.0)),
        "n_above_5_5s": int(np.sum(rt > 5.5)),
        "n_above_6s": int(np.sum(rt > 6.0)),
        "proportion_above_5s": float(np.mean(rt > 5.0)),
        "top_rounded_rt": [
            {"rt_seconds": float(value), "n": int(count)}
            for value, count in top.items()
        ],
        "modal_rounded_rt": float(top.index[0]),
        "modal_count": int(top.iloc[0]),
        "modal_fraction": float(top.iloc[0] / n),
        "expected_count_at_modal_bin": float(n / np.unique(rounded).size),
    }


def _profile_likelihood(
    df: pd.DataFrame,
    fit,
    n_terms: int,
    n_points: int = 15,
    span: float | None = None,
    n_se: float = 8.0,
    maxiter: int = 300,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    x_amount = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    full = np.array([fit.params[name] for name in PARAMETER_NAMES], dtype=float)
    a_hat = float(full[0])
    bounds = fit.bounds
    if bounds is None:
        raise ValueError("Profile likelihood requires stored parameter bounds.")
    remaining_bounds = [tuple(pair) for pair in bounds[1:]]

    # The grid must be fine enough to resolve a 95% interval, which is roughly
    # +/-2 SE wide. A fixed span is either too coarse (degenerate interval) or
    # pointlessly wide, so scale it to the Hessian standard error when one is
    # available.
    se_a = None
    if getattr(fit, "se", None) is not None:
        se_a = float(fit.se["a"])
    if span is None:
        if se_a is not None and np.isfinite(se_a) and se_a > 0:
            span = n_se * se_a
        else:
            span = 0.25 * (float(bounds[0][1]) - float(bounds[0][0]))
    span = float(max(span, 1e-4))

    lower_edge = max(float(bounds[0][0]), a_hat - span)
    upper_edge = min(float(bounds[0][1]), a_hat + span)
    grid = np.linspace(lower_edge, upper_edge, int(n_points))
    grid = np.unique(np.concatenate((grid, [a_hat], [lower_edge], [upper_edge])))

    rows = []
    for a_value in grid:
        def objective(theta: np.ndarray) -> float:
            values = np.concatenate(([a_value], theta))
            ll = log_likelihood(
                DDMParameters.from_array(values),
                rt,
                choice,
                x_amount,
                x_delay,
                n_terms=n_terms,
            )
            return 1e12 if not np.isfinite(ll) else -ll

        result = minimize(
            objective,
            full[1:],
            method="L-BFGS-B",
            bounds=remaining_bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-5},
        )
        ll = -float(result.fun) if np.isfinite(result.fun) else np.nan
        rows.append(
            {
                "a": float(a_value),
                "log_likelihood": ll,
                "converged": bool(result.success),
                "t0": float(result.x[0]),
                "v0": float(result.x[1]),
                "beta_amount": float(result.x[2]),
                "beta_delay": float(result.x[3]),
                "nfev": int(result.nfev),
            }
        )
    table = pd.DataFrame(rows)
    best = table["log_likelihood"].max()
    table["deviance"] = 2.0 * (best - table["log_likelihood"])
    critical = float(chi2.ppf(0.95, 1))

    def _crossing(side: str) -> float | None:
        ordered = table.sort_values("a")
        a_values = ordered["a"].to_numpy(dtype=float)
        deviance = ordered["deviance"].to_numpy(dtype=float)
        if side == "lower":
            candidates = np.flatnonzero(a_values <= a_hat)
            candidates = candidates[np.argsort(a_values[candidates], kind="stable")]
            # Walk away from a_hat until the threshold is crossed.
            previous_a = None
            previous_d = None
            for index in candidates[::-1]:
                if deviance[index] <= critical:
                    previous_a = a_values[index]
                    previous_d = deviance[index]
                elif previous_a is not None:
                    a_out = a_values[index]
                    d_out = deviance[index]
                    weight = (critical - previous_d) / (d_out - previous_d)
                    return float(previous_a + weight * (a_out - previous_a))
                else:
                    return None
            return None
        candidates = np.flatnonzero(a_values >= a_hat)
        candidates = candidates[np.argsort(a_values[candidates], kind="stable")]
        previous_a = None
        previous_d = None
        for index in candidates:
            if deviance[index] <= critical:
                previous_a = a_values[index]
                previous_d = deviance[index]
            elif previous_a is not None:
                a_out = a_values[index]
                d_out = deviance[index]
                weight = (critical - previous_d) / (d_out - previous_d)
                return float(previous_a + weight * (a_out - previous_a))
            else:
                return None
        return None

    lower = _crossing("lower")
    upper = _crossing("upper")
    summary = {
        "a_hat": a_hat,
        "hessian_se_a": se_a,
        "grid_span": span,
        "grid_min": float(table["a"].min()),
        "grid_max": float(table["a"].max()),
        "max_profile_log_likelihood": float(best),
        "full_fit_log_likelihood": float(fit.log_likelihood),
        "deviance_critical_95": critical,
        "profile_ci_lower": lower,
        "profile_ci_upper": upper,
        "profile_ci_bounded_by_grid": bool(
            lower is not None
            and upper is not None
            and lower > float(table["a"].min()) + 1e-9
            and upper < float(table["a"].max()) - 1e-9
        ),
        "profile_ci_method": "linear interpolation of the deviance crossing",
        "n_points": int(len(table)),
    }
    return table, summary


def _plot_ccdf(
    df: pd.DataFrame,
    simulated_rt: np.ndarray,
    simulated_choice: np.ndarray,
    path: Path,
) -> None:
    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    grid = np.linspace(0.0, float(np.quantile(rt, 0.995)) * 1.05, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panels = (
        ("all", rt, simulated_rt, np.ones_like(choice, dtype=bool)),
        ("SS", rt[choice == 0], simulated_rt[simulated_choice == 0], choice == 0),
        ("LL", rt[choice == 1], simulated_rt[simulated_choice == 1], choice == 1),
    )
    for ax, (label, obs, pred, _) in zip(axes, panels):
        ax.plot(grid, _survival(obs, grid), label="observed", color="black")
        ax.plot(grid, _survival(pred, grid), label="predicted", color="tab:red", linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("RT (s)")
        ax.set_title(label)
        ax.set_ylim(1e-4, 1.05)
    axes[0].set_ylabel("P(RT > t)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{STUDY}: observed and posterior-predictive RT survival curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_profile(table: pd.DataFrame, critical: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["a"], table["deviance"], "o-")
    ax.axhline(critical, color="grey", linestyle=":", label="95% profile threshold")
    ax.set_xlabel("Boundary separation a")
    ax.set_ylabel("Profile deviance")
    ax.set_title(f"{STUDY}: profile likelihood over a")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_participants(deviations: pd.DataFrame, path: Path) -> None:
    q90 = deviations[deviations["quantile"] == 0.9].copy()
    q90 = q90.sort_values("difference")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.22 * len(q90))))
    ax.barh(
        np.arange(len(q90)),
        q90["difference"],
        color=np.where(q90["difference"] > 0, "tab:red", "tab:blue"),
    )
    ax.set_yticks(np.arange(len(q90)))
    ax.set_yticklabels([str(s) for s in q90["subj_ident"]], fontsize=7)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Predicted minus observed q90 (s)")
    ax.set_title(f"{STUDY}: participant-level q90 deviations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_investigation(
    df: pd.DataFrame,
    fit,
    out_dir: Path,
    ppc_samples: dict[str, np.ndarray] | None = None,
    n_terms: int = 100,
    profile_points: int = 15,
) -> dict[str, Any]:
    """Run all Lempert_2015 diagnostics and write outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    params = DDMParameters.from_array([fit.params[name] for name in PARAMETER_NAMES])
    if ppc_samples is None:
        # Use the same posterior-predictive draw as the main analysis so that
        # the numbers reported here agree with `results/fits/*_diagnostics.json`.
        from src.evaluation.diagnostics import ppc_summary

        ppc = ppc_summary(df, params, include_samples=True)
        ppc_samples = {
            "rt": np.asarray(ppc["simulated_rt"], dtype=float),
            "choice": np.asarray(ppc["simulated_choice"], dtype=int),
        }

    observed_rt = df["rt"].to_numpy(dtype=float)
    simulated_rt = np.asarray(ppc_samples["rt"], dtype=float)
    simulated_choice = np.asarray(ppc_samples["choice"], dtype=int)

    tails = _tail_table(observed_rt, simulated_rt)
    by_choice_tails = []
    for value, label in ((0, "SS"), (1, "LL")):
        obs = observed_rt[df["choice"].to_numpy(dtype=int) == value]
        pred = simulated_rt[simulated_choice == value]
        by_choice_tails.append(
            {
                "choice": label,
                "observed_q90": float(np.quantile(obs, 0.9)),
                "predicted_q90": float(np.quantile(pred, 0.9)),
                "difference_q90": float(
                    np.quantile(pred, 0.9) - np.quantile(obs, 0.9)
                ),
                "observed_max": float(np.max(obs)),
                "predicted_q99": float(np.quantile(pred, 0.99)),
            }
        )

    deadline = _deadline_checks(df)
    deviations = participant_quantile_deviations(
        df, params, quantiles=(0.5, 0.9, 0.95), n_terms=12, n_grid=400
    )
    deviations.to_csv(
        out_dir / f"{PREFIX}_lempert_2015_participant_deviations.csv", index=False
    )
    q90 = deviations[deviations["quantile"] == 0.9]
    q95 = deviations[deviations["quantile"] == 0.95]
    participant_summary = {
        "n_participants": int(q90.shape[0]),
        "q90_overpredicted_n": int(np.sum(q90["difference"] > 0)),
        "q90_underpredicted_n": int(np.sum(q90["difference"] < 0)),
        "q90_mean_difference": float(q90["difference"].mean()),
        "q90_median_difference": float(q90["difference"].median()),
        "q90_worst_positive": float(q90["difference"].max()),
        "q90_worst_negative": float(q90["difference"].min()),
        "q90_correlation_observed_predicted": float(
            np.corrcoef(q90["observed"], q90["predicted"])[0, 1]
        ),
        "q90_slope_predicted_on_observed": float(
            np.polyfit(q90["observed"], q90["predicted"], 1)[0]
        ),
        "q90_rmse": float(np.sqrt(np.mean(q90["difference"] ** 2))),
        "q95_mean_difference": float(q95["difference"].mean()),
    }

    model_quantiles = pd.DataFrame(
        model_quantile_table(df, params, n_terms=12, n_grid=500)
    )
    pit = conditional_pit(df, params, n_terms=12, n_grid=400)
    calibration = pit_calibration(pit)

    profile_table, profile_summary = _profile_likelihood(
        df, fit, n_terms=n_terms, n_points=profile_points
    )
    profile_table.to_csv(
        out_dir / f"{PREFIX}_lempert_2015_profile_likelihood_a.csv", index=False
    )
    try:
        from src.fitting.clustered import clustered_covariance

        clustered = clustered_covariance(fit, df, n_terms=n_terms)
        profile_summary["clustered_se_a"] = next(
            row["clustered_se"]
            for row in clustered["rows"]
            if row["parameter"] == "a"
        )
    except Exception:  # noqa: BLE001 - the comparison is optional context
        profile_summary["clustered_se_a"] = None

    _plot_ccdf(
        df,
        simulated_rt,
        simulated_choice,
        fig_dir / f"{PREFIX}_lempert_2015_reaction_time_survival.png",
    )
    _plot_profile(
        profile_table,
        float(profile_summary["deviance_critical_95"]),
        fig_dir / f"{PREFIX}_lempert_2015_profile_likelihood_a.png",
    )
    _plot_participants(
        deviations,
        fig_dir / f"{PREFIX}_lempert_2015_participant_q90_deviations.png",
    )

    payload = {
        "study": STUDY,
        "n_trials": int(len(df)),
        "n_subjects": int(df["subj_ident"].nunique()),
        "parameters": {name: float(fit.params[name]) for name in PARAMETER_NAMES},
        "t0_margin_seconds": float(float(np.min(observed_rt)) - params.t0),
        "tail_diagnostics": tails,
        "tail_by_choice": by_choice_tails,
        "deadline_checks": deadline,
        "participant_summary": participant_summary,
        "model_quantiles": model_quantiles.to_dict(orient="records"),
        "pit_calibration": calibration,
        "profile": profile_summary,
        "interpretation_notes": [
            "With constant drift and a midpoint start, choice and RT are "
            "independent within a trial. Across trials the model can produce "
            "an SS/LL RT difference only through drift variation; it predicts "
            "a small difference with the opposite sign to the observed one.",
            "The upper-tail overprediction cannot be removed by changing only "
            "the boundary separation if the profile likelihood is flat or if "
            "the profile-implied a remains in the same tail-misfit region.",
        ],
    }
    _save_json(
        payload, out_dir / f"{PREFIX}_lempert_2015_investigation.json"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_build_markdown(payload), encoding="utf-8")
    return payload


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Lempert_2015: diagnostic investigation\n"]
    lines.append(
        "This investigation does **not** modify the model, the cleaning rules, "
        "or the fitted parameters. It documents where the classical DDM fails "
        "for this study.\n"
    )
    lines.append("## Fitted parameters\n")
    lines.append(
        "| Parameter | Estimate |\n|---|---:|\n"
        + "\n".join(
            f"| {name} | {value:.4f} |"
            for name, value in payload["parameters"].items()
        )
    )
    lines.append("")

    lines.append("## Tail diagnostics\n")
    rows = [
        {
            "Threshold / quantile": (
                f">{row['threshold_seconds']:.1f} s"
                if "threshold_seconds" in row
                else f"q{int(row['quantile'] * 100)}"
            ),
            "Observed": row.get("observed_proportion", row.get("observed_seconds")),
            "Predicted": row.get(
                "predicted_proportion", row.get("predicted_seconds")
            ),
            "Difference": row["difference"],
        }
        for row in payload["tail_diagnostics"]
    ]
    lines.append(pd.DataFrame(rows).round(4).to_markdown(index=False))
    lines.append("")
    lines.append(
        "Positive differences mean the model predicts more slow responses than "
        "were observed.\n"
    )

    lines.append("## Deadline and pile-up checks\n")
    deadline = payload["deadline_checks"]
    lines.append(
        f"- Trials: {deadline['n_trials']:,}; min RT {deadline['min_rt']:.3f} s; "
        f"max RT {deadline['max_rt']:.3f} s.\n"
        f"- Trials above 5 s: {deadline['n_above_5s']} "
        f"({deadline['proportion_above_5s']:.4%}); above 5.5 s: "
        f"{deadline['n_above_5_5s']}; above 6 s: {deadline['n_above_6s']}.\n"
        f"- Most common 0.01-s RT bin: {deadline['modal_rounded_rt']:.2f} s with "
        f"{deadline['modal_count']} trials ({deadline['modal_fraction']:.3%}), "
        f"versus {deadline['expected_count_at_modal_bin']:.1f} expected if RTs "
        "were spread evenly across distinct bins.\n"
    )
    if (
        deadline["n_above_5_5s"] <= 0.005 * deadline["n_trials"]
        and deadline["modal_fraction"] < 0.05
    ):
        lines.append(
            "There is no evidence of an unrecorded deadline or a response "
            "pile-up. The misfit is therefore not explained by truncated RTs.\n"
        )
    else:
        lines.append(
            "The tail or modal checks are unusual enough that a deadline or "
            "contaminant process cannot be ruled out from these data alone.\n"
        )

    lines.append("## Participant-level deviations\n")
    ps = payload["participant_summary"]
    lines.append(
        f"- q90 is overpredicted for {ps['q90_overpredicted_n']} of "
        f"{ps['n_participants']} participants and underpredicted for "
        f"{ps['q90_underpredicted_n']}.\n"
        f"- Mean q90 deviation: {ps['q90_mean_difference']:+.3f} s; median "
        f"{ps['q90_median_difference']:+.3f} s; RMSE "
        f"{ps['q90_rmse']:.3f} s.\n"
        f"- Worst overprediction: {ps['q90_worst_positive']:+.3f} s; worst "
        f"underprediction: {ps['q90_worst_negative']:+.3f} s.\n"
        f"- Observed-versus-predicted q90 correlation: "
        f"{ps['q90_correlation_observed_predicted']:.3f}; slope "
        f"{ps['q90_slope_predicted_on_observed']:.3f}.\n"
    )
    lines.append(
        "A slope far below 1 means the model compresses participant differences: "
        "predicted q90 varies much less across participants than observed q90. "
        "With a single boundary and non-decision time per study, the model has no "
        "participant-level parameter to explain stable individual differences in "
        "RT scale. The per-participant CSV gives the exact pattern.\n"
    )

    lines.append("## PIT calibration\n")
    cal = payload["pit_calibration"]["conditional"]
    lines.append(
        f"Conditional PIT mean = {cal['mean']:.4f}, KS = "
        f"{cal['ks_statistic']:.4f} (p = {cal['ks_pvalue']:.2g}), maximum "
        f"bin deviation = {cal['max_bin_deviation']:.4f}.\n"
    )

    lines.append("## Profile likelihood over boundary separation\n")
    prof = payload["profile"]
    se_text = (
        f"{prof['hessian_se_a']:.4f}" if prof.get("hessian_se_a") is not None else "n/a"
    )
    clustered_se = prof.get("clustered_se_a")
    clustered_text = (
        f"{clustered_se:.4f}" if clustered_se is not None else "not available"
    )
    lines.append(
        f"- Hessian (trial-level) SE for a: {se_text}.\n"
        f"- Participant-clustered SE for a: {clustered_text}.\n"
        f"- Profile grid: a_hat +/- {prof['grid_span']:.4f}, "
        f"{prof['n_points']} points from {prof['grid_min']:.4f} to "
        f"{prof['grid_max']:.4f}.\n"
        f"- Full-fit log-likelihood: {prof['full_fit_log_likelihood']:.3f}.\n"
        f"- Maximum profile log-likelihood: "
        f"{prof['max_profile_log_likelihood']:.3f}.\n"
        f"- 95% profile interval for a: "
        f"[{prof['profile_ci_lower']:.4f}, {prof['profile_ci_upper']:.4f}].\n"
        f"- Interval bounded inside the grid: "
        f"{prof['profile_ci_bounded_by_grid']}.\n"
    )
    if (
        clustered_se is not None
        and prof["profile_ci_lower"] is not None
        and prof["profile_ci_upper"] is not None
    ):
        ratio = (prof["profile_ci_upper"] - prof["profile_ci_lower"]) / (
            2.0 * 1.96 * clustered_se
        )
        lines.append(
            "The profile interval is likelihood-based and treats trials as "
            "independent, so it inherits the naive precision rather than the "
            f"participant-clustered one: it is about {ratio:.2f} times the "
            "width of a Wald interval built from the clustered SE. The profile "
            "is reported here as a shape diagnostic (is the likelihood "
            "single-peaked near the estimate?), not as the study's uncertainty "
            "interval; the clustered SE is the interval to use.\n"
        )
    if not prof["profile_ci_bounded_by_grid"]:
        lines.append(
            "The profile interval reaches the edge of the searched grid, so the "
            "boundary separation is weakly identified by the RT tails alone. "
            "This is consistent with the tail misfit being structural rather "
            "than a single-parameter mis-estimate.\n"
        )
    else:
        lines.append(
            "The profile is single-peaked and sharply curved around the "
            "estimate, so the boundary separation is well identified. Moving "
            "a away from the estimate makes the overall fit worse, so the tail "
            "misfit is not a consequence of a suboptimal boundary estimate.\n"
        )

    lines.append("## Honest conclusion\n")
    lines.append(
        "The Lempert_2015 misfit is a genuine lack of fit of the classical DDM, "
        "not a numerical artefact. The model overpredicts the slow tail, shows "
        "PIT miscalibration, and predicts a small SS/LL RT difference with the "
        "opposite sign to the observed difference. Because constant drift makes "
        "choice and RT independent within a trial, the model can generate an "
        "SS/LL difference only through drift variation, which is too weak here. "
        "No deadline or pile-up explains the pattern. The next step should be a "
        "model with across-trial variability (drift, non-decision time, and/or "
        "start point), specified in advance and compared on held-out "
        "participants; it should not be a post hoc adjustment to this fit.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = ROOT / "results" / "lempert_2015"
    if "--report-only" in sys.argv[1:]:
        payload = json.loads(
            (out_dir / f"{PREFIX}_lempert_2015_investigation.json").read_text(
                encoding="utf-8"
            )
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(_build_markdown(payload), encoding="utf-8")
        print(
            "Lempert report rebuilt from "
            f"{out_dir / (PREFIX + '_lempert_2015_investigation.json')}"
        )
        return

    config_path = Path(__file__).with_name("config.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    from src.data.clean import add_predictors, clean_trials, select_primary
    from src.data.load import (
        load_trials,
        resolve_configured_path,
        verify_dataset,
    )

    data_path = resolve_configured_path(cfg["data"]["path"], ROOT)
    verify_dataset(data_path, check_hash=True)
    raw = load_trials(data_path)
    cleaned, _ = clean_trials(raw, cfg["cleaning"])
    primary = select_primary(cleaned)
    df = primary[primary["paper"] == STUDY].copy()
    df, _ = add_predictors(df)
    fit = fit_classical_ddm(
        df,
        STUDY,
        n_starts=int(cfg["fitting"]["n_starts"]),
        n_terms=int(cfg["model"]["n_terms"]),
        maxiter=int(cfg["fitting"]["maxiter"]),
    )
    run_investigation(
        df,
        fit,
        out_dir,
        n_terms=int(cfg["model"]["n_terms"]),
        profile_points=int(cfg["lempert"]["profile_points"]),
    )
    print(f"Lempert investigation written to {out_dir}")


if __name__ == "__main__":
    main()
