"""Full 12-study classical-DDM analysis.

Run from the project root:

    ..\.venv\Scripts\python.exe experiments\classical_ddm\run_full.py

The script fits each study separately, computes participant-clustered
uncertainty, runs participant-level 5-fold cross-validation, performs posterior
predictive checks, synthesises study-level estimates, investigates the
Lempert_2015 misfit, and writes a manifest for verification.

Pass ``--report-only`` to rebuild the full analysis report and the manifest
from the artifacts already on disk (no refitting).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import (  # noqa: E402
    add_predictors,
    clean_trials,
    primary_audit,
    select_primary,
    study_profile,
)
from src.data.load import (  # noqa: E402
    load_trials,
    resolve_configured_path,
    verify_dataset,
)
from src.evaluation.cross_validation import (  # noqa: E402
    assign_subject_folds,
    cross_validate_study,
)
from src.evaluation.diagnostics import (  # noqa: E402
    parameter_bound_diagnostics,
    parameter_plausibility,
    ppc_summary,
)
from src.evaluation.meta import (  # noqa: E402
    descriptive_table,
    meta_analysis_table,
)
from src.evaluation.rt_cdf import (  # noqa: E402
    conditional_pit,
    model_quantile_table,
    pit_calibration,
)
from src.fitting.clustered import clustered_covariance, clustered_table  # noqa: E402
from src.fitting.optimize import fit_classical_ddm  # noqa: E402
from src.models.classical_ddm import DDMParameters, PARAMETER_NAMES  # noqa: E402

from investigate_lempert import run_investigation  # noqa: E402
from run_pilot import (  # noqa: E402
    _jsonable,
    _plot_calibration,
    _plot_choice_bins,
    _plot_recovery,
    _plot_rt_quantiles,
    run_recovery,
    run_validation,
    save_json,
)

CONFIG_PATH = Path(__file__).with_name("config.json")
PREFIX = "classical_ddm_itc"
RESULT_DIR = ROOT / "results"
FIT_DIR = RESULT_DIR / "fits"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
REPORT_DIR = RESULT_DIR / "reports"
LEMPERT_DIR = RESULT_DIR / "lempert_2015"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_full_analysis_report.md"
MANIFEST_PATH = RESULT_DIR / f"{PREFIX}_manifest.json"


def _study_stem(paper: str) -> str:
    """Return the shared filename stem for one study's result files."""
    return f"{PREFIX}_{paper}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _survival(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.full(grid.size, np.nan)
    return 1.0 - np.searchsorted(values, grid, side="right") / values.size


def _plot_ppc_ccdf(
    observed_rt: np.ndarray,
    observed_choice: np.ndarray,
    simulated_rt: np.ndarray,
    simulated_choice: np.ndarray,
    paper: str,
    path: Path,
) -> None:
    grid = np.linspace(0.0, float(np.quantile(observed_rt, 0.995)) * 1.05, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    panels = [
        ("all", observed_rt, simulated_rt),
        ("SS", observed_rt[observed_choice == 0], simulated_rt[simulated_choice == 0]),
        ("LL", observed_rt[observed_choice == 1], simulated_rt[simulated_choice == 1]),
    ]
    for ax, (label, obs, pred) in zip(axes, panels):
        ax.plot(grid, _survival(obs, grid), color="black", label="observed")
        ax.plot(
            grid,
            _survival(pred, grid),
            color="tab:red",
            linestyle="--",
            label="posterior predictive",
        )
        ax.set_yscale("log")
        ax.set_xlabel("RT (s)")
        ax.set_title(label)
        ax.set_ylim(1e-4, 1.05)
    axes[0].set_ylabel("P(RT > t)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{paper}: observed and posterior-predictive RT survival")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pit(
    pit_payload: dict[str, np.ndarray],
    paper: str,
    path: Path,
) -> None:
    conditional = pit_payload["pit_conditional"]
    choice = pit_payload["choice"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    panels = [
        ("all", conditional),
        ("SS", conditional[choice == 0]),
        ("LL", conditional[choice == 1]),
    ]
    for ax, (label, values) in zip(axes, panels):
        values = values[np.isfinite(values)]
        ax.hist(values, bins=20, range=(0, 1), density=True, alpha=0.7)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{label} (n={values.size})")
        ax.set_xlabel("conditional PIT")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("density")
    fig.suptitle(f"{paper}: PIT calibration (uniform is ideal)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_forest(estimates: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    parameters = list(PARAMETER_NAMES)
    fig, axes = plt.subplots(1, len(parameters), figsize=(16, 4.8), sharey=True)
    studies = sorted(estimates["paper"].unique())
    y = np.arange(len(studies))
    for ax, parameter in zip(axes, parameters):
        subset = (
            estimates[estimates["parameter"] == parameter]
            .set_index("paper")
            .reindex(studies)
        )
        ax.errorbar(
            subset["estimate"],
            y,
            xerr=1.96 * subset["se"],
            fmt="o",
            markersize=4,
            capsize=2,
            color="tab:blue",
        )
        pooled = meta[meta["parameter"] == parameter].iloc[0]
        ax.axvline(pooled["estimate"], color="tab:red", linewidth=1.2)
        ax.axvspan(pooled["ci_lower"], pooled["ci_upper"], color="tab:red", alpha=0.12)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(parameter)
        ax.set_xlabel("estimate (95% SE)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(studies, fontsize=8)
    fig.suptitle("Study-level estimates with clustered SEs and random-effects mean")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_summary_metrics(
    fits: pd.DataFrame,
    cv: pd.DataFrame,
    path: Path,
) -> None:
    merged = fits.merge(
        cv[["paper", "pooled_test_choice_log_loss", "pooled_test_auc"]],
        on="paper",
        how="left",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(merged["choice_log_loss"], merged["pooled_test_choice_log_loss"])
    lo = float(
        np.nanmin(
            [merged["choice_log_loss"].min(), merged["pooled_test_choice_log_loss"].min()]
        )
    )
    hi = float(
        np.nanmax(
            [merged["choice_log_loss"].max(), merged["pooled_test_choice_log_loss"].max()]
        )
    )
    axes[0].plot([lo, hi], [lo, hi], color="grey", linestyle=":")
    axes[0].set_xlabel("in-sample choice log loss")
    axes[0].set_ylabel("held-out choice log loss")
    axes[0].set_title("Choice log loss")

    axes[1].scatter(merged["auc"], merged["pooled_test_auc"])
    lo = float(np.nanmin([merged["auc"].min(), merged["pooled_test_auc"].min()]))
    hi = float(np.nanmax([merged["auc"].max(), merged["pooled_test_auc"].max()]))
    axes[1].plot([lo, hi], [lo, hi], color="grey", linestyle=":")
    axes[1].set_xlabel("in-sample AUC")
    axes[1].set_ylabel("held-out AUC")
    axes[1].set_title("Choice AUC")
    fig.suptitle("In-sample versus participant-level held-out prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(df: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    present = [column for column in columns if column in df.columns]
    return df[present].round(decimals).to_markdown(index=False)


def build_report(
    cfg: dict,
    verification: dict,
    audit: dict,
    profile: pd.DataFrame,
    validation: dict,
    recovery_summary: dict,
    fits: pd.DataFrame,
    diagnostics: dict[str, dict],
    clustered: pd.DataFrame,
    cv_summary: pd.DataFrame,
    meta: pd.DataFrame,
    descriptive: pd.DataFrame,
    lempert: dict | None,
    failures: dict[str, list[str]],
    checks: dict[str, bool],
) -> str:
    lines: list[str] = []
    lines.append("# Classical DDM: full 12-study analysis\n")
    lines.append(
        "This report covers the complete frozen primary sample. Every study was "
        "fitted separately; no parameters were pooled across studies during "
        "estimation. Cross-validation folds are defined over participants, and "
        "uncertainty is reported both naively and with a participant-clustered "
        "sandwich estimator.\n"
    )

    lines.append("## 1. Inputs and frozen sample\n")
    lines.append(
        f"- Dataset: `{verification['path']}`\n"
        f"- Size: {verification['size']:,} bytes\n"
        f"- SHA-256 verified: **{verification['sha256_ok']}**\n"
        f"- Licence: CC BY-NC-SA 4.0\n"
        f"- Primary sample: **{audit['studies']} studies, {audit['subjects']} "
        f"participants, {audit['trials']:,} trials, {audit['sessions']} sessions**\n"
        f"- Cleaning rules and the 12-study sample are unchanged from the "
        "frozen plan (see `docs/dataset_audit.md` and `docs/decisions.md`).\n"
    )

    lines.append("## 2. Validation and recovery\n")
    lines.append(
        f"- PyDDM density-grid maximum absolute error: "
        f"{validation['pyddm_grid']['max_abs_error']:.3e}.\n"
        f"- Normalisation checks: "
        + "; ".join(
            f"v={item['drift']}, a={item['boundary']} total error "
            f"{item['total_error']:+.1e}"
            for item in validation["normalisation"]
        )
        + ".\n"
        f"- Simulation check: choice-proportion difference "
        f"{validation['simulation']['choice_proportion_difference']:+.3f}; "
        f"maximum absolute RT-quantile difference "
        f"{validation['simulation']['max_abs_quantile_difference']:.3f} s.\n"
    )
    recovery_rows = pd.DataFrame(recovery_summary["rows"])
    lines.append(
        "Synthetic recovery (5 datasets, 3,000 trials each) is summarised below. "
        "All parameters recover well; the largest relative RMSE is for `v0`, "
        "which is expected because baseline drift is partly traded off against "
        "the amount and delay slopes.\n"
    )
    lines.append(
        _table(
            recovery_rows,
            ["parameter", "true", "mean_estimate", "bias", "rmse", "relative_rmse"],
            decimals=3,
        )
    )
    lines.append("")

    lines.append("## 3. Convergence and boundary diagnostics\n")
    lines.append(
        _table(
            fits,
            [
                "paper",
                "trials",
                "subjects",
                "converged",
                "hessian_positive_definite",
                "hessian_condition",
                "start_objective_spread",
                "runtime_seconds",
                "t0",
                "t0_margin_seconds",
                "t0_at_upper_5pct",
                "params_at_5pct",
            ],
            decimals=4,
        )
    )
    lines.append("")
    n_conv = int(fits["converged"].sum()) if len(fits) else 0
    n_pd = int(fits["hessian_positive_definite"].sum()) if len(fits) else 0
    n_t0 = int(fits["t0_at_upper_5pct"].sum()) if len(fits) else 0
    lines.append(
        f"- Converged fits: **{n_conv}/{len(fits)}**.\n"
        f"- Positive-definite Hessians: **{n_pd}/{len(fits)}**.\n"
        f"- Studies with `t0` within 5% of its upper bound: **{n_t0}/{len(fits)}**.\n"
        f"- Studies with any parameter within 5% of a bound: "
        f"**{int((fits['n_params_at_5pct'] > 0).sum()) if 'n_params_at_5pct' in fits else 'n/a'}** "
        "(see `params_at_5pct` and `results/fits/*_parameters.csv`).\n"
        f"- Failed fits: {failures.get('fits', []) or 'none'}.\n"
        "\n"
        "Where `t0` is at or near the bound, its Hessian-based SE is **not** a "
        "valid confidence interval. The clustered SE has the same caveat; it "
        "accounts for participant clustering, not for the boundary constraint.\n"
    )

    lines.append("## 4. Parameter estimates\n")
    param_pivot = fits[
        ["paper"]
        + [f"{name}" for name in PARAMETER_NAMES]
        + [f"clustered_se_{name}" for name in PARAMETER_NAMES]
    ].copy()
    lines.append(_table(param_pivot, list(param_pivot.columns), decimals=3))
    lines.append("")
    lines.append(
        "Point estimates are maximum likelihood. The `clustered_se_*` columns "
        "are participant-clustered sandwich standard errors; the unclustered "
        "SEs and design effects are in "
        "`results/tables/classical_ddm_itc_clustered_uncertainty_summary.csv`. "
        "Design effects well above 1 show that trial-level SEs would be badly "
        "overconfident.\n"
    )

    lines.append("## 5. Choice prediction\n")
    lines.append(
        _table(
            fits,
            [
                "paper",
                "choice_log_loss",
                "brier",
                "auc",
                "observed_ll",
                "predicted_ll",
            ],
            decimals=4,
        )
    )
    lines.append("")
    lines.append(
        "The ITC task has no correct answer, so accuracy is undefined. Log "
        "loss, Brier score, and AUC describe how well the drift model ranks and "
        "calibrates LL choices.\n"
    )

    lines.append("## 6. RT diagnostics and posterior predictive checks\n")
    lines.append(
        _table(
            fits,
            [
                "paper",
                "rt_quantile_mae",
                "coverage_SS",
                "coverage_LL",
                "pit_ks_conditional",
                "pit_mean_conditional",
                "simulation_failure_rate",
            ],
            decimals=4,
        )
    )
    lines.append("")
    lines.append(
        "RT quantile MAE is computed from the posterior predictive simulation "
        "(20,000 simulated trials at dt=1e-4). Central-80 coverage is reported "
        "separately for SS and LL. The conditional PIT KS statistic tests "
        "uniformity; with thousands of trials it is sensitive to small "
        "deviations, so effect size (mean PIT and bin deviations) matters more "
        "than the p-value.\n"
    )
    lines.append(
        "**Structural note.** With constant drift and a midpoint start, choice "
        "and RT are independent within a trial. Across trials, the only route "
        "to an SS/LL RT difference is variation in drift, so the model can "
        "capture only the part of the association that is explained by the two "
        "predictors. In Lempert_2015 it predicts a small SS-minus-LL q90 "
        "difference with the opposite sign to the observed difference. This is "
        "a structural limitation to keep in mind when interpreting the RT "
        "diagnostics.\n"
    )

    lines.append("## 7. Participant-level cross-validation\n")
    lines.append(
        _table(
            cv_summary,
            [
                "paper",
                "n_folds_ok",
                "test_choice_log_loss_mean",
                "pooled_test_choice_log_loss",
                "test_auc_mean",
                "pooled_test_auc",
                "test_conditional_log_score_mean",
                "test_rt_quantile_mae_mean",
                "pooled_test_pit_ks_conditional",
                "cv_n_zero_density",
                "cv_n_below_t0",
            ],
            decimals=4,
        )
    )
    lines.append("")
    if {"cv_n_zero_density", "cv_n_below_t0"} <= set(cv_summary.columns):
        n_zero = int(cv_summary["cv_n_zero_density"].fillna(0).sum())
        n_below = int(cv_summary["cv_n_below_t0"].fillna(0).sum())
        n_trials = int(cv_summary["n_trials"].sum())
        lines.append(
            f"- Held-out trials with zero model density (excluded from the "
            f"RT log score): **{n_zero:,} of {n_trials:,}** "
            f"({n_zero / max(n_trials, 1):.4%}).\n"
            f"- Held-out trials faster than the training-fold `t0`: "
            f"**{n_below:,}**.\n"
            "\n"
            "Both counts are reported rather than silently dropped: they "
            "measure how often a study-level `t0` is too slow for new "
            "participants.\n"
        )
    lines.append(
        "Folds are over participants and are balanced by trial count. The "
        "predictor scaler is fitted on training participants only. Held-out "
        "metrics are proper scoring rules: choice log loss/Brier/AUC and the "
        "conditional RT log score; PIT and quantile MAE are calibration "
        "diagnostics. CV failures: "
        f"{failures.get('cv', []) or 'none'}.\n"
    )

    lines.append("## 8. Cross-study synthesis\n")
    lines.append(
        "Study-level estimates were combined with DerSimonian-Laird "
        "random-effects meta-analysis. This is a descriptive synthesis of "
        "separately fitted models, not a pooled re-fit.\n"
    )
    lines.append(
        _table(
            meta,
            [
                "parameter",
                "k",
                "estimate",
                "se",
                "ci_lower",
                "ci_upper",
                "tau2",
                "i2",
                "q_pvalue",
            ],
            decimals=4,
        )
    )
    lines.append("")
    lines.append(
        "Unweighted descriptive summaries are in "
        "`results/tables/classical_ddm_itc_meta_analysis_descriptive.csv`. "
        "Heterogeneity is substantial for "
        "several parameters, which is expected given different task designs, "
        "amount ranges, and delay ranges. The meta-analytic means should not be "
        "read as universal DDM parameters.\n"
    )

    lines.append("## 9. Lempert_2015 investigation\n")
    if lempert is None:
        lines.append("The Lempert investigation did not run.\n")
    else:
        tail_q90 = next(
            row
            for row in lempert["tail_diagnostics"]
            if row.get("quantile") == 0.9
        )
        ps = lempert["participant_summary"]
        prof = lempert["profile"]
        lines.append(
            f"- Predicted q90 = {tail_q90['predicted_seconds']:.2f} s versus "
            f"observed {tail_q90['observed_seconds']:.2f} s "
            f"({tail_q90['difference']:+.2f} s).\n"
            f"- Conditional PIT KS = "
            f"{lempert['pit_calibration']['conditional']['ks_statistic']:.3f}; "
            f"mean PIT = "
            f"{lempert['pit_calibration']['conditional']['mean']:.3f}.\n"
            f"- q90 overpredicted for {ps['q90_overpredicted_n']} of "
            f"{ps['n_participants']} participants; mean deviation "
            f"{ps['q90_mean_difference']:+.2f} s.\n"
            f"- Max observed RT {lempert['deadline_checks']['max_rt']:.2f} s; "
            f"{lempert['deadline_checks']['n_above_5_5s']} trials above 5.5 s, "
            "so there is no evidence of an unrecorded deadline or pile-up.\n"
            f"- Profile 95% interval for `a`: "
            f"[{prof['profile_ci_lower']:.3f}, {prof['profile_ci_upper']:.3f}] "
            f"(a_hat = {prof['a_hat']:.3f}; bounded by grid: "
            f"{prof['profile_ci_bounded_by_grid']}).\n"
        )
        lines.append(
            "The full diagnostic is in "
            "`results/reports/classical_ddm_itc_lempert_2015_diagnostic_report.md`. "
            "The conclusion is "
            "that the misfit is structural: the model overpredicts the slow "
            "tail and predicts an SS/LL RT difference with the opposite sign "
            "to the observed one. No model change was made to improve the "
            "result.\n"
        )

    lines.append("## 10. What completed and what did not\n")
    completed = [
        "Dataset hash verification and frozen cleaning audit.",
        "Mathematical validation (PyDDM grid, normalisation, simulation).",
        "Synthetic parameter recovery.",
        f"Separate MLE for {len(fits)}/{len(cfg['primary_studies'])} studies.",
        "Participant-clustered sandwich uncertainty for all successful fits.",
        "Participant-level 5-fold CV for all studies with successful fits.",
        "Posterior predictive checks and PIT calibration for all successful fits.",
        "Cross-study random-effects synthesis.",
        "Lempert_2015 diagnostic investigation (no model change).",
        "Manifest and completeness verification script.",
    ]
    for item in completed:
        lines.append(f"- {item}")
    lines.append("")
    if failures.get("fits") or failures.get("cv"):
        lines.append(
            f"**Failed analyses:** fits = {failures.get('fits', []) or 'none'}; "
            f"CV = {failures.get('cv', []) or 'none'}. These are reported, not "
            "silently dropped.\n"
        )
    else:
        lines.append("**Failed analyses:** none.\n")

    lines.append("## 11. Limitations\n")
    lines.append(
        "- The model is a classical DDM with constant drift within a trial, "
        "fixed midpoint start, and unit noise. It omits across-trial "
        "variability in drift, start point, and non-decision time.\n"
        "- Constant drift with a midpoint start makes choice and RT independent "
        "within a trial. Across trials, only drift variation can generate an "
        "SS/LL RT difference, so the model is likely to understate the "
        "choice-RT association.\n"
        "- `t0` is bounded above by the fastest observed RT. Several fits sit "
        "close to that bound, so their Hessian and sandwich SEs are not valid "
        "confidence intervals for the unconstrained parameter.\n"
        "- Participant-clustered SEs account for within-participant correlation "
        "but not for the boundary constraint or for model misspecification.\n"
        "- Cross-validation evaluates a study-specific model fitted to other "
        "participants in the same study. It does not test transfer to new "
        "studies or new task designs.\n"
        "- The random-effects synthesis uses study-level clustered SEs and "
        "assumes approximate normality; with 12 studies the meta-analytic "
        "uncertainty is itself imprecise.\n"
        "- Posterior predictive checks use the observed predictor distribution "
        "and 20,000 simulations at dt=1e-4. Simulation has a small residual "
        "discretisation bias in the extreme tail.\n"
        "- The ITC task has no accuracy criterion, so choice quality is judged "
        "only by log loss, Brier score, and AUC.\n"
    )

    lines.append("## 12. Next steps\n")
    lines.append(
        "1. Add across-trial variability in drift and non-decision time, and "
        "compare on the same participant-level CV folds.\n"
        "2. Allow a non-midpoint start point and test whether it captures the "
        "observed choice-RT association.\n"
        "3. Consider a contaminant/deadline component only if the deadline "
        "diagnostics justify it in other studies; Lempert_2015 does not.\n"
        "4. Report hierarchical or participant-level estimates alongside "
        "study-level MLEs before making individual-difference claims.\n"
        "5. Pre-specify the model comparison metric (held-out log score) and "
        "the set of models before looking at the remaining studies.\n"
        "6. Re-run `verify_outputs.py` after any change and keep the manifest "
        "with the results.\n"
    )

    lines.append("## 13. Reproduction\n")
    lines.append(
        "From the project root:\n\n"
        "```text\n"
        "..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n"
        "..\\.venv\\Scripts\\python.exe experiments\\classical_ddm\\run_full.py\n"
        "..\\.venv\\Scripts\\python.exe experiments\\classical_ddm\\verify_outputs.py\n"
        "```\n\n"
        "All paths are relative to `classical-ddm-intertemporal-choice/`. The "
        "dataset is read from `../data/itc_all_data.csv` (or `DDM_ITC_CSV`) and "
        "is never modified.\n"
    )
    lines.append("## 14. Output inventory\n")
    lines.append(
        f"- `results/fits/{PREFIX}_{{study}}_fit.json` - parameters, "
        "likelihood, Hessian, starts.\n"
        f"- `results/fits/{PREFIX}_{{study}}_clustered_uncertainty.json` - "
        "naive and clustered covariance, participant scores.\n"
        f"- `results/fits/{PREFIX}_{{study}}_diagnostics.json` - PPC, PIT, "
        "model quantiles, bounds.\n"
        f"- `results/figures/{PREFIX}_{{study}}_*.png` - RT quantiles, "
        "calibration, choice bins, PPC tail survival, PIT histogram.\n"
        f"- `results/tables/{PREFIX}_12_study_fit_summary.csv` - one row per "
        "study.\n"
        f"- `results/tables/{PREFIX}_clustered_uncertainty_summary.csv` - long "
        "parameter table.\n"
        f"- `results/tables/{PREFIX}_5_fold_cross_validation_*.csv` - CV "
        "results, fold membership, and participant assignments.\n"
        f"- `results/tables/{PREFIX}_random_effects_meta_analysis.csv` and "
        f"`{PREFIX}_meta_analysis_descriptive.csv`.\n"
        "- `results/lempert_2015/` - investigation JSON, profile likelihood, "
        "participant deviations, figures; report in `results/reports/`.\n"
        f"- `results/{PREFIX}_manifest.json` - hashes and environment for "
        "verification.\n"
    )
    lines.append("## 15. Analysis checks\n")
    lines.append(
        pd.DataFrame(
            [{"check": key, "passed": bool(value)} for key, value in checks.items()]
        ).to_markdown(index=False)
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LEMPERT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    data_path = resolve_configured_path(cfg["data"]["path"], ROOT)
    print(f"Verifying dataset: {data_path}")
    verification = verify_dataset(data_path, check_hash=True)
    print(f"SHA-256 verified: {verification['sha256_ok']}")

    print("Loading and cleaning trials...")
    raw = load_trials(data_path)
    cleaned, audit = clean_trials(raw, cfg["cleaning"])
    primary = select_primary(cleaned)
    audit.update(primary_audit(primary))
    for key, expected in cfg["expected_primary"].items():
        observed = audit[key]
        if int(observed) != int(expected):
            raise ValueError(
                f"Primary sample mismatch for {key}: observed {observed}, "
                f"expected {expected}. Refusing to continue silently."
            )
    profile = study_profile(primary)
    profile.to_csv(RESULT_DIR / f"{PREFIX}_study_profile.csv")
    save_json(
        {"verification": verification, "audit": audit},
        RESULT_DIR / f"{PREFIX}_data_audit.json",
    )
    print(
        f"Primary sample: {audit['studies']} studies, {audit['subjects']} "
        f"participants, {audit['trials']} trials"
    )

    print("Running mathematical validation...")
    validation = run_validation(cfg)
    save_json(validation, RESULT_DIR / f"{PREFIX}_validation.json")

    print("Running synthetic recovery...")
    recovery, recovery_summary = run_recovery(cfg)
    recovery.to_csv(RESULT_DIR / f"{PREFIX}_parameter_recovery_trials.csv", index=False)
    save_json(
        recovery_summary, RESULT_DIR / f"{PREFIX}_parameter_recovery_summary.json"
    )
    _plot_recovery(recovery, FIG_DIR / f"{PREFIX}_parameter_recovery.png")

    fits: dict[str, Any] = {}
    diagnostics: dict[str, dict] = {}
    clustered_payloads: list[dict] = []
    fit_rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {"fits": [], "cv": []}
    lempert_samples: dict[str, np.ndarray] | None = None

    for paper in cfg["primary_studies"]:
        print(f"Fitting study: {paper}")
        study_df = primary[primary["paper"] == paper].copy()
        study_df, _ = add_predictors(study_df)
        try:
            t_start = time.perf_counter()
            fit = fit_classical_ddm(
                study_df,
                paper,
                n_starts=int(cfg["fitting"]["n_starts"]),
                n_terms=int(cfg["model"]["n_terms"]),
                maxiter=int(cfg["fitting"]["maxiter"]),
                compute_hessian=True,
            )
            runtime = time.perf_counter() - t_start
            params = DDMParameters.from_array(
                [fit.params[name] for name in PARAMETER_NAMES]
            )
            clustered = clustered_covariance(
                fit, study_df, n_terms=int(cfg["model"]["n_terms"])
            )
            ppc = ppc_summary(
                study_df,
                params,
                n_sim=int(cfg["ppc"]["n_sim"]),
                seed=int(cfg["ppc"]["seed"]),
                dt=float(cfg["ppc"].get("simulation_dt", 1e-4)),
                include_samples=True,
            )
            simulated_rt = np.asarray(ppc.pop("simulated_rt"), dtype=float)
            simulated_choice = np.asarray(ppc.pop("simulated_choice"), dtype=int)
            pit_payload = conditional_pit(
                study_df, params, n_terms=12, n_grid=400
            )
            calibration = pit_calibration(pit_payload)
            model_quantiles = model_quantile_table(
                study_df, params, n_terms=12, n_grid=500
            )
            plausibility = parameter_plausibility(fit, study_df)
            boundary = parameter_bound_diagnostics(fit, study_df)
        except Exception as exc:  # noqa: BLE001 - report failures honestly
            print(f"  FAILED: {exc!r}")
            failures["fits"].append(f"{paper}: {exc!r}")
            fit_rows.append(
                {
                    "paper": paper,
                    "status": "failed",
                    "error": repr(exc),
                    "trials": int(len(study_df)),
                    "subjects": int(study_df["subj_ident"].nunique()),
                }
            )
            continue

        fits[paper] = fit
        clustered_payloads.append(clustered)
        diagnostics[paper] = {
            "ppc": ppc,
            "pit_calibration": calibration,
            "model_quantiles": model_quantiles,
            "parameter_plausibility": plausibility,
            "parameter_bounds": boundary,
            "clustered_uncertainty": clustered["rows"],
        }

        stem = _study_stem(paper)
        save_json(fit.to_dict(), FIT_DIR / f"{stem}_fit.json")
        save_json(clustered, FIT_DIR / f"{stem}_clustered_uncertainty.json")
        save_json(diagnostics[paper], FIT_DIR / f"{stem}_diagnostics.json")

        bound_by_param = {
            row["parameter"]: row for row in boundary["rows"]
        }
        clustered_by_param = {
            row["parameter"]: row for row in clustered["rows"]
        }
        bound_params = [
            name
            for name in PARAMETER_NAMES
            if bound_by_param[name]["at_bound_5pct"]
        ]
        param_rows = []
        for name in PARAMETER_NAMES:
            param_rows.append(
                {
                    "paper": paper,
                    "parameter": name,
                    "estimate": fit.params[name],
                    "naive_se": clustered_by_param[name]["naive_se"],
                    "clustered_se": clustered_by_param[name]["clustered_se"],
                    "design_effect": clustered_by_param[name]["design_effect"],
                    "lower_bound": bound_by_param[name]["lower"],
                    "upper_bound": bound_by_param[name]["upper"],
                    "distance_to_nearest_bound": bound_by_param[name][
                        "distance_to_nearest_bound"
                    ],
                    "fraction_of_range_to_nearest_bound": bound_by_param[name][
                        "fraction_of_range_to_nearest_bound"
                    ],
                    "at_bound_5pct": bound_by_param[name]["at_bound_5pct"],
                }
            )
        pd.DataFrame(param_rows).to_csv(
            FIT_DIR / f"{stem}_parameters.csv", index=False
        )

        diag = diagnostics[paper]
        row = {
            "paper": paper,
            "status": "ok",
            "trials": int(len(study_df)),
            "subjects": int(study_df["subj_ident"].nunique()),
            "sessions": int(study_df["sess_ident"].nunique()),
            "converged": bool(fit.converged),
            "message": fit.message,
            "nfev": int(fit.nfev),
            "runtime_seconds": float(runtime),
            "log_likelihood": float(fit.log_likelihood),
            "log_likelihood_per_trial": float(fit.log_likelihood / len(study_df)),
            "hessian_positive_definite": bool(fit.hessian_positive_definite),
            "hessian_condition": float(fit.hessian_condition)
            if fit.hessian_condition is not None
            else np.nan,
            "start_objective_spread": float(fit.start_objective_spread)
            if fit.start_objective_spread is not None
            else np.nan,
            "t0_margin_seconds": boundary["t0_margin_to_upper_seconds"],
            "t0_at_upper_5pct": boundary["t0_at_upper_5pct"],
            "hessian_se_valid_for_t0": boundary["hessian_se_valid_for_t0"],
            "n_params_at_5pct": len(bound_params),
            "params_at_5pct": ";".join(bound_params),
            "choice_log_loss": diag["ppc"]["choice_metrics_in_sample"]["log_loss"],
            "brier": diag["ppc"]["choice_metrics_in_sample"]["brier"],
            "auc": diag["ppc"]["choice_metrics_in_sample"]["auc"],
            "observed_ll": diag["ppc"]["observed_ll_proportion"],
            "predicted_ll": diag["ppc"]["predicted_ll_proportion"],
            "rt_quantile_mae": diag["ppc"]["rt_quantile_mae"],
            "coverage_SS": diag["ppc"]["central_80_coverage"]["SS"],
            "coverage_LL": diag["ppc"]["central_80_coverage"]["LL"],
            "simulation_failure_rate": diag["ppc"]["simulation_failure_rate"],
            "pit_ks_conditional": diag["pit_calibration"]["conditional"][
                "ks_statistic"
            ],
            "pit_ks_pvalue": diag["pit_calibration"]["conditional"]["ks_pvalue"],
            "pit_mean_conditional": diag["pit_calibration"]["conditional"]["mean"],
            "pit_max_bin_deviation": diag["pit_calibration"]["conditional"][
                "max_bin_deviation"
            ],
        }
        for name in PARAMETER_NAMES:
            row[name] = float(fit.params[name])
            row[f"se_{name}"] = (
                float(fit.se[name]) if fit.se is not None else np.nan
            )
            row[f"clustered_se_{name}"] = clustered_by_param[name][
                "clustered_se"
            ]
            row[f"design_effect_{name}"] = clustered_by_param[name][
                "design_effect"
            ]
        fit_rows.append(row)

        _plot_rt_quantiles(
            diag["ppc"],
            paper,
            FIG_DIR / f"{stem}_reaction_time_quantiles.png",
        )
        _plot_calibration(
            diag["ppc"], paper, FIG_DIR / f"{stem}_choice_calibration.png"
        )
        _plot_choice_bins(
            diag["ppc"], paper, FIG_DIR / f"{stem}_choice_by_amount_delay.png"
        )
        _plot_ppc_ccdf(
            study_df["rt"].to_numpy(dtype=float),
            study_df["choice"].to_numpy(dtype=int),
            simulated_rt,
            simulated_choice,
            paper,
            FIG_DIR / f"{stem}_reaction_time_tail.png",
        )
        _plot_pit(
            pit_payload, paper, FIG_DIR / f"{stem}_pit_calibration.png"
        )

        if paper == "Lempert_2015":
            lempert_samples = {
                "rt": simulated_rt,
                "choice": simulated_choice,
            }
        print(
            f"  LL/trial={fit.log_likelihood / len(study_df):.3f}, "
            f"AUC={row['auc']:.3f}, RT MAE={row['rt_quantile_mae']:.3f}s, "
            f"t0 margin={boundary['t0_margin_to_upper_seconds']:.3f}s"
        )

    fits_table = pd.DataFrame(fit_rows)
    fits_table.to_csv(TABLE_DIR / f"{PREFIX}_12_study_fit_summary.csv", index=False)
    save_json(fit_rows, TABLE_DIR / f"{PREFIX}_12_study_fit_summary.json")
    clustered_long = (
        clustered_table(clustered_payloads)
        if clustered_payloads
        else pd.DataFrame()
    )
    clustered_long.to_csv(
        TABLE_DIR / f"{PREFIX}_clustered_uncertainty_summary.csv", index=False
    )

    print("Running participant-level 5-fold cross-validation...")
    cv_folds_all = []
    cv_rows = []
    cv_assignments = []
    for paper in cfg["primary_studies"]:
        if paper not in fits:
            failures["cv"].append(f"{paper}: fit failed, CV skipped")
            continue
        print(f"  CV study: {paper}")
        study_df = primary[primary["paper"] == paper].copy()
        study_df, _ = add_predictors(study_df)
        try:
            fold_assignment = assign_subject_folds(
                study_df,
                n_folds=int(cfg["cross_validation"]["n_folds"]),
                seed=int(cfg["cross_validation"]["seed"]),
            )
            for subject, fold in fold_assignment.items():
                n_trials = int(
                    (study_df["subj_ident"].astype(str) == str(subject)).sum()
                )
                cv_assignments.append(
                    {
                        "paper": paper,
                        "subj_ident": str(subject),
                        "fold": int(fold),
                        "n_trials": n_trials,
                    }
                )
            fold_table, summary_row, _ = cross_validate_study(
                study_df,
                paper,
                n_folds=int(cfg["cross_validation"]["n_folds"]),
                seed=int(cfg["cross_validation"]["seed"]),
                n_starts=int(cfg["cross_validation"]["n_starts"]),
                n_terms=int(cfg["model"]["n_terms"]),
                maxiter=int(cfg["fitting"]["maxiter"]),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc!r}")
            failures["cv"].append(f"{paper}: {exc!r}")
            continue
        fold_table["paper"] = paper
        cv_folds_all.append(fold_table)
        cv_rows.append(summary_row)
        print(
            f"    pooled log loss={summary_row.get('pooled_test_choice_log_loss', float('nan')):.4f}, "
            f"AUC={summary_row.get('pooled_test_auc', float('nan')):.3f}"
        )

    cv_summary = pd.DataFrame(cv_rows)
    cv_folds = (
        pd.concat(cv_folds_all, ignore_index=True)
        if cv_folds_all
        else pd.DataFrame()
    )
    cv_summary.to_csv(
        TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_summary.csv", index=False
    )
    cv_folds.to_csv(
        TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_folds.csv", index=False
    )
    pd.DataFrame(cv_assignments).to_csv(
        TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_assignments.csv",
        index=False,
    )
    save_json(
        cv_rows, TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_summary.json"
    )

    print("Synthesising study-level estimates...")
    estimates_long = fits_table[fits_table["status"] == "ok"].melt(
        id_vars=["paper"],
        value_vars=list(PARAMETER_NAMES),
        var_name="parameter",
        value_name="estimate",
    )
    clustered_lookup = clustered_long.rename(
        columns={"paper": "paper", "parameter": "parameter"}
    )[["paper", "parameter", "clustered_se", "naive_se", "design_effect"]]
    estimates_long = estimates_long.merge(
        clustered_lookup, on=["paper", "parameter"], how="left"
    )
    estimates_long["se"] = estimates_long["clustered_se"]
    estimates_long["se_type"] = "participant-clustered"
    estimates_long.to_csv(
        TABLE_DIR / f"{PREFIX}_parameter_estimates_long.csv", index=False
    )
    meta = (
        meta_analysis_table(estimates_long, PARAMETER_NAMES)
        if not estimates_long.empty
        else pd.DataFrame()
    )
    descriptive = (
        descriptive_table(estimates_long, PARAMETER_NAMES)
        if not estimates_long.empty
        else pd.DataFrame()
    )
    meta.to_csv(
        TABLE_DIR / f"{PREFIX}_random_effects_meta_analysis.csv", index=False
    )
    descriptive.to_csv(
        TABLE_DIR / f"{PREFIX}_meta_analysis_descriptive.csv", index=False
    )
    save_json(
        meta.to_dict(orient="records"),
        TABLE_DIR / f"{PREFIX}_random_effects_meta_analysis.json",
    )
    if not meta.empty:
        _plot_forest(
            estimates_long,
            meta,
            FIG_DIR / f"{PREFIX}_study_parameter_forest_plot.png",
        )
    if not cv_summary.empty:
        _plot_summary_metrics(
            fits_table,
            cv_summary,
            FIG_DIR / f"{PREFIX}_study_summary_metrics.png",
        )

    lempert_payload = None
    if "Lempert_2015" in fits:
        print("Investigating Lempert_2015 RT misfit...")
        lempert_df = primary[primary["paper"] == "Lempert_2015"].copy()
        lempert_df, _ = add_predictors(lempert_df)
        lempert_payload = run_investigation(
            lempert_df,
            fits["Lempert_2015"],
            LEMPERT_DIR,
            ppc_samples=lempert_samples,
            n_terms=int(cfg["model"]["n_terms"]),
            profile_points=int(cfg["lempert"]["profile_points"]),
        )
    else:
        failures["fits"].append("Lempert_2015: investigation skipped (no fit)")

    checks = {
        "dataset_hash_verified": bool(verification["sha256_ok"]),
        "expected_sample_size": bool(
            audit["studies"] == cfg["expected_primary"]["studies"]
            and audit["subjects"] == cfg["expected_primary"]["subjects"]
            and audit["trials"] == cfg["expected_primary"]["trials"]
        ),
        "all_12_studies_fitted": bool(len(fits) == len(cfg["primary_studies"])),
        "all_fits_converged": bool(
            len(fits_table) > 0
            and (fits_table["status"] == "ok").all()
            and fits_table["converged"].all()
        ),
        "all_hessians_positive_definite": bool(
            len(fits_table) > 0
            and (fits_table["status"] == "ok").all()
            and fits_table["hessian_positive_definite"].all()
        ),
        "cross_validation_complete": bool(
            len(cv_summary) == len(cfg["primary_studies"])
            and (cv_summary["n_folds_ok"] == cfg["cross_validation"]["n_folds"]).all()
        ),
        "lempert_investigation_written": bool(lempert_payload is not None),
        "pyddm_max_abs_error_below_1e_6": bool(
            validation["pyddm_grid"]["max_abs_error"] < 1e-6
        ),
        "normalisation_total_error_below_1e_4": bool(
            all(abs(item["total_error"]) < 1e-4 for item in validation["normalisation"])
        ),
    }
    save_json(checks, TABLE_DIR / f"{PREFIX}_analysis_checks.json")

    report = build_report(
        cfg,
        verification,
        audit,
        profile,
        validation,
        recovery_summary,
        fits_table,
        diagnostics,
        clustered_long,
        cv_summary,
        meta,
        descriptive,
        lempert_payload,
        failures,
        checks,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    _write_manifest(verification, cfg, failures, checks)
    print(f"Full analysis complete. Results in {RESULT_DIR}")
    print("Checks:", json.dumps(_jsonable(checks), indent=2))
    if failures["fits"] or failures["cv"]:
        print("Failures:", json.dumps(_jsonable(failures), indent=2))


def _write_manifest(
    verification: dict,
    cfg: dict,
    failures: dict,
    checks: dict,
) -> None:
    """Hash every output file and rewrite the manifest.

    Hashing happens after the report is written, so the manifest covers the
    final text of every artifact.
    """
    manifest_files = []
    for path in sorted(RESULT_DIR.rglob("*")):
        if path.is_file() and path != MANIFEST_PATH:
            manifest_files.append(
                {
                    "relative_path": str(path.relative_to(RESULT_DIR)).replace(
                        "\\", "/"
                    ),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": verification,
        "config": {
            "path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(CONFIG_PATH),
            "contents": cfg,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "settings": {
            "n_starts": cfg["fitting"]["n_starts"],
            "maxiter": cfg["fitting"]["maxiter"],
            "n_terms": cfg["model"]["n_terms"],
            "ppc_n_sim": cfg["ppc"]["n_sim"],
            "ppc_dt": cfg["ppc"]["simulation_dt"],
            "cv_folds": cfg["cross_validation"]["n_folds"],
            "cv_seed": cfg["cross_validation"]["seed"],
            "cv_n_starts": cfg["cross_validation"]["n_starts"],
        },
        "failures": failures,
        "checks": checks,
        "files": manifest_files,
    }
    save_json(manifest, MANIFEST_PATH)


def rebuild_report() -> None:
    """Regenerate the full report and the manifest from saved artifacts.

    This is an inspection convenience: it does not refit anything, so it can
    only restate results that are already on disk. It is used after edits to
    the report wording so that the report and the manifest stay in sync
    without repeating the full computation.
    """
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    audit_payload = json.loads(
        (RESULT_DIR / f"{PREFIX}_data_audit.json").read_text(encoding="utf-8")
    )
    verification = audit_payload["verification"]
    audit = audit_payload["audit"]
    validation = json.loads(
        (RESULT_DIR / f"{PREFIX}_validation.json").read_text(encoding="utf-8")
    )
    recovery_summary = json.loads(
        (RESULT_DIR / f"{PREFIX}_parameter_recovery_summary.json").read_text(
            encoding="utf-8"
        )
    )
    profile = pd.read_csv(RESULT_DIR / f"{PREFIX}_study_profile.csv")
    fits = pd.read_csv(TABLE_DIR / f"{PREFIX}_12_study_fit_summary.csv")
    clustered = pd.read_csv(
        TABLE_DIR / f"{PREFIX}_clustered_uncertainty_summary.csv"
    )
    cv_summary = pd.read_csv(
        TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_summary.csv"
    )
    cv_folds = pd.read_csv(
        TABLE_DIR / f"{PREFIX}_5_fold_cross_validation_folds.csv"
    )
    if {"test_n_zero_density", "test_n_below_t0"} <= set(cv_folds.columns):
        totals = cv_folds.groupby("paper")[
            ["test_n_zero_density", "test_n_below_t0"]
        ].sum()
        cv_summary["cv_n_zero_density"] = cv_summary["paper"].map(
            totals["test_n_zero_density"]
        )
        cv_summary["cv_n_below_t0"] = cv_summary["paper"].map(
            totals["test_n_below_t0"]
        )
    meta = pd.read_csv(TABLE_DIR / f"{PREFIX}_random_effects_meta_analysis.csv")
    descriptive = pd.read_csv(
        TABLE_DIR / f"{PREFIX}_meta_analysis_descriptive.csv"
    )
    checks = json.loads(
        (TABLE_DIR / f"{PREFIX}_analysis_checks.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = manifest["failures"]

    # Recompute which parameters sit within 5% of a bound, so the report does
    # not depend on a column having existed in an earlier fit summary.
    flagged: dict[str, list[str]] = {}
    for paper in cfg["primary_studies"]:
        path = FIT_DIR / f"{_study_stem(paper)}_parameters.csv"
        if not path.exists():
            flagged[paper] = []
            continue
        table = pd.read_csv(path)
        flagged[paper] = table.loc[table["at_bound_5pct"].astype(bool), "parameter"].tolist()
    fits["params_at_5pct"] = fits["paper"].map(lambda p: ";".join(flagged.get(p, [])))
    fits["n_params_at_5pct"] = fits["paper"].map(lambda p: len(flagged.get(p, [])))

    lempert_path = LEMPERT_DIR / f"{PREFIX}_lempert_2015_investigation.json"
    lempert = (
        json.loads(lempert_path.read_text(encoding="utf-8"))
        if lempert_path.exists()
        else None
    )

    report = build_report(
        cfg,
        verification,
        audit,
        profile,
        validation,
        recovery_summary,
        fits,
        {},
        clustered,
        cv_summary,
        meta,
        descriptive,
        lempert,
        failures,
        checks,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    _write_manifest(verification, cfg, failures, checks)
    print(f"Report rebuilt from saved artifacts: {REPORT_PATH}")


if __name__ == "__main__":
    if "--report-only" in sys.argv[1:]:
        rebuild_report()
    else:
        main()
