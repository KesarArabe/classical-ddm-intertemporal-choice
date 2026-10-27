"""Model-implied RT distributions, probability integral transforms, and quantiles.

All functions use the validated first-passage density in
``src.models.classical_ddm``. Conditional-on-choice CDFs are obtained by
numerically integrating the joint density and dividing by the analytic choice
probability. The integration grid is log-spaced in decision time so that the
sharp early density is resolved without an excessive number of points.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

from src.evaluation.metrics import quantile_mae
from src.models.classical_ddm import (
    DDMParameters,
    choice_probability,
    ddm_log_density,
    drift_from_predictors,
)

DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _as_parameters(params) -> DDMParameters:
    if isinstance(params, DDMParameters):
        return params
    return DDMParameters.from_array(params)


def _cdf_on_grid(
    drift: np.ndarray,
    boundary: float,
    grid: np.ndarray,
    n_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unconditional CDFs at ``grid`` for each drift value.

    ``drift`` has shape ``(n,)`` and ``grid`` has shape ``(m,)``. The returned
    arrays have shape ``(n, m)`` and give ``P(T <= t, choice = c)`` measured
    from the end of the non-decision time.
    """
    rt = np.asarray(grid, dtype=float)[None, :]
    v = np.asarray(drift, dtype=float)[:, None]
    log_upper, log_lower = ddm_log_density(
        rt, v, float(boundary), t0=0.0, n_terms=int(n_terms)
    )
    upper = np.exp(np.clip(log_upper, -745.0, 700.0))
    lower = np.exp(np.clip(log_lower, -745.0, 700.0))
    cdf_upper = cumulative_trapezoid(upper, x=grid, axis=1, initial=0.0)
    cdf_lower = cumulative_trapezoid(lower, x=grid, axis=1, initial=0.0)
    return cdf_upper, cdf_lower


def _interpolate_rows(
    grid: np.ndarray,
    values: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of one row of ``values`` at each ``point``."""
    out = np.empty(points.size, dtype=float)
    for i, point in enumerate(points):
        out[i] = np.interp(point, grid, values[i])
    return out


def _decision_grid(min_time: float, max_time: float, n_grid: int) -> np.ndarray:
    lower = max(float(min_time) * 0.5, 1e-8)
    upper = max(float(max_time) * 1.001, lower * 1.01)
    return np.geomspace(lower, upper, int(n_grid))


def conditional_pit(
    df: pd.DataFrame,
    params,
    n_terms: int = 12,
    n_grid: int = 400,
    chunk_size: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute conditional and marginal PIT values for observed responses.

    The conditional PIT is ``P(T <= rt | choice)`` under the fitted model. If
    the model is correct it is uniformly distributed on (0, 1), separately for
    each choice. The marginal PIT is ``P(T <= rt)`` ignoring choice.

    Trials with ``rt <= t0`` receive ``nan``; their count is returned so that
    they are reported rather than silently discarded.
    """
    p = _as_parameters(params)
    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    x_amount = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    drift = drift_from_predictors(p, x_amount, x_delay)
    decision_time = rt - p.t0

    n = rt.size
    pit_cond = np.full(n, np.nan, dtype=float)
    pit_marg = np.full(n, np.nan, dtype=float)
    valid = decision_time > 0
    if not np.any(valid):
        return {
            "pit_conditional": pit_cond,
            "pit_marginal": pit_marg,
            "choice": choice,
            "n_below_t0": int(n),
        }

    grid = _decision_grid(
        float(np.min(decision_time[valid])),
        float(np.max(decision_time[valid])),
        n_grid,
    )
    p_ll = np.clip(choice_probability(drift, p.a), 1e-12, 1.0 - 1e-12)

    valid_idx = np.flatnonzero(valid)
    for start in range(0, valid_idx.size, int(chunk_size)):
        idx = valid_idx[start : start + int(chunk_size)]
        cdf_upper, cdf_lower = _cdf_on_grid(
            drift[idx], p.a, grid, n_terms=n_terms
        )
        chosen = np.where(
            choice[idx][:, None] == 1, cdf_upper, cdf_lower
        )
        chosen_at = _interpolate_rows(grid, chosen, decision_time[idx])
        marginal_at = _interpolate_rows(
            grid, cdf_upper + cdf_lower, decision_time[idx]
        )
        denominator = np.where(choice[idx] == 1, p_ll[idx], 1.0 - p_ll[idx])
        pit_cond[idx] = chosen_at / denominator
        pit_marg[idx] = marginal_at

    pit_cond = np.clip(pit_cond, 0.0, 1.0)
    pit_marg = np.clip(pit_marg, 0.0, 1.0)
    return {
        "pit_conditional": pit_cond,
        "pit_marginal": pit_marg,
        "choice": choice,
        "n_below_t0": int(np.sum(~valid)),
    }


def _calibration_block(pit: np.ndarray, n_bins: int = 10) -> dict[str, object]:
    values = np.asarray(pit, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "ks_statistic": np.nan, "ks_pvalue": np.nan, "bins": []}
    result = kstest(values, "uniform")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts, _ = np.histogram(values, bins=edges)
    expected = values.size / float(n_bins)
    bins = []
    for i in range(int(n_bins)):
        bins.append(
            {
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "n": int(counts[i]),
                "expected": float(expected),
                "proportion": float(counts[i] / values.size),
            }
        )
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "ks_statistic": float(result.statistic),
        "ks_pvalue": float(result.pvalue),
        "max_bin_deviation": float(
            max(abs(b["proportion"] - 1.0 / n_bins) for b in bins)
        ),
        "bins": bins,
    }


def pit_calibration(
    pit_payload: dict[str, np.ndarray],
    n_bins: int = 10,
) -> dict[str, object]:
    """Summarise conditional PIT uniformity overall and by choice."""
    conditional = np.asarray(pit_payload["pit_conditional"], dtype=float)
    marginal = np.asarray(pit_payload["pit_marginal"], dtype=float)
    choice = np.asarray(pit_payload["choice"], dtype=int)
    return {
        "n_below_t0": int(pit_payload.get("n_below_t0", 0)),
        "conditional": _calibration_block(conditional, n_bins=n_bins),
        "conditional_SS": _calibration_block(conditional[choice == 0], n_bins=n_bins),
        "conditional_LL": _calibration_block(conditional[choice == 1], n_bins=n_bins),
        "marginal": _calibration_block(marginal, n_bins=n_bins),
    }


def mixture_quantiles(
    df: pd.DataFrame,
    params,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    by_choice: bool = True,
    n_terms: int = 12,
    n_grid: int = 500,
    chunk_size: int = 2000,
    max_time: float | None = None,
) -> dict[str, list[float]]:
    """Model-implied RT quantiles for the trial mix in ``df``.

    The predicted distribution is the mixture of trial-specific first-passage
    distributions, weighted by 1 for the marginal distribution or by the
    model-implied choice probability for the choice-conditional distributions.
    This avoids Monte Carlo noise when comparing quantiles.
    """
    p = _as_parameters(params)
    rt = df["rt"].to_numpy(dtype=float)
    x_amount = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    drift = drift_from_predictors(p, x_amount, x_delay)
    p_ll = np.clip(choice_probability(drift, p.a), 1e-12, 1.0 - 1e-12)

    min_decision = max(float(np.min(rt)) - p.t0, 1e-6)
    if max_time is None:
        max_time = max(40.0, 6.0 * float(np.max(rt)))
    grid = _decision_grid(min_decision, float(max_time), n_grid)

    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "all": (
            np.ones_like(p_ll),
            np.ones_like(p_ll),
            np.ones_like(p_ll),
        ),
    }
    if by_choice:
        # cdf_upper already contains the factor P(LL) and cdf_lower contains
        # P(SS), because they integrate the joint density. The conditional
        # mixture therefore sums the joint CDFs and normalises by the summed
        # choice probability; multiplying by p again would double-count it.
        groups["LL"] = (np.ones_like(p_ll), np.zeros_like(p_ll), p_ll)
        groups["SS"] = (
            np.zeros_like(p_ll),
            np.ones_like(p_ll),
            1.0 - p_ll,
        )

    accumulated = {
        name: np.zeros(grid.size, dtype=float) for name in groups
    }
    weights = {name: 0.0 for name in groups}

    for start in range(0, rt.size, int(chunk_size)):
        sl = slice(start, start + int(chunk_size))
        cdf_upper, cdf_lower = _cdf_on_grid(
            drift[sl], p.a, grid, n_terms=n_terms
        )
        for name, (w_upper, w_lower, w_norm) in groups.items():
            contribution = (
                w_upper[sl][:, None] * cdf_upper
                + w_lower[sl][:, None] * cdf_lower
            )
            accumulated[name] += contribution.sum(axis=0)
            weights[name] += float(np.sum(w_norm[sl]))

    result: dict[str, list[float]] = {}
    for name, cdf in accumulated.items():
        if weights[name] <= 0:
            result[name] = [float("nan")] * len(quantiles)
            continue
        cdf = cdf / weights[name]
        cdf = np.maximum.accumulate(cdf)
        if cdf[-1] <= 0:
            result[name] = [float("nan")] * len(quantiles)
            continue
        cdf = cdf / cdf[-1]
        result[name] = [
            float(np.interp(q, cdf, grid + p.t0)) for q in quantiles
        ]
    return result


def model_quantile_table(
    df: pd.DataFrame,
    params,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    n_terms: int = 12,
    n_grid: int = 400,
) -> list[dict[str, float]]:
    """Observed and model-implied RT quantiles, marginally and by choice."""
    qs = tuple(float(q) for q in quantiles)
    predicted = mixture_quantiles(
        df, params, quantiles=qs, by_choice=True, n_terms=n_terms, n_grid=n_grid
    )
    observed_rt = df["rt"].to_numpy(dtype=float)
    observed_choice = df["choice"].to_numpy(dtype=int)
    rows: list[dict[str, float]] = []
    for label in ("all", "SS", "LL"):
        if label == "all":
            obs = observed_rt
        else:
            value = 0 if label == "SS" else 1
            obs = observed_rt[observed_choice == value]
        predicted_values = predicted[label]
        for q, pred in zip(qs, predicted_values):
            obs_q = float(np.quantile(obs, q)) if obs.size else float("nan")
            rows.append(
                {
                    "choice": label,
                    "quantile": q,
                    "observed": obs_q,
                    "predicted": pred,
                    "difference": pred - obs_q,
                }
            )
    return rows


def participant_quantile_deviations(
    df: pd.DataFrame,
    params,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95),
    n_terms: int = 12,
    n_grid: int = 400,
) -> pd.DataFrame:
    """Observed versus model-implied marginal quantiles for each participant."""
    rows = []
    for subject, group in df.groupby("subj_ident", sort=True):
        predicted = mixture_quantiles(
            group,
            params,
            quantiles=quantiles,
            by_choice=False,
            n_terms=n_terms,
            n_grid=n_grid,
        )["all"]
        for q, pred in zip(quantiles, predicted):
            obs = float(np.quantile(group["rt"].to_numpy(dtype=float), q))
            rows.append(
                {
                    "subj_ident": str(subject),
                    "n_trials": int(len(group)),
                    "quantile": float(q),
                    "observed": obs,
                    "predicted": float(pred),
                    "difference": float(pred - obs),
                }
            )
    return pd.DataFrame(rows)


def quantile_row_mae(rows: list[dict[str, float]]) -> float:
    return quantile_mae(rows)
