"""Prediction and goodness-of-fit metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import rankdata


def _clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)


def choice_metrics(choice: np.ndarray, p_ll: np.ndarray) -> dict[str, float]:
    """Log loss, Brier score, and AUC for LL-choice predictions."""
    y = np.asarray(choice, dtype=int)
    p = _clip_probability(p_ll)
    if y.size == 0:
        return {"log_loss": np.nan, "brier": np.nan, "auc": np.nan, "n": 0}
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    if np.unique(y).size < 2:
        auc = np.nan
    else:
        ranks = rankdata(p)
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        auc = float(
            (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        )
    return {"log_loss": log_loss, "brier": brier, "auc": auc, "n": int(y.size)}


def quantile_table(
    observed_rt: np.ndarray,
    observed_choice: np.ndarray,
    predicted_rt: np.ndarray,
    predicted_choice: np.ndarray,
    quantiles: Iterable[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> list[dict[str, float]]:
    """Observed and predicted RT quantiles, separately by choice."""
    rows: list[dict[str, float]] = []
    for choice_value, label in ((0, "SS"), (1, "LL")):
        obs = np.asarray(observed_rt)[np.asarray(observed_choice) == choice_value]
        pred = np.asarray(predicted_rt)[np.asarray(predicted_choice) == choice_value]
        for q in quantiles:
            obs_q = float(np.quantile(obs, q)) if obs.size else np.nan
            pred_q = float(np.quantile(pred, q)) if pred.size else np.nan
            rows.append(
                {
                    "choice": label,
                    "quantile": float(q),
                    "observed": obs_q,
                    "predicted": pred_q,
                    "difference": pred_q - obs_q,
                }
            )
    return rows


def quantile_mae(rows: list[dict[str, float]]) -> float:
    diffs = [
        abs(r["difference"])
        for r in rows
        if np.isfinite(r["difference"])
    ]
    return float(np.mean(diffs)) if diffs else np.nan


def central_interval_coverage(
    observed_rt: np.ndarray,
    predicted_rt: np.ndarray,
    level: float = 0.8,
) -> float:
    """Coverage of a central prediction interval from simulated RTs."""
    if observed_rt.size == 0 or predicted_rt.size == 0:
        return np.nan
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(predicted_rt, alpha))
    hi = float(np.quantile(predicted_rt, 1.0 - alpha))
    return float(np.mean((observed_rt >= lo) & (observed_rt <= hi)))


def choice_calibration(
    choice: np.ndarray,
    p_ll: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Observed versus predicted LL proportion in probability bins."""
    y = np.asarray(choice, dtype=int)
    p = np.asarray(p_ll, dtype=float)
    if y.size == 0:
        return []
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bins = np.array_split(np.arange(y.size), int(n_bins))
    rows = []
    for members in bins:
        if members.size == 0:
            continue
        rows.append(
            {
                "n": int(members.size),
                "predicted": float(np.mean(p_sorted[members])),
                "observed": float(np.mean(y_sorted[members])),
                "lower": float(np.min(p_sorted[members])),
                "upper": float(np.max(p_sorted[members])),
            }
        )
    return rows


def choice_by_quantile_bin(
    values: np.ndarray,
    choice: np.ndarray,
    p_ll: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, float]]:
    """Observed and predicted LL proportion across bins of one predictor."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(choice, dtype=int)
    p = np.asarray(p_ll, dtype=float)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i,
                "lower": float(lo),
                "upper": float(hi),
                "n": int(np.sum(mask)),
                "observed_ll": float(np.mean(y[mask])),
                "predicted_ll": float(np.mean(p[mask])),
            }
        )
    return rows
