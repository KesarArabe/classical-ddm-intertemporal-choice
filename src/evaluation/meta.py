"""Descriptive and random-effects meta-analysis of study-level estimates.

This is a synthesis of separately fitted study-level parameters. It is not a
pooled re-fit and does not propagate participant-level clustering beyond the
study-level standard errors supplied to it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def random_effects(effects: Iterable[float], ses: Iterable[float]) -> dict[str, float]:
    """DerSimonian-Laird random-effects meta-analysis for one parameter."""
    y = np.asarray(list(effects), dtype=float)
    se = np.asarray(list(ses), dtype=float)
    valid = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y = y[valid]
    se = se[valid]
    k = int(y.size)
    if k == 0:
        return {
            "k": 0,
            "estimate": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "tau2": np.nan,
            "i2": np.nan,
            "q": np.nan,
            "q_df": np.nan,
            "q_pvalue": np.nan,
        }
    w = 1.0 / se**2
    fixed = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed) ** 2))
    df = k - 1
    if k > 1:
        c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
        tau2 = float(max(0.0, (q - df) / c)) if c > 0 else 0.0
        i2 = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
        from scipy.stats import chi2

        q_pvalue = float(chi2.sf(q, df))
    else:
        tau2 = 0.0
        i2 = 0.0
        q_pvalue = np.nan

    w_re = 1.0 / (se**2 + tau2)
    estimate = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z = estimate / se_re if se_re > 0 else np.nan
    pvalue = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "k": k,
        "estimate": estimate,
        "se": se_re,
        "ci_lower": estimate - 1.96 * se_re,
        "ci_upper": estimate + 1.96 * se_re,
        "z": float(z),
        "pvalue": pvalue,
        "fixed_effect": fixed,
        "tau2": tau2,
        "i2": i2,
        "q": q,
        "q_df": float(df),
        "q_pvalue": q_pvalue,
    }


def meta_analysis_table(
    estimates: pd.DataFrame,
    parameters: Iterable[str],
    estimate_col: str = "estimate",
    se_col: str = "se",
) -> pd.DataFrame:
    """Run random-effects meta-analysis for each parameter."""
    rows = []
    for parameter in parameters:
        subset = estimates[estimates["parameter"] == parameter]
        result = random_effects(
            subset[estimate_col].to_numpy(dtype=float),
            subset[se_col].to_numpy(dtype=float),
        )
        rows.append({"parameter": parameter, **result})
    return pd.DataFrame(rows)


def descriptive_table(
    estimates: pd.DataFrame,
    parameters: Iterable[str],
    estimate_col: str = "estimate",
    se_col: str = "se",
) -> pd.DataFrame:
    """Unweighted and precision-weighted descriptive summaries."""
    rows = []
    for parameter in parameters:
        subset = estimates[estimates["parameter"] == parameter]
        y = subset[estimate_col].to_numpy(dtype=float)
        se = subset[se_col].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(se) & (se > 0)
        y = y[valid]
        se = se[valid]
        if y.size == 0:
            rows.append({"parameter": parameter, "k": 0})
            continue
        w = 1.0 / se**2
        rows.append(
            {
                "parameter": parameter,
                "k": int(y.size),
                "mean_unweighted": float(np.mean(y)),
                "sd_unweighted": float(np.std(y, ddof=1)) if y.size > 1 else np.nan,
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "median": float(np.median(y)),
                "precision_weighted_mean": float(np.sum(w * y) / np.sum(w)),
            }
        )
    return pd.DataFrame(rows)
