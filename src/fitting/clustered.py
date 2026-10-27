"""Participant-clustered (sandwich) uncertainty for trial-level MLEs.

Trials from the same participant are not independent: they share parameters and
unobserved participant-level traits. The usual inverse-Hessian standard errors
therefore understate uncertainty. This module computes the cluster-robust
sandwich estimator

    Cov = (G / (G - 1)) * H^-1 * (sum_g s_g s_g') * H^-1

where ``H`` is the Hessian of the summed negative log-likelihood, ``s_g`` is the
sum of per-trial score contributions for participant ``g``, and ``G`` is the
number of participants. The factor ``G / (G - 1)`` is a small-sample correction.
No further correction is applied because the number of trials is large relative
to the five parameters.

The bread matrix is the numerical Hessian already stored by
``fit_classical_ddm``. Scores are central differences of the per-trial
log-likelihood at the fitted parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.classical_ddm import (
    DDMParameters,
    PARAMETER_NAMES,
    ddm_log_density,
    drift_from_predictors,
)


def _trial_log_likelihood(
    values: np.ndarray,
    rt: np.ndarray,
    choice: np.ndarray,
    x_amount: np.ndarray,
    x_delay: np.ndarray,
    n_terms: int,
) -> np.ndarray:
    params = DDMParameters.from_array(values)
    if not np.isfinite(params.as_array()).all():
        return np.full_like(rt, -np.inf, dtype=float)
    drift = drift_from_predictors(params, x_amount, x_delay)
    log_upper, log_lower = ddm_log_density(
        rt, drift, params.a, params.t0, n_terms=n_terms
    )
    return np.where(np.asarray(choice, dtype=int) == 1, log_upper, log_lower)


def participant_scores(
    df: pd.DataFrame,
    fit,
    n_terms: int = 100,
    rel_step: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-participant score sums at the fitted parameters.

    Returns
    -------
    subjects, scores, trial_scores
        ``subjects`` is the sorted array of participant identifiers,
        ``scores`` has shape ``(n_subjects, n_parameters)`` and contains the
        summed score for each participant, and ``trial_scores`` is the
        trial-level score matrix.
    """
    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    x_amount = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    x = np.array([fit.params[name] for name in PARAMETER_NAMES], dtype=float)

    def evaluate(values: np.ndarray) -> np.ndarray:
        return _trial_log_likelihood(
            values, rt, choice, x_amount, x_delay, n_terms=n_terms
        )

    base = evaluate(x)
    if not np.all(np.isfinite(base)):
        raise ValueError(
            "Cannot compute scores: some trials have non-finite log density "
            "at the fitted parameters."
        )

    n_params = x.size
    trial_scores = np.empty((len(df), n_params), dtype=float)
    for j in range(n_params):
        h = rel_step * max(1.0, abs(x[j]))
        step = np.zeros(n_params, dtype=float)
        step[j] = h
        log_plus = evaluate(x + step)
        log_minus = evaluate(x - step)
        if not (np.all(np.isfinite(log_plus)) and np.all(np.isfinite(log_minus))):
            raise ValueError(
                f"Score for parameter {PARAMETER_NAMES[j]!r} is non-finite at "
                "the numerical step size."
            )
        trial_scores[:, j] = (log_plus - log_minus) / (2.0 * h)

    labels, inverse = np.unique(
        df["subj_ident"].astype(str).to_numpy(), return_inverse=True
    )
    scores = np.zeros((labels.size, n_params), dtype=float)
    np.add.at(scores, inverse, trial_scores)
    return labels, scores, trial_scores


def clustered_covariance(
    fit,
    df: pd.DataFrame,
    n_terms: int = 100,
    small_sample_correction: bool = True,
) -> dict[str, Any]:
    """Naive and participant-clustered covariance for one study fit."""
    if fit.hessian is None:
        raise ValueError("Fit has no Hessian; clustered SEs require one.")
    hessian = np.asarray(fit.hessian, dtype=float)
    if not bool(fit.hessian_positive_definite):
        raise ValueError(
            "Hessian is not positive definite; the sandwich bread matrix is "
            "not well defined for this fit."
        )
    bread = np.linalg.inv(hessian)
    subjects, scores, _ = participant_scores(df, fit, n_terms=n_terms)
    n_clusters = int(scores.shape[0])
    if n_clusters < 2:
        raise ValueError("At least two participants are required for clustering.")

    meat = scores.T @ scores
    correction = n_clusters / (n_clusters - 1.0) if small_sample_correction else 1.0
    clustered = correction * (bread @ meat @ bread)
    naive = bread

    naive_se = np.sqrt(np.maximum(np.diag(naive), 0.0))
    clustered_se = np.sqrt(np.maximum(np.diag(clustered), 0.0))
    design_effect = np.divide(
        clustered_se**2,
        naive_se**2,
        out=np.full_like(naive_se, np.nan),
        where=naive_se > 0,
    )

    estimates = np.array([fit.params[name] for name in PARAMETER_NAMES], dtype=float)
    z = np.divide(
        estimates,
        clustered_se,
        out=np.full_like(estimates, np.nan),
        where=clustered_se > 0,
    )

    rows = []
    for j, name in enumerate(PARAMETER_NAMES):
        rows.append(
            {
                "parameter": name,
                "estimate": float(estimates[j]),
                "naive_se": float(naive_se[j]),
                "clustered_se": float(clustered_se[j]),
                "design_effect": float(design_effect[j]),
                "z_clustered": float(z[j]),
            }
        )

    return {
        "paper": str(fit.paper),
        "n_trials": int(len(df)),
        "n_subjects": n_clusters,
        "small_sample_correction": float(correction),
        "hessian_condition": float(fit.hessian_condition)
        if fit.hessian_condition is not None
        else None,
        "rows": rows,
        "naive_covariance": naive.tolist(),
        "clustered_covariance": clustered.tolist(),
        "parameter_names": list(PARAMETER_NAMES),
        "subject_scores": {
            str(subject): score.tolist()
            for subject, score in zip(subjects, scores)
        },
    }


def clustered_table(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten several clustered-covariance payloads into a table."""
    rows = []
    for payload in payloads:
        for row in payload["rows"]:
            rows.append(
                {
                    "paper": payload["paper"],
                    "n_trials": payload["n_trials"],
                    "n_subjects": payload["n_subjects"],
                    **row,
                }
            )
    return pd.DataFrame(rows)
