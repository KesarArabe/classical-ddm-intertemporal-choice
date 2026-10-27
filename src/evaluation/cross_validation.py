"""Participant-level cross-validation for the classical DDM.

Folds are formed over participants, never over trials, and are greedily
balanced by trial count. The predictor standardisation is fitted on the
training participants of each fold and then applied unchanged to the held-out
participants, so no test-set information enters the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.clean import apply_predictor_scaler, fit_predictor_scaler
from src.evaluation.metrics import choice_metrics, quantile_mae
from src.evaluation.rt_cdf import (
    conditional_pit,
    model_quantile_table,
    pit_calibration,
)
from src.fitting.optimize import fit_classical_ddm
from src.models.classical_ddm import (
    DDMParameters,
    PARAMETER_NAMES,
    choice_probability,
    ddm_log_density,
    drift_from_predictors,
)


def assign_subject_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 20261024,
) -> pd.Series:
    """Assign participants to folds, greedily balancing trial counts.

    The returned Series maps ``subj_ident`` to a zero-based fold index. The
    ``seed`` is accepted for reproducibility and used only to break ties among
    participants with identical trial counts.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    counts = (
        df.groupby("subj_ident", sort=True)
        .size()
        .rename("trials")
        .reset_index()
    )
    if counts["subj_ident"].nunique() < n_folds:
        raise ValueError(
            f"Cannot form {n_folds} folds from "
            f"{counts['subj_ident'].nunique()} participants."
        )
    rng = np.random.default_rng(int(seed))
    counts["tie_break"] = rng.random(len(counts))
    counts = counts.sort_values(
        ["trials", "tie_break"], ascending=[False, True]
    ).reset_index(drop=True)

    loads = np.zeros(int(n_folds), dtype=float)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        fold = int(np.argmin(loads))
        assignment[str(row.subj_ident)] = fold
        loads[fold] += float(row.trials)
    return pd.Series(assignment, name="fold").rename_axis("subj_ident")


def heldout_metrics(
    df: pd.DataFrame,
    params,
    n_terms: int = 100,
    cdf_terms: int = 12,
) -> dict[str, Any]:
    """Proper scoring rules and calibration for held-out trials."""
    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    x_amount = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    drift = drift_from_predictors(params, x_amount, x_delay)
    p_ll = np.clip(choice_probability(drift, params.a), 1e-12, 1.0 - 1e-12)
    log_upper, log_lower = ddm_log_density(
        rt, drift, params.a, params.t0, n_terms=n_terms
    )
    chosen = np.where(choice == 1, log_upper, log_lower)
    p_chosen = np.where(choice == 1, p_ll, 1.0 - p_ll)
    conditional_log_score = chosen - np.log(p_chosen)
    marginal_log_score = np.logaddexp(log_upper, log_lower)

    finite_cond = np.isfinite(conditional_log_score)
    finite_marg = np.isfinite(marginal_log_score)
    if np.any(~finite_cond):
        conditional_log_score = conditional_log_score[finite_cond]
    if np.any(~finite_marg):
        marginal_log_score = marginal_log_score[finite_marg]

    pit_payload = conditional_pit(df, params, n_terms=cdf_terms)
    calibration = pit_calibration(pit_payload)
    quantiles = model_quantile_table(df, params, n_terms=cdf_terms)
    metrics = choice_metrics(choice, p_ll)
    return {
        "n_trials": int(len(df)),
        "observed_ll_proportion": float(np.mean(choice)),
        "predicted_ll_proportion": float(np.mean(p_ll)),
        "choice_log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "auc": metrics["auc"],
        "conditional_log_score": float(np.mean(conditional_log_score))
        if conditional_log_score.size
        else float("nan"),
        "marginal_log_score": float(np.mean(marginal_log_score))
        if marginal_log_score.size
        else float("nan"),
        "n_zero_density": int(np.sum(~finite_cond)),
        "n_below_t0": int(pit_payload["n_below_t0"]),
        "rt_quantile_mae": quantile_mae(quantiles),
        "quantiles": quantiles,
        "pit_calibration": calibration,
        "pit_conditional": pit_payload["pit_conditional"],
        "pit_marginal": pit_payload["pit_marginal"],
        "p_ll": p_ll,
        "conditional_log_score_per_trial": conditional_log_score,
        "marginal_log_score_per_trial": marginal_log_score,
    }


def _pooled_payload(predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (
        "choice",
        "p_ll",
        "pit_conditional",
        "pit_marginal",
        "conditional_log_score_per_trial",
        "marginal_log_score_per_trial",
    )
    return {
        key: np.concatenate([p[key] for p in predictions]) for key in keys
    }


def cross_validate_study(
    df: pd.DataFrame,
    paper: str,
    n_folds: int = 5,
    seed: int = 20261024,
    n_starts: int = 2,
    n_terms: int = 100,
    maxiter: int = 400,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run participant-level K-fold CV for one study.

    Returns ``(fold_table, summary_row, pooled_calibration)``.
    """
    required = {"x_amt_raw", "x_delay_raw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CV requires raw predictors: missing {missing}")

    folds = assign_subject_folds(df, n_folds=n_folds, seed=seed)
    work = df.copy()
    work["fold"] = work["subj_ident"].astype(str).map(folds)
    if work["fold"].isna().any():
        raise ValueError("Fold assignment left some participants unassigned.")

    fold_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, np.ndarray]] = []
    all_subjects = set(work["subj_ident"].astype(str))

    for fold in range(int(n_folds)):
        train = work[work["fold"] != fold].copy()
        test = work[work["fold"] == fold].copy()
        train_subjects = set(train["subj_ident"].astype(str))
        test_subjects = set(test["subj_ident"].astype(str))
        if train_subjects & test_subjects:
            raise RuntimeError(
                f"Participant leakage in fold {fold}: "
                f"{sorted(train_subjects & test_subjects)[:5]}"
            )
        if train.empty or test.empty:
            raise RuntimeError(f"Empty train or test set in fold {fold}.")
        if train_subjects | test_subjects != all_subjects:
            raise RuntimeError(f"Fold {fold} does not cover all participants.")

        scaler = fit_predictor_scaler(train[["paper", "x_amt_raw", "x_delay_raw"]])
        train_scaled = apply_predictor_scaler(train, scaler)
        test_scaled = apply_predictor_scaler(test, scaler)

        row: dict[str, Any] = {
            "paper": paper,
            "fold": fold,
            "n_train": int(len(train_scaled)),
            "n_test": int(len(test_scaled)),
            "n_train_subjects": int(train_scaled["subj_ident"].nunique()),
            "n_test_subjects": int(test_scaled["subj_ident"].nunique()),
        }
        try:
            fit = fit_classical_ddm(
                train_scaled,
                f"{paper}_fold{fold}",
                n_starts=int(n_starts),
                n_terms=int(n_terms),
                maxiter=int(maxiter),
                compute_hessian=False,
            )
        except Exception as exc:  # noqa: BLE001 - reported explicitly
            row.update(
                {
                    "converged": False,
                    "error": repr(exc),
                    "test_choice_log_loss": np.nan,
                    "test_brier": np.nan,
                    "test_auc": np.nan,
                    "test_conditional_log_score": np.nan,
                    "test_marginal_log_score": np.nan,
                    "test_rt_quantile_mae": np.nan,
                    "test_pit_ks_conditional": np.nan,
                    "test_pit_ks_pvalue": np.nan,
                    "test_n_zero_density": np.nan,
                }
            )
            fold_rows.append(row)
            continue

        params = DDMParameters.from_array(
            [fit.params[name] for name in PARAMETER_NAMES]
        )
        metrics = heldout_metrics(test_scaled, params, n_terms=n_terms)
        row.update(
            {
                "converged": bool(fit.converged),
                "log_likelihood_train": float(fit.log_likelihood),
                "log_likelihood_per_trial_train": float(
                    fit.log_likelihood / len(train_scaled)
                ),
                "test_choice_log_loss": metrics["choice_log_loss"],
                "test_brier": metrics["brier"],
                "test_auc": metrics["auc"],
                "test_conditional_log_score": metrics["conditional_log_score"],
                "test_marginal_log_score": metrics["marginal_log_score"],
                "test_rt_quantile_mae": metrics["rt_quantile_mae"],
                "test_pit_ks_conditional": metrics["pit_calibration"][
                    "conditional"
                ]["ks_statistic"],
                "test_pit_ks_pvalue": metrics["pit_calibration"]["conditional"][
                    "ks_pvalue"
                ],
                "test_n_zero_density": metrics["n_zero_density"],
                "test_n_below_t0": metrics["n_below_t0"],
                "observed_ll_proportion": metrics["observed_ll_proportion"],
                "predicted_ll_proportion": metrics["predicted_ll_proportion"],
            }
        )
        for name in PARAMETER_NAMES:
            row[f"param_{name}"] = float(fit.params[name])
        fold_rows.append(row)
        predictions.append(
            {
                "choice": test_scaled["choice"].to_numpy(dtype=int),
                "p_ll": metrics["p_ll"],
                "pit_conditional": metrics["pit_conditional"],
                "pit_marginal": metrics["pit_marginal"],
                "conditional_log_score_per_trial": metrics[
                    "conditional_log_score_per_trial"
                ],
                "marginal_log_score_per_trial": metrics[
                    "marginal_log_score_per_trial"
                ],
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    ok = (
        fold_table[fold_table["error"].isna()]
        if "error" in fold_table.columns
        else fold_table
    )
    summary: dict[str, Any] = {
        "paper": paper,
        "n_trials": int(len(work)),
        "n_subjects": int(work["subj_ident"].nunique()),
        "n_folds": int(n_folds),
        "n_folds_ok": int(len(ok)),
        "n_folds_failed": int(n_folds - len(ok)),
        "all_converged": bool(len(ok) == n_folds and ok["converged"].all()),
    }

    if len(ok) > 0:
        for key in (
            "test_choice_log_loss",
            "test_brier",
            "test_auc",
            "test_conditional_log_score",
            "test_marginal_log_score",
            "test_rt_quantile_mae",
            "test_pit_ks_conditional",
        ):
            summary[f"{key}_mean"] = float(ok[key].mean())
            summary[f"{key}_sd"] = float(ok[key].std(ddof=1)) if len(ok) > 1 else np.nan

    pooled_calibration: dict[str, Any] = {}
    if predictions:
        pooled = _pooled_payload(predictions)
        pooled_choice = choice_metrics(pooled["choice"], pooled["p_ll"])
        finite_cond = np.isfinite(pooled["conditional_log_score_per_trial"])
        finite_marg = np.isfinite(pooled["marginal_log_score_per_trial"])
        pooled_calibration = pit_calibration(
            {
                "pit_conditional": pooled["pit_conditional"],
                "pit_marginal": pooled["pit_marginal"],
                "choice": pooled["choice"],
                "n_below_t0": int(np.sum(~finite_cond)),
            }
        )
        summary.update(
            {
                "pooled_test_choice_log_loss": pooled_choice["log_loss"],
                "pooled_test_brier": pooled_choice["brier"],
                "pooled_test_auc": pooled_choice["auc"],
                "pooled_test_conditional_log_score": float(
                    np.mean(pooled["conditional_log_score_per_trial"][finite_cond])
                )
                if np.any(finite_cond)
                else np.nan,
                "pooled_test_marginal_log_score": float(
                    np.mean(pooled["marginal_log_score_per_trial"][finite_marg])
                )
                if np.any(finite_marg)
                else np.nan,
                "pooled_test_n_zero_density": int(np.sum(~finite_cond)),
                "pooled_test_pit_ks_conditional": pooled_calibration[
                    "conditional"
                ]["ks_statistic"],
                "pooled_test_pit_ks_pvalue": pooled_calibration["conditional"][
                    "ks_pvalue"
                ],
            }
        )
    return fold_table, summary, pooled_calibration
