"""Cleaning rules and trial-level predictors for the ITC Database.

The rules in this module are frozen for the classical-DDM experiment. They are
also recorded in docs/dataset_audit.md and docs/decisions.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRIMARY_STUDIES = [
    "Bixter_2015_Exp1",
    "Bixter_2015_Exp2",
    "Bruder_2021",
    "Gluth_2017_Exp1",
    "Gluth_2017_Exp2",
    "Herting_2010",
    "Knauth_2022",
    "Konstantinidis_2020",
    "Lempert_2015",
    "Robinson_2015",
    "Zhou_2022",
    "Zilker_2021",
]

DEFAULT_RULES: dict[str, Any] = {
    "rt_min": 0.2,
    "rt_max": 20.0,
    "require_positive_amounts": True,
    "require_ss_time_lt_ll_time": True,
    "require_valid_choice": True,
    "exclude_time_pressure": True,
    "required_presentation": "Simultaneous",
    "exclude_mouse_finger": True,
    "required_rt_in_full_seconds": 0,
    "required_rt_data_analyzed": 1,
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def clean_trials(
    df: pd.DataFrame,
    rules: dict[str, Any] | None = None,
    apply_design_filters: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the frozen trial- and study-level cleaning rules.

    Parameters
    ----------
    df:
        Raw trial-level data.
    rules:
        Optional override of ``DEFAULT_RULES``.
    apply_design_filters:
        If False, only the trial-level rules are applied. This is useful for
        auditing how much data each design filter removes.

    Returns
    -------
    cleaned, audit
        The cleaned DataFrame and a step-by-step count of removed rows.
    """
    cfg = {**DEFAULT_RULES, **(rules or {})}
    audit: dict[str, int] = {"raw_rows": int(len(df))}
    out = df.copy()

    # Subject-level exclusion: drop all rows for any flagged subject.
    flagged_subjects = set(
        out.loc[_as_bool(out["subj_excl"]), "subj_ident"].unique()
    )
    if flagged_subjects:
        out = out[~out["subj_ident"].isin(flagged_subjects)]
    audit["after_subject_exclusion"] = int(len(out))

    # Compiler trial exclusion.
    out = out[~_as_bool(out["trial_excl"])]
    audit["after_trial_exclusion"] = int(len(out))

    # RT range.
    out = out[out["rt"].between(cfg["rt_min"], cfg["rt_max"], inclusive="both")]
    audit["after_rt_range"] = int(len(out))

    # Valid amounts and delays.
    if cfg["require_positive_amounts"]:
        out = out[(out["ss_value"] > 0) & (out["ll_value"] > 0)]
    if cfg["require_ss_time_lt_ll_time"]:
        out = out[out["ss_time"] < out["ll_time"]]
    if cfg["require_valid_choice"]:
        out = out[out["choice"].isin([0, 1])]
    audit["after_valid_stimuli_choice"] = int(len(out))

    if apply_design_filters:
        if cfg["exclude_time_pressure"]:
            out = out[out["time_pressure"] == 0]
        if cfg["required_presentation"] is not None:
            out = out[
                out["presentation_of_information"].fillna("")
                == cfg["required_presentation"]
            ]
        if cfg["exclude_mouse_finger"] and not out.empty:
            methods = out["additional_methods"].fillna("").astype(str)
            out = out[~methods.str.contains("Mouse|Finger", case=False, regex=True)]
        if cfg["required_rt_in_full_seconds"] is not None:
            out = out[out["rt_in_full_seconds"] == cfg["required_rt_in_full_seconds"]]
        if cfg["required_rt_data_analyzed"] is not None:
            out = out[out["rt_data_analyzed"] == cfg["required_rt_data_analyzed"]]
        audit["after_design_filters"] = int(len(out))

    return out.reset_index(drop=True), audit


def study_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-study counts and design flags, computed on cleaned data."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("paper", sort=True)
    profile = pd.DataFrame(
        {
            "trials": grouped.size(),
            "subjects": grouped["subj_ident"].nunique(),
            "median_trials_per_subject": grouped.apply(
                lambda x: x.groupby("subj_ident").size().median(),
                include_groups=False,
            ),
            "rt_median": grouped["rt"].median(),
            "choice_ll": grouped["choice"].mean(),
            "time_pressure": grouped["time_pressure"].max(),
            "rt_in_full_seconds": grouped["rt_in_full_seconds"].max(),
            "rt_data_analyzed": grouped["rt_data_analyzed"].min(),
        }
    )
    profile["mouse_or_finger"] = grouped["additional_methods"].apply(
        lambda s: bool(s.fillna("").str.contains("Mouse|Finger", case=False).any())
    )
    profile["presentation"] = grouped["presentation_of_information"].first()
    return profile


def select_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict cleaned data to the frozen 12-study primary sample."""
    missing = sorted(set(PRIMARY_STUDIES) - set(df["paper"].unique()))
    if missing:
        raise ValueError(f"Primary studies missing after cleaning: {missing}")
    return df[df["paper"].isin(PRIMARY_STUDIES)].reset_index(drop=True)


def primary_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise the primary sample and verify its expected size."""
    if df.empty:
        raise ValueError("Primary sample is empty.")
    per_subject = df.groupby("subj_ident").size()
    return {
        "studies": int(df["paper"].nunique()),
        "subjects": int(df["subj_ident"].nunique()),
        "trials": int(len(df)),
        "sessions": int(df["sess_ident"].nunique()),
        "median_trials_per_subject": float(per_subject.median()),
        "min_trials_per_subject": int(per_subject.min()),
        "max_trials_per_subject": int(per_subject.max()),
        "subjects_ge_150_trials": int((per_subject >= 150).sum()),
        "subjects_ge_300_trials": int((per_subject >= 300).sum()),
        "studies_list": sorted(df["paper"].unique().tolist()),
    }


def add_predictors(
    df: pd.DataFrame,
    scaler: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add standardised amount and delay predictors.

    ``x_amt_raw = log(ll_value / ss_value)``
    ``x_delay_raw = log((ll_time + 1) / (ss_time + 1))``

    Standardisation is within study. If ``scaler`` is provided, it is applied
    unchanged (important for cross-validation, where the scaler must be fitted
    on training data only).
    """
    out = df.copy()
    out["x_amt_raw"] = np.log(out["ll_value"] / out["ss_value"])
    out["x_delay_raw"] = np.log(
        (out["ll_time"] + 1.0) / (out["ss_time"] + 1.0)
    )

    if scaler is None:
        scaler = fit_predictor_scaler(out)
    out = apply_predictor_scaler(out, scaler)
    return out, scaler


def fit_predictor_scaler(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit per-study means and SDs for the two predictors."""
    scaler: dict[str, dict[str, float]] = {}
    for paper, group in df.groupby("paper"):
        amt = group["x_amt_raw"].to_numpy(dtype=float)
        delay = group["x_delay_raw"].to_numpy(dtype=float)
        scaler[str(paper)] = {
            "amt_mean": float(np.mean(amt)),
            "amt_sd": float(np.std(amt, ddof=0)) or 1.0,
            "delay_mean": float(np.mean(delay)),
            "delay_sd": float(np.std(delay, ddof=0)) or 1.0,
        }
    return scaler


def apply_predictor_scaler(
    df: pd.DataFrame,
    scaler: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply a previously fitted per-study predictor scaler."""
    out = df.copy()
    out["x_amt"] = np.nan
    out["x_delay"] = np.nan
    for paper, group in out.groupby("paper"):
        if paper not in scaler:
            raise KeyError(f"No predictor scaler for study {paper!r}")
        s = scaler[paper]
        idx = group.index
        out.loc[idx, "x_amt"] = (
            group["x_amt_raw"] - s["amt_mean"]
        ) / s["amt_sd"]
        out.loc[idx, "x_delay"] = (
            group["x_delay_raw"] - s["delay_mean"]
        ) / s["delay_sd"]
    if out[["x_amt", "x_delay"]].isna().any().any():
        raise ValueError("Predictor standardisation produced missing values.")
    return out
