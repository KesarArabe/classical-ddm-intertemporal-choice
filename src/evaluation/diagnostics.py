"""Diagnostics and posterior predictive checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    central_interval_coverage,
    choice_by_quantile_bin,
    choice_calibration,
    choice_metrics,
    quantile_mae,
    quantile_table,
)
from src.models.classical_ddm import (
    DDMParameters,
    choice_probability,
    drift_from_predictors,
    simulate_from_predictors,
)


def parameter_plausibility(fit, df: pd.DataFrame) -> dict[str, object]:
    """Simple, explicit checks on fitted parameters."""
    p = DDMParameters.from_array([fit.params[k] for k in (
        "a", "t0", "v0", "beta_amount", "beta_delay"
    )])
    min_rt = float(df["rt"].min())
    drift = drift_from_predictors(
        p, df["x_amt"].to_numpy(), df["x_delay"].to_numpy()
    )
    checks = {
        "a_positive": bool(p.a > 0),
        "a_within_bounds": bool(0.2 <= p.a <= 5.0),
        "t0": float(p.t0),
        "t0_nonnegative": bool(p.t0 >= 0),
        "t0_below_min_rt": bool(p.t0 < min_rt),
        "t0_margin_seconds": float(min_rt - p.t0),
        "drift_finite": bool(np.all(np.isfinite(drift))),
        "drift_min": float(np.min(drift)),
        "drift_max": float(np.max(drift)),
        "min_rt": min_rt,
        "observed_rt_min": min_rt,
    }
    return checks


def parameter_bound_diagnostics(fit, df: pd.DataFrame) -> dict[str, object]:
    """Distance of each estimate from its optimisation bounds.

    The ``t0`` bound is the only one that is routinely active in this model:
    it is set just below the fastest observed RT. A ``t0`` estimate close to
    that bound means the Hessian-based standard error is not a valid confidence
    interval, because the likelihood is truncated by the constraint.
    """
    names = ("a", "t0", "v0", "beta_amount", "beta_delay")
    if fit.bounds is None:
        raise ValueError("Fit has no stored bounds.")
    rows = []
    for name, (low, high) in zip(names, fit.bounds):
        value = float(fit.params[name])
        span = float(high) - float(low)
        distance = min(value - float(low), float(high) - value)
        rows.append(
            {
                "parameter": name,
                "estimate": value,
                "lower": float(low),
                "upper": float(high),
                "distance_to_nearest_bound": distance,
                "fraction_of_range_to_nearest_bound": (
                    distance / span if span > 0 else np.nan
                ),
                "at_bound_1pct": bool(span > 0 and distance / span <= 0.01),
                "at_bound_5pct": bool(span > 0 and distance / span <= 0.05),
                "at_lower_bound": bool(value - float(low) <= 1e-8),
                "at_upper_bound": bool(float(high) - value <= 1e-8),
            }
        )
    t0_row = next(row for row in rows if row["parameter"] == "t0")
    return {
        "rows": rows,
        "t0_margin_to_upper_seconds": float(
            t0_row["upper"] - t0_row["estimate"]
        ),
        "t0_fraction_of_range_to_upper": float(
            t0_row["fraction_of_range_to_nearest_bound"]
        ),
        "t0_at_upper_5pct": bool(t0_row["at_upper_bound"] or t0_row["at_bound_5pct"]),
        "hessian_se_valid_for_t0": bool(
            not (t0_row["at_upper_bound"] or t0_row["at_bound_5pct"])
        ),
        "min_rt": float(df["rt"].min()),
        "observed_rt_min": float(df["rt"].min()),
    }


def ppc_summary(
    df: pd.DataFrame,
    params: DDMParameters,
    n_sim: int = 20000,
    seed: int = 20261024,
    dt: float = 1e-4,
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    include_samples: bool = False,
) -> dict[str, object]:
    """Posterior predictive check using the observed predictor distribution.

    Predictors are resampled with replacement from the study, then choices and
    RTs are simulated from the fitted model. This preserves the stimulus design
    while generating a new sample of responses.

    ``dt`` is the Euler-Maruyama step. At 1e-3 the simulation has a small
    positive bias in the upper RT quantiles (about 0.1 s at q90 for the pilot
    parameters); 1e-4 reduces it to about 0.02 s.
    """
    rng = np.random.default_rng(seed)
    n_obs = len(df)
    idx = rng.integers(0, n_obs, size=int(n_sim))
    x_amt = df["x_amt"].to_numpy(dtype=float)[idx]
    x_delay = df["x_delay"].to_numpy(dtype=float)[idx]
    sim_rt, sim_choice = simulate_from_predictors(
        params, x_amt, x_delay, dt=dt, max_time=30.0, seed=seed + 1
    )
    valid = np.isfinite(sim_rt) & (sim_choice >= 0)
    sim_rt = sim_rt[valid]
    sim_choice = sim_choice[valid]

    observed_choice = df["choice"].to_numpy(dtype=int)
    drift = drift_from_predictors(
        params,
        df["x_amt"].to_numpy(dtype=float),
        df["x_delay"].to_numpy(dtype=float),
    )
    p_ll = choice_probability(drift, params.a)

    qtable = quantile_table(
        df["rt"].to_numpy(dtype=float),
        observed_choice,
        sim_rt,
        sim_choice,
        quantiles=quantiles,
    )
    coverage = {}
    for choice_value, label in ((0, "SS"), (1, "LL")):
        obs = df.loc[observed_choice == choice_value, "rt"].to_numpy(dtype=float)
        pred = sim_rt[sim_choice == choice_value]
        coverage[label] = central_interval_coverage(obs, pred, level=0.8)

    return {
        "n_simulated": int(sim_rt.size),
        "simulation_failure_rate": float(1.0 - sim_rt.size / max(n_sim, 1)),
        "observed_ll_proportion": float(np.mean(observed_choice)),
        "predicted_ll_proportion": float(np.mean(sim_choice)),
        "choice_metrics_in_sample": choice_metrics(observed_choice, p_ll),
        "rt_quantiles": qtable,
        "rt_quantile_mae": quantile_mae(qtable),
        "central_80_coverage": coverage,
        "calibration": choice_calibration(observed_choice, p_ll, n_bins=10),
        "choice_by_amount": choice_by_quantile_bin(
            df["x_amt"].to_numpy(dtype=float), observed_choice, p_ll, n_bins=5
        ),
        "choice_by_delay": choice_by_quantile_bin(
            df["x_delay"].to_numpy(dtype=float), observed_choice, p_ll, n_bins=5
        ),
        **(
            {"simulated_rt": sim_rt, "simulated_choice": sim_choice}
            if include_samples
            else {}
        ),
    }
