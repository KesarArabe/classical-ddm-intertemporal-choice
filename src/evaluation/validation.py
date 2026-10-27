"""Independent numerical validation of the classical-DDM implementation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.special import expit

from src.models.classical_ddm import ddm_density, simulate_ddm


def pyddm_grid_check(
    drifts=(0.0, 0.5, -1.0, 2.0),
    boundaries=(0.6, 1.0, 2.0),
    times=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    n_terms: int = 100,
) -> dict:
    """Compare the implemented density with PyDDM's analytical solver."""
    from pyddm.analytic import analytic_ddm

    max_abs = 0.0
    max_rel = 0.0
    rows = []
    for v in drifts:
        for a in boundaries:
            t = np.asarray(times, dtype=float)
            ours_u, ours_l = ddm_density(t, v, a, t0=0.0, n_terms=n_terms)
            ref_u, ref_l = analytic_ddm(
                v, 1.0, a / 2.0, t, shift=0.5, force_python=True
            )
            for choice, ours, ref in (("upper", ours_u, ref_u), ("lower", ours_l, ref_l)):
                abs_err = float(np.max(np.abs(ours - ref)))
                denom = float(np.max(np.abs(ref)))
                rel_err = abs_err / denom if denom > 0 else abs_err
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)
                rows.append(
                    {
                        "drift": float(v),
                        "boundary": float(a),
                        "choice": choice,
                        "max_abs_error": abs_err,
                        "max_rel_error": rel_err,
                    }
                )
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "rows": rows,
    }


def density_normalisation_check(
    drift: float = 0.5,
    boundary: float = 1.2,
    n_terms: int = 100,
    n_grid: int = 80000,
) -> dict:
    """Integrate the two densities and compare with the analytic choice probability."""
    t = np.logspace(-6, np.log10(100.0), int(n_grid))
    f_upper, f_lower = ddm_density(t, drift, boundary, t0=0.0, n_terms=n_terms)
    integral_upper = float(simpson(f_upper, x=t))
    integral_lower = float(simpson(f_lower, x=t))
    total = integral_upper + integral_lower
    analytic_upper = float(expit(drift * boundary))
    return {
        "drift": float(drift),
        "boundary": float(boundary),
        "integral_upper": integral_upper,
        "integral_lower": integral_lower,
        "integral_total": total,
        "analytic_upper": analytic_upper,
        "total_error": total - 1.0,
        "upper_error": integral_upper - analytic_upper,
    }


def _analytic_quantiles(
    drift: float,
    boundary: float,
    choice_value: int,
    quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
    n_terms: int = 100,
    n_grid: int = 80000,
) -> np.ndarray:
    t = np.logspace(-6, np.log10(100.0), int(n_grid))
    f_upper, f_lower = ddm_density(t, drift, boundary, t0=0.0, n_terms=n_terms)
    density = f_upper if choice_value == 1 else f_lower
    cdf = np.cumsum(density * np.gradient(t))
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, t)


def simulation_check(
    drift: float = 0.8,
    boundary: float = 1.2,
    t0: float = 0.3,
    n_trials: int = 5000,
    dt: float = 1e-4,
    seed: int = 12345,
    quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict:
    """Compare Euler-Maruyama simulation with the analytic model."""
    rt, choice = simulate_ddm(
        drift,
        boundary,
        t0=t0,
        dt=dt,
        max_time=30.0,
        seed=seed,
        n_trials=n_trials,
    )
    valid = np.isfinite(rt) & (choice >= 0)
    rt = rt[valid]
    choice = choice[valid]
    empirical_p = float(np.mean(choice == 1))
    analytic_p = float(expit(drift * boundary))

    quantile_rows = []
    for choice_value, label in ((0, "SS"), (1, "LL")):
        observed = rt[choice == choice_value]
        analytic = _analytic_quantiles(drift, boundary, choice_value, quantiles)
        for q, aq in zip(quantiles, analytic):
            quantile_rows.append(
                {
                    "choice": label,
                    "quantile": float(q),
                    "simulated": float(np.quantile(observed, q)),
                    "analytic": float(aq) + t0,
                    "difference": float(np.quantile(observed, q) - (aq + t0)),
                }
            )
    return {
        "drift": float(drift),
        "boundary": float(boundary),
        "t0": float(t0),
        "n_trials": int(rt.size),
        "simulation_failure_rate": float(1.0 - rt.size / n_trials),
        "empirical_ll_proportion": empirical_p,
        "analytic_ll_proportion": analytic_p,
        "choice_proportion_difference": empirical_p - analytic_p,
        "quantile_rows": quantile_rows,
        "max_abs_quantile_difference": float(
            max(abs(r["difference"]) for r in quantile_rows)
        ),
    }
