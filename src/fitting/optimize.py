"""Maximum-likelihood fitting for the classical DDM."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.classical_ddm import (
    DDMParameters,
    PARAMETER_NAMES,
    log_likelihood,
)


@dataclass
class FitResult:
    paper: str
    n_trials: int
    n_subjects: int
    params: dict[str, float]
    se: dict[str, float] | None
    log_likelihood: float
    converged: bool
    message: str
    nfev: int
    runtime_seconds: float
    min_rt: float | None = None
    bounds: list[list[float]] | None = None
    hessian: list[list[float]] | None = None
    hessian_condition: float | None = None
    hessian_positive_definite: bool | None = None
    start_objective_spread: float | None = None
    start_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _negative_log_likelihood(
    params: np.ndarray,
    rt: np.ndarray,
    choice: np.ndarray,
    x_amount: np.ndarray,
    x_delay: np.ndarray,
    n_terms: int,
) -> float:
    ll = log_likelihood(
        DDMParameters.from_array(params),
        rt,
        choice,
        x_amount,
        x_delay,
        n_terms=n_terms,
    )
    if not np.isfinite(ll):
        return 1e12
    return -ll


def _bounds(min_rt: float) -> list[tuple[float, float]]:
    # t0 must leave a small positive decision time for the fastest response.
    t0_upper = max(0.01, float(min_rt) - 0.005)
    return [
        (0.2, 5.0),          # a
        (0.001, t0_upper),   # t0
        (-20.0, 20.0),       # v0
        (-20.0, 20.0),       # beta_amount
        (-20.0, 20.0),       # beta_delay
    ]


def _starting_values(min_rt: float, n_starts: int) -> list[np.ndarray]:
    t0_hi = max(0.01, float(min_rt) - 0.005)
    base = [
        np.array([1.0, min(0.30, 0.50 * t0_hi), 0.0, 1.0, -0.5]),
        np.array([1.5, min(0.50, 0.70 * t0_hi), 0.5, 0.5, -0.2]),
        np.array([0.8, min(0.15, 0.30 * t0_hi), -0.5, 1.5, -1.0]),
        np.array([2.0, min(0.60, 0.80 * t0_hi), 0.0, 0.8, -0.8]),
        np.array([1.2, min(0.20, 0.40 * t0_hi), 0.2, 1.2, -0.3]),
    ]
    return base[: int(n_starts)]


def numerical_hessian(
    func,
    x: np.ndarray,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = rel_step * np.maximum(1.0, np.abs(x))
    hess = np.zeros((n, n), dtype=float)
    f0 = float(func(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        f_plus = float(func(x + ei))
        f_minus = float(func(x - ei))
        hess[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            fpp = float(func(x + ei + ej))
            fpm = float(func(x + ei - ej))
            fmp = float(func(x - ei + ej))
            fmm = float(func(x - ei - ej))
            value = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            hess[i, j] = value
            hess[j, i] = value
    return hess


def fit_classical_ddm(
    df: pd.DataFrame,
    paper: str,
    n_starts: int = 5,
    n_terms: int = 100,
    maxiter: int = 400,
    compute_hessian: bool = True,
) -> FitResult:
    """Fit the classical DDM to one study by maximum likelihood.

    ``compute_hessian=False`` is used for the many cross-validation fits, where
    only the point estimates and held-out predictions are needed.
    """
    required = {"rt", "choice", "x_amt", "x_delay"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for fitting: {missing}")
    if df.empty:
        raise ValueError(f"No trials for study {paper!r}")

    rt = df["rt"].to_numpy(dtype=float)
    choice = df["choice"].to_numpy(dtype=int)
    x_amt = df["x_amt"].to_numpy(dtype=float)
    x_delay = df["x_delay"].to_numpy(dtype=float)
    min_rt = float(np.min(rt))

    bounds = _bounds(min_rt)
    starts = _starting_values(min_rt, n_starts)

    def objective(params: np.ndarray) -> float:
        return _negative_log_likelihood(
            params, rt, choice, x_amt, x_delay, n_terms=n_terms
        )

    start_results: list[dict[str, Any]] = []
    best = None
    t_start = time.perf_counter()

    for start_index, x0 in enumerate(starts):
        x0 = np.array(
            [np.clip(v, lo, hi) for v, (lo, hi) in zip(x0, bounds)],
            dtype=float,
        )
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "ftol": 1e-9,
                "gtol": 1e-5,
                "maxls": 50,
            },
        )
        entry = {
            "start": start_index,
            "x0": x0.tolist(),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "fun": float(result.fun),
            "x": np.asarray(result.x, dtype=float).tolist(),
        }
        start_results.append(entry)
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    runtime = time.perf_counter() - t_start
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"All optimisation starts failed for study {paper!r}")

    x_best = np.asarray(best.x, dtype=float)
    finite_funs = [r["fun"] for r in start_results if np.isfinite(r["fun"])]
    spread = float(max(finite_funs) - min(finite_funs)) if finite_funs else np.nan
    if compute_hessian:
        hessian = numerical_hessian(objective, x_best)
        hessian_pd = bool(np.all(np.linalg.eigvalsh(hessian) > 0))
        condition = float(np.linalg.cond(hessian))
        if hessian_pd:
            cov = np.linalg.inv(hessian)
            se_values = np.sqrt(np.maximum(np.diag(cov), 0.0))
            se = dict(zip(PARAMETER_NAMES, se_values.tolist()))
        else:
            se = None
        hessian_list = hessian.tolist()
    else:
        hessian = None
        hessian_pd = None
        condition = None
        se = None
        hessian_list = None

    params = dict(zip(PARAMETER_NAMES, x_best.tolist()))
    return FitResult(
        paper=str(paper),
        n_trials=int(len(df)),
        n_subjects=int(df["subj_ident"].nunique()),
        params=params,
        se=se,
        log_likelihood=float(-best.fun),
        converged=bool(best.success),
        message=str(best.message),
        nfev=int(sum(r["nfev"] for r in start_results)),
        runtime_seconds=float(runtime),
        min_rt=min_rt,
        bounds=[[float(lo), float(hi)] for lo, hi in bounds],
        hessian=hessian_list,
        hessian_condition=condition,
        hessian_positive_definite=hessian_pd,
        start_objective_spread=spread,
        start_results=start_results,
    )
