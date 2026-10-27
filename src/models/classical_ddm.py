"""Classical drift diffusion model: likelihood, choice probabilities, simulation.

The model has constant boundaries, a fixed midpoint starting point, unit
diffusion noise, and trial-specific drift. The first-passage density uses the
eigenfunction/reflection series for constant boundaries (Anderson, 1960;
Navarro & Fuss, 2009). The implementation is vectorised over trials so that
continuous trial-level drift can be fitted efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import expit

PARAMETER_NAMES = ("a", "t0", "v0", "beta_amount", "beta_delay")

# Numerical floor for densities. Observed RTs in this dataset have densities
# far above this, but the floor keeps log-likelihoods finite in bad proposals.
_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class DDMParameters:
    a: float
    t0: float
    v0: float
    beta_amount: float
    beta_delay: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.a, self.t0, self.v0, self.beta_amount, self.beta_delay],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "DDMParameters":
        return cls(*[float(v) for v in values])


def _series_sum(t: np.ndarray, half_boundary: float, n_terms: int) -> np.ndarray:
    """Reflection-series sum for constant symmetric boundaries.

    This is the vectorised form of the Anderson/Navarro-Fuss expansion used by
    PyDDM's analytical solver. Starting point is the midpoint, so the two
    boundaries are ``+b`` and ``-b`` with ``b = a / 2``.

    The reflection series converges quickly for short decision times but
    suffers catastrophic cancellation when ``t`` is large relative to ``a**2``
    (the terms are large and nearly cancel). It is therefore only used for
    ``t < _SWITCH_FRACTION * a**2``; see ``_eigen_series`` for the long-time
    regime.
    """
    b = float(half_boundary)
    tmp = -4.0 * b / t
    total = np.zeros_like(t, dtype=float)
    for n in range(int(n_terms)):
        c1 = (4 * n + 1) * b
        c2 = (4 * n + 3) * b
        e1 = np.exp(tmp * n * (2 * n + 1) * b)
        e2 = np.exp(tmp * (n + 1) * (2 * n + 1) * b)
        total += c1 * e1 - c2 * e2
    return total


def _log_eigen_series(t: np.ndarray, boundary: float, n_terms: int) -> np.ndarray:
    """Log of the sine-series sum used for long decision times.

    For a midpoint start the upper and lower first-passage densities share the
    same series

        G(t) = (pi / a**2) * sum_{k odd} k * (-1)**((k-1)/2)
               * exp(-k**2 * pi**2 * t / (2 * a**2)),

    and ``f_upper = G * exp(v*a/2 - v**2*t/2)``,
    ``f_lower = G * exp(-v*a/2 - v**2*t/2)``.

    For ``t >= a**2 / 2`` the first term dominates and the alternating sum is
    free of cancellation. Factoring out ``exp(-c)`` with
    ``c = pi**2 * t / (2 * a**2)`` keeps the result finite in log space even
    when ``G`` itself would underflow.
    """
    a = float(boundary)
    c = (np.pi * np.pi * t) / (2.0 * a * a)
    bracket = np.zeros_like(t, dtype=float)
    for k in range(1, int(n_terms) + 1, 2):
        sign = 1.0 if ((k - 1) // 2) % 2 == 0 else -1.0
        bracket += sign * k * np.exp(-((k * k - 1) * c))
    return np.log(np.pi / (a * a)) - c + np.log(bracket)


# Relative decision time at which the two expansions are both accurate to
# about machine precision (checked against 50-digit arithmetic).
_SWITCH_FRACTION = 0.5


def _log_densities(
    t: np.ndarray,
    drift: np.ndarray,
    boundary: float,
    n_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Log first-passage densities for ``t > 0`` using the stable expansion.

    Short times use the reflection series; long times use the eigenfunction
    (sine) series. Both are exact; the split avoids the cancellation that
    destroys the reflection series when ``t`` is much larger than ``a**2``.
    """
    a = float(boundary)
    b = a / 2.0
    log_upper = np.empty_like(t, dtype=float)
    log_lower = np.empty_like(t, dtype=float)
    switch = _SWITCH_FRACTION * a * a
    short = t < switch
    long = ~short

    if np.any(short):
        ts = t[short]
        vs = drift[short]
        series = np.maximum(_series_sum(ts, b, n_terms), _TINY)
        log_norm = -0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(ts)
        log_upper[short] = (
            log_norm - (b - vs * ts) ** 2 / (2.0 * ts) + np.log(series)
        )
        log_lower[short] = (
            log_norm - (b + vs * ts) ** 2 / (2.0 * ts) + np.log(series)
        )

    if np.any(long):
        tl = t[long]
        vl = drift[long]
        log_g = _log_eigen_series(tl, a, n_terms)
        log_upper[long] = log_g + vl * a / 2.0 - vl * vl * tl / 2.0
        log_lower[long] = log_g - vl * a / 2.0 - vl * vl * tl / 2.0

    return log_upper, log_lower


def ddm_density(
    rt: np.ndarray | float,
    drift: np.ndarray | float,
    boundary: float,
    t0: float = 0.0,
    n_terms: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint density of choice and RT for the classical DDM.

    Parameters
    ----------
    rt:
        Observed response times in seconds.
    drift:
        Trial-wise drift rate (positive favours the upper/LL boundary).
    boundary:
        Total boundary separation ``a``.
    t0:
        Non-decision time in seconds.
    n_terms:
        Number of series terms.

    Returns
    -------
    (f_upper, f_lower)
        Densities at the upper (LL) and lower (SS) boundaries. Values are
        zero where ``rt <= t0``.
    """
    rt_arr = np.asarray(rt, dtype=float)
    drift_arr = np.asarray(drift, dtype=float)
    rt_arr, drift_arr = np.broadcast_arrays(rt_arr, drift_arr)
    t = rt_arr - float(t0)

    f_upper = np.zeros_like(t, dtype=float)
    f_lower = np.zeros_like(t, dtype=float)
    valid = t > 0
    if not np.any(valid):
        return f_upper, f_lower

    tv = t[valid]
    vv = drift_arr[valid]
    log_upper_valid, log_lower_valid = _log_densities(
        tv, vv, float(boundary), n_terms
    )

    # Clip to avoid overflow when exponentiating extreme proposals.
    f_upper[valid] = np.exp(np.clip(log_upper_valid, -745.0, 700.0))
    f_lower[valid] = np.exp(np.clip(log_lower_valid, -745.0, 700.0))
    return f_upper, f_lower


def ddm_log_density(
    rt: np.ndarray | float,
    drift: np.ndarray | float,
    boundary: float,
    t0: float = 0.0,
    n_terms: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Log joint density of choice and RT.

    Returns ``-inf`` where ``rt <= t0``. This is the numerically stable form
    used by the likelihood.
    """
    rt_arr = np.asarray(rt, dtype=float)
    drift_arr = np.asarray(drift, dtype=float)
    rt_arr, drift_arr = np.broadcast_arrays(rt_arr, drift_arr)
    t = rt_arr - float(t0)

    log_upper = np.full_like(t, -np.inf, dtype=float)
    log_lower = np.full_like(t, -np.inf, dtype=float)
    valid = t > 0
    if not np.any(valid):
        return log_upper, log_lower

    tv = t[valid]
    vv = drift_arr[valid]
    log_upper[valid], log_lower[valid] = _log_densities(
        tv, vv, float(boundary), n_terms
    )
    return log_upper, log_lower


def drift_from_predictors(
    params: DDMParameters | Iterable[float],
    x_amount: np.ndarray,
    x_delay: np.ndarray,
) -> np.ndarray:
    p = params if isinstance(params, DDMParameters) else DDMParameters.from_array(params)
    return p.v0 + p.beta_amount * np.asarray(x_amount) + p.beta_delay * np.asarray(x_delay)


def log_likelihood(
    params: DDMParameters | Iterable[float],
    rt: np.ndarray,
    choice: np.ndarray,
    x_amount: np.ndarray,
    x_delay: np.ndarray,
    n_terms: int = 100,
) -> float:
    """Summed log-likelihood for one study."""
    p = params if isinstance(params, DDMParameters) else DDMParameters.from_array(params)
    if not np.isfinite([p.a, p.t0, p.v0, p.beta_amount, p.beta_delay]).all():
        return -np.inf
    if p.a <= 0 or p.t0 < 0:
        return -np.inf

    drift = drift_from_predictors(p, x_amount, x_delay)
    log_upper, log_lower = ddm_log_density(
        rt, drift, p.a, p.t0, n_terms=n_terms
    )
    choice_arr = np.asarray(choice).astype(int)
    log_f = np.where(choice_arr == 1, log_upper, log_lower)
    if np.any(~np.isfinite(log_f)):
        return -np.inf
    return float(np.sum(log_f))


def choice_probability(
    drift: np.ndarray | float,
    boundary: float,
) -> np.ndarray:
    """Probability of the upper (LL) choice for constant drift.

    With unit noise and a midpoint start, ``P(LL) = expit(v * a)``.
    """
    return expit(np.asarray(drift, dtype=float) * float(boundary))


def simulate_ddm(
    drift: np.ndarray | float,
    boundary: float,
    t0: float = 0.0,
    dt: float = 0.001,
    max_time: float = 30.0,
    seed: int | None = None,
    n_trials: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the classical DDM with Euler-Maruyama.

    Returns ``(rt, choice)`` where ``choice == 1`` means the upper/LL
    boundary. Trials that do not terminate by ``max_time`` are returned with
    ``rt = nan`` and ``choice = -1``. If ``drift`` is scalar, pass
    ``n_trials`` to simulate more than one trial.
    """
    drift_arr = np.asarray(drift, dtype=float)
    if drift_arr.ndim == 0:
        if n_trials is None:
            n_trials = 1
        drift_arr = np.full(int(n_trials), float(drift_arr))
    n = drift_arr.size
    rng = np.random.default_rng(seed)

    x = np.full(n, float(boundary) / 2.0, dtype=float)
    rt = np.full(n, np.nan, dtype=float)
    choice = np.full(n, -1, dtype=int)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(np.ceil(max_time / dt))

    for step in range(1, n_steps + 1):
        active = choice < 0
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        x[idx] += drift_arr[idx] * dt + sqrt_dt * rng.standard_normal(idx.size)
        hit_upper = idx[x[idx] >= boundary]
        hit_lower = idx[x[idx] <= 0.0]
        if hit_upper.size:
            rt[hit_upper] = step * dt + t0
            choice[hit_upper] = 1
        if hit_lower.size:
            rt[hit_lower] = step * dt + t0
            choice[hit_lower] = 0

    return rt, choice


def simulate_from_predictors(
    params: DDMParameters | Iterable[float],
    x_amount: np.ndarray,
    x_delay: np.ndarray,
    dt: float = 0.001,
    max_time: float = 30.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = params if isinstance(params, DDMParameters) else DDMParameters.from_array(params)
    drift = drift_from_predictors(p, x_amount, x_delay)
    return simulate_ddm(drift, p.a, p.t0, dt=dt, max_time=max_time, seed=seed)
