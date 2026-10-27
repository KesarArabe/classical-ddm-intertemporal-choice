"""Synthetic parameter-recovery test for the classical DDM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fitting.optimize import fit_classical_ddm  # noqa: E402
from src.models.classical_ddm import (  # noqa: E402
    DDMParameters,
    log_likelihood,
    simulate_from_predictors,
)


class TestRecovery(unittest.TestCase):
    def test_recover_known_parameters(self):
        true = DDMParameters(
            a=1.2, t0=0.30, v0=0.20, beta_amount=0.8, beta_delay=-0.6
        )
        rng = np.random.default_rng(20261024)
        n = 4000
        z = rng.standard_normal((n, 2))
        x_amt = z[:, 0]
        x_delay = 0.2 * z[:, 0] + np.sqrt(1 - 0.2**2) * z[:, 1]
        rt, choice = simulate_from_predictors(
            true, x_amt, x_delay, dt=0.001, max_time=30.0, seed=99
        )
        valid = np.isfinite(rt) & (choice >= 0)
        df = pd.DataFrame(
            {
                "paper": "synthetic",
                "subj_ident": "S1",
                "rt": rt[valid],
                "choice": choice[valid],
                "x_amt": x_amt[valid],
                "x_delay": x_delay[valid],
            }
        )
        fit = fit_classical_ddm(df, "synthetic", n_starts=2, n_terms=100)
        self.assertTrue(fit.converged)
        self.assertTrue(fit.hessian_positive_definite)
        ll_true = log_likelihood(
            true,
            df["rt"].to_numpy(),
            df["choice"].to_numpy(),
            df["x_amt"].to_numpy(),
            df["x_delay"].to_numpy(),
        )
        self.assertGreaterEqual(fit.log_likelihood, ll_true - 1e-6)

        est = fit.params
        self.assertLess(abs(est["a"] - true.a) / true.a, 0.25)
        self.assertLess(abs(est["t0"] - true.t0), 0.15)
        self.assertLess(abs(est["v0"] - true.v0), 0.50)
        self.assertLess(abs(est["beta_amount"] - true.beta_amount) / abs(true.beta_amount), 0.35)
        self.assertLess(abs(est["beta_delay"] - true.beta_delay) / abs(true.beta_delay), 0.35)


if __name__ == "__main__":
    unittest.main()
