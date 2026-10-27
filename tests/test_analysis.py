"""Tests for cross-validation isolation, clustered SEs, PIT, and bound checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.cross_validation import assign_subject_folds  # noqa: E402
from src.evaluation.diagnostics import parameter_bound_diagnostics  # noqa: E402
from src.evaluation.rt_cdf import (  # noqa: E402
    conditional_pit,
    model_quantile_table,
    pit_calibration,
)
from src.fitting.clustered import clustered_covariance  # noqa: E402
from src.fitting.optimize import fit_classical_ddm  # noqa: E402
from src.models.classical_ddm import (  # noqa: E402
    DDMParameters,
    PARAMETER_NAMES,
    simulate_from_predictors,
)


def _simulate_trials(
    n_subjects: int = 12,
    trials_per_subject: int = 80,
    dt: float = 1e-3,
    seed: int = 20261025,
) -> pd.DataFrame:
    true = DDMParameters(
        a=1.2, t0=0.30, v0=0.20, beta_amount=0.8, beta_delay=-0.6
    )
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        n = trials_per_subject
        z = rng.standard_normal((n, 2))
        x_amt = z[:, 0]
        x_delay = 0.2 * z[:, 0] + np.sqrt(1 - 0.2**2) * z[:, 1]
        rt, choice = simulate_from_predictors(
            true, x_amt, x_delay, dt=dt, max_time=30.0, seed=seed + subject
        )
        valid = np.isfinite(rt) & (choice >= 0)
        rows.append(
            pd.DataFrame(
                {
                    "paper": "synthetic",
                    "subj_ident": f"S{subject:02d}",
                    "rt": rt[valid],
                    "choice": choice[valid],
                    "x_amt": x_amt[valid],
                    "x_delay": x_delay[valid],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TestFolds(unittest.TestCase):
    def test_folds_are_subject_isolated_and_cover_everyone(self):
        df = pd.DataFrame(
            {
                "subj_ident": [
                    f"S{i:02d}" for i in range(23) for _ in range(5 + i % 7)
                ]
            }
        )
        folds = assign_subject_folds(df, n_folds=5, seed=7)
        self.assertEqual(set(folds.unique()), set(range(5)))
        self.assertEqual(folds.index.nunique(), 23)
        for fold in range(5):
            train = set(folds[folds != fold].index)
            test = set(folds[folds == fold].index)
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(folds.index))
        loads = (
            df.assign(fold=df["subj_ident"].map(folds))
            .groupby("fold")
            .size()
        )
        self.assertLessEqual(loads.max() - loads.min(), 7)


class TestClustered(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _simulate_trials(n_subjects=10, trials_per_subject=70)
        cls.fit = fit_classical_ddm(
            cls.df,
            "synthetic",
            n_starts=1,
            n_terms=40,
            maxiter=200,
            compute_hessian=True,
        )

    def test_sandwich_dimensions_and_positive_diagonal(self):
        payload = clustered_covariance(
            self.fit, self.df, n_terms=40
        )
        cov = np.asarray(payload["clustered_covariance"], dtype=float)
        naive = np.asarray(payload["naive_covariance"], dtype=float)
        self.assertEqual(cov.shape, (5, 5))
        self.assertEqual(naive.shape, (5, 5))
        self.assertTrue(np.all(np.diag(cov) > 0))
        self.assertTrue(np.all(np.isfinite(cov)))
        self.assertEqual(payload["n_subjects"], 10)
        self.assertEqual(len(payload["subject_scores"]), 10)
        effects = np.asarray(
            [row["design_effect"] for row in payload["rows"]], dtype=float
        )
        self.assertTrue(np.all(np.isfinite(effects)))
        self.assertTrue(np.all(effects > 0))


class TestPIT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.true = DDMParameters(
            a=1.2, t0=0.30, v0=0.20, beta_amount=0.8, beta_delay=-0.6
        )
        rng = np.random.default_rng(4242)
        n = 2500
        z = rng.standard_normal((n, 2))
        x_amt = z[:, 0]
        x_delay = 0.2 * z[:, 0] + np.sqrt(1 - 0.2**2) * z[:, 1]
        rt, choice = simulate_from_predictors(
            cls.true, x_amt, x_delay, dt=1e-4, max_time=30.0, seed=31
        )
        valid = np.isfinite(rt) & (choice >= 0)
        cls.df = pd.DataFrame(
            {
                "rt": rt[valid],
                "choice": choice[valid],
                "x_amt": x_amt[valid],
                "x_delay": x_delay[valid],
            }
        )
        cls.payload = conditional_pit(cls.df, cls.true, n_terms=12)
        cls.calibration = pit_calibration(cls.payload)

    def test_pit_is_approximately_uniform(self):
        cal = self.calibration["conditional"]
        self.assertEqual(cal["n"], len(self.df))
        self.assertLess(abs(cal["mean"] - 0.5), 0.05)
        self.assertLess(cal["ks_statistic"], 0.07)
        self.assertLess(cal["max_bin_deviation"], 0.06)
        self.assertEqual(self.calibration["n_below_t0"], 0)

    def test_conditional_and_marginal_pit_coincide_for_constant_drift(self):
        # With constant drift and a midpoint start, the boundary-density ratio
        # is constant, so choice and RT are independent.
        np.testing.assert_allclose(
            self.payload["pit_conditional"],
            self.payload["pit_marginal"],
            atol=2e-3,
        )


class TestModelQuantiles(unittest.TestCase):
    def test_quantiles_are_ordered_and_choice_invariant_for_constant_drift(self):
        # With constant predictors the drift is identical on every trial, so
        # the choice-conditional RT distributions coincide.
        df = _simulate_trials(n_subjects=4, trials_per_subject=60)
        df["x_amt"] = 0.0
        df["x_delay"] = 0.0
        params = DDMParameters(
            a=1.2, t0=0.30, v0=0.20, beta_amount=0.8, beta_delay=-0.6
        )
        rows = model_quantile_table(df, params, n_terms=12, n_grid=300)
        table = pd.DataFrame(rows)
        for label in ("all", "SS", "LL"):
            values = table[table["choice"] == label].sort_values("quantile")[
                "predicted"
            ].to_numpy(dtype=float)
            self.assertTrue(np.all(np.diff(values) > 0))
        ss = table[table["choice"] == "SS"].sort_values("quantile")[
            "predicted"
        ].to_numpy(dtype=float)
        ll = table[table["choice"] == "LL"].sort_values("quantile")[
            "predicted"
        ].to_numpy(dtype=float)
        np.testing.assert_allclose(ss, ll, atol=1e-6)


class TestBounds(unittest.TestCase):
    def test_t0_near_upper_bound_is_flagged(self):
        min_rt = 0.50
        upper = min_rt - 0.005
        fit = SimpleNamespace(
            params={
                "a": 1.2,
                "t0": upper - 0.005,
                "v0": 0.2,
                "beta_amount": 0.8,
                "beta_delay": -0.6,
            },
            bounds=[
                [0.2, 5.0],
                [0.001, upper],
                [-20.0, 20.0],
                [-20.0, 20.0],
                [-20.0, 20.0],
            ],
        )
        df = pd.DataFrame({"rt": [min_rt, 0.8, 1.1]})
        payload = parameter_bound_diagnostics(fit, df)
        self.assertTrue(payload["t0_at_upper_5pct"])
        self.assertFalse(payload["hessian_se_valid_for_t0"])
        self.assertAlmostEqual(payload["t0_margin_to_upper_seconds"], 0.005)


if __name__ == "__main__":
    unittest.main()
