"""Mathematical and numerical validation of the classical-DDM likelihood."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.validation import (  # noqa: E402
    density_normalisation_check,
    pyddm_grid_check,
    simulation_check,
)
from src.models.classical_ddm import (  # noqa: E402
    ddm_density,
    ddm_log_density,
    log_likelihood,
)


class TestDensity(unittest.TestCase):
    def test_matches_pyddm(self):
        result = pyddm_grid_check()
        self.assertLess(result["max_rel_error"], 1e-10)
        self.assertLess(result["max_abs_error"], 1e-12)

    def test_normalisation_and_choice_probability(self):
        for drift, boundary in ((0.0, 1.0), (0.5, 1.2), (2.0, 2.0), (-1.0, 0.8)):
            result = density_normalisation_check(drift, boundary)
            self.assertAlmostEqual(result["integral_total"], 1.0, places=8)
            self.assertAlmostEqual(
                result["integral_upper"], result["analytic_upper"], places=8
            )

    def test_log_density_matches_density(self):
        t = np.array([0.1, 0.3, 1.0, 3.0])
        drift = np.array([-0.5, 0.0, 0.8, 1.5])
        fu, fl = ddm_density(t, drift, boundary=1.3, t0=0.2)
        lu, ll = ddm_log_density(t, drift, boundary=1.3, t0=0.2)
        for density, log_density in ((fu, lu), (fl, ll)):
            positive = density > 0
            np.testing.assert_allclose(
                np.log(density[positive]), log_density[positive],
                rtol=1e-12, atol=1e-12,
            )
            self.assertTrue(np.all(np.isneginf(log_density[~positive])))

    def test_nondecision_shift(self):
        t = np.array([0.5, 1.0, 2.0])
        fu0, fl0 = ddm_density(t, 0.4, 1.1, t0=0.0)
        fu1, fl1 = ddm_density(t + 0.3, 0.4, 1.1, t0=0.3)
        np.testing.assert_allclose(fu0, fu1, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fl0, fl1, rtol=1e-12, atol=1e-14)

    def test_long_time_density_matches_high_precision(self):
        """Regression test for the reflection-series cancellation bug.

        For t >> a**2 the reflection series cancels to zero in double
        precision. Reference values were computed with 140-digit decimal
        arithmetic. The old implementation returned a clamped near-zero
        density for the first case and zero for the others.
        """
        cases = [
            (10.0, -2.0, 0.4, -325.847826184322, -325.047826184322),
            (10.0, 0.5, 1.5, -22.473654555010, -23.223654555010),
            (5.0, 0.0, 0.8, -36.962125203277, -36.962125203277),
            (30.0, 0.3, 2.7, -22.094601645952, -22.904601645952),
        ]
        for t, drift, boundary, expected_upper, expected_lower in cases:
            upper, lower = ddm_log_density(
                np.array([t]), drift, boundary, t0=0.0
            )
            self.assertAlmostEqual(upper[0], expected_upper, places=8)
            self.assertAlmostEqual(lower[0], expected_lower, places=8)
            # The two boundaries must differ by exactly exp(drift * a).
            self.assertAlmostEqual(
                upper[0] - lower[0], drift * boundary, places=10
            )

    def test_likelihood_rejects_rt_before_t0(self):
        ll = log_likelihood(
            [1.0, 0.5, 0.0, 1.0, -0.5],
            np.array([0.4, 0.6]),
            np.array([1, 0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        self.assertEqual(ll, -np.inf)


class TestSimulation(unittest.TestCase):
    def test_simulation_matches_analytic_choice_and_quantiles(self):
        result = simulation_check(n_trials=5000, seed=123)
        self.assertLess(abs(result["choice_proportion_difference"]), 0.03)
        self.assertLess(result["max_abs_quantile_difference"], 0.08)


if __name__ == "__main__":
    unittest.main()
