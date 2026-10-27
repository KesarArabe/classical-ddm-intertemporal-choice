"""Tests for the data-cleaning and predictor pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import (  # noqa: E402
    add_predictors,
    clean_trials,
    fit_predictor_scaler,
)


def toy_frame() -> pd.DataFrame:
    rows = []
    base = {
        "paper": "Toy_2026",
        "sess_ident": "s1",
        "trial_idx": 1,
        "ss_value": 10.0,
        "ss_time": 0.0,
        "ll_value": 20.0,
        "ll_time": 30.0,
        "choice": 1,
        "rt": 1.0,
        "subj_excl": False,
        "trial_excl": False,
        "time_pressure": 0,
        "rt_data_analyzed": 1,
        "rt_in_full_seconds": 0,
        "additional_methods": np.nan,
        "presentation_of_information": "Simultaneous",
    }
    # 0-1: valid trials
    for i in range(2):
        row = dict(base, subj_ident=f"S{i+1}", trial_idx=i + 1)
        rows.append(row)
    # 2: trial excluded
    rows.append(dict(base, subj_ident="S1", trial_idx=3, trial_excl=True))
    # 3: RT too fast
    rows.append(dict(base, subj_ident="S1", trial_idx=4, rt=0.1))
    # 4: zero amount
    rows.append(dict(base, subj_ident="S1", trial_idx=5, ss_value=0.0))
    # 5: equal delays
    rows.append(dict(base, subj_ident="S1", trial_idx=6, ss_time=30.0))
    # 6: invalid choice
    rows.append(dict(base, subj_ident="S1", trial_idx=7, choice=2))
    # 7: flagged subject (all rows must go)
    rows.append(dict(base, subj_ident="S3", trial_idx=8, subj_excl=True))
    rows.append(dict(base, subj_ident="S3", trial_idx=9, subj_excl=True))
    return pd.DataFrame(rows)


class TestCleaning(unittest.TestCase):
    def test_trial_level_rules(self):
        cleaned, audit = clean_trials(toy_frame(), apply_design_filters=False)
        # Two valid trials for S1/S2; S3 removed entirely.
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(set(cleaned["subj_ident"]), {"S1", "S2"})
        self.assertEqual(audit["raw_rows"], 9)
        self.assertIn("after_rt_range", audit)

    def test_design_filter_rejects_time_pressure(self):
        df = toy_frame()
        df["time_pressure"] = 1
        cleaned, _ = clean_trials(df, apply_design_filters=True)
        self.assertEqual(len(cleaned), 0)

    def test_design_filter_rejects_mouse_tracking(self):
        df = toy_frame()
        df["additional_methods"] = "Mouse tracking"
        cleaned, _ = clean_trials(df, apply_design_filters=True)
        self.assertEqual(len(cleaned), 0)

    def test_predictor_standardisation(self):
        df = pd.DataFrame(
            {
                "paper": ["A"] * 4 + ["B"] * 3,
                "ss_value": [10, 10, 10, 10, 5, 5, 5],
                "ll_value": [20, 30, 20, 30, 10, 15, 20],
                "ss_time": [0, 0, 10, 10, 0, 0, 0],
                "ll_time": [30, 60, 30, 60, 30, 60, 90],
            }
        )
        df, scaler = add_predictors(df)
        for paper, group in df.groupby("paper"):
            self.assertAlmostEqual(float(group["x_amt"].mean()), 0.0, places=10)
            self.assertAlmostEqual(float(group["x_amt"].std(ddof=0)), 1.0, places=10)
            self.assertAlmostEqual(float(group["x_delay"].mean()), 0.0, places=10)
            self.assertAlmostEqual(float(group["x_delay"].std(ddof=0)), 1.0, places=10)
        # Reapplying the same scaler must be idempotent.
        again, _ = add_predictors(df[["paper", "ss_value", "ll_value", "ss_time", "ll_time"]], scaler=scaler)
        np.testing.assert_allclose(again["x_amt"], df["x_amt"], atol=1e-12)
        np.testing.assert_allclose(again["x_delay"], df["x_delay"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
