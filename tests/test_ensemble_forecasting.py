"""
Unit tests for Phase 5: Ensemble Demand Forecasting Module.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ensemble_forecasting import (
    calculate_model_weights,
    combine_forecasts,
    train_predict_ensemble
)


class TestEnsembleForecasting(unittest.TestCase):

    def setUp(self):
        # 90 days of synthetic demand for an eligible test product
        dates = pd.date_range("2011-08-01", "2011-10-30", freq="D")
        np.random.seed(42)
        dow = dates.dayofweek
        base_demand = np.where(dow == 5, 0, np.random.randint(15, 60, size=len(dates)))
        self.history_df = pd.DataFrame({
            "Date": dates,
            "StockCode": ["TEST_ITEM"] * len(dates),
            "Description": ["Test Ensemble Product"] * len(dates),
            "Demand": base_demand
        })

    def test_weight_calculation_and_sum(self):
        val_errors = {
            "Prophet": 10.0,
            "LightGBM": 20.0,
            "Moving Average 28D": 5.0
        }
        weights = calculate_model_weights(val_errors, power=1.0)

        # 1. Weights must sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        # 2. Lower error must yield strictly higher weight:
        # MA28 (err=5) > Prophet (err=10) > LightGBM (err=20)
        self.assertGreater(weights["Moving Average 28D"], weights["Prophet"])
        self.assertGreater(weights["Prophet"], weights["LightGBM"])

        # 3. All weights must be strictly positive
        for m, w in weights.items():
            self.assertGreater(w, 0.0)

    def test_combine_forecasts(self):
        preds = {
            "ModelA": np.array([10.0, 20.0, 30.0]),
            "ModelB": np.array([20.0, 40.0, 60.0])
        }
        weights = {"ModelA": 0.5, "ModelB": 0.5}
        combined = combine_forecasts(preds, weights)

        # Expected: 0.5*10 + 0.5*20 = 15.0, 0.5*20 + 0.5*40 = 30.0, 0.5*30 + 0.5*60 = 45.0
        np.testing.assert_allclose(combined, [15.0, 30.0, 45.0])

    def test_no_negative_forecasts(self):
        # Even if a component model produced negative values, ensemble must clip to 0
        preds = {
            "ModelA": np.array([-10.0, 5.0]),
            "ModelB": np.array([-20.0, 10.0])
        }
        weights = {"ModelA": 0.5, "ModelB": 0.5}
        combined = combine_forecasts(preds, weights)
        self.assertTrue((combined >= 0.0).all())
        self.assertEqual(combined[0], 0.0)

    def test_train_predict_ensemble(self):
        horizon = 14
        weights = {"Prophet": 0.4, "LightGBM": 0.3, "Moving Average 28D": 0.3}
        out_df = train_predict_ensemble(self.history_df, horizon=horizon, weights=weights)

        self.assertEqual(len(out_df), horizon)
        self.assertEqual(out_df["Model"].iloc[0], "Ensemble")
        self.assertIn("Predicted_Demand", out_df.columns)
        self.assertIn("Prophet_Prediction", out_df.columns)
        self.assertIn("LightGBM_Prediction", out_df.columns)
        self.assertIn("Baseline_Prediction", out_df.columns)
        self.assertIn("Ensemble_Prediction", out_df.columns)

        # Verify predictions are non-negative
        self.assertTrue((out_df["Predicted_Demand"] >= 0).all())

        # Check date alignment
        expected_start = pd.to_datetime("2011-10-31")
        self.assertEqual(out_df["Date"].min(), expected_start)

    def test_zero_demand_handling(self):
        history_zero = pd.DataFrame({
            "Date": pd.date_range("2011-09-01", periods=60, freq="D"),
            "StockCode": ["ZERO_ITEM"] * 60,
            "Description": ["Zero Item"] * 60,
            "Demand": [0] * 60
        })
        out_df = train_predict_ensemble(history_zero, horizon=7)
        self.assertEqual(len(out_df), 7)
        self.assertTrue((out_df["Predicted_Demand"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
