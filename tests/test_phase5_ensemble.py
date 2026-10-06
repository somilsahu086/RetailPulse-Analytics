"""
Unit tests for Phase 5: PyTorch LSTM & Hybrid Prophet + LSTM Ensemble.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lstm_forecasting import (
    MinMaxDemandScaler,
    create_sequences,
    DemandLSTM,
    train_predict_lstm
)
from ensemble_forecasting import (
    optimize_ensemble_weights,
    combine_prophet_lstm_predictions,
    evaluate_product_phase5
)
from evaluation import evaluate_forecast


class TestPhase5Ensemble(unittest.TestCase):

    def setUp(self):
        # 100 days of daily continuous demand
        dates = pd.date_range("2011-06-01", "2011-09-08", freq="D")
        np.random.seed(42)
        # Demand pattern with zero on Saturdays (dow==5)
        dow = dates.dayofweek
        demand = np.where(dow == 5, 0, np.random.randint(10, 50, size=len(dates)))
        self.history_df = pd.DataFrame({
            "Date": dates,
            "Demand": demand
        })

    def test_minmax_scaler_no_leakage(self):
        scaler = MinMaxDemandScaler()
        train_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        scaler.fit(train_vals)

        scaled = scaler.transform(train_vals)
        self.assertAlmostEqual(scaled[0], 0.0)
        self.assertAlmostEqual(scaled[-1], 1.0)

        # Unscale back
        unscaled = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(unscaled, train_vals)

    def test_sequence_window_generation(self):
        series = np.arange(10, dtype=float)
        lookback = 3
        X, y = create_sequences(series, lookback=lookback)

        # Total sequences: 10 - 3 = 7
        self.assertEqual(len(X), 7)
        self.assertEqual(len(y), 7)
        # First sequence: [0, 1, 2] -> target 3
        np.testing.assert_array_equal(X[0].flatten(), np.array([0, 1, 2]))
        self.assertEqual(y[0][0], 3)
        # Last sequence: [6, 7, 8] -> target 9
        np.testing.assert_array_equal(X[-1].flatten(), np.array([6, 7, 8]))
        self.assertEqual(y[-1][0], 9)

    def test_lstm_forecast_horizon_and_non_negative(self):
        horizon = 15
        pred_df = train_predict_lstm(
            self.history_df,
            horizon=horizon,
            lookback=14,
            epochs=5,
            hidden_dim=16,
            seed=42
        )
        self.assertEqual(len(pred_df), horizon, "Prediction length must match requested horizon")
        self.assertEqual(pred_df["Model"].iloc[0], "LSTM")
        self.assertTrue((pred_df["Predicted_Demand"] >= 0).all(), "Predictions must be non-negative")

        # Chronological future dates
        expected_first = pd.to_datetime("2011-09-09")
        self.assertEqual(pred_df["Date"].min(), expected_first)

    def test_ensemble_weight_optimization(self):
        y_val_true = np.array([10.0, 20.0, 30.0])
        # Suppose Prophet is closer to ground truth
        p_val = np.array([11.0, 19.0, 31.0])
        # Suppose LSTM is further
        l_val = np.array([5.0, 10.0, 15.0])

        best_w, best_score, log = optimize_ensemble_weights(
            y_val_true=y_val_true,
            prophet_val_preds=p_val,
            lstm_val_preds=l_val,
            metric="MAE"
        )
        # Optimal weight for Prophet should be high (>= 0.7)
        self.assertGreaterEqual(best_w, 0.7)
        self.assertIn("w_prophet_0.50", log)

    def test_combine_prophet_lstm_predictions(self):
        dates = pd.date_range("2011-11-10", periods=5, freq="D")
        p_df = pd.DataFrame({
            "Date": dates,
            "Predicted_Demand": [20.0, 25.0, 30.0, 35.0, 40.0],
            "Lower_Bound": [15.0, 20.0, 25.0, 30.0, 35.0],
            "Upper_Bound": [25.0, 30.0, 35.0, 40.0, 45.0],
            "Model": "Prophet"
        })
        l_df = pd.DataFrame({
            "Date": dates,
            "Predicted_Demand": [10.0, 15.0, 20.0, 25.0, 30.0],
            "Model": "LSTM"
        })

        # Test 50/50 weighting: 0.5 * 20 + 0.5 * 10 = 15.0
        ens_df = combine_prophet_lstm_predictions(p_df, l_df, prophet_weight=0.5)

        self.assertEqual(len(ens_df), 5)
        self.assertAlmostEqual(ens_df["Predicted_Demand"].iloc[0], 15.0)
        self.assertAlmostEqual(ens_df["Predicted_Demand"].iloc[-1], 35.0)
        self.assertTrue((ens_df["Predicted_Demand"] >= 0).all())
        self.assertIn("Lower_Bound", ens_df.columns)
        self.assertIn("Upper_Bound", ens_df.columns)

    def test_zero_demand_evaluation(self):
        y_true = np.array([0.0, 0.0, 10.0, 20.0])
        y_pred = np.array([0.0, 2.0, 12.0, 18.0])

        metrics = evaluate_forecast(y_true, y_pred)
        self.assertFalse(np.isnan(metrics["MAE"]))
        self.assertFalse(np.isnan(metrics["RMSE"]))
        self.assertFalse(np.isnan(metrics["MAPE"]))
        self.assertFalse(np.isnan(metrics["sMAPE"]))
        self.assertTrue(metrics["sMAPE"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
