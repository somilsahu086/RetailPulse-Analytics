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

    def test_log1p_expm1_reversibility_and_bounds(self):
        """Verify log1p and expm1 reversibility across scale and non-negativity."""
        test_vals = np.array([0.0, 1.0, 2.0, 10.0, 50.0, 500.0, 5000.0])
        transformed = np.log1p(test_vals)
        recovered = np.expm1(transformed)
        np.testing.assert_allclose(recovered, test_vals, rtol=1e-5, atol=1e-5)

        # Boundary / edge case: zero must invert to exactly zero
        self.assertEqual(np.expm1(np.log1p(0.0)), 0.0)

        # Negative noise suppression
        clipped = np.maximum(0.0, np.expm1(-0.05))
        self.assertEqual(clipped, 0.0)

    def test_log1p_lstm_training_and_non_negativity(self):
        """Verify train_predict_lstm with use_log1p=True produces non-negative forecasts."""
        horizon = 14
        pred_df = train_predict_lstm(
            self.history_df,
            horizon=horizon,
            lookback=14,
            epochs=5,
            hidden_dim=16,
            seed=42,
            use_log1p=True
        )
        self.assertEqual(len(pred_df), horizon)
        self.assertEqual(pred_df["Model"].iloc[0], "LSTM")
        self.assertTrue((pred_df["Predicted_Demand"] >= 0.0).all(), "Log1p-LSTM predictions must be non-negative")
        self.assertEqual(pred_df.columns.tolist(), ["Date", "Predicted_Demand", "Model"])

    def test_saturday_masking_toggle_and_preservation(self):
        """Verify Saturday masking sets Saturdays to zero when enabled and preserves baseline when disabled."""
        # Create a 14-day date range covering exactly 2 Saturdays
        dates = pd.date_range("2011-10-10", periods=14, freq="D")
        p_df = pd.DataFrame({
            "Date": dates,
            "Predicted_Demand": [20.0] * 14,
            "Lower_Bound": [15.0] * 14,
            "Upper_Bound": [25.0] * 14,
            "Model": "Prophet"
        })
        l_df = pd.DataFrame({
            "Date": dates,
            "Predicted_Demand": [10.0] * 14,
            "Model": "LSTM"
        })

        # 1. Disabled (baseline behavior preserved)
        ens_unmasked = combine_prophet_lstm_predictions(p_df, l_df, prophet_weight=0.5, apply_saturday_mask=False)
        self.assertTrue((ens_unmasked["Predicted_Demand"] == 15.0).all())

        # 2. Enabled (Saturday mask applied)
        ens_masked = combine_prophet_lstm_predictions(p_df, l_df, prophet_weight=0.5, apply_saturday_mask=True)
        saturdays = dates[dates.dayofweek == 5]
        non_saturdays = dates[dates.dayofweek != 5]

        # Verify Saturdays are zeroed out
        sat_preds = ens_masked[ens_masked["Date"].isin(saturdays)]
        self.assertEqual(len(sat_preds), 2)
        self.assertTrue((sat_preds["Predicted_Demand"] == 0.0).all())
        self.assertTrue((sat_preds["Lower_Bound"] == 0.0).all())
        self.assertTrue((sat_preds["Upper_Bound"] == 0.0).all())

        # Verify non-Saturdays retain exact weighted prediction
        non_sat_preds = ens_masked[ens_masked["Date"].isin(non_saturdays)]
        self.assertEqual(len(non_sat_preds), 12)
        self.assertTrue((non_sat_preds["Predicted_Demand"] == 15.0).all())

    def test_candidate_c_inventory_and_stockout_risk_integration(self):
        """Verify Candidate C forecasts integrate seamlessly with inventory_planning without errors."""
        from inventory_planning import generate_inventory_recommendations

        dates = pd.date_range("2011-11-10", periods=30, freq="D")
        dow = dates.dayofweek
        # Candidate C style: weekly demand with Saturday=0
        demand_profile = np.where(dow == 5, 0.0, np.array([25.0] * 30))
        forecast_dict = {"TEST_CAND_C": demand_profile}

        inv_df, risk_df = generate_inventory_recommendations(
            forecast_series_dict=forecast_dict,
            history_daily_means={"TEST_CAND_C": 22.0},
            lead_time_days=7,
            service_level=0.95
        )

        self.assertEqual(len(inv_df), 1)
        self.assertEqual(len(risk_df), 1)
        self.assertGreater(inv_df["Safety_Stock"].iloc[0], 0)
        self.assertGreater(inv_df["Reorder_Point"].iloc[0], 0)
        self.assertIn(risk_df["Stockout_Risk"].iloc[0], ["LOW", "MEDIUM", "HIGH"])

    def test_forecast_date_sku_alignment_and_horizon(self):
        """Verify date alignment and 30-day chronological continuity between models."""
        dates = pd.date_range("2011-11-10", periods=30, freq="D")
        p_df = pd.DataFrame({"Date": dates, "Predicted_Demand": np.linspace(10, 40, 30), "Model": "Prophet"})
        l_df = pd.DataFrame({"Date": dates, "Predicted_Demand": np.linspace(12, 38, 30), "Model": "LSTM"})

        ens_df = combine_prophet_lstm_predictions(p_df, l_df, prophet_weight=0.6, apply_saturday_mask=True)
        self.assertEqual(len(ens_df), 30)
        self.assertTrue((ens_df["Date"] == dates).all())
        # Chronological order verified
        self.assertTrue(ens_df["Date"].is_monotonic_increasing)

    def test_phase5_output_csv_and_json_schemas(self):
        """Verify that Phase 5 schema contracts adhere to dashboard and downstream requirements."""
        from pathlib import Path
        import json

        expected_pred_cols = ["Date", "StockCode", "Actual_Demand", "Predicted_Demand", "Model"]
        expected_inv_cols = [
            "StockCode", "Description", "Forecast_Horizon_Days", "Total_Forecast_Demand",
            "Daily_Demand_Mean", "Daily_Demand_Std", "Lead_Time_Days", "Service_Level_Pct",
            "Safety_Stock", "Reorder_Point", "Suggested_Max_Inventory"
        ]
        expected_risk_cols = [
            "StockCode", "Description", "Stockout_Risk", "Surge_Factor",
            "Coeff_of_Variation", "Safety_Stock", "Reorder_Point"
        ]
        expected_json_keys = [
            "evaluation_date", "phase", "training_window", "holdout_window",
            "product_count", "lstm_configuration", "prophet_configuration",
            "ensemble_weighting_method", "average_prophet_weight", "best_model",
            "metrics_summary", "leakage_safeguard"
        ]

        # Resolve artifact directory: prefer current phase5 artifacts, fallback to baseline backup
        project_root = Path(__file__).resolve().parent.parent
        current_dir = project_root / "outputs" / "forecasting" / "phase5"
        backup_dir = project_root / "outputs" / "forecasting" / "phase5_baseline_backup"

        required_filenames = [
            "ensemble_predictions.csv",
            "inventory_recommendations.csv",
            "stockout_risk.csv",
            "best_ensemble_summary.json",
        ]

        current_missing = [f for f in required_filenames if not (current_dir / f).is_file()]
        backup_missing = [f for f in required_filenames if not (backup_dir / f).is_file()]

        if not current_missing:
            target_dir = current_dir
        elif not backup_missing:
            target_dir = backup_dir
        else:
            self.fail(
                "Phase 5 output schema verification failed: neither artifact directory contains all four required files.\n"
                f"  Current directory ({current_dir}) missing: {current_missing}\n"
                f"  Baseline backup directory ({backup_dir}) missing: {backup_missing}"
            )

        # Verify CSV schemas
        ens_df = pd.read_csv(target_dir / "ensemble_predictions.csv", nrows=2)
        self.assertEqual(
            ens_df.columns.tolist(),
            expected_pred_cols,
            f"Schema mismatch in ensemble_predictions.csv: expected {expected_pred_cols}, got {ens_df.columns.tolist()}"
        )

        inv_df = pd.read_csv(target_dir / "inventory_recommendations.csv", nrows=2)
        self.assertEqual(
            inv_df.columns.tolist(),
            expected_inv_cols,
            f"Schema mismatch in inventory_recommendations.csv: expected {expected_inv_cols}, got {inv_df.columns.tolist()}"
        )

        risk_df = pd.read_csv(target_dir / "stockout_risk.csv", nrows=2)
        self.assertEqual(
            risk_df.columns.tolist(),
            expected_risk_cols,
            f"Schema mismatch in stockout_risk.csv: expected {expected_risk_cols}, got {risk_df.columns.tolist()}"
        )

        # Verify summary JSON schema
        summary_path = target_dir / "best_ensemble_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            for k in expected_json_keys:
                self.assertIn(
                    k,
                    summary_data,
                    f"Missing expected key '{k}' in {summary_path}"
                )


if __name__ == "__main__":
    unittest.main()
