"""
Unit tests for Baseline Forecasting Module.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baseline_forecasting import (
    naive_forecast,
    moving_average_forecast,
    evaluate_single_product_baseline,
    run_baseline_benchmark
)


class TestBaselineForecasting(unittest.TestCase):

    def setUp(self):
        # 40 days of historical daily demand for a test product
        dates = pd.date_range("2011-10-01", "2011-11-09", freq="D")
        np.random.seed(42)
        # Synthetic demand with positive values
        demands = np.random.randint(5, 50, size=len(dates))
        self.history_df = pd.DataFrame({
            "Date": dates,
            "Demand": demands
        })

    def test_naive_forecast_horizon_and_dates(self):
        horizon = 30
        pred_df = naive_forecast(self.history_df, horizon=horizon, strategy="last")

        self.assertEqual(len(pred_df), horizon, f"Forecast must contain exactly {horizon} days")
        self.assertEqual(pred_df["Date"].min(), pd.Timestamp("2011-11-10"), "First forecast date must be day after history max")
        self.assertEqual(pred_df["Date"].max(), pd.Timestamp("2011-12-09"), "Last forecast date must match 30-day horizon")
        self.assertEqual(pred_df["Model"].iloc[0], "Naive (Last Value)")

    def test_moving_average_forecast(self):
        # Test 7-day moving average
        last_7_mean = float(np.mean(self.history_df["Demand"].iloc[-7:]))
        pred_df_7 = moving_average_forecast(self.history_df, window=7, horizon=30)

        self.assertEqual(len(pred_df_7), 30)
        self.assertAlmostEqual(pred_df_7["Predicted_Demand"].iloc[0], last_7_mean, places=4)
        self.assertTrue(all(pred_df_7["Predicted_Demand"] == last_7_mean), "MA prediction must be constant across horizon")

        # Test 14-day moving average
        last_14_mean = float(np.mean(self.history_df["Demand"].iloc[-14:]))
        pred_df_14 = moving_average_forecast(self.history_df, window=14, horizon=30)
        self.assertAlmostEqual(pred_df_14["Predicted_Demand"].iloc[0], last_14_mean, places=4)

        # Test 28-day moving average
        last_28_mean = float(np.mean(self.history_df["Demand"].iloc[-28:]))
        pred_df_28 = moving_average_forecast(self.history_df, window=28, horizon=30)
        self.assertAlmostEqual(pred_df_28["Predicted_Demand"].iloc[0], last_28_mean, places=4)

    def test_no_negative_predictions(self):
        # Create history with zero or edge values
        history_zero = pd.DataFrame({
            "Date": pd.date_range("2011-10-01", "2011-11-09", freq="D"),
            "Demand": [0] * 40
        })
        pred_naive = naive_forecast(history_zero, horizon=15)
        pred_ma = moving_average_forecast(history_zero, window=7, horizon=15)

        self.assertTrue((pred_naive["Predicted_Demand"] >= 0).all(), "Naive predictions must be non-negative")
        self.assertTrue((pred_ma["Predicted_Demand"] >= 0).all(), "MA predictions must be non-negative")

    def test_metric_calculation_and_evaluation(self):
        train_df = self.history_df.iloc[:30]
        test_df = self.history_df.iloc[30:].copy()

        metrics, pred_df = evaluate_single_product_baseline(
            train_df=train_df,
            test_df=test_df,
            forecast_func=moving_average_forecast,
            model_kwargs={"window": 7},
            stock_code="TEST_PROD"
        )

        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("sMAPE", metrics)
        self.assertGreaterEqual(metrics["RMSE"], metrics["MAE"], "RMSE is always >= MAE")
        self.assertFalse(pred_df.empty)

    def test_benchmark_runner_and_train_test_separation(self):
        # Create small multi-product dataset
        dates = pd.date_range("2011-10-01", "2011-12-09", freq="D")
        df_list = []
        for code in ["PROD_A", "PROD_B"]:
            df_list.append(pd.DataFrame({
                "Date": dates,
                "StockCode": code,
                "Description": f"Desc {code}",
                "Demand": np.random.randint(1, 20, size=len(dates))
            }))
        multi_df = pd.concat(df_list, ignore_index=True)

        summary_df, preds_df = run_baseline_benchmark(
            daily_demand_df=multi_df,
            eligible_stock_codes=["PROD_A", "PROD_B"],
            split_date="2011-11-09",
            horizon=30,
            windows=(7, 14, 28)
        )

        # Expected 4 models: Naive, MA7, MA14, MA28
        self.assertEqual(len(summary_df), 4)
        self.assertEqual(summary_df["Product Count"].iloc[0], 2)
        # Verify predictions cover 30 days
        self.assertEqual(preds_df["Date"].min(), pd.Timestamp("2011-11-10"))
        self.assertEqual(preds_df["Date"].max(), pd.Timestamp("2011-12-09"))


if __name__ == "__main__":
    unittest.main()
