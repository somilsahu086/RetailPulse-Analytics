"""
Unit tests for Phase 4: Advanced Forecasting Models & Feature Engineering.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from feature_engineering import (
    create_calendar_features,
    create_lag_and_rolling_features,
    build_product_feature_matrix,
    get_feature_column_names
)
from forecasting import (
    train_predict_prophet,
    train_predict_lightgbm,
    run_phase4_benchmark
)


class TestAdvancedForecasting(unittest.TestCase):

    def setUp(self):
        # 120 days of daily continuous demand for a test product
        dates = pd.date_range("2011-07-01", "2011-10-28", freq="D")
        np.random.seed(42)
        # Seasonal base with random fluctuations
        dow = dates.dayofweek
        base_demand = np.where(dow == 5, 0, np.random.randint(10, 50, size=len(dates)))
        self.history_df = pd.DataFrame({
            "Date": dates,
            "StockCode": ["TEST_PROD"] * len(dates),
            "Description": ["Test Product Item"] * len(dates),
            "Demand": base_demand
        })

    def test_calendar_features(self):
        feat_df = create_calendar_features(self.history_df)
        self.assertIn("day_of_week", feat_df.columns)
        self.assertIn("is_saturday", feat_df.columns)
        self.assertIn("is_weekend", feat_df.columns)
        self.assertIn("sin_dow", feat_df.columns)
        self.assertIn("cos_dow", feat_df.columns)
        self.assertIn("sin_month", feat_df.columns)
        self.assertIn("cos_month", feat_df.columns)

        # Verify Saturday flag accuracy
        sat_indices = feat_df[feat_df["day_of_week"] == 5].index
        self.assertTrue((feat_df.loc[sat_indices, "is_saturday"] == 1).all())

    def test_lag_and_rolling_features_no_leakage(self):
        feat_df = create_lag_and_rolling_features(
            self.history_df,
            lags=(1, 7),
            windows=(7, 14)
        )
        # Verify that lag_1 at row 10 is equal to Demand at row 9
        self.assertEqual(feat_df["lag_1"].iloc[10], float(self.history_df["Demand"].iloc[9]))
        # Verify that lag_7 at row 10 is equal to Demand at row 3
        self.assertEqual(feat_df["lag_7"].iloc[10], float(self.history_df["Demand"].iloc[3]))

        # Verify rolling mean excludes current day t (computed on shift=1)
        # rolling_mean_7 at row 10 must equal mean of rows 3 to 9 (7 days)
        expected_rm7 = float(np.mean(self.history_df["Demand"].iloc[3:10]))
        self.assertAlmostEqual(feat_df["rolling_mean_7"].iloc[10], expected_rm7, places=4)

    def test_build_feature_matrix_dimensions(self):
        matrix = build_product_feature_matrix(self.history_df, drop_initial_na=True)
        feat_cols = get_feature_column_names()
        for col in feat_cols:
            self.assertIn(col, matrix.columns)
        self.assertFalse(matrix.empty)
        # Ensure no NaN in feature columns after dropping initial lags
        self.assertEqual(matrix[feat_cols].isna().sum().sum(), 0)

    def test_prophet_forecasting(self):
        horizon = 14
        pred_df = train_predict_prophet(
            self.history_df,
            horizon=horizon,
            add_uk_holidays=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        self.assertEqual(len(pred_df), horizon)
        self.assertEqual(pred_df["Model"].iloc[0], "Prophet")
        self.assertIn("Lower_Bound", pred_df.columns)
        self.assertIn("Upper_Bound", pred_df.columns)
        # Non-negative constraint
        self.assertTrue((pred_df["Predicted_Demand"] >= 0).all())
        self.assertTrue((pred_df["Lower_Bound"] >= 0).all())
        # First date must be day after history end
        expected_start = pd.to_datetime("2011-10-29")
        self.assertEqual(pred_df["Date"].min(), expected_start)

    def test_lightgbm_forecasting(self):
        horizon = 14
        pred_df = train_predict_lightgbm(
            self.history_df,
            horizon=horizon,
            lags=(1, 7),
            windows=(7, 14)
        )
        self.assertEqual(len(pred_df), horizon)
        # Non-negative constraint
        self.assertTrue((pred_df["Predicted_Demand"] >= 0).all())
        # First date must be day after history end
        expected_start = pd.to_datetime("2011-10-29")
        self.assertEqual(pred_df["Date"].min(), expected_start)

    def test_phase4_benchmark_runner(self):
        # Create small test dataset with 2 products
        dates = pd.date_range("2011-06-01", "2011-11-20", freq="D")
        np.random.seed(42)
        df_list = []
        for code in ["ITEM_A", "ITEM_B"]:
            df_list.append(pd.DataFrame({
                "Date": dates,
                "StockCode": [code] * len(dates),
                "Description": [f"Desc {code}"] * len(dates),
                "Demand": np.random.randint(5, 30, size=len(dates))
            }))
        df_multi = pd.concat(df_list, ignore_index=True)

        summary_df, preds_df = run_phase4_benchmark(
            daily_demand_df=df_multi,
            eligible_stock_codes=["ITEM_A", "ITEM_B"],
            split_date="2011-11-06",
            horizon=14,
            include_prophet=True,
            include_lightgbm=True
        )

        self.assertFalse(summary_df.empty)
        # Verify both baselines and advanced models are present
        model_names = summary_df["Model"].tolist()
        self.assertIn("Naive (Last Value)", model_names)
        self.assertIn("Moving Average 28D", model_names)
        self.assertIn("LightGBM", model_names)
        self.assertIn("Prophet", model_names)
        self.assertFalse(preds_df.empty)


if __name__ == "__main__":
    unittest.main()
