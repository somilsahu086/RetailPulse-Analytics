"""
Unit tests for Demand Forecasting Preprocessing & Evaluation Modules.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import (
    clean_sales_data,
    build_daily_product_timeseries,
    analyze_product_eligibility,
    train_test_split_timeseries,
    validate_daily_demand
)
from evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    evaluate_forecast
)


class TestPreprocessingAndEvaluation(unittest.TestCase):

    def setUp(self):
        # Synthetic raw transactions mimicking UCI retail dataset
        self.raw_data = pd.DataFrame({
            "Invoice": ["500001", "C500002", "500003", "500004", "500005", "500006", "500001"],
            "StockCode": ["85123A", "85123A", "POST", "85123A", "22423", "22423", "85123A"],
            "Description": ["WHITE HEART", "WHITE HEART", "POSTAGE", "WHITE HEART", "CAKE STAND", "CAKE STAND", "WHITE HEART"],
            "Quantity": [10, -5, 1, 20, -2, 15, 10],  # Includes cancellation, negative qty, duplicate
            "InvoiceDate": [
                "2011-01-01 10:00:00",
                "2011-01-01 11:00:00",
                "2011-01-02 09:00:00",
                "2011-01-03 14:00:00",
                "2011-01-04 12:00:00",
                "2011-01-05 16:00:00",
                "2011-01-01 10:00:00"  # Exact duplicate of row 0
            ],
            "Price": [2.5, 2.5, 5.0, 2.5, 0.0, 10.0, 2.5],
            "Customer ID": [12345.0, 12345.0, 12346.0, np.nan, 12347.0, np.nan, 12345.0],
            "Country": ["United Kingdom"] * 7
        })

    def test_cancellation_removal(self):
        clean_df, report = clean_sales_data(self.raw_data, verbose=False)
        self.assertFalse(any(clean_df["Invoice"].str.startswith("C")), "Cancelled invoices must be removed")
        self.assertEqual(report["cancellations_removed"], 1)

    def test_negative_quantity_removal(self):
        clean_df, _ = clean_sales_data(self.raw_data, verbose=False)
        self.assertTrue(all(clean_df["Quantity"] > 0), "All retained quantities must be strictly positive")

    def test_null_customer_id_preservation(self):
        clean_df, report = clean_sales_data(self.raw_data, verbose=False)
        # Row 3 and Row 5 have null Customer IDs and valid positive sales
        null_customers = clean_df["Customer ID"].isna().sum()
        self.assertGreater(null_customers, 0, "Guest checkouts with null Customer ID must NOT be dropped")
        self.assertEqual(report["preserved_null_customer_ids"], null_customers)

    def test_non_merchandise_code_removal(self):
        clean_df, report = clean_sales_data(self.raw_data, verbose=False)
        self.assertNotIn("POST", clean_df["StockCode"].values, "POST code must be removed")
        self.assertEqual(report["non_merchandise_codes_removed"], 1)

    def test_duplicate_removal(self):
        clean_df, report = clean_sales_data(self.raw_data, verbose=False)
        self.assertEqual(report["exact_duplicates_removed"], 1, "Duplicate row must be removed")

    def test_daily_aggregation_and_missing_calendar_dates(self):
        clean_df, _ = clean_sales_data(self.raw_data, verbose=False)
        # Clean data for 85123A has sales on 2011-01-01 (10 units) and 2011-01-03 (20 units)
        ts = build_daily_product_timeseries(
            clean_df,
            stock_code="85123A",
            start_date="2011-01-01",
            end_date="2011-01-04"
        )
        # Expected calendar days: Jan 1, Jan 2, Jan 3, Jan 4 (4 days)
        self.assertEqual(len(ts), 4, "Time series must have continuous daily frequency for full date range")

        # Jan 2 had no sales of 85123A, so demand should be filled with 0
        jan_2_demand = ts[ts["Date"] == "2011-01-02"]["Demand"].iloc[0]
        self.assertEqual(jan_2_demand, 0, "Missing calendar dates must be zero-filled")

        # Jan 1 demand should be 10 units
        jan_1_demand = ts[ts["Date"] == "2011-01-01"]["Demand"].iloc[0]
        self.assertEqual(jan_1_demand, 10)

        # Jan 3 demand should be 20 units
        jan_3_demand = ts[ts["Date"] == "2011-01-03"]["Demand"].iloc[0]
        self.assertEqual(jan_3_demand, 20)

    def test_train_test_chronological_split_and_no_leakage(self):
        # Create 10 days of continuous data
        dates = pd.date_range("2011-11-01", "2011-11-10", freq="D")
        ts_df = pd.DataFrame({"Date": dates, "Demand": [10] * 10})

        train_df, test_df = train_test_split_timeseries(ts_df, split_date="2011-11-05")

        self.assertEqual(len(train_df), 5, "Train should contain 5 days (Nov 1 to Nov 5)")
        self.assertEqual(len(test_df), 5, "Test should contain 5 days (Nov 6 to Nov 10)")

        # Verify strict chronology
        self.assertLess(train_df["Date"].max(), test_df["Date"].min(), "No temporal overlap allowed")

    def test_daily_validation_checks(self):
        dates = pd.date_range("2011-01-01", "2011-01-05", freq="D")
        valid_df = pd.DataFrame({
            "Date": dates,
            "StockCode": ["85123A"] * 5,
            "Demand": [10, 20, 0, 15, 30]
        })
        report = validate_daily_demand(valid_df)
        self.assertTrue(report["passed"])

        # Test failure on negative demand
        invalid_df = valid_df.copy()
        invalid_df.loc[0, "Demand"] = -5
        with self.assertRaises(ValueError):
            validate_daily_demand(invalid_df)

    def test_zero_safe_evaluation_metrics(self):
        y_true = np.array([0, 10, 20, 0, 50])
        y_pred = np.array([2, 12, 18, 0, 48])

        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = root_mean_squared_error(y_true, y_pred)
        mape_val = mean_absolute_percentage_error(y_true, y_pred)
        smape_val = symmetric_mean_absolute_percentage_error(y_true, y_pred)

        # MAE: mean(|0-2|, |10-12|, |20-18|, |0-0|, |50-48|) = (2+2+2+0+2)/5 = 1.6
        self.assertAlmostEqual(mae_val, 1.6)

        # RMSE must be positive and >= MAE
        self.assertGreaterEqual(rmse_val, mae_val)

        # MAPE must not raise ZeroDivisionError and must be finite
        self.assertTrue(np.isfinite(mape_val))

        # sMAPE must be bounded and finite
        self.assertTrue(0.0 <= smape_val <= 200.0)

        # Test evaluate_forecast bundle
        metrics = evaluate_forecast(y_true, y_pred)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("sMAPE", metrics)


if __name__ == "__main__":
    unittest.main()
