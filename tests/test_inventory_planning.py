"""
Unit tests for Phase 5: Inventory Planning & Stockout Risk Layer.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from inventory_planning import (
    get_z_score,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_order_up_to_level,
    assess_stockout_risk,
    generate_inventory_recommendations
)


class TestInventoryPlanning(unittest.TestCase):

    def test_z_score_calculation(self):
        # 95% service level -> z ~ 1.645
        z_95 = get_z_score(0.95)
        self.assertAlmostEqual(z_95, 1.6449, places=3)

        # 99% service level -> z ~ 2.326
        z_99 = get_z_score(0.99)
        self.assertAlmostEqual(z_99, 2.3263, places=3)

        # Out-of-bounds error check
        with self.assertRaises(ValueError):
            get_z_score(0.20)
        with self.assertRaises(ValueError):
            get_z_score(1.05)

    def test_safety_stock_formula(self):
        # daily_std = 10, lead_time = 7, service_level = 0.95 (Z ~ 1.645)
        # SS = ceil(1.6449 * 10 * sqrt(7)) = ceil(16.449 * 2.64575) = ceil(43.52) = 44
        ss = calculate_safety_stock(daily_demand_std=10.0, lead_time_days=7, service_level=0.95)
        self.assertEqual(ss, 44)
        self.assertIsInstance(ss, int)

        # Zero volatility -> SS must be 0
        ss_zero = calculate_safety_stock(daily_demand_std=0.0, lead_time_days=7)
        self.assertEqual(ss_zero, 0)

    def test_reorder_point_formula(self):
        # daily_mean = 20, lead_time = 7, SS = 44
        # ROP = ceil(20 * 7 + 44) = 184
        rop = calculate_reorder_point(daily_demand_mean=20.0, safety_stock=44, lead_time_days=7)
        self.assertEqual(rop, 184)
        self.assertIsInstance(rop, int)

    def test_order_up_to_level_formula(self):
        # daily_mean = 20, lead_time = 7, review_period = 14, SS = 44
        # S = ceil(20 * (7 + 14) + 44) = ceil(420 + 44) = 464
        max_lvl = calculate_order_up_to_level(
            daily_demand_mean=20.0,
            safety_stock=44,
            lead_time_days=7,
            review_period_days=14
        )
        self.assertEqual(max_lvl, 464)
        self.assertGreater(max_lvl, 184)

    def test_stockout_risk_classification(self):
        # 1. High risk due to surge (forecast 150 vs history 100 -> surge = 1.50)
        risk_high_surge = assess_stockout_risk(historical_daily_mean=100.0, forecast_daily_mean=150.0, forecast_daily_std=20.0)
        self.assertEqual(risk_high_surge["stockout_risk"], "HIGH")

        # 2. High risk due to extreme coefficient of variation (std 40 vs mean 30 -> CV = 1.33)
        risk_high_cv = assess_stockout_risk(historical_daily_mean=30.0, forecast_daily_mean=30.0, forecast_daily_std=40.0)
        self.assertEqual(risk_high_cv["stockout_risk"], "HIGH")

        # 3. Medium risk (surge = 1.15, CV = 0.6)
        risk_medium = assess_stockout_risk(historical_daily_mean=100.0, forecast_daily_mean=115.0, forecast_daily_std=70.0)
        self.assertEqual(risk_medium["stockout_risk"], "MEDIUM")

        # 4. Low risk (stable demand, surge = 1.02, CV = 0.2)
        risk_low = assess_stockout_risk(historical_daily_mean=100.0, forecast_daily_mean=102.0, forecast_daily_std=20.0)
        self.assertEqual(risk_low["stockout_risk"], "LOW")

    def test_generate_inventory_recommendations_catalog(self):
        forecasts = {
            "PROD_A": np.array([20.0] * 30),
            "PROD_B": np.array([50.0] * 30)
        }
        hist_means = {"PROD_A": 18.0, "PROD_B": 45.0}
        descriptions = {"PROD_A": "Widget A", "PROD_B": "Widget B"}

        inv_df, risk_df = generate_inventory_recommendations(
            forecast_series_dict=forecasts,
            history_daily_means=hist_means,
            product_descriptions=descriptions,
            lead_time_days=7,
            service_level=0.95
        )

        self.assertEqual(len(inv_df), 2)
        self.assertEqual(len(risk_df), 2)
        self.assertIn("Safety_Stock", inv_df.columns)
        self.assertIn("Reorder_Point", inv_df.columns)
        self.assertIn("Suggested_Max_Inventory", inv_df.columns)
        self.assertIn("Stockout_Risk", risk_df.columns)


if __name__ == "__main__":
    unittest.main()
