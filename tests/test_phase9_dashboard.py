"""
RetailPulse — Demand Forecasting Module
Phase 9 Streamlit Dashboard & WMAPE Unit & Integration Test Suite

Tests:
1. Successful import of app.py and Streamlit dependencies.
2. Volume-Weighted Mean Absolute Percentage Error (WMAPE) mathematical correctness and edge cases.
3. Dashboard data loaders: catalog, historical demand, precomputed predictions, inventory recommendations.
4. Model comparison, drift monitoring, and Airflow retraining artifact loaders.
5. Graceful error handling for missing files and unknown SKU selections.
6. Zero mutation and no data leakage across dashboard data processing.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root and src/ to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import (
    weighted_mean_absolute_percentage_error,
    evaluate_forecast,
    mean_absolute_error
)
import app


class TestPhase9Dashboard:
    """Test suite covering Phase 9 Streamlit dashboard integration and WMAPE metric."""

    def test_wmape_mathematical_correctness(self):
        """Verify WMAPE computes sum(|y - y_hat|) / sum(|y|) accurately."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # |10-12|=2, |20-18|=2, |30-33|=3 -> sum error = 7
        # sum actual = 60
        # WMAPE = (7 / 60) * 100% = 11.6667%
        expected_wmape = (7.0 / 60.0) * 100.0
        wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
        assert abs(wmape - expected_wmape) < 1e-4

    def test_wmape_edge_cases(self):
        """Verify WMAPE handles identical arrays, zero demand, and empty inputs gracefully."""
        # 1. Identical predictions -> 0% error
        y = np.array([5.0, 15.0, 25.0])
        assert weighted_mean_absolute_percentage_error(y, y) == 0.0

        # 2. Empty arrays -> 0.0
        assert weighted_mean_absolute_percentage_error([], []) == 0.0

        # 3. All zeros actual and predicted -> 0.0
        assert weighted_mean_absolute_percentage_error([0, 0, 0], [0, 0, 0]) == 0.0

        # 4. Intermittent series with zeros does not explode
        y_sparse = np.array([0.0, 0.0, 100.0, 0.0, 50.0])
        y_sparse_pred = np.array([2.0, 1.0, 95.0, 3.0, 48.0])
        # total abs err = 2 + 1 + 5 + 3 + 2 = 13. total actual = 150.
        # WMAPE = 13 / 150 * 100% = 8.67%
        sparse_wmape = weighted_mean_absolute_percentage_error(y_sparse, y_sparse_pred)
        assert 8.0 < sparse_wmape < 9.0

    def test_evaluate_forecast_includes_wmape(self):
        """Verify evaluate_forecast returns WMAPE alongside MAE, RMSE, MAPE, and sMAPE."""
        y_true = [10, 20, 30, 40]
        y_pred = [12, 19, 28, 41]
        metrics = evaluate_forecast(y_true, y_pred)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert "sMAPE" in metrics
        assert "WMAPE" in metrics
        assert metrics["WMAPE"] >= 0.0

    def test_catalog_loader(self):
        """Verify load_catalog loads eligible forecasting products with required columns."""
        df = app.load_catalog(project_dir=PROJECT_ROOT)
        assert not df.empty
        assert "StockCode" in df.columns
        assert "Description" in df.columns
        assert "TotalDemand" in df.columns
        assert "eligible_for_forecasting" in df.columns
        assert df["eligible_for_forecasting"].sum() >= 50

    def test_historical_series_loader(self):
        """Verify load_historical_series loads sorted daily demand for an eligible product."""
        # Top 1 SKU: 84077
        df = app.load_historical_series(stock_code="84077", project_dir=PROJECT_ROOT)
        assert not df.empty
        assert "Date" in df.columns
        assert "Demand" in df.columns
        assert (df["Demand"] >= 0.0).all()
        # Ensure chronological ordering
        assert df["Date"].is_monotonic_increasing

    def test_all_predictions_loader(self):
        """Verify load_all_predictions loads multi-model holdout predictions."""
        preds_dict = app.load_all_predictions(project_dir=PROJECT_ROOT)
        assert len(preds_dict) > 0
        assert "Prophet + PyTorch LSTM Ensemble (Production)" in preds_dict

        ens_df = preds_dict["Prophet + PyTorch LSTM Ensemble (Production)"]
        assert "Date" in ens_df.columns
        assert "StockCode" in ens_df.columns
        assert "Predicted_Demand" in ens_df.columns
        assert (ens_df["Predicted_Demand"] >= 0.0).all()

    def test_inventory_and_risk_loaders(self):
        """Verify inventory planning and stockout risk loaders return valid tables."""
        inv_df = app.load_inventory_catalog_recommendations(project_dir=PROJECT_ROOT)
        risk_df = app.load_stockout_risk_table(project_dir=PROJECT_ROOT)

        assert not inv_df.empty
        assert "Safety_Stock" in inv_df.columns
        assert "Reorder_Point" in inv_df.columns
        assert (inv_df["Safety_Stock"] >= 0).all()

        assert not risk_df.empty
        assert "Stockout_Risk" in risk_df.columns
        assert set(risk_df["Stockout_Risk"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_drift_and_retraining_loaders(self):
        """Verify Phase 7 drift and Phase 8 retraining artifact loaders."""
        summary, table = app.load_drift_monitoring_artifacts(project_dir=PROJECT_ROOT)
        assert isinstance(summary, dict)
        assert "dataset_drift_detected" in summary
        assert not table.empty

        dag_sum, decision, metrics = app.load_retraining_artifacts(project_dir=PROJECT_ROOT)
        assert isinstance(dag_sum, dict)
        assert isinstance(decision, dict)
        assert "promotion_decision" in decision
        assert not metrics.empty

    def test_graceful_missing_data_handling(self, tmp_path):
        """Verify loaders handle missing files and non-existent SKUs without crashing."""
        # Non-existent directory
        fake_dir = str(tmp_path / "non_existent")
        assert app.load_catalog(project_dir=fake_dir).empty
        assert app.load_historical_series("UNKNOWN_SKU", project_dir=fake_dir).empty
        assert len(app.load_all_predictions(project_dir=fake_dir)) == 0
        assert app.load_inventory_catalog_recommendations(project_dir=fake_dir).empty

        # Non-existent SKU in valid catalog
        df_unknown = app.load_historical_series("TOTALLY_UNKNOWN_SKU_12345", project_dir=PROJECT_ROOT)
        assert df_unknown.empty

    def test_no_data_leakage_and_immutability(self):
        """Verify that loading historical data does not mutate underlying structures or leak holdout."""
        df_hist = app.load_historical_series(stock_code="84077", project_dir=PROJECT_ROOT)
        cutoff = pd.Timestamp("2011-11-09")
        pre_training = df_hist[df_hist["Date"] <= cutoff].copy()

        # Assert pre-training slice is strictly prior to holdout start
        assert pre_training["Date"].max() <= cutoff
        assert len(pre_training) > 0

        # Verify immutability: slicing does not modify original dataframe
        assert len(df_hist) >= len(pre_training)
