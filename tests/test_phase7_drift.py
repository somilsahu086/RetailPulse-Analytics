"""
RetailPulse — Demand Forecasting Module
Phase 7 Drift Monitoring & Data Quality Unit Test Suite

Tests:
1. Chronological ordering of Reference and Current periods.
2. Anti-leakage safeguards (rejection of overlapping or inverted time periods).
3. Successful execution of Evidently drift monitoring.
4. Configurable statistical significance thresholds and drift share ratios.
5. Missing, null, and non-finite value handling.
6. Generation and integrity of output reports (HTML, CSV, JSON, PNG).
7. Comprehensive metadata generation and serialization.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from drift_monitoring import (
    prepare_monitoring_datasets,
    run_drift_monitoring,
    plot_drift_summary,
    build_monitoring_metadata,
    run_prediction_drift_monitoring,
    run_model_performance_monitoring,
    run_schema_and_quality_audit,
    CORE_MONITORED_FEATURES
)


@pytest.fixture
def synthetic_retail_timeline_df():
    """Create 200 days of synthetic daily retail demand for multiple products."""
    dates = pd.date_range("2011-01-01", periods=200, freq="D")
    records = []
    np.random.seed(42)

    for code in ["PROD_A", "PROD_B"]:
        base_demand = 30.0 if code == "PROD_A" else 50.0
        # Add slight drift in the second half (after day 140)
        demand = [
            base_demand + 10.0 * np.sin(i / 7.0) + (25.0 if i > 140 else 0.0) + np.random.normal(0, 3)
            for i in range(len(dates))
        ]
        for d, q in zip(dates, demand):
            records.append({
                "Date": d,
                "StockCode": code,
                "Demand": max(1.0, round(q, 1))
            })

    return pd.DataFrame(records)


def test_reference_current_chronological_separation(synthetic_retail_timeline_df):
    """Verify that Reference and Current datasets are strictly separated chronologically."""
    ref_df, curr_df, features = prepare_monitoring_datasets(
        synthetic_retail_timeline_df,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15"
    )

    assert not ref_df.empty
    assert not curr_df.empty
    assert len(features) > 0
    assert ref_df["Date"].max() < curr_df["Date"].min()
    assert ref_df["Date"].max() <= pd.Timestamp("2011-04-30")
    assert curr_df["Date"].min() >= pd.Timestamp("2011-05-01")


def test_reference_current_leakage_exception(synthetic_retail_timeline_df):
    """Verify that inverted or overlapping reference and current periods raise a ValueError."""
    # Case 1: Inverted dates (ref_end > curr_start)
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        prepare_monitoring_datasets(
            synthetic_retail_timeline_df,
            ref_start="2011-01-01",
            ref_end="2011-06-01",
            curr_start="2011-05-01",
            curr_end="2011-07-15"
        )

    # Case 2: Identical dates (ref_end == curr_start)
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        prepare_monitoring_datasets(
            synthetic_retail_timeline_df,
            ref_start="2011-01-01",
            ref_end="2011-05-01",
            curr_start="2011-05-01",
            curr_end="2011-07-15"
        )


def test_drift_monitoring_runs_successfully(synthetic_retail_timeline_df):
    """Verify that Evidently drift monitoring executes and returns expected data structures."""
    ref_df, curr_df, features = prepare_monitoring_datasets(
        synthetic_retail_timeline_df,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15",
        stock_codes=["PROD_A"]
    )

    drift_summary, quality_summary, feature_drift_df = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=features[:4],
        drift_threshold=0.05,
        save_html=False
    )

    assert isinstance(drift_summary["dataset_drift_detected"], bool)
    assert drift_summary["number_of_features"] == 4
    assert len(feature_drift_df) == 4
    assert set(["feature", "drift_detected", "drift_score", "threshold"]).issubset(feature_drift_df.columns)
    assert quality_summary["reference_row_count"] == len(ref_df)
    assert quality_summary["current_row_count"] == len(curr_df)


def test_drift_threshold_configurable(synthetic_retail_timeline_df):
    """Verify that custom statistical thresholds are correctly applied."""
    ref_df, curr_df, features = prepare_monitoring_datasets(
        synthetic_retail_timeline_df,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15",
        stock_codes=["PROD_A"]
    )

    drift_summary_strict, _, feature_df_strict = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=["Demand"],
        drift_threshold=0.01,
        save_html=False
    )

    drift_summary_lenient, _, feature_df_lenient = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=["Demand"],
        drift_threshold=0.25,
        save_html=False
    )

    assert drift_summary_strict["drift_threshold"] == 0.01
    assert drift_summary_lenient["drift_threshold"] == 0.25
    assert feature_df_strict["threshold"].iloc[0] == 0.01
    assert feature_df_lenient["threshold"].iloc[0] == 0.25


def test_missing_and_null_handling(synthetic_retail_timeline_df):
    """Verify that datasets containing missing/null values are handled safely without crashing."""
    df_with_nulls = synthetic_retail_timeline_df.copy()
    # Introduce nulls in demand
    df_with_nulls.loc[5:10, "Demand"] = np.nan

    ref_df, curr_df, features = prepare_monitoring_datasets(
        df_with_nulls,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15",
        stock_codes=["PROD_A"]
    )

    # Monitored features should have nulls filled safely
    for col in features:
        assert ref_df[col].isna().sum() == 0
        assert curr_df[col].isna().sum() == 0

    drift_summary, quality_summary, _ = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=features[:3],
        save_html=False
    )

    assert quality_summary["missing_values"]["reference_missing_count"] == 0


def test_invalid_non_finite_values_handling(synthetic_retail_timeline_df):
    """Verify that infinite and non-numeric values are safely coerced and logged."""
    df_with_inf = synthetic_retail_timeline_df.copy()
    df_with_inf.loc[15, "Demand"] = np.inf

    ref_df, curr_df, features = prepare_monitoring_datasets(
        df_with_inf,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15"
    )

    assert not np.isinf(ref_df["Demand"]).any()
    assert not np.isnan(ref_df["Demand"]).any()


def test_expected_output_files_generated(synthetic_retail_timeline_df, tmp_path):
    """Verify that all Phase 7 artifact files (HTML, CSV, JSON, PNG) are generated properly."""
    ref_df, curr_df, features = prepare_monitoring_datasets(
        synthetic_retail_timeline_df,
        ref_start="2011-01-01",
        ref_end="2011-04-30",
        curr_start="2011-05-01",
        curr_end="2011-07-15",
        stock_codes=["PROD_A"]
    )

    drift_summary, quality_summary, feature_drift_df = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=features[:3],
        output_dir=str(tmp_path),
        save_html=True,
        html_filename="test_drift_report.html"
    )

    # Save CSV and JSON artifacts
    drift_json = tmp_path / "drift_summary.json"
    drift_json.write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")

    feature_csv = tmp_path / "feature_drift.csv"
    feature_drift_df.to_csv(feature_csv, index=False)

    quality_json = tmp_path / "data_quality_summary.json"
    quality_json.write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

    meta = build_monitoring_metadata(
        reference_period="2011-01-01 to 2011-04-30",
        current_period="2011-05-01 to 2011-07-15",
        drift_summary=drift_summary,
        data_quality_summary=quality_summary
    )
    meta_json = tmp_path / "monitoring_metadata.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    plot_file = tmp_path / "test_drift_plot.png"
    plot_drift_summary(feature_drift_df, str(plot_file))

    # Assertions
    assert (tmp_path / "test_drift_report.html").exists()
    assert (tmp_path / "test_drift_report.html").stat().st_size > 500
    assert drift_json.exists()
    assert feature_csv.exists()
    assert quality_json.exists()
    assert meta_json.exists()
    assert plot_file.exists()


def test_prediction_drift_monitoring_stable_and_detected():
    """Verify prediction-distribution drift accurately flags stable vs drifted predictions."""
    np.random.seed(42)
    ref_preds = np.random.normal(50.0, 8.0, 500)
    curr_stable = np.random.normal(50.5, 8.0, 500)
    curr_drifted = np.random.normal(75.0, 15.0, 500)

    res_stable = run_prediction_drift_monitoring(ref_preds, curr_stable, drift_threshold=0.01)
    res_drifted = run_prediction_drift_monitoring(ref_preds, curr_drifted, drift_threshold=0.01)

    assert not res_stable["prediction_drift_detected"]
    assert res_stable["drift_status"] == "STABLE"
    assert res_stable["p_value"] > 0.01

    assert res_drifted["prediction_drift_detected"]
    assert res_drifted["drift_status"] == "DRIFT_DETECTED"
    assert res_drifted["p_value"] < 0.01
    assert res_drifted["percent_mean_shift"] > 30.0
    assert "interpretation" in res_drifted


def test_model_performance_monitoring_metrics():
    """Verify performance metrics, signed bias, tracking signal, and status categorization."""
    actuals = [100.0, 150.0, 200.0, 120.0, 180.0]

    # Healthy forecast (close predictions)
    healthy_preds = [102.0, 148.0, 205.0, 118.0, 182.0]
    perf_healthy = run_model_performance_monitoring(
        actual_demand=actuals,
        predicted_demand=healthy_preds,
        mape_warning_threshold=15.0,
        bias_warning_threshold=10.0
    )

    assert perf_healthy["performance_status"] == "PERFORMANCE_HEALTHY"
    assert perf_healthy["portfolio_mape_pct"] < 5.0
    assert abs(perf_healthy["signed_bias_pct"]) < 5.0
    assert "interpretation" in perf_healthy

    # Degraded forecast (severe under-forecast)
    degraded_preds = [70.0, 100.0, 130.0, 80.0, 120.0]
    perf_degraded = run_model_performance_monitoring(
        actual_demand=actuals,
        predicted_demand=degraded_preds,
        mape_warning_threshold=15.0,
        mape_critical_threshold=25.0
    )

    assert perf_degraded["performance_status"] in ["PERFORMANCE_DEGRADED", "PERFORMANCE_CRITICAL"]
    assert perf_degraded["portfolio_mape_pct"] > 25.0
    assert perf_degraded["signed_bias_pct"] < -25.0


def test_schema_and_quality_audit(synthetic_retail_timeline_df):
    """Verify schema validation, missing column checks, and quality thresholds."""
    df_ref = synthetic_retail_timeline_df[synthetic_retail_timeline_df["Date"] < "2011-04-01"].copy()
    df_curr = synthetic_retail_timeline_df[synthetic_retail_timeline_df["Date"] >= "2011-04-01"].copy()

    # Case 1: Valid schema
    res_pass = run_schema_and_quality_audit(
        reference_df=df_ref,
        current_df=df_curr,
        required_columns=["Demand", "StockCode"]
    )
    assert res_pass["schema_valid"]
    assert res_pass["quality_pass"]
    assert res_pass["status"] == "PASS"

    # Case 2: Missing required column in current
    df_curr_missing = df_curr.drop(columns=["Demand"])
    res_missing = run_schema_and_quality_audit(
        reference_df=df_ref,
        current_df=df_curr_missing,
        required_columns=["Demand", "StockCode"]
    )
    assert not res_missing["schema_valid"]
    assert "Demand" in res_missing["current_missing_columns"]
    assert res_missing["status"] == "FAIL"

    # Case 3: Empty input
    res_empty = run_schema_and_quality_audit(
        reference_df=pd.DataFrame(),
        current_df=df_curr
    )
    assert not res_empty["schema_valid"]
    assert res_empty["status"] == "FAIL"


def test_holdout_quarantine_enforcement(synthetic_retail_timeline_df):
    """Verify that attempting to monitor into the quarantined holdout raises a clear breach exception."""
    with pytest.raises(ValueError, match="HOLDOUT QUARANTINE BREACH"):
        prepare_monitoring_datasets(
            synthetic_retail_timeline_df,
            ref_start="2011-01-01",
            ref_end="2011-08-11",
            curr_start="2011-08-12",
            curr_end="2011-11-25",  # Deep into quarantined holdout
            enforce_holdout_quarantine=True
        )


def test_empty_predictions_handling():
    """Verify that empty or non-finite inputs to prediction and performance monitoring return safe responses."""
    empty_pred_res = run_prediction_drift_monitoring([], [])
    assert empty_pred_res["status"] == "INSUFFICIENT_DATA"
    assert not empty_pred_res["prediction_drift_detected"]

    empty_perf_res = run_model_performance_monitoring([], [])
    assert empty_perf_res["performance_status"] == "INSUFFICIENT_DATA"


def test_holdout_quarantine_fails_before_data_access():
    """Regression test: Proves that requesting dates extending beyond 2011-11-09
    fails immediately during date validation before any data loading, querying, or slicing occurs."""
    class GuardedDataFrame:
        """Mock DataFrame object that raises an AssertionError if any attribute or method is queried."""
        def __getattr__(self, name):
            raise AssertionError(f"UNSAFE DATA ACCESS: Attribute or method '{name}' was queried!")

        @property
        def empty(self):
            raise AssertionError("UNSAFE DATA ACCESS: 'empty' property was queried!")

    # Attempting to monitor into the holdout (2011-11-10+) must fail during date validation
    # without touching or querying the dataframe object.
    with pytest.raises(ValueError, match="HOLDOUT QUARANTINE BREACH"):
        prepare_monitoring_datasets(
            GuardedDataFrame(),
            ref_start="2011-01-01",
            ref_end="2011-08-11",
            curr_start="2011-08-12",
            curr_end="2011-11-10"  # Exact holdout start boundary
        )


def test_holdout_quarantine_default_enforcement(synthetic_retail_timeline_df):
    """Verify that enforce_holdout_quarantine defaults to True without passing explicit flag."""
    with pytest.raises(ValueError, match="HOLDOUT QUARANTINE BREACH"):
        prepare_monitoring_datasets(
            synthetic_retail_timeline_df,
            ref_start="2011-01-01",
            ref_end="2011-08-11",
            curr_start="2011-08-12",
            curr_end="2011-11-15"  # No enforce_holdout_quarantine argument passed
        )


def test_aggregate_volume_error_metric_semantics():
    """Verify that performance monitoring reports aggregate_volume_error_pct and scope notes."""
    actuals = [100.0, 150.0, 200.0, 120.0, 180.0]
    preds = [110.0, 140.0, 210.0, 110.0, 190.0]

    perf = run_model_performance_monitoring(actual_demand=actuals, predicted_demand=preds)
    assert "aggregate_volume_error_pct" in perf
    assert "metric_scope_note" in perf
    assert "official mean weekly portfolio MAPE" in perf["metric_scope_note"]
    assert perf["aggregate_volume_error_pct"] == perf["portfolio_mape_pct"]

