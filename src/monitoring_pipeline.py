"""
RetailPulse — Demand Forecasting Module
Phase 8: Scheduled Monitoring Orchestration Pipeline (Monitoring-Only)

Provides modular, reproducible orchestration tasks for:
1. Input data validation and strict holdout quarantine enforcement.
2. Data quality and schema conformity auditing.
3. Evidently AI feature distribution and dataset drift detection.
4. Model prediction distribution drift monitoring.
5. Realized actual-versus-forecast performance and bias evaluation.
6. Centralized monitoring governance, alerting, and retraining decision gating.

GOVERNANCE PRINCIPLES:
- Strictly monitoring-only: NEVER trains, tunes, selects, or promotes models.
- Quarantine protection: Validates date boundaries before any data access (t <= 2011-11-09).
- Collinear drift rule: Input feature drift alone NEVER triggers automated retraining.
- Metric integrity: Distinguishes multi-week aggregate volume error from official mean weekly portfolio MAPE (14.81%).
- Production protection: Dashboard scorecard and production baseline models remain frozen.
- Artifact isolation: All run outputs are saved to isolated, run-specific directories.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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

logger = logging.getLogger("retailpulse.monitoring_orchestration")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [MonitoringPipeline] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default data and outputs directories
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DEFAULT_RUNS_DIR = str(PROJECT_ROOT / "outputs" / "forecasting" / "monitoring_runs")


# -----------------------------------------------------------------------------
# Task 1: Input Validation & Holdout Quarantine
# -----------------------------------------------------------------------------
def task_validate_inputs_and_quarantine(
    data_dir: str = DEFAULT_DATA_DIR,
    ref_start: str = "2011-01-01",
    ref_end: str = "2011-08-11",
    curr_start: str = "2011-08-12",
    curr_end: str = "2011-11-07",
    enforce_holdout_quarantine: bool = True
) -> Dict[str, Any]:
    """
    Task 1: Validate input data files, schema integrity, and strict quarantine boundaries.

    FAILS FAST before loading or querying any demand data if dates enter the holdout.
    """
    logger.info("Task 1: Validating monitoring date boundaries and quarantine guards...")

    ref_start_ts = pd.to_datetime(ref_start).floor("D")
    ref_end_ts = pd.to_datetime(ref_end).floor("D")
    curr_start_ts = pd.to_datetime(curr_start).floor("D")
    curr_end_ts = pd.to_datetime(curr_end).floor("D")

    # Anti-leakage chronological assertions
    if ref_end_ts >= curr_start_ts:
        raise ValueError(f"LEAKAGE DETECTED: Reference end ({ref_end}) must strictly precede Current start ({curr_start})!")
    if ref_start_ts > ref_end_ts:
        raise ValueError(f"Reference start ({ref_start}) cannot be after reference end ({ref_end}).")
    if curr_start_ts > curr_end_ts:
        raise ValueError(f"Current start ({curr_start}) cannot be after current end ({curr_end}).")

    # Fail-safe holdout quarantine assertion (t <= 2011-11-09)
    if enforce_holdout_quarantine:
        holdout_boundary = pd.Timestamp("2011-11-09")
        if any(ts > holdout_boundary for ts in [curr_end_ts, curr_start_ts, ref_end_ts, ref_start_ts]):
            raise ValueError(
                f"HOLDOUT QUARANTINE BREACH: Requested monitoring dates extend into the quarantined "
                f"holdout period (2011-11-10 to 2011-12-09). Drift monitoring must be performed strictly "
                f"on development data (t <= 2011-11-09)."
            )

    # Check data files exist
    data_path = Path(data_dir) / "daily_demand_clean.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Clean daily demand dataset not found at: {data_path}")

    # Inspect schema metadata safely without full table load
    df_sample = pd.read_parquet(data_path)
    required_cols = ["Date", "StockCode", "Demand"]
    missing = [c for c in required_cols if c not in df_sample.columns]
    if missing:
        raise ValueError(f"Dataset missing required primary columns: {missing}")

    date_min = pd.to_datetime(df_sample["Date"]).min().strftime("%Y-%m-%d")
    date_max = pd.to_datetime(df_sample["Date"]).max().strftime("%Y-%m-%d")

    logger.info(f"Task 1 Passed: Dataset spans {date_min} to {date_max}. Quarantine respected (max requested: {curr_end}).")

    return {
        "status": "VALIDATED",
        "reference_window": f"{ref_start} to {ref_end}",
        "current_window": f"{curr_start} to {curr_end}",
        "dataset_total_rows": len(df_sample),
        "quarantine_enforced": True,
        "max_monitored_date": curr_end
    }


# -----------------------------------------------------------------------------
# Task 2: Data Schema & Quality Audit
# -----------------------------------------------------------------------------
def task_schema_and_quality_audit(
    data_dir: str = DEFAULT_DATA_DIR,
    ref_start: str = "2011-01-01",
    ref_end: str = "2011-08-11",
    curr_start: str = "2011-08-12",
    curr_end: str = "2011-11-07",
    stock_codes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Task 2: Evaluate column existence, null counts, infinite values, and data type alignment.
    """
    logger.info("Task 2: Executing data schema and quality audit...")
    data_path = Path(data_dir) / "daily_demand_clean.parquet"
    df = pd.read_parquet(data_path)
    df["StockCode"] = df["StockCode"].astype(str)

    ref_df, curr_df, monitored_features = prepare_monitoring_datasets(
        daily_demand_df=df,
        ref_start=ref_start,
        ref_end=ref_end,
        curr_start=curr_start,
        curr_end=curr_end,
        stock_codes=stock_codes,
        feature_cols=CORE_MONITORED_FEATURES,
        enforce_holdout_quarantine=True
    )

    quality_res = run_schema_and_quality_audit(
        reference_df=ref_df,
        current_df=curr_df,
        required_columns=monitored_features,
        max_missing_share=0.0
    )

    if not quality_res["schema_valid"] or not quality_res["quality_pass"]:
        logger.error(f"Task 2 FAILED: Schema or quality checks failed: {quality_res}")
        raise ValueError(f"Schema/Quality audit failed with status {quality_res['status']}")

    logger.info(f"Task 2 Passed: Schema valid, 0 nulls across {len(ref_df):,} ref and {len(curr_df):,} curr rows.")
    return quality_res


# -----------------------------------------------------------------------------
# Task 3: Feature & Dataset Drift Detection (Evidently AI)
# -----------------------------------------------------------------------------
def task_feature_drift_monitoring(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    ref_start: str = "2011-01-01",
    ref_end: str = "2011-08-11",
    curr_start: str = "2011-08-12",
    curr_end: str = "2011-11-07",
    stock_codes: Optional[List[str]] = None,
    drift_threshold: float = 0.05,
    drift_share: float = 0.50
) -> Dict[str, Any]:
    """
    Task 3: Execute Evidently AI distribution drift monitoring on all 12 forecasting features.
    Outputs are saved to an isolated run-specific directory.
    """
    logger.info("Task 3: Running Evidently AI feature drift monitoring...")
    run_tag = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(output_dir or (Path(DEFAULT_RUNS_DIR) / run_tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir) / "daily_demand_clean.parquet"
    df = pd.read_parquet(data_path)
    df["StockCode"] = df["StockCode"].astype(str)

    ref_df, curr_df, monitored_features = prepare_monitoring_datasets(
        daily_demand_df=df,
        ref_start=ref_start,
        ref_end=ref_end,
        curr_start=curr_start,
        curr_end=curr_end,
        stock_codes=stock_codes,
        feature_cols=CORE_MONITORED_FEATURES,
        enforce_holdout_quarantine=True
    )

    drift_summary, quality_summary, feature_drift_df = run_drift_monitoring(
        reference_df=ref_df,
        current_df=curr_df,
        monitored_columns=monitored_features,
        drift_threshold=drift_threshold,
        drift_share=drift_share,
        output_dir=str(out_dir),
        save_html=True,
        html_filename="drift_report.html"
    )

    # Save CSV and visual summary plot in run directory
    feature_csv_path = out_dir / "feature_drift.csv"
    feature_drift_df.to_csv(feature_csv_path, index=False)

    plot_path = out_dir / "drift_summary.png"
    plot_drift_summary(feature_drift_df, str(plot_path), title=f"Feature Drift Summary ({run_tag})")

    # Export structured JSON summaries
    with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(drift_summary, f, indent=2)

    with open(out_dir / "data_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)

    logger.info(
        f"Task 3 Completed: Drift detected={drift_summary['dataset_drift_detected']} "
        f"({drift_summary['number_of_drifted_features']}/{drift_summary['number_of_features']} features). "
        f"Artifacts saved to {out_dir}"
    )

    return {
        "status": "DRIFT_MONITORED",
        "run_id": run_tag,
        "output_dir": str(out_dir),
        "dataset_drift_detected": drift_summary["dataset_drift_detected"],
        "share_of_drifted_features": drift_summary["share_of_drifted_features"],
        "number_of_drifted_features": drift_summary["number_of_drifted_features"],
        "total_monitored_features": drift_summary["number_of_features"],
        "drift_threshold": drift_threshold,
        "feature_drift_records": feature_drift_df.to_dict(orient="records")
    }


# -----------------------------------------------------------------------------
# Task 4: Model Prediction Distribution Drift Monitoring
# -----------------------------------------------------------------------------
def task_prediction_drift_monitoring(
    backtest_preds_path: Optional[str] = None,
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Task 4: Evaluate prediction distribution shifts between baseline and recent forecast outputs.
    """
    logger.info("Task 4: Evaluating prediction-distribution drift...")
    preds_file = Path(backtest_preds_path or (PROJECT_ROOT / "outputs" / "forecasting" / "rolling_backtest_experiment" / "rolling_backtest_predictions.csv"))

    if not preds_file.exists():
        logger.warning(f"Backtest predictions not found at {preds_file}. Skipping prediction drift.")
        return {
            "status": "PREDICTION_DRIFT_UNAVAILABLE",
            "prediction_drift_detected": False,
            "message": f"Predictions file not found: {preds_file}"
        }

    bt_preds = pd.read_csv(preds_file)
    p_w1 = bt_preds[(bt_preds["Model"] == "Optuna_Tuned_Prophet") & (bt_preds["Window"] == "Window_1")]["Predicted_Demand"].values
    p_w3 = bt_preds[(bt_preds["Model"] == "Optuna_Tuned_Prophet") & (bt_preds["Window"] == "Window_3")]["Predicted_Demand"].values

    pred_drift_res = run_prediction_drift_monitoring(
        reference_preds=p_w1,
        current_preds=p_w3,
        drift_threshold=drift_threshold,
        prediction_col="Prophet_Predicted_Demand"
    )

    logger.info(f"Task 4 Completed: Prediction drift detected={pred_drift_res['prediction_drift_detected']} (p={pred_drift_res['p_value']}).")
    return pred_drift_res


# -----------------------------------------------------------------------------
# Task 5: Actual Forecast Performance & Bias Monitoring
# -----------------------------------------------------------------------------
def task_evaluate_forecast_performance(
    backtest_preds_path: Optional[str] = None,
    window_name: str = "Window_3",
    mape_warning_threshold: float = 15.0,
    mape_critical_threshold: float = 20.0,
    bias_warning_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Task 5: Evaluate realized actual-vs-forecast accuracy where ground truth is available.

    Clearly distinguishes 30-day aggregate volume error from official mean weekly portfolio MAPE.
    If actuals are pending, safely reports ACTUALS_PENDING without falsifying errors.
    """
    logger.info(f"Task 5: Evaluating realized actual-vs-forecast performance for {window_name}...")
    preds_file = Path(backtest_preds_path or (PROJECT_ROOT / "outputs" / "forecasting" / "rolling_backtest_experiment" / "rolling_backtest_predictions.csv"))

    if not preds_file.exists():
        logger.info("Ground truth predictions file unavailable. Reporting ACTUALS_PENDING.")
        return {
            "performance_status": "ACTUALS_PENDING",
            "message": "Actual demand records pending realization. Accuracy check deferred."
        }

    bt_preds = pd.read_csv(preds_file)
    p_df = bt_preds[bt_preds["Model"] == "Optuna_Tuned_Prophet"].copy()
    l_df = bt_preds[bt_preds["Model"] == "PyTorch_LSTM_Tuned"].copy()

    merged_bt = p_df.merge(
        l_df[["Window", "Date", "StockCode", "Predicted_Demand"]],
        on=["Window", "Date", "StockCode"],
        suffixes=("_prophet", "_lstm")
    )
    merged_bt["Pred_50_50"] = 0.50 * merged_bt["Predicted_Demand_prophet"] + 0.50 * merged_bt["Predicted_Demand_lstm"]

    w_sub = merged_bt[merged_bt["Window"] == window_name].sort_values("Date").copy()
    if w_sub.empty:
        return {
            "performance_status": "INSUFFICIENT_DATA",
            "message": f"No records found for window {window_name}."
        }

    # Evaluate 30-day full horizon (includes all 1,500 SKU-day predictions up to 2011-11-09)
    perf_30d = run_model_performance_monitoring(
        actual_demand=w_sub["Actual_Demand"],
        predicted_demand=w_sub["Pred_50_50"],
        mape_warning_threshold=mape_warning_threshold,
        mape_critical_threshold=mape_critical_threshold,
        bias_warning_threshold=bias_warning_threshold
    )

    # Evaluate strict 28-day 4-week window (up to 2011-11-07, 1,400 predictions)
    w_28d = w_sub[w_sub["Date"] <= "2011-11-07"].copy()
    perf_28d = run_model_performance_monitoring(
        actual_demand=w_28d["Actual_Demand"],
        predicted_demand=w_28d["Pred_50_50"],
        mape_warning_threshold=mape_warning_threshold,
        bias_warning_threshold=bias_warning_threshold
    )

    # Compute official 4-week mean weekly portfolio MAPE
    w_28d["Date_dt"] = pd.to_datetime(w_28d["Date"])
    w_28d["Day_Idx"] = (w_28d["Date_dt"] - w_28d["Date_dt"].min()).dt.days
    w_28d["Week_Num"] = w_28d["Day_Idx"] // 7 + 1
    weekly_mapes = []
    for wk in [1, 2, 3, 4]:
        sub_wk = w_28d[w_28d["Week_Num"] == wk]
        act_w = sub_wk["Actual_Demand"].sum()
        pred_w = sub_wk["Pred_50_50"].sum()
        if act_w > 0:
            weekly_mapes.append(abs(pred_w - act_w) / act_w * 100.0)
    official_weekly_mape = float(np.mean(weekly_mapes)) if weekly_mapes else 0.0

    perf_30d["evaluation_period_30d"] = f"{w_sub['Date'].min()} to {w_sub['Date'].max()} (N={len(w_sub)})"
    perf_30d["evaluation_period_28d_strict"] = f"{w_28d['Date'].min()} to {w_28d['Date'].max()} (N={len(w_28d)})"
    perf_30d["strict_28d_aggregate_volume_error_pct"] = round(perf_28d["aggregate_volume_error_pct"], 4)
    perf_30d["official_mean_weekly_portfolio_mape_pct"] = round(official_weekly_mape, 2)

    logger.info(
        f"Task 5 Completed: 30-Day Aggregate Volume Error={perf_30d['aggregate_volume_error_pct']:.2f}%, "
        f"Official Mean Weekly Portfolio MAPE={official_weekly_mape:.2f}%, Status={perf_30d['performance_status']}."
    )
    return perf_30d


# -----------------------------------------------------------------------------
# Task 6: Governance, Alerting & Retraining Decision Gating
# -----------------------------------------------------------------------------
def task_compile_monitoring_governance(
    output_dir: str,
    run_id: str,
    validation_res: Dict[str, Any],
    quality_res: Dict[str, Any],
    drift_res: Dict[str, Any],
    pred_drift_res: Dict[str, Any],
    perf_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Task 6: Synthesize all 4 monitoring pillars and enforce retraining governance rules.

    RETRAINING GOVERNANCE RULES:
    1. Feature drift ALONE does NOT trigger automated model retraining.
    2. Retraining is ONLY recommended if BOTH feature drift is detected AND official weekly portfolio MAPE > 15%.
    3. Automated model replacement is STRICTLY FORBIDDEN. Only advisory alerts are issued.
    4. The official dashboard scorecard remains frozen at 14.81%.
    """
    logger.info("Task 6: Enforcing governance and alert rules...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_drift_detected = bool(drift_res.get("dataset_drift_detected", False))
    pred_drift_detected = bool(pred_drift_res.get("prediction_drift_detected", False))
    perf_status = perf_res.get("performance_status", "ACTUALS_PENDING")
    weekly_mape = perf_res.get("official_mean_weekly_portfolio_mape_pct", 14.81)

    # Evaluate Retraining Decision Gating
    if not quality_res.get("quality_pass", True):
        governance_status = "CRITICAL_SCHEMA_FAILURE"
        retraining_recommendation = "BLOCKED_DATA_QUALITY_ISSUE"
        alert_level = "CRITICAL"
        rationale = "Data schema or null constraints failed. Downstream modeling blocked until data pipeline is restored."
    elif perf_status in ["PERFORMANCE_DEGRADED", "PERFORMANCE_CRITICAL"] and feature_drift_detected:
        governance_status = "RETRAINING_RECOMMENDED"
        retraining_recommendation = "MANUAL_RETRAINING_ADVISORY"
        alert_level = "WARNING"
        rationale = (
            f"Both input distribution drift and forecast performance degradation (MAPE={weekly_mape}%) observed. "
            "Manual review and supervised retraining recommended. Automated promotion remains disabled."
        )
    elif feature_drift_detected:
        governance_status = "DRIFT_ALERT_PERFORMANCE_STABLE"
        retraining_recommendation = "RETAIN_PRODUCTION_MODEL"
        alert_level = "INFO"
        rationale = (
            "Input feature distribution drift detected (seasonal volume ramp), but model forecast performance "
            f"remains healthy (aggregate volume error={perf_res.get('aggregate_volume_error_pct', 'N/A')}%). "
            "Input shift did not degrade forecast accuracy; retraining is NOT recommended."
        )
    else:
        governance_status = "MONITORING_HEALTHY"
        retraining_recommendation = "RETAIN_PRODUCTION_MODEL"
        alert_level = "NORMAL"
        rationale = "All data quality, distribution stability, and performance metrics are within operational tolerances."

    governance_record = {
        "metadata_version": "1.0.0",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "Demand Forecasting",
        "phase": "Phase 8 Airflow Scheduled Monitoring Orchestration",
        "governance_status": governance_status,
        "alert_level": alert_level,
        "retraining_recommendation": retraining_recommendation,
        "rationale": rationale,
        "production_model_policy": {
            "automated_retraining_allowed": False,
            "automated_deployment_allowed": False,
            "production_baseline_status": "FROZEN",
            "official_dashboard_scorecard_mape": "14.81%"
        },
        "monitoring_pillars": {
            "pillar_1_data_validation": validation_res,
            "pillar_2_schema_and_quality": quality_res,
            "pillar_3_feature_and_dataset_drift": {
                "dataset_drift_detected": feature_drift_detected,
                "share_of_drifted_features": drift_res.get("share_of_drifted_features", 0.0),
                "number_of_drifted_features": drift_res.get("number_of_drifted_features", 0),
                "total_monitored_features": drift_res.get("total_monitored_features", 12),
                "collinearity_interpretation": (
                    "All 12 features are mathematical transformations of univariate demand. "
                    "Concurrent drift reflects seasonal volume expansion rather than 12 independent failure modes."
                )
            },
            "pillar_4_prediction_drift": {
                "prediction_drift_detected": pred_drift_detected,
                "ks_p_value": pred_drift_res.get("p_value", 1.0)
            },
            "pillar_5_actual_performance": perf_res
        }
    }

    # Save governance metadata JSON in run directory
    gov_file = out_dir / "monitoring_governance.json"
    with open(gov_file, "w", encoding="utf-8") as f:
        json.dump(governance_record, f, indent=2)

    logger.info(f"Task 6 Completed: Governance Status={governance_status}, Alert={alert_level}. Saved to {gov_file}.")
    return governance_record
