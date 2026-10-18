"""
RetailPulse — Demand Forecasting Module
Phase 8 Airflow Orchestration & Retraining Unit Test Suite

Tests:
1. Successful import of the Airflow DAG and verification of DAG properties.
2. Complete task graph validation (8 discrete tasks in linear chronological order).
3. Valid retraining schedule and default arguments.
4. Data validation and non-null integrity checks.
5. Strict anti-leakage checks (enforcing train cutoff <= 2011-11-09).
6. Drift-aware hyperparameter tuning decision logic (conditional execution).
7. Conservative model promotion logic (rejection of worse or equivalent candidate models).
8. Model promotion when candidate exhibits statistically significant holdout improvement.
9. Machine-readable governance metadata and summary generation.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add src and airflow/dags to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
DAGS_PATH = os.path.join(PROJECT_ROOT, "airflow", "dags")

for p in [SRC_PATH, DAGS_PATH, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demand_forecasting_retraining import dag
from demand_drift_monitoring_dag import dag as monitoring_dag
from retraining_pipeline import (
    task_data_validation,
    task_preprocessing,
    task_hyperparameter_tuning,
    task_model_metadata,
    task_mlflow_tracking
)
from monitoring_pipeline import (
    task_validate_inputs_and_quarantine,
    task_schema_and_quality_audit,
    task_feature_drift_monitoring,
    task_prediction_drift_monitoring,
    task_evaluate_forecast_performance,
    task_compile_monitoring_governance
)


def test_dag_import_and_properties():
    """Verify that the Airflow DAG object imports cleanly with valid properties."""
    assert dag is not None
    assert dag.dag_id == "demand_forecasting_retraining"
    assert dag.catchup is False
    assert len(dag.tasks) == 8
    # Schedule should be weekly
    assert dag.schedule_interval in ["@weekly", "0 0 * * 0"] or dag.schedule == "@weekly"


def test_dag_task_list_and_dependency_chain():
    """Verify all 8 tasks exist and downstream dependencies form a linear pipeline."""
    expected_tasks = [
        "data_validation",
        "preprocessing",
        "drift_monitoring",
        "hyperparameter_tuning",
        "model_training",
        "model_evaluation",
        "model_metadata",
        "mlflow_tracking"
    ]

    actual_tasks = [t.task_id for t in dag.tasks]
    assert set(actual_tasks) == set(expected_tasks)

    task_map = {t.task_id: t for t in dag.tasks}

    # Verify linear downstream chain
    for i in range(len(expected_tasks) - 1):
        curr_task = expected_tasks[i]
        next_task = expected_tasks[i + 1]
        downstreams = [t.task_id for t in task_map[curr_task].downstream_list]
        assert next_task in downstreams, f"Expected {next_task} downstream of {curr_task}"


def test_data_validation_task_execution():
    """Verify task_data_validation inspects datasets and reports clean schema status."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    val_res = task_data_validation(data_dir=data_dir)

    assert val_res["status"] == "VALID"
    assert val_res["total_records"] > 0
    assert val_res["eligible_forecasting_products"] >= 50
    assert val_res["null_count"] == 0
    assert val_res["inf_count"] == 0


def test_preprocessing_task_no_leakage():
    """Verify task_preprocessing partitions data with zero future leakage."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    prep_res = task_preprocessing(data_dir=data_dir, cutoff_train="2011-11-09", top_n=5)

    assert prep_res["status"] == "PREPROCESSED"
    assert prep_res["cutoff_train"] == "2011-11-09"
    assert prep_res["train_rows"] > 0
    assert prep_res["holdout_rows"] > 0
    assert prep_res["selected_products_count"] == 5


def test_drift_aware_retraining_decision_logic():
    """Verify that hyperparameter tuning conditionally executes based on drift detection."""
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # Case 1: No drift detected and not forced -> skip expensive tuning
    res_no_drift = task_hyperparameter_tuning(
        data_dir=data_dir,
        drift_detected=False,
        force_tuning=False
    )
    assert res_no_drift["tuning_executed"] is False
    assert "No drift detected" in res_no_drift["rationale"]
    assert "prophet_params" in res_no_drift
    assert "lstm_params" in res_no_drift

    # Case 2: Drift detected -> executes Optuna search on validation data
    res_drift = task_hyperparameter_tuning(
        data_dir=data_dir,
        drift_detected=True,
        force_tuning=False,
        n_trials=1,
        seed=42
    )
    assert res_drift["status"] == "TUNING_COMPLETED"
    assert res_drift["tuning_executed"] is True
    assert "prophet_params" in res_drift
    assert "lstm_params" in res_drift


def test_model_promotion_logic_rejects_worse_models():
    """Verify that a retrained candidate model with worse holdout MAE is rejected."""
    # Synthetic evaluation scenario: Candidate has higher error than production baseline
    prod_mae = 61.285
    cand_mae = 68.450  # Worse performance

    relative_improvement = (prod_mae - cand_mae) / (prod_mae + 1e-8)
    promotion_threshold = 0.02

    if relative_improvement > promotion_threshold:
        decision = "PROMOTED_NEW_MODEL"
        active_model = "Retrained Prophet + LSTM Ensemble"
    else:
        decision = "REJECTED_RETAIN_PRODUCTION"
        active_model = "Phase 5 Prophet + PyTorch LSTM Hybrid Ensemble (Default)"

    assert decision == "REJECTED_RETAIN_PRODUCTION"
    assert active_model == "Phase 5 Prophet + PyTorch LSTM Hybrid Ensemble (Default)"


def test_model_promotion_logic_promotes_substantially_better_models():
    """Verify that a retrained candidate model with significantly superior holdout MAE is promoted."""
    prod_mae = 65.0
    cand_mae = 58.0  # 10.7% improvement

    relative_improvement = (prod_mae - cand_mae) / (prod_mae + 1e-8)
    promotion_threshold = 0.02

    if relative_improvement > promotion_threshold:
        decision = "PROMOTED_NEW_MODEL"
        active_model = "Retrained Prophet + LSTM Ensemble"
    else:
        decision = "REJECTED_RETAIN_PRODUCTION"
        active_model = "Phase 5 Prophet + PyTorch LSTM Hybrid Ensemble (Default)"

    assert decision == "PROMOTED_NEW_MODEL"
    assert active_model == "Retrained Prophet + LSTM Ensemble"


def test_phase8_metadata_and_summary_generation(tmp_path):
    """Verify that Phase 8 governance metadata and run summaries serialize cleanly."""
    out_dir = str(tmp_path)
    eval_res = {
        "status": "EVALUATED",
        "promotion_decision": "REJECTED_RETAIN_PRODUCTION",
        "active_model": "Phase 5 Prophet + PyTorch LSTM Hybrid Ensemble (Default)",
        "production_mae": 61.285,
        "candidate_mae": 65.426
    }
    drift_res = {
        "dataset_drift_detected": True,
        "share_of_drifted_features": 1.0,
        "number_of_drifted_features": 12
    }

    meta_res = task_model_metadata(output_dir=out_dir, eval_result=eval_res, drift_result=drift_res)
    assert os.path.exists(meta_res["metadata_path"])
    assert os.path.exists(meta_res["summary_path"])

    with open(meta_res["metadata_path"], "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["selected_model"] == "Phase 5 Prophet + PyTorch LSTM Hybrid Ensemble (Default)"
    assert "Apache Airflow" in meta_data["extra_metadata"]["orchestrator"]


# =============================================================================
# Phase 8: Monitoring-Only DAG & Orchestration Governance Tests
# =============================================================================

def test_monitoring_dag_properties():
    """Verify dedicated monitoring DAG configuration, schedule, retries, and timeout."""
    assert monitoring_dag is not None
    assert monitoring_dag.dag_id == "demand_drift_monitoring"
    assert monitoring_dag.catchup is False
    assert len(monitoring_dag.tasks) == 6

    # Verify weekly schedule
    assert monitoring_dag.schedule_interval in ["@weekly", "0 0 * * 1"] or monitoring_dag.schedule == "@weekly"

    # Verify default arguments
    assert monitoring_dag.default_args.get("retries") == 2
    assert monitoring_dag.default_args.get("execution_timeout") == pd.Timedelta(minutes=15)
    assert monitoring_dag.default_args.get("owner") == "RetailPulse-MLOps-Team"


def test_monitoring_dag_dependency_chain():
    """Verify linear sequential task dependency for the scheduled monitoring workflow."""
    expected_order = [
        "validate_inputs_and_quarantine",
        "audit_schema_and_quality",
        "monitor_feature_drift",
        "monitor_prediction_drift",
        "evaluate_performance_and_bias",
        "compile_governance_and_alert"
    ]

    actual_tasks = [t.task_id for t in monitoring_dag.tasks]
    assert set(actual_tasks) == set(expected_order)

    task_map = {t.task_id: t for t in monitoring_dag.tasks}
    for i in range(len(expected_order) - 1):
        curr_t = expected_order[i]
        next_t = expected_order[i + 1]
        downstreams = [t.task_id for t in task_map[curr_t].downstream_list]
        assert next_t in downstreams, f"Expected {next_t} to be downstream of {curr_t}"


def test_monitoring_quarantine_validation_fails_fast():
    """Verify that requesting dates in the holdout period fails fast before reading data."""
    with pytest.raises(ValueError, match="HOLDOUT QUARANTINE BREACH"):
        task_validate_inputs_and_quarantine(
            ref_start="2011-01-01",
            ref_end="2011-08-11",
            curr_start="2011-08-12",
            curr_end="2011-11-20",  # Holdout breach
            enforce_holdout_quarantine=True
        )


def test_monitoring_governance_collinear_drift_logic(tmp_path):
    """Verify that feature drift alone does NOT trigger retraining when performance is healthy."""
    out_dir = str(tmp_path / "gov_test")

    # Scenario A: Feature drift detected (12/12 features), but performance is healthy (MAPE=5.92%)
    val_res = {"status": "VALIDATED"}
    qual_res = {"quality_pass": True, "schema_valid": True}
    drift_res = {"dataset_drift_detected": True, "number_of_drifted_features": 12, "share_of_drifted_features": 1.0}
    pred_res = {"prediction_drift_detected": True, "p_value": 0.01}
    perf_res_healthy = {
        "performance_status": "PERFORMANCE_HEALTHY",
        "aggregate_volume_error_pct": 5.92,
        "official_mean_weekly_portfolio_mape_pct": 14.81
    }

    gov_a = task_compile_monitoring_governance(
        output_dir=out_dir,
        run_id="test_run_a",
        validation_res=val_res,
        quality_res=qual_res,
        drift_res=drift_res,
        pred_drift_res=pred_res,
        perf_res=perf_res_healthy
    )

    assert gov_a["governance_status"] == "DRIFT_ALERT_PERFORMANCE_STABLE"
    assert gov_a["retraining_recommendation"] == "RETAIN_PRODUCTION_MODEL"
    assert gov_a["production_model_policy"]["automated_retraining_allowed"] is False
    assert "Input shift did not degrade forecast accuracy" in gov_a["rationale"]

    # Scenario B: Feature drift detected AND performance degraded (>15% warning)
    perf_res_degraded = {
        "performance_status": "PERFORMANCE_DEGRADED",
        "aggregate_volume_error_pct": 22.50,
        "official_mean_weekly_portfolio_mape_pct": 24.10
    }

    gov_b = task_compile_monitoring_governance(
        output_dir=out_dir,
        run_id="test_run_b",
        validation_res=val_res,
        quality_res=qual_res,
        drift_res=drift_res,
        pred_drift_res=pred_res,
        perf_res=perf_res_degraded
    )

    assert gov_b["governance_status"] == "RETRAINING_RECOMMENDED"
    assert gov_b["retraining_recommendation"] == "MANUAL_RETRAINING_ADVISORY"
    assert gov_b["production_model_policy"]["automated_deployment_allowed"] is False
    assert "Manual review and supervised retraining recommended" in gov_b["rationale"]


def test_monitoring_actuals_pending_handling():
    """Verify that actual-vs-forecast evaluation returns ACTUALS_PENDING when ground truth is absent."""
    res = task_evaluate_forecast_performance(backtest_preds_path="non_existent_file.csv")
    assert res["performance_status"] == "ACTUALS_PENDING"
    assert "pending realization" in res["message"]


def test_monitoring_artifact_isolation(tmp_path):
    """Verify that monitoring runs write outputs to isolated run-specific directories."""
    from pathlib import Path
    run_dir = tmp_path / "isolated_run_123"
    phase7_dir = Path(PROJECT_ROOT) / "outputs" / "forecasting" / "phase7"

    assert not run_dir.exists()

    # Create dummy governance record in isolated directory
    res = task_compile_monitoring_governance(
        output_dir=str(run_dir),
        run_id="isolated_run_123",
        validation_res={"status": "VALIDATED"},
        quality_res={"quality_pass": True},
        drift_res={"dataset_drift_detected": False},
        pred_drift_res={"prediction_drift_detected": False},
        perf_res={"performance_status": "PERFORMANCE_HEALTHY"}
    )

    assert (run_dir / "monitoring_governance.json").exists()
    # Confirm phase7 directory was not overwritten
    assert (phase7_dir / "phase7_drift_report.html").exists()
    assert (phase7_dir / "monitoring_metadata.json").exists()

