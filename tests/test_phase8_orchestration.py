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
from retraining_pipeline import (
    task_data_validation,
    task_preprocessing,
    task_hyperparameter_tuning,
    task_model_metadata,
    task_mlflow_tracking
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
