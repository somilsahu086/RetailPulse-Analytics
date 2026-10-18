"""
RetailPulse — Demand Forecasting Module
Apache Airflow DAG: Scheduled Drift Monitoring & Data Quality Orchestration (Monitoring-Only)

Objective:
Orchestrates the scheduled execution of the Phase 7 drift monitoring pipeline with:
- Strict holdout quarantine enforcement (t <= 2011-11-09).
- Schema and distribution drift evaluation using Evidently AI.
- Prediction-distribution shift auditing.
- Actual-vs-forecast accuracy evaluation (where realized).
- Centralized retraining governance (no automated retraining or replacement).
- Isolated run output directories preventing artifact overwrites.
- Full idempotency and transparent failure logging.

Schedule: Weekly (@weekly, 0 0 * * 1)
"""

import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root and src to path for Airflow scheduler/worker
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from monitoring_pipeline import (
    task_validate_inputs_and_quarantine,
    task_schema_and_quality_audit,
    task_feature_drift_monitoring,
    task_prediction_drift_monitoring,
    task_evaluate_forecast_performance,
    task_compile_monitoring_governance,
    DEFAULT_DATA_DIR,
    DEFAULT_RUNS_DIR
)


# DAG Default Arguments
default_args = {
    "owner": "RetailPulse-MLOps-Team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15)
}


def _get_run_output_dir(context: dict) -> tuple:
    """Helper to generate an isolated, deterministic, run-specific directory."""
    raw_run_id = context.get("run_id", "manual_run")
    # Clean run_id for filesystem safety
    clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(raw_run_id))
    out_dir = os.path.join(DEFAULT_RUNS_DIR, f"run_{clean_id}")
    os.makedirs(out_dir, exist_ok=True)
    return clean_id, out_dir


# Task Execution Callables
def run_validate_inputs_task(**kwargs):
    result = task_validate_inputs_and_quarantine(
        data_dir=DEFAULT_DATA_DIR,
        ref_start="2011-01-01",
        ref_end="2011-08-11",
        curr_start="2011-08-12",
        curr_end="2011-11-07",
        enforce_holdout_quarantine=True
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="validation_result", value=result)
    return result


def run_schema_quality_task(**kwargs):
    result = task_schema_and_quality_audit(
        data_dir=DEFAULT_DATA_DIR,
        ref_start="2011-01-01",
        ref_end="2011-08-11",
        curr_start="2011-08-12",
        curr_end="2011-11-07"
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="quality_result", value=result)
    return result


def run_feature_drift_task(**kwargs):
    clean_id, out_dir = _get_run_output_dir(kwargs)
    result = task_feature_drift_monitoring(
        data_dir=DEFAULT_DATA_DIR,
        output_dir=out_dir,
        run_id=clean_id,
        ref_start="2011-01-01",
        ref_end="2011-08-11",
        curr_start="2011-08-12",
        curr_end="2011-11-07",
        drift_threshold=0.05,
        drift_share=0.50
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="drift_result", value=result)
    return result


def run_prediction_drift_task(**kwargs):
    result = task_prediction_drift_monitoring(drift_threshold=0.05)
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="prediction_drift_result", value=result)
    return result


def run_performance_evaluation_task(**kwargs):
    result = task_evaluate_forecast_performance(
        window_name="Window_3",
        mape_warning_threshold=15.0,
        bias_warning_threshold=10.0
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="performance_result", value=result)
    return result


def run_governance_task(**kwargs):
    ti = kwargs.get("ti")
    clean_id, out_dir = _get_run_output_dir(kwargs)

    validation_res = ti.xcom_pull(task_ids="validate_inputs_and_quarantine", key="validation_result") if ti else {}
    quality_res = ti.xcom_pull(task_ids="audit_schema_and_quality", key="quality_result") if ti else {}
    drift_res = ti.xcom_pull(task_ids="monitor_feature_drift", key="drift_result") if ti else {}
    pred_drift_res = ti.xcom_pull(task_ids="monitor_prediction_drift", key="prediction_drift_result") if ti else {}
    perf_res = ti.xcom_pull(task_ids="evaluate_performance_and_bias", key="performance_result") if ti else {}

    result = task_compile_monitoring_governance(
        output_dir=out_dir,
        run_id=clean_id,
        validation_res=validation_res or {},
        quality_res=quality_res or {},
        drift_res=drift_res or {},
        pred_drift_res=pred_drift_res or {},
        perf_res=perf_res or {}
    )
    if ti:
        ti.xcom_push(key="governance_result", value=result)
    return result


# DAG Definition
with DAG(
    dag_id="demand_drift_monitoring",
    default_args=default_args,
    description="Scheduled monitoring workflow for demand feature drift, data quality, and model accuracy",
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["retailpulse", "drift-monitoring", "data-quality", "mlops", "phase8"]
) as dag:

    # Task 1: Input Validation & Holdout Quarantine
    validate_inputs = PythonOperator(
        task_id="validate_inputs_and_quarantine",
        python_callable=run_validate_inputs_task,
        provide_context=True
    )

    # Task 2: Schema & Quality Audit
    schema_quality = PythonOperator(
        task_id="audit_schema_and_quality",
        python_callable=run_schema_quality_task,
        provide_context=True
    )

    # Task 3: Evidently AI Feature Drift Monitoring
    feature_drift = PythonOperator(
        task_id="monitor_feature_drift",
        python_callable=run_feature_drift_task,
        provide_context=True
    )

    # Task 4: Model Prediction Distribution Shift Audit
    prediction_drift = PythonOperator(
        task_id="monitor_prediction_drift",
        python_callable=run_prediction_drift_task,
        provide_context=True
    )

    # Task 5: Realized Actuals vs Forecast Performance
    performance_eval = PythonOperator(
        task_id="evaluate_performance_and_bias",
        python_callable=run_performance_evaluation_task,
        provide_context=True
    )

    # Task 6: Governance, Alerting & Retraining Decision Gating
    governance_alert = PythonOperator(
        task_id="compile_governance_and_alert",
        python_callable=run_governance_task,
        provide_context=True
    )

    # Define Linear Chronological Workflow Dependencies
    validate_inputs >> schema_quality >> feature_drift >> prediction_drift >> performance_eval >> governance_alert
