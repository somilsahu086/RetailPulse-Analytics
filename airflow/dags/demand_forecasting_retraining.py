"""
RetailPulse — Demand Forecasting Module
Apache Airflow DAG: Automated Demand Forecasting Retraining Pipeline

Workflow Architecture:
1. data_validation: Audit daily demand parquet, eligibility table, nulls, and date boundaries.
2. preprocessing: Partition chronological history with zero future/holdout leakage.
3. drift_monitoring: Run Evidently AI distribution drift check comparing reference vs current data.
4. hyperparameter_tuning: Drift-aware Optuna optimization on internal validation data.
5. model_training: Train candidate forecasting models (Prophet, PyTorch LSTM, Ensemble).
6. model_evaluation: Evaluate on untouched holdout test window and decide model promotion.
7. model_metadata: Compile machine-readable governance metadata and software package manifest.
8. mlflow_tracking: Log experiment run, metrics, parameters, and artifacts to MLflow registry.

Schedule: Weekly (@weekly) with manual trigger support.
Anti-Leakage Guarantee: Holdout window (2011-11-10 to 2011-12-09) strictly sequestered.
"""

import os
import sys
from datetime import datetime, timedelta
import warnings

# Add project root and src to path for Airflow scheduler/worker
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from retraining_pipeline import (
    task_data_validation,
    task_preprocessing,
    task_drift_monitoring,
    task_hyperparameter_tuning,
    task_model_training,
    task_model_evaluation,
    task_model_metadata,
    task_mlflow_tracking,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR
)


# DAG Default Configuration
default_args = {
    "owner": "RetailPulse-Forecasting-Team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}


# Wrapper functions for Airflow task execution and XCom integration
def run_validation_task(**kwargs):
    result = task_data_validation(data_dir=DEFAULT_DATA_DIR)
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="validation_result", value=result)
    return result


def run_preprocessing_task(**kwargs):
    result = task_preprocessing(data_dir=DEFAULT_DATA_DIR, cutoff_train="2011-11-09")
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="preprocessing_result", value=result)
    return result


def run_drift_task(**kwargs):
    result = task_drift_monitoring(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR)
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="drift_result", value=result)
    return result


def run_tuning_task(**kwargs):
    drift_result = {}
    if kwargs.get("ti"):
        drift_result = kwargs["ti"].xcom_pull(task_ids="drift_monitoring", key="drift_result") or {}
    drift_detected = drift_result.get("dataset_drift_detected", True)

    result = task_hyperparameter_tuning(
        data_dir=DEFAULT_DATA_DIR,
        drift_detected=drift_detected,
        force_tuning=False,
        n_trials=5
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="tuning_result", value=result)
    return result


def run_training_task(**kwargs):
    tuning_result = {}
    if kwargs.get("ti"):
        tuning_result = kwargs["ti"].xcom_pull(task_ids="hyperparameter_tuning", key="tuning_result") or {}

    p_params = tuning_result.get("prophet_params")
    l_params = tuning_result.get("lstm_params")

    result = task_model_training(
        data_dir=DEFAULT_DATA_DIR,
        prophet_params=p_params,
        lstm_params=l_params
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="training_result", value=result)
    return result


def run_evaluation_task(**kwargs):
    tuning_result = {}
    if kwargs.get("ti"):
        tuning_result = kwargs["ti"].xcom_pull(task_ids="hyperparameter_tuning", key="tuning_result") or {}

    p_params = tuning_result.get("prophet_params")
    l_params = tuning_result.get("lstm_params")

    result = task_model_evaluation(
        data_dir=DEFAULT_DATA_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        prophet_params=p_params,
        lstm_params=l_params,
        top_n=10
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="evaluation_result", value=result)
    return result


def run_metadata_task(**kwargs):
    eval_result = {}
    drift_result = {}
    if kwargs.get("ti"):
        eval_result = kwargs["ti"].xcom_pull(task_ids="model_evaluation", key="evaluation_result") or {}
        drift_result = kwargs["ti"].xcom_pull(task_ids="drift_monitoring", key="drift_result") or {}

    result = task_model_metadata(
        output_dir=DEFAULT_OUTPUT_DIR,
        eval_result=eval_result,
        drift_result=drift_result
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="metadata_result", value=result)
    return result


def run_mlflow_task(**kwargs):
    eval_result = {}
    drift_result = {}
    if kwargs.get("ti"):
        eval_result = kwargs["ti"].xcom_pull(task_ids="model_evaluation", key="evaluation_result") or {}
        drift_result = kwargs["ti"].xcom_pull(task_ids="drift_monitoring", key="drift_result") or {}

    result = task_mlflow_tracking(
        output_dir=DEFAULT_OUTPUT_DIR,
        eval_result=eval_result,
        drift_result=drift_result
    )
    if kwargs.get("ti"):
        kwargs["ti"].xcom_push(key="mlflow_result", value=result)
    return result


# Initialize the Airflow DAG
with DAG(
    dag_id="demand_forecasting_retraining",
    default_args=default_args,
    description="Automated weekly demand forecasting retraining pipeline with drift detection and conservative promotion",
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["retailpulse", "forecasting", "retraining", "mlops", "phase8"]
) as dag:

    # Task 1: Data Validation
    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=run_validation_task,
        provide_context=True
    )

    # Task 2: Preprocessing & Chronological Splitting
    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing_task,
        provide_context=True
    )

    # Task 3: Drift Monitoring (Evidently AI)
    drift_monitoring = PythonOperator(
        task_id="drift_monitoring",
        python_callable=run_drift_task,
        provide_context=True
    )

    # Task 4: Hyperparameter Tuning (Optuna)
    hyperparameter_tuning = PythonOperator(
        task_id="hyperparameter_tuning",
        python_callable=run_tuning_task,
        provide_context=True
    )

    # Task 5: Model Training
    model_training = PythonOperator(
        task_id="model_training",
        python_callable=run_training_task,
        provide_context=True
    )

    # Task 6: Model Evaluation & Promotion Decision
    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_evaluation_task,
        provide_context=True
    )

    # Task 7: Model Metadata Compilation
    model_metadata = PythonOperator(
        task_id="model_metadata",
        python_callable=run_metadata_task,
        provide_context=True
    )

    # Task 8: MLflow Experiment Tracking & Run Registration
    mlflow_tracking = PythonOperator(
        task_id="mlflow_tracking",
        python_callable=run_mlflow_task,
        provide_context=True
    )

    # Define Linear Chronological Workflow Dependencies
    data_validation >> preprocessing >> drift_monitoring >> hyperparameter_tuning >> model_training >> model_evaluation >> model_metadata >> mlflow_tracking
