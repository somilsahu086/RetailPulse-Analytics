"""
RetailPulse — Demand Forecasting Module
MLOps Experiment Tracking & Run Management using MLflow

Features:
- Local file-based and SQLite MLflow tracking without requiring a remote server.
- Full tracking of model names, types, horizons, lookback windows, date ranges,
  hyperparameters, ensemble weights, random seeds, and tags.
- Detailed metric logging: MAE, RMSE, MAPE, sMAPE across validation and holdout windows.
- Artifact logging: predictions CSV, benchmark comparison CSV, evaluation plots, and metadata JSON.
- Run retrieval and comparison utilities for model selection and registry auditing.
"""

import os
import json
import tempfile
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Ensure file store compatibility and silence agent hint in MLflow 3.x+
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

try:
    import mlflow
    from mlflow.entities import ViewType
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


DEFAULT_EXPERIMENT_NAME = "RetailPulse-Demand-Forecasting"


def resolve_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """
    Resolve and normalize a tracking URI for local file-based or SQLite execution.
    """
    if tracking_uri is not None and str(tracking_uri).strip():
        uri_str = str(tracking_uri).strip()
        # If already formatted as file: or sqlite:, return as is
        if uri_str.startswith("file:") or uri_str.startswith("sqlite:") or uri_str.startswith("http:") or uri_str.startswith("https:"):
            return uri_str
        # Otherwise, treat as local directory path and ensure absolute path
        abs_path = os.path.abspath(uri_str).replace("\\", "/")
        return abs_path

    # Default fallback to project outputs mlruns
    default_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasting", "phase6", "mlruns")
    ).replace("\\", "/")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def init_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
) -> str:
    """
    Initialize local MLflow tracking URI and set the active experiment.

    Parameters
    ----------
    tracking_uri : Optional[str]
        Path to local tracking folder or SQLite database URI.
    experiment_name : str
        Name of the MLflow experiment (default: 'RetailPulse-Demand-Forecasting').

    Returns
    -------
    str
        Experiment ID.
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed in the current environment.")

    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_forecasting_run(
    model_name: str,
    model_type: str,
    forecast_horizon: int = 30,
    lookback_window: Optional[int] = None,
    training_start_date: Optional[str] = None,
    training_end_date: Optional[str] = None,
    validation_start_date: Optional[str] = None,
    validation_end_date: Optional[str] = None,
    holdout_start_date: Optional[str] = None,
    holdout_end_date: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    ensemble_weights: Optional[Union[Dict[str, float], float, str]] = None,
    random_seed: Optional[int] = 42,
    val_metrics: Optional[Dict[str, float]] = None,
    holdout_metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[List[str]] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None
) -> str:
    """
    Log a complete forecasting experiment run to MLflow.

    Tracks:
    - Model name, model type, forecast horizon, lookback window
    - Date ranges: training, validation, holdout
    - Hyperparameters (e.g., hidden_dim, epochs, learning_rate, changepoint_prior_scale)
    - Ensemble weights and random seed
    - Validation metrics (val_mae, val_rmse, val_mape, val_smape)
    - Holdout metrics (holdout_mae, holdout_rmse, holdout_mape, holdout_smape)
    - Artifact files (predictions CSV, benchmark comparison CSV, plots, JSON metadata)

    Parameters
    ----------
    model_name : str
        Display name of the model (e.g. 'Prophet', 'LSTM', 'Hybrid Ensemble').
    model_type : str
        Model architecture type (e.g. 'additive_decomposable', 'pytorch_lstm', 'hybrid_ensemble').
    forecast_horizon : int
        Forecast horizon in days (default: 30).
    lookback_window : Optional[int]
        Autoregressive lookback window (days).
    training_start_date : Optional[str]
        Start of training timeline (e.g. '2009-12-01').
    training_end_date : Optional[str]
        End of training timeline (e.g. '2011-11-09').
    validation_start_date : Optional[str]
        Start of internal validation window (e.g. '2011-10-11').
    validation_end_date : Optional[str]
        End of internal validation window (e.g. '2011-11-09').
    holdout_start_date : Optional[str]
        Start of holdout test window (e.g. '2011-11-10').
    holdout_end_date : Optional[str]
        End of holdout test window (e.g. '2011-12-09').
    hyperparameters : Optional[Dict[str, Any]]
        Model-specific hyperparameters.
    ensemble_weights : Optional[Union[Dict, float, str]]
        Ensemble weights representation.
    random_seed : Optional[int]
        Random seed for reproducibility.
    val_metrics : Optional[Dict[str, float]]
        Metrics evaluated on internal validation split.
    holdout_metrics : Optional[Dict[str, float]]
        Metrics evaluated on untouched holdout test set.
    artifacts : Optional[List[str]]
        Filepaths of artifacts to log to MLflow.
    run_name : Optional[str]
        Custom MLflow run name. Defaults to f"{model_name}_h{forecast_horizon}".
    tags : Optional[Dict[str, str]]
        Additional metadata tags.
    experiment_name : str
        Target experiment name.
    tracking_uri : Optional[str]
        Target tracking URI.

    Returns
    -------
    str
        MLflow run_id.
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed in the current environment.")

    init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    active_run = mlflow.active_run()
    created_new_run = False

    if active_run is None:
        effective_run_name = run_name or f"{model_name}_h{forecast_horizon}"
        run_context = mlflow.start_run(run_name=effective_run_name)
        created_new_run = True
    else:
        run_context = active_run

    try:
        # 1. Log Standard Parameters
        params_to_log: Dict[str, Any] = {
            "model_name": str(model_name),
            "model_type": str(model_type),
            "forecast_horizon": int(forecast_horizon),
            "lookback_window": int(lookback_window) if lookback_window is not None else "N/A",
            "training_start_date": str(training_start_date or "N/A"),
            "training_end_date": str(training_end_date or "N/A"),
            "validation_start_date": str(validation_start_date or "N/A"),
            "validation_end_date": str(validation_end_date or "N/A"),
            "holdout_start_date": str(holdout_start_date or "N/A"),
            "holdout_end_date": str(holdout_end_date or "N/A"),
            "random_seed": int(random_seed) if random_seed is not None else "N/A"
        }

        if ensemble_weights is not None:
            if isinstance(ensemble_weights, (dict, list)):
                params_to_log["ensemble_weights"] = json.dumps(ensemble_weights)
            else:
                params_to_log["ensemble_weights"] = str(ensemble_weights)

        # Unpack model-specific hyperparameters
        if hyperparameters:
            for hp_k, hp_v in hyperparameters.items():
                params_to_log[f"param_{hp_k}"] = hp_v

        # Log parameters in bulk
        mlflow.log_params(params_to_log)

        # 2. Log Tags
        all_tags = {
            "phase": "Phase 6 MLOps",
            "framework": "RetailPulse-DemandForecasting"
        }
        if tags:
            all_tags.update(tags)
        mlflow.set_tags(all_tags)

        # 3. Log Metrics
        metrics_to_log: Dict[str, float] = {}

        if val_metrics:
            for k, v in val_metrics.items():
                if v is not None and not np.isnan(float(v)):
                    metric_name = k if k.startswith("val_") else f"val_{k.lower()}"
                    metrics_to_log[metric_name] = float(v)

        if holdout_metrics:
            for k, v in holdout_metrics.items():
                if v is not None and not np.isnan(float(v)):
                    metric_name = k if k.startswith("holdout_") else f"holdout_{k.lower()}"
                    metrics_to_log[metric_name] = float(v)
                    # Also log canonical root names for primary dashboard metrics
                    k_upper = k.upper()
                    if k_upper in ["MAE", "RMSE", "MAPE", "SMAPE"]:
                        metrics_to_log[k_upper] = float(v)

        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)

        # 4. Log Artifacts
        if artifacts:
            for artifact_path in artifacts:
                if artifact_path and os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)

        current_run_id = mlflow.active_run().info.run_id
        return current_run_id

    finally:
        if created_new_run:
            mlflow.end_run()


def log_predictions_table(
    predictions_df: pd.DataFrame,
    artifact_filename: str = "predictions.csv"
) -> Optional[str]:
    """
    Log a predictions DataFrame as a CSV artifact within the active MLflow run.
    """
    if not HAS_MLFLOW or mlflow.active_run() is None:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = os.path.join(tmp_dir, artifact_filename)
        predictions_df.to_csv(tmp_csv, index=False)
        mlflow.log_artifact(tmp_csv)
        return artifact_filename


def log_dict_artifact(
    data: dict,
    artifact_filename: str = "summary.json"
) -> Optional[str]:
    """
    Log a python dictionary as a JSON artifact within the active MLflow run.
    """
    if not HAS_MLFLOW or mlflow.active_run() is None:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_json = os.path.join(tmp_dir, artifact_filename)
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(tmp_json)
        return artifact_filename


def get_experiment_runs(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None
) -> pd.DataFrame:
    """
    Retrieve all logged runs for the specified experiment as a cleaned pandas DataFrame.
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed in the current environment.")

    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY
    )
    return runs_df
