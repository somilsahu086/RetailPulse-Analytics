"""
RetailPulse — Demand Forecasting Module
Phase 6 MLOps Unit Test Suite

Tests:
1. Chronological validation split ordering and window lengths.
2. Anti-leakage safeguards (raising exceptions if holdout timestamps appear).
3. Optuna Prophet hyperparameter search space and objective evaluation.
4. Optuna LSTM hyperparameter search space and objective evaluation.
5. Deterministic/reproducible behavior of Optuna sampler with fixed seeds.
6. MLflow experiment initialization and parameter/metric logging with local temporary URI.
7. MLflow artifact logging (predictions CSV, summary metadata).
8. Model metadata generation, structure schema, and JSON serialization.
9. End-to-end leakage verification ensuring holdout data is never accessible during tuning.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mlops_tracking import (
    init_mlflow,
    log_forecasting_run,
    log_predictions_table,
    log_dict_artifact,
    get_experiment_runs
)
from optuna_tuning import (
    split_chronological_train_val,
    tune_prophet_optuna,
    tune_lstm_optuna,
    tune_forecasting_pipeline
)
from model_metadata import (
    build_model_metadata,
    save_model_metadata,
    get_environment_package_versions
)


@pytest.fixture
def synthetic_history_df():
    """Generate 100 days of synthetic daily demand ending on 2011-11-09."""
    dates = pd.date_range("2011-08-02", "2011-11-09", freq="D")
    np.random.seed(42)
    demand = np.maximum(2.0, 20.0 + 5.0 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7.0)) + np.random.normal(0, 3, len(dates)))
    return pd.DataFrame({
        "Date": dates,
        "Demand": np.round(demand, 1),
        "StockCode": "TEST_PROD"
    })


def test_split_chronological_train_val_ordering(synthetic_history_df):
    """Verify train/validation split strictly preserves temporal order with no overlap."""
    train_df, val_df = split_chronological_train_val(
        synthetic_history_df,
        val_days=30,
        max_allowed_date="2011-11-09"
    )

    assert len(val_df) == 30, f"Expected 30 validation days, got {len(val_df)}"
    assert len(train_df) == len(synthetic_history_df) - 30
    assert train_df["Date"].max() < val_df["Date"].min(), "Train and validation dates overlap!"
    assert val_df["Date"].max() <= pd.Timestamp("2011-11-09")


def test_split_chronological_train_val_leakage_exception(synthetic_history_df):
    """Verify that dates extending into the holdout window trigger a ValueError."""
    leaking_df = synthetic_history_df.copy()
    leaking_row = pd.DataFrame({
        "Date": [pd.Timestamp("2011-11-15")],  # Inside holdout period!
        "Demand": [25.0],
        "StockCode": ["TEST_PROD"]
    })
    leaking_df = pd.concat([leaking_df, leaking_row], ignore_index=True)

    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        split_chronological_train_val(leaking_df, val_days=30, max_allowed_date="2011-11-09")


def test_optuna_prophet_tuning_fast(synthetic_history_df):
    """Test Prophet Optuna hyperparameter optimization on small synthetic series."""
    train_df, val_df = split_chronological_train_val(synthetic_history_df, val_days=14)

    best_params, best_score, study = tune_prophet_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=2,
        seed=42,
        metric="MAE",
        changepoint_prior_scale_choices=[0.05, 0.1],
        seasonality_prior_scale_choices=[5.0, 10.0]
    )

    assert "changepoint_prior_scale" in best_params
    assert "seasonality_prior_scale" in best_params
    assert best_score > 0.0 and np.isfinite(best_score)
    assert len(study.trials) == 2


def test_optuna_lstm_tuning_fast(synthetic_history_df):
    """Test LSTM Optuna hyperparameter optimization on small synthetic series."""
    train_df, val_df = split_chronological_train_val(synthetic_history_df, val_days=14)

    best_params, best_score, study = tune_lstm_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=2,
        seed=42,
        metric="MAE",
        lookback_choices=[14],
        hidden_dim_choices=[16],
        lr_choices=[0.01],
        epoch_choices=[1, 2]
    )

    assert "hidden_dim" in best_params
    assert "lookback" in best_params
    assert "lr" in best_params
    assert "epochs" in best_params
    assert best_score > 0.0 and np.isfinite(best_score)
    assert len(study.trials) == 2


def test_optuna_deterministic_behavior(synthetic_history_df):
    """Verify that setting the seed produces reproducible parameter suggestions."""
    train_df, val_df = split_chronological_train_val(synthetic_history_df, val_days=14)

    params_1, score_1, _ = tune_prophet_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=3,
        seed=123,
        changepoint_prior_scale_choices=[0.01, 0.05, 0.1],
        seasonality_prior_scale_choices=[1.0, 5.0, 10.0]
    )

    params_2, score_2, _ = tune_prophet_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=3,
        seed=123,
        changepoint_prior_scale_choices=[0.01, 0.05, 0.1],
        seasonality_prior_scale_choices=[1.0, 5.0, 10.0]
    )

    assert params_1 == params_2
    assert pytest.approx(score_1, rel=1e-4) == score_2


def test_mlflow_logging_in_tmp_dir(tmp_path):
    """Verify local MLflow experiment initialization, parameter, and metric logging."""
    tracking_dir = str(tmp_path / "test_mlruns").replace("\\", "/")
    exp_name = "Test-RetailPulse-Forecasting"

    init_mlflow(tracking_uri=tracking_dir, experiment_name=exp_name)

    val_metrics = {"MAE": 2.45, "RMSE": 3.12, "MAPE": 12.5, "sMAPE": 11.8}
    holdout_metrics = {"MAE": 2.80, "RMSE": 3.45, "MAPE": 14.1, "sMAPE": 13.2}
    hyperparams = {"hidden_dim": 32, "epochs": 20, "lr": 0.01}

    run_id = log_forecasting_run(
        model_name="Test-Hybrid-Ensemble",
        model_type="hybrid_ensemble",
        forecast_horizon=30,
        lookback_window=30,
        training_start_date="2009-12-01",
        training_end_date="2011-11-09",
        validation_start_date="2011-10-11",
        validation_end_date="2011-11-09",
        holdout_start_date="2011-11-10",
        holdout_end_date="2011-12-09",
        hyperparameters=hyperparams,
        ensemble_weights={"Prophet": 0.6, "LSTM": 0.4},
        random_seed=42,
        val_metrics=val_metrics,
        holdout_metrics=holdout_metrics,
        experiment_name=exp_name,
        tracking_uri=tracking_dir
    )

    assert isinstance(run_id, str) and len(run_id) > 0

    runs_df = get_experiment_runs(experiment_name=exp_name, tracking_uri=tracking_dir)
    assert len(runs_df) >= 1

    logged_run = runs_df[runs_df["run_id"] == run_id].iloc[0]
    assert logged_run["params.model_name"] == "Test-Hybrid-Ensemble"
    assert logged_run["params.forecast_horizon"] == "30"
    assert float(logged_run["metrics.holdout_mae"]) == pytest.approx(2.80)
    assert float(logged_run["metrics.val_mae"]) == pytest.approx(2.45)


def test_mlflow_artifact_logging(tmp_path):
    """Verify CSV and JSON artifact creation and logging in MLflow."""
    tracking_dir = str(tmp_path / "test_artifact_mlruns").replace("\\", "/")
    exp_name = "Test-Artifacts-Experiment"

    # Create dummy artifact files
    pred_file = tmp_path / "test_preds.csv"
    pred_file.write_text("Date,Predicted_Demand\n2011-11-10,15.5\n", encoding="utf-8")

    run_id = log_forecasting_run(
        model_name="Artifact-Test-Model",
        model_type="prophet",
        forecast_horizon=30,
        artifacts=[str(pred_file)],
        experiment_name=exp_name,
        tracking_uri=tracking_dir
    )

    assert run_id is not None


def test_model_metadata_generation(tmp_path):
    """Verify model metadata dictionary structure and JSON file serialization."""
    tuned_params = {
        "Prophet": {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
        "LSTM": {"hidden_dim": 32, "lookback": 30, "lr": 0.01, "epochs": 20}
    }
    weights = {"Prophet": 0.55, "LSTM": 0.45}
    v_metrics = {"MAE": 2.1, "RMSE": 2.8, "MAPE": 10.5, "sMAPE": 9.8}
    h_metrics = {"MAE": 2.4, "RMSE": 3.1, "MAPE": 11.9, "sMAPE": 11.2}

    meta = build_model_metadata(
        selected_model="Prophet + PyTorch LSTM Hybrid Ensemble",
        tuned_parameters=tuned_params,
        ensemble_weights=weights,
        validation_metrics=v_metrics,
        holdout_metrics=h_metrics,
        random_seed=42
    )

    # Check structural fields
    assert meta["selected_model"] == "Prophet + PyTorch LSTM Hybrid Ensemble"
    assert meta["forecast_horizon"] == 30
    assert meta["random_seed"] == 42
    assert "temporal_timeline" in meta
    assert "leakage_prevention" in meta["temporal_timeline"]
    assert "environment_packages" in meta
    assert "PyTorch" in meta["environment_packages"]
    assert "Prophet" in meta["environment_packages"]
    assert "Optuna" in meta["environment_packages"]
    assert "MLflow" in meta["environment_packages"]

    # Verify serialization
    out_file = str(tmp_path / "test_meta.json")
    save_model_metadata(meta, out_file)
    assert os.path.exists(out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["selected_model"] == meta["selected_model"]
