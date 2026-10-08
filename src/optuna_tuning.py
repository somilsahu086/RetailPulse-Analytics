"""
RetailPulse — Demand Forecasting Module
Optuna Hyperparameter Optimization Pipeline

Key Principles:
1. Strict Anti-Leakage:
   - Hyperparameter optimization operates EXCLUSIVELY on historical train/validation timeline.
   - The final 30-day holdout test set (2011-11-10 to 2011-12-09) is NEVER seen during tuning.
2. Chronological Validation Split:
   - Pre-training timeline: 2009-12-01 through 2011-10-10.
   - Internal validation window: 2011-10-11 through 2011-11-09 (30 days).
3. Reproducibility & Determinism:
   - Fixed random seeds passed to Optuna's TPESampler, PyTorch, and NumPy.
4. Parameter Candidates:
   - LSTM: lookback window, hidden dimension, learning rate, training epochs.
   - Prophet: changepoint_prior_scale, seasonality_prior_scale.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from evaluation import evaluate_forecast
from forecasting import train_predict_prophet
from lstm_forecasting import train_predict_lstm
from ensemble_forecasting import optimize_ensemble_weights


# Standard historical split cutoff (never cross into holdout!)
DEFAULT_MAX_TRAIN_DATE = "2011-11-09"
DEFAULT_VAL_DAYS = 30


def split_chronological_train_val(
    history_df: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    max_allowed_date: str = DEFAULT_MAX_TRAIN_DATE,
    date_col: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split historical series into pre-training and internal validation sets chronologically.

    Strict Anti-Leakage Assertion:
    - If any record has a Date strictly greater than `max_allowed_date`,
      a ValueError is raised immediately to prevent holdout contamination.

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical product demand dataset.
    val_days : int
        Duration of the internal validation window (default: 30 days).
    max_allowed_date : str
        Latest permissible date (default: '2011-11-09').
    date_col : str
        Date column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df) strictly preserving chronological ordering.
    """
    if history_df.empty:
        raise ValueError("Cannot split empty history DataFrame.")

    df_sorted = history_df.sort_values(by=date_col).copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col]).dt.floor("D")
    max_date = pd.to_datetime(max_allowed_date).floor("D")

    if (df_sorted[date_col] > max_date).any():
        leaking_dates = df_sorted[df_sorted[date_col] > max_date][date_col].tolist()
        raise ValueError(
            f"LEAKAGE DETECTED: Input data contains {len(leaking_dates)} dates "
            f"after max allowed training cutoff {max_allowed_date}! "
            f"First leaking date: {leaking_dates[0]}"
        )

    unique_dates = df_sorted[date_col].drop_duplicates().sort_values().values
    if len(unique_dates) <= val_days:
        raise ValueError(
            f"History contains only {len(unique_dates)} dates, which is <= validation days ({val_days})."
        )

    val_start_date = unique_dates[-val_days]
    train_df = df_sorted[df_sorted[date_col] < val_start_date].copy()
    val_df = df_sorted[df_sorted[date_col] >= val_start_date].copy()

    # Integrity verification
    assert train_df[date_col].max() < val_df[date_col].min(), "Chronological train/val overlap detected!"
    assert val_df[date_col].max() <= max_date, "Validation window extends past max_allowed_date!"

    return train_df, val_df


def tune_prophet_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 10,
    seed: int = 42,
    metric: str = "MAE",
    date_col: str = "Date",
    demand_col: str = "Demand",
    changepoint_prior_scale_choices: Optional[List[float]] = None,
    seasonality_prior_scale_choices: Optional[List[float]] = None
) -> Tuple[Dict[str, Any], float, Any]:
    """
    Tune Prophet changepoint_prior_scale and seasonality_prior_scale using Optuna.

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical pre-training dataset.
    val_df : pd.DataFrame
        Internal validation dataset.
    n_trials : int
        Number of Optuna trials.
    seed : int
        Random seed for TPESampler.
    metric : str
        Validation metric to minimize ('MAE', 'RMSE', 'sMAPE').
    changepoint_prior_scale_choices : Optional[List[float]]
        Candidate values for changepoint flexibility.
    seasonality_prior_scale_choices : Optional[List[float]]
        Candidate values for seasonality strength.

    Returns
    -------
    Tuple[Dict[str, Any], float, optuna.Study]
        Best hyperparameters dictionary, best validation metric score, and the Optuna Study.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed in the current environment.")

    cp_choices = changepoint_prior_scale_choices or [0.01, 0.05, 0.1, 0.5]
    sp_choices = seasonality_prior_scale_choices or [1.0, 5.0, 10.0]

    val_horizon = len(val_df)
    y_val_true = val_df[demand_col].astype(float).values

    def objective(trial: optuna.Trial) -> float:
        cp_scale = trial.suggest_categorical("changepoint_prior_scale", cp_choices)
        sp_scale = trial.suggest_categorical("seasonality_prior_scale", sp_choices)

        try:
            preds_df = train_predict_prophet(
                history_df=train_df,
                horizon=val_horizon,
                date_col=date_col,
                demand_col=demand_col,
                changepoint_prior_scale=cp_scale,
                seasonality_prior_scale=sp_scale
            )
            y_pred = preds_df["Predicted_Demand"].values
            scores = evaluate_forecast(y_val_true, y_pred)
            return float(scores.get(metric.upper(), scores["MAE"]))
        except Exception:
            return float("inf")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value, study


def tune_lstm_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 10,
    seed: int = 42,
    metric: str = "MAE",
    date_col: str = "Date",
    demand_col: str = "Demand",
    lookback_choices: Optional[List[int]] = None,
    hidden_dim_choices: Optional[List[int]] = None,
    lr_choices: Optional[List[float]] = None,
    epoch_choices: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], float, Any]:
    """
    Tune PyTorch LSTM hyperparameters (hidden_dim, lookback, lr, epochs) using Optuna.

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical pre-training dataset.
    val_df : pd.DataFrame
        Internal validation dataset.
    n_trials : int
        Number of Optuna trials.
    seed : int
        Random seed for TPESampler and neural network initialization.
    metric : str
        Validation metric to minimize ('MAE', 'RMSE', 'sMAPE').
    lookback_choices : Optional[List[int]]
        Lookback sequence window choices (default: [14, 30, 45]).
    hidden_dim_choices : Optional[List[int]]
        Hidden state dimension choices (default: [16, 32, 64]).
    lr_choices : Optional[List[float]]
        Learning rate choices (default: [0.005, 0.01, 0.02]).
    epoch_choices : Optional[List[int]]
        Training epoch choices (default: [15, 25]).

    Returns
    -------
    Tuple[Dict[str, Any], float, optuna.Study]
        Best hyperparameters dictionary, best validation metric score, and the Optuna Study.
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed in the current environment.")

    lb_choices = lookback_choices or [14, 30, 45]
    hd_choices = hidden_dim_choices or [16, 32, 64]
    lr_c = lr_choices or [0.005, 0.01, 0.02]
    ep_c = epoch_choices or [15, 25]

    val_horizon = len(val_df)
    y_val_true = val_df[demand_col].astype(float).values

    def objective(trial: optuna.Trial) -> float:
        lb = trial.suggest_categorical("lookback", lb_choices)
        hd = trial.suggest_categorical("hidden_dim", hd_choices)
        lr = trial.suggest_categorical("lr", lr_c)
        epochs = trial.suggest_categorical("epochs", ep_c)

        try:
            preds_df = train_predict_lstm(
                history_df=train_df,
                horizon=val_horizon,
                lookback=lb,
                hidden_dim=hd,
                lr=lr,
                epochs=epochs,
                seed=seed,
                date_col=date_col,
                demand_col=demand_col
            )
            y_pred = preds_df["Predicted_Demand"].values
            scores = evaluate_forecast(y_val_true, y_pred)
            return float(scores.get(metric.upper(), scores["MAE"]))
        except Exception:
            return float("inf")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value, study


def tune_forecasting_pipeline(
    history_df: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    max_allowed_date: str = DEFAULT_MAX_TRAIN_DATE,
    prophet_trials: int = 10,
    lstm_trials: int = 10,
    seed: int = 42,
    metric: str = "MAE",
    date_col: str = "Date",
    demand_col: str = "Demand"
) -> Dict[str, Any]:
    """
    Run end-to-end hyperparameter optimization for Prophet, LSTM, and Hybrid Ensemble.

    Steps:
    1. Strictly split history into pre-train and internal validation without holdout access.
    2. Optimize Prophet parameters (changepoint and seasonality prior scales).
    3. Optimize LSTM parameters (lookback, hidden_dim, learning rate, epochs).
    4. Generate validation predictions using the best tuned configurations.
    5. Optimize ensemble weights (w_prophet, 1 - w_prophet) on the validation window.
    6. Compile study results, metrics, and best parameters.

    Returns
    -------
    Dict[str, Any]
        Structured tuning dictionary with best parameters, validation scores, and studies.
    """
    # 1. Chronological Train/Val Split with Anti-Leakage Guard
    train_df, val_df = split_chronological_train_val(
        history_df=history_df,
        val_days=val_days,
        max_allowed_date=max_allowed_date,
        date_col=date_col
    )

    # 2. Tune Prophet
    best_prophet_params, best_prophet_score, prophet_study = tune_prophet_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=prophet_trials,
        seed=seed,
        metric=metric,
        date_col=date_col,
        demand_col=demand_col
    )

    # 3. Tune LSTM
    best_lstm_params, best_lstm_score, lstm_study = tune_lstm_optuna(
        train_df=train_df,
        val_df=val_df,
        n_trials=lstm_trials,
        seed=seed,
        metric=metric,
        date_col=date_col,
        demand_col=demand_col
    )

    # 4. Generate Validation Predictions with Best Tuned Parameters
    val_horizon = len(val_df)
    y_val_true = val_df[demand_col].astype(float).values

    val_prophet_df = train_predict_prophet(
        train_df,
        horizon=val_horizon,
        date_col=date_col,
        demand_col=demand_col,
        **best_prophet_params
    )
    val_lstm_df = train_predict_lstm(
        train_df,
        horizon=val_horizon,
        date_col=date_col,
        demand_col=demand_col,
        seed=seed,
        **best_lstm_params
    )

    p_val_preds = val_prophet_df["Predicted_Demand"].values
    l_val_preds = val_lstm_df["Predicted_Demand"].values

    # 5. Optimize Ensemble Weights on Validation Window
    opt_w, best_ensemble_score, search_log = optimize_ensemble_weights(
        y_val_true=y_val_true,
        prophet_val_preds=p_val_preds,
        lstm_val_preds=l_val_preds,
        metric=metric
    )

    combo_val_preds = np.maximum(0.0, opt_w * p_val_preds + (1.0 - opt_w) * l_val_preds)
    val_ensemble_metrics = evaluate_forecast(y_val_true, combo_val_preds)
    val_prophet_metrics = evaluate_forecast(y_val_true, p_val_preds)
    val_lstm_metrics = evaluate_forecast(y_val_true, l_val_preds)

    # Convert Optuna studies to summary DataFrames
    prophet_trials_df = prophet_study.trials_dataframe()
    lstm_trials_df = lstm_study.trials_dataframe()

    return {
        "best_prophet_params": best_prophet_params,
        "best_lstm_params": best_lstm_params,
        "best_ensemble_weight_prophet": opt_w,
        "best_ensemble_weight_lstm": round(1.0 - opt_w, 4),
        "validation_metrics": {
            "Prophet": val_prophet_metrics,
            "LSTM": val_lstm_metrics,
            "Ensemble": val_ensemble_metrics
        },
        "prophet_trials_df": prophet_trials_df,
        "lstm_trials_df": lstm_trials_df,
        "weight_search_log": search_log
    }
