"""
RetailPulse — Demand Forecasting Module
Evaluation Metrics Framework

Provides robust, zero-safe implementations of:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- WMAPE (Volume-Weighted Mean Absolute Percentage Error)

Zero Actual Demand Handling:
In retail time series, zero-demand days are common (store closed, intermittent purchases).
- Standard MAPE has a division-by-zero singularity when y_true == 0.
  Here, MAPE masks out or ignores zero-actual points (reporting on non-zero demand days),
  or applies an epsilon offset.
- sMAPE (Symmetric MAPE) bounds percentage errors between 0% and 200% by dividing
  by (|y_true| + |y_pred| + eps), gracefully handling zeros in y_true without infinite values.
- WMAPE (Weighted MAPE) computes total absolute errors divided by total actual demand,
  providing a business-grounded metric that does not explode on low-volume retail days.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def mean_absolute_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """
    Compute Mean Absolute Error (MAE):
        MAE = (1/n) * sum(|y_true - y_pred|)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """
    Compute Root Mean Squared Error (RMSE):
        RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-5,
    zero_strategy: str = "filter"
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Zero Demand Handling:
    1. 'filter' (Default): Computes MAPE only on observations where y_true > 0.
       This prevents division-by-zero explosion and reflects accuracy on active sales days.
    2. 'epsilon': Adds a small numerical offset epsilon to the denominator.

    Returns percentage value in [0, 100+].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        return 0.0

    if zero_strategy == "filter":
        non_zero_mask = y_t > 0
        if not np.any(non_zero_mask):
            return 0.0  # All actuals are zero
        mape_val = np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0
    elif zero_strategy == "epsilon":
        mape_val = np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + epsilon))) * 100.0
    else:
        raise ValueError(f"Unknown zero_strategy: {zero_strategy}. Use 'filter' or 'epsilon'.")

    return float(mape_val)


def symmetric_mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8
) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE):
        sMAPE = (100% / n) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )

    Zero Demand Handling:
    When both y_true == 0 and y_pred == 0, error is 0.
    sMAPE naturally avoids division by zero and is bounded in [0%, 200%].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        return 0.0

    denominator = np.abs(y_t) + np.abs(y_p) + epsilon
    numerator = 2.0 * np.abs(y_t - y_p)
    smape_val = np.mean(numerator / denominator) * 100.0
    return float(smape_val)


def weighted_mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8
) -> float:
    """
    Compute Volume-Weighted Mean Absolute Percentage Error (WMAPE):
        WMAPE = ( sum(|y_true - y_pred|) / (sum(|y_true|) + eps) ) * 100%

    Unlike unweighted MAPE, WMAPE weights errors by transaction volume,
    preventing artificial error explosions on low-volume/intermittent demand days.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        return 0.0

    total_actual = float(np.sum(np.abs(y_t)))
    total_abs_error = float(np.sum(np.abs(y_t - y_p)))

    if total_actual == 0.0:
        return 0.0 if total_abs_error == 0.0 else 100.0

    return float((total_abs_error / (total_actual + epsilon)) * 100.0)


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list]
) -> Dict[str, float]:
    """
    Compute all standard forecast evaluation metrics.

    Returns
    -------
    dict
        {'MAE': float, 'RMSE': float, 'MAPE': float, 'sMAPE': float, 'WMAPE': float}
    """
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 4),
        "sMAPE": round(symmetric_mean_absolute_percentage_error(y_true, y_pred), 4),
        "WMAPE": round(weighted_mean_absolute_percentage_error(y_true, y_pred), 4)
    }
