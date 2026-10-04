"""
RetailPulse — Demand Forecasting Module
Baseline Forecasting Models

Implements benchmark heuristic models:
- Naive Forecast (Last observed value)
- Seasonal Naive (7-day seasonal cycle)
- Moving Average Forecasts (7-day, 14-day, 28-day windows)

These baselines serve as the statistical minimum performance thresholds
that future ML/DL models (Prophet, LSTM) must outperform.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from evaluation import evaluate_forecast


def naive_forecast(
    history_df: pd.DataFrame,
    horizon: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand",
    strategy: str = "last"
) -> pd.DataFrame:
    """
    Generate naive demand forecast for a single product.

    Strategies:
    - 'last': Persistent naive. Future demand equals the last observed day's demand:
              y_hat[T + h] = y[T]
    - 'seasonal_7': Seasonal naive. Future demand equals demand from 7 days prior:
                    y_hat[T + h] = y[T + h - 7]

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical time-series DataFrame containing date and demand columns.
    horizon : int
        Number of calendar days ahead to forecast (default 30).
    date_col : str
        Name of the date column.
    demand_col : str
        Name of the demand column.
    strategy : str
        'last' or 'seasonal_7'.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns: ['Date', 'Predicted_Demand', 'Model']
    """
    if history_df.empty:
        raise ValueError("Cannot forecast on an empty history DataFrame.")

    df_sorted = history_df.sort_values(by=date_col).copy()
    last_date = pd.to_datetime(df_sorted[date_col].iloc[-1]).floor("D")
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    history_values = df_sorted[demand_col].values

    if strategy == "last":
        last_val = history_values[-1] if len(history_values) > 0 else 0.0
        preds = np.full(horizon, float(last_val))
        model_name = "Naive (Last Value)"
    elif strategy == "seasonal_7":
        if len(history_values) < 7:
            # Fallback to last if insufficient seasonal history
            preds = np.full(horizon, float(history_values[-1]))
        else:
            # Replicate the last 7 days cyclically across the horizon
            last_7 = history_values[-7:]
            repeats = int(np.ceil(horizon / 7))
            preds = np.tile(last_7, repeats)[:horizon].astype(float)
        model_name = "Seasonal Naive (7D)"
    else:
        raise ValueError(f"Unknown naive strategy: '{strategy}'. Choose 'last' or 'seasonal_7'.")

    # Enforce non-negative demand constraint
    preds = np.maximum(0.0, preds)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Demand": preds,
        "Model": model_name
    })


def moving_average_forecast(
    history_df: pd.DataFrame,
    window: int = 7,
    horizon: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand"
) -> pd.DataFrame:
    """
    Generate moving average demand forecast for a single product.

    The forecast for all horizon steps is the unweighted mean of the last
    `window` historical observations:
        y_hat[T + h] = (1 / W) * sum_{i=0}^{W-1} y[T - i]

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical time-series DataFrame containing date and demand columns.
    window : int
        Number of historical days to average (e.g. 7, 14, 28).
    horizon : int
        Number of calendar days ahead to forecast (default 30).
    date_col : str
        Name of the date column.
    demand_col : str
        Name of the demand column.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns: ['Date', 'Predicted_Demand', 'Model']
    """
    if history_df.empty:
        raise ValueError("Cannot forecast on an empty history DataFrame.")

    df_sorted = history_df.sort_values(by=date_col).copy()
    last_date = pd.to_datetime(df_sorted[date_col].iloc[-1]).floor("D")
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    history_values = df_sorted[demand_col].values
    effective_window = min(window, len(history_values))
    ma_val = float(np.mean(history_values[-effective_window:]))

    # Enforce non-negative demand constraint
    ma_val = max(0.0, ma_val)
    preds = np.full(horizon, ma_val)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Demand": preds,
        "Model": f"Moving Average ({window}D)"
    })


def evaluate_single_product_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_func: Callable,
    model_kwargs: Optional[dict] = None,
    stock_code: Optional[str] = None,
    date_col: str = "Date",
    demand_col: str = "Demand"
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run and evaluate a baseline model on a single product train/holdout split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical training data for the product.
    test_df : pd.DataFrame
        Holdout test data for the product.
    forecast_func : callable
        Forecasting function (e.g. naive_forecast or moving_average_forecast).
    model_kwargs : dict, optional
        Arguments passed to forecast_func (e.g. {'window': 7}).
    stock_code : str, optional
        Product identifier for metadata tagging.

    Returns
    -------
    metrics : dict
        Evaluation metrics: {'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'sMAPE': ...}
    pred_df : pd.DataFrame
        Merged DataFrame with ['Date', 'StockCode', 'Actual_Demand', 'Predicted_Demand', 'Model']
    """
    kwargs = model_kwargs or {}
    horizon = len(test_df)

    pred_df = forecast_func(train_df, horizon=horizon, date_col=date_col, demand_col=demand_col, **kwargs)

    # Align predictions with test ground truth
    test_sorted = test_df.sort_values(by=date_col).copy()
    y_true = test_sorted[demand_col].values
    y_pred = pred_df["Predicted_Demand"].values

    metrics = evaluate_forecast(y_true, y_pred)

    merged_df = pred_df.copy()
    merged_df["StockCode"] = stock_code if stock_code is not None else "UNKNOWN"
    merged_df["Actual_Demand"] = y_true

    return metrics, merged_df[["Date", "StockCode", "Actual_Demand", "Predicted_Demand", "Model"]]


def run_baseline_benchmark(
    daily_demand_df: pd.DataFrame,
    eligible_stock_codes: List[str],
    split_date: str = "2011-11-09",
    horizon: int = 30,
    windows: Tuple[int, ...] = (7, 14, 28)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute comprehensive benchmark experiments comparing:
    - Naive (Last Value)
    - Moving Average 7D
    - Moving Average 14D
    - Moving Average 28D

    Parameters
    ----------
    daily_demand_df : pd.DataFrame
        Cleaned daily demand dataset (e.g. loaded from daily_demand_clean.parquet).
    eligible_stock_codes : list of str
        List of product StockCodes to evaluate.
    split_date : str
        Chronological cutoff date for training.
    horizon : int
        Holdout horizon length in days.
    windows : tuple of int
        Moving average window sizes to test.

    Returns
    -------
    summary_df : pd.DataFrame
        Aggregated benchmark metrics across all evaluated products.
    all_predictions_df : pd.DataFrame
        Detailed daily predictions for every product and model.
    """
    # Filter dataset to selected products
    subset_df = daily_demand_df[daily_demand_df["StockCode"].isin(eligible_stock_codes)].copy()
    subset_df["Date"] = pd.to_datetime(subset_df["Date"]).dt.floor("D")
    split_ts = pd.to_datetime(split_date).floor("D")

    train_data = subset_df[subset_df["Date"] <= split_ts].copy()
    test_data = subset_df[subset_df["Date"] > split_ts].copy()

    # Define model configurations
    model_configs: List[Tuple[str, Callable, dict]] = [
        ("Naive (Last Value)", naive_forecast, {"strategy": "last"}),
    ]
    for w in windows:
        model_configs.append((f"Moving Average {w}D", moving_average_forecast, {"window": w}))

    # Store results
    per_product_records = []
    prediction_frames = []

    num_products = len(eligible_stock_codes)

    for code in eligible_stock_codes:
        p_train = train_data[train_data["StockCode"] == code]
        p_test = test_data[test_data["StockCode"] == code]

        if len(p_train) == 0 or len(p_test) == 0:
            continue

        for model_name, func, kwargs in model_configs:
            metrics, pred_df = evaluate_single_product_baseline(
                train_df=p_train,
                test_df=p_test,
                forecast_func=func,
                model_kwargs=kwargs,
                stock_code=code
            )
            # Record per-product metrics
            per_product_records.append({
                "StockCode": code,
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "sMAPE": metrics["sMAPE"]
            })
            prediction_frames.append(pred_df)

    prod_results_df = pd.DataFrame(per_product_records)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # Compute macro-average across all evaluated products
    summary_list = []
    for model_name in [cfg[0] for cfg in model_configs]:
        m_subset = prod_results_df[prod_results_df["Model"] == model_name]
        summary_list.append({
            "Model": model_name,
            "Product Count": m_subset["StockCode"].nunique(),
            "MAE": round(m_subset["MAE"].mean(), 4),
            "RMSE": round(m_subset["RMSE"].mean(), 4),
            "MAPE": round(m_subset["MAPE"].mean(), 2),
            "sMAPE": round(m_subset["sMAPE"].mean(), 2)
        })

    summary_df = pd.DataFrame(summary_list)
    return summary_df, all_predictions_df
