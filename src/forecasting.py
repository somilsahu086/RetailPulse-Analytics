"""
RetailPulse — Demand Forecasting Module
Advanced Forecasting Framework

Implements production-grade advanced time-series forecasting:
1. Prophet: Decomposable time-series model (Trend + UK Holidays + Weekly & Annual Seasonality)
2. LightGBM / Gradient Boosting: Non-linear regression with lag, rolling, and calendar features
3. Unified Benchmarking: Direct head-to-head comparison against Phase 3 Baselines (Naive, MA28)
"""

import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Suppress noisy Prophet / cmdstanpy logs during batch runs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

from evaluation import evaluate_forecast
from baseline_forecasting import naive_forecast, moving_average_forecast
from feature_engineering import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    build_product_feature_matrix,
    create_calendar_features,
    get_feature_column_names
)

# Optional LightGBM with scikit-learn GradientBoosting fallback
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def train_predict_prophet(
    history_df: pd.DataFrame,
    horizon: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand",
    add_uk_holidays: bool = True,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0
) -> pd.DataFrame:
    """
    Train Facebook Prophet model on historical demand and generate multi-step forecasts.

    Features & Configuration:
    - Additive/linear trend modeling.
    - Weekly seasonality explicitly capturing Saturday store closures.
    - Yearly seasonality capturing annual pre-holiday retail ramp.
    - Official UK Bank Holidays to anticipate bank holiday trading dips.
    - Tunable changepoint_prior_scale and seasonality_prior_scale.
    - Non-negative clipping on predicted demand and intervals.

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical product time series.
    horizon : int
        Number of calendar days ahead to forecast.
    date_col : str
        Date column name.
    demand_col : str
        Demand column name.
    add_uk_holidays : bool
        If True, adds UK statutory holidays.
    weekly_seasonality : bool
        Enable weekly seasonality.
    yearly_seasonality : bool
        Enable yearly seasonality.
    changepoint_prior_scale : float
        Flexibility of the automatic changepoint selection (default 0.05).
    seasonality_prior_scale : float
        Strength of the seasonality component (default 10.0).

    Returns
    -------
    pd.DataFrame
        Predictions with columns: ['Date', 'Predicted_Demand', 'Lower_Bound', 'Upper_Bound', 'Model']
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet package is not installed.")

    if history_df.empty:
        raise ValueError("Cannot train Prophet on an empty history DataFrame.")

    # Prepare standard Prophet schema: 'ds' (datestamp) and 'y' (target)
    df_p = pd.DataFrame({
        "ds": pd.to_datetime(history_df[date_col]),
        "y": history_df[demand_col].astype(float)
    }).sort_values("ds").reset_index(drop=True)

    # Initialize Prophet model
    model = Prophet(
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=0.80  # 80% confidence interval for inventory safety stock
    )

    if add_uk_holidays:
        try:
            model.add_country_holidays(country_name="UK")
        except Exception:
            pass  # Fallback gracefully if holidays library has locale issue

    model.fit(df_p)

    # Generate future horizon dataframe
    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    forecast = model.predict(future)

    # Enforce non-negativity constraint
    preds = np.maximum(0.0, forecast["yhat"].values)
    lower = np.maximum(0.0, forecast["yhat_lower"].values)
    upper = np.maximum(0.0, forecast["yhat_upper"].values)

    return pd.DataFrame({
        "Date": forecast["ds"],
        "Predicted_Demand": preds,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Model": "Prophet"
    })


def train_predict_lightgbm(
    history_df: pd.DataFrame,
    horizon: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand",
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    windows: Tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    lgb_params: Optional[dict] = None
) -> pd.DataFrame:
    """
    Train Gradient Boosted Decision Trees (LightGBM) and forecast 30 days ahead
    using recursive multi-step forecasting with strict leakage prevention.

    LEAKAGE PREVENTION:
    - Features are computed strictly on shifted past observations (shift=1).
    - During recursive multi-step forecasting, at day t+h, lag and rolling features
      are updated using earlier model predictions, NEVER future ground truth!

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical product time series.
    horizon : int
        Number of calendar days ahead to forecast.
    date_col : str
        Date column name.
    demand_col : str
        Demand column name.
    lags : tuple of int
        Lag steps.
    windows : tuple of int
        Rolling window sizes.
    lgb_params : dict, optional
        Custom LightGBM parameters.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: ['Date', 'Predicted_Demand', 'Model']
    """
    if history_df.empty:
        raise ValueError("Cannot train LightGBM on an empty history DataFrame.")

    # 1. Build training feature matrix
    feat_df = build_product_feature_matrix(
        df=history_df,
        demand_col=demand_col,
        date_col=date_col,
        lags=lags,
        windows=windows,
        drop_initial_na=True
    )

    feature_cols = get_feature_column_names(lags=lags, windows=windows)

    if len(feat_df) < 10:
        # Fallback to MA28 if history is too short for ML lags
        return moving_average_forecast(history_df, window=28, horizon=horizon, date_col=date_col, demand_col=demand_col)

    X_train = feat_df[feature_cols]
    y_train = feat_df[demand_col].values

    # 2. Fit Regressor
    if HAS_LIGHTGBM:
        default_params = {
            "n_estimators": 60,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1
        }
        if lgb_params:
            default_params.update(lgb_params)
        model = lgb.LGBMRegressor(**default_params)
    else:
        model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    model.fit(X_train, y_train)

    # 3. Recursive Multi-Step Out-of-Sample Forecasting
    history_sorted = history_df.sort_values(by=date_col).copy()
    last_date = pd.to_datetime(history_sorted[date_col].iloc[-1]).floor("D")
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Working series containing history + recursive predictions
    working_demand = list(history_sorted[demand_col].values)
    working_dates = list(pd.to_datetime(history_sorted[date_col]))

    predictions = []
    max_lag = max(lags)
    max_window = max(windows)
    trend_start_idx = len(history_sorted)

    for step_i, cur_date in enumerate(future_dates):
        # Extract features for current step
        row_feat = {}
        # Calendar
        row_feat["day_of_week"] = cur_date.dayofweek
        row_feat["day_of_month"] = cur_date.day
        row_feat["month"] = cur_date.month
        row_feat["year"] = cur_date.year
        row_feat["week_of_year"] = cur_date.isocalendar()[1]
        row_feat["day_of_year"] = cur_date.dayofyear
        row_feat["is_weekend"] = int(cur_date.dayofweek in [5, 6])
        row_feat["is_saturday"] = int(cur_date.dayofweek == 5)
        row_feat["sin_dow"] = np.sin(2 * np.pi * cur_date.dayofweek / 7.0)
        row_feat["cos_dow"] = np.cos(2 * np.pi * cur_date.dayofweek / 7.0)
        row_feat["sin_month"] = np.sin(2 * np.pi * cur_date.month / 12.0)
        row_feat["cos_month"] = np.cos(2 * np.pi * cur_date.month / 12.0)
        row_feat["trend_index"] = trend_start_idx + step_i

        # Lags from working demand (shifted by 1)
        for lag in lags:
            val = working_demand[-lag] if len(working_demand) >= lag else 0.0
            row_feat[f"lag_{lag}"] = val

        # Rolling statistics
        for w in windows:
            slice_w = working_demand[-w:] if len(working_demand) >= w else working_demand
            row_feat[f"rolling_mean_{w}"] = float(np.mean(slice_w))
            row_feat[f"rolling_std_{w}"] = float(np.std(slice_w)) if len(slice_w) > 1 else 0.0

        # Construct single-row DataFrame
        X_step = pd.DataFrame([row_feat])[feature_cols]
        pred_val = float(model.predict(X_step)[0])
        # Non-negative demand constraint
        pred_val = max(0.0, pred_val)

        predictions.append(pred_val)
        working_demand.append(pred_val)
        working_dates.append(cur_date)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Demand": predictions,
        "Model": "LightGBM" if HAS_LIGHTGBM else "GradientBoosting"
    })


def run_phase4_benchmark(
    daily_demand_df: pd.DataFrame,
    eligible_stock_codes: List[str],
    split_date: str = "2011-11-09",
    horizon: int = 30,
    include_prophet: bool = True,
    include_lightgbm: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full Phase 4 comparative benchmark across:
    - Phase 3 Baselines: Naive, MA7, MA14, MA28
    - Phase 4 Advanced Models: Prophet, LightGBM

    Parameters
    ----------
    daily_demand_df : pd.DataFrame
        Complete daily demand DataFrame (e.g. from Parquet cache).
    eligible_stock_codes : list of str
        List of product StockCodes to evaluate.
    split_date : str
        Chronological cutoff date for training.
    horizon : int
        Holdout horizon length in days.
    include_prophet : bool
        Include Prophet model in evaluation.
    include_lightgbm : bool
        Include LightGBM model in evaluation.

    Returns
    -------
    summary_df : pd.DataFrame
        Overall comparative ranking table (MAE, RMSE, MAPE, sMAPE).
    all_predictions_df : pd.DataFrame
        Detailed predictions for all models and products.
    """
    subset_df = daily_demand_df[daily_demand_df["StockCode"].isin(eligible_stock_codes)].copy()
    subset_df["Date"] = pd.to_datetime(subset_df["Date"]).dt.floor("D")
    split_ts = pd.to_datetime(split_date).floor("D")

    train_data = subset_df[subset_df["Date"] <= split_ts].copy()
    test_data = subset_df[subset_df["Date"] > split_ts].copy()

    # Define model suite
    models = [
        ("Naive (Last Value)", naive_forecast, {"strategy": "last"}),
        ("Moving Average 7D", moving_average_forecast, {"window": 7}),
        ("Moving Average 14D", moving_average_forecast, {"window": 14}),
        ("Moving Average 28D", moving_average_forecast, {"window": 28}),
    ]
    if include_lightgbm:
        models.append(("LightGBM", train_predict_lightgbm, {}))
    if include_prophet and HAS_PROPHET:
        models.append(("Prophet", train_predict_prophet, {}))

    records = []
    prediction_frames = []

    for code in eligible_stock_codes:
        p_train = train_data[train_data["StockCode"] == code].sort_values("Date")
        p_test = test_data[test_data["StockCode"] == code].sort_values("Date")

        if len(p_train) == 0 or len(p_test) == 0:
            continue

        y_true = p_test["Demand"].values

        for model_name, func, kwargs in models:
            try:
                preds_df = func(p_train, horizon=horizon, **kwargs)
                y_pred = preds_df["Predicted_Demand"].values

                metrics = evaluate_forecast(y_true, y_pred)
                records.append({
                    "StockCode": code,
                    "Model": model_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                    "sMAPE": metrics["sMAPE"]
                })

                p_frame = preds_df.copy()
                p_frame["StockCode"] = code
                p_frame["Actual_Demand"] = y_true
                prediction_frames.append(p_frame[["Date", "StockCode", "Actual_Demand", "Predicted_Demand", "Model"]])
            except Exception as e:
                # Log error and continue to next model
                continue

    eval_df = pd.DataFrame(records)
    all_preds_df = pd.concat(prediction_frames, ignore_index=True)

    # Compute macro averages
    summary_rows = []
    for model_name, _, _ in models:
        m_sub = eval_df[eval_df["Model"] == model_name]
        if not m_sub.empty:
            summary_rows.append({
                "Model": model_name,
                "Product Count": m_sub["StockCode"].nunique(),
                "MAE": round(m_sub["MAE"].mean(), 4),
                "RMSE": round(m_sub["RMSE"].mean(), 4),
                "MAPE": round(m_sub["MAPE"].mean(), 2),
                "sMAPE": round(m_sub["sMAPE"].mean(), 2)
            })

    summary_df = pd.DataFrame(summary_rows)
    # Sort by MAE ascending
    summary_df = summary_df.sort_values(by="MAE").reset_index(drop=True)
    return summary_df, all_preds_df
