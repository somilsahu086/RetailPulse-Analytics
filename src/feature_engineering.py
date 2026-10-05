"""
RetailPulse — Demand Forecasting Module
Time-Series Feature Engineering Framework

Provides modular, time-series safe feature engineering:
- Lag features (t-1, t-7, t-14, t-21, t-28)
- Rolling window means (7D, 14D, 28D)
- Rolling window standard deviations (7D, 14D, 28D)
- Calendar & seasonal features (DayOfWeek, Month, DayOfMonth, WeekOfYear, DayOfYear)
- Cyclical trigonometric encodings (sin/cos of DayOfWeek and Month)
- Retail business flags (is_weekend, is_saturday)
- Trend index

CRITICAL SAFEGUARD:
All rolling and lag statistics are computed on shifted series (shift=1)
to strictly prevent lookahead data leakage into the forecast target.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Default standard lags and rolling windows
DEFAULT_LAGS: Tuple[int, ...] = (1, 7, 14, 21, 28)
DEFAULT_ROLLING_WINDOWS: Tuple[int, ...] = (7, 14, 28)


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Extract calendar, cyclical, and retail operating features from a Date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added calendar feature columns.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])

    # Core calendar fields
    out["day_of_week"] = dates.dt.dayofweek.astype(int)          # 0=Monday, 6=Sunday
    out["day_of_month"] = dates.dt.day.astype(int)
    out["month"] = dates.dt.month.astype(int)
    out["year"] = dates.dt.year.astype(int)
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["day_of_year"] = dates.dt.dayofyear.astype(int)

    # Retail operating indicators (Saturday closure pattern)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_saturday"] = (out["day_of_week"] == 5).astype(int)

    # Cyclical trigonometric encodings for continuous periodicity
    # Day of week cycle (period = 7)
    out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    # Month cycle (period = 12)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    demand_col: str = "Demand",
    date_col: str = "Date",
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    windows: Tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Generate lag and rolling window features for a single product time series.

    LEAKAGE PREVENTION:
    All features are derived from demand shifted by 1 day:
        lag_k[t] = y[t - k]
        rolling_mean_W[t] = mean(y[t-1], y[t-2], ..., y[t-W])
    No information from time t or any future time is accessible at step t.

    Parameters
    ----------
    df : pd.DataFrame
        Product daily time series sorted chronologically.
    demand_col : str
        Name of the target demand column.
    date_col : str
        Name of the date column.
    lags : tuple of int
        Lag steps to extract (e.g. 1, 7, 14, 21, 28).
    windows : tuple of int
        Rolling window sizes for moving statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with lag and rolling feature columns.
    """
    out = df.sort_values(by=date_col).copy()
    y = out[demand_col].astype(float)

    # Shifted series (t-1)
    y_shifted = y.shift(1)

    # Lag features
    for lag in lags:
        out[f"lag_{lag}"] = y.shift(lag)

    # Rolling window mean and standard deviation
    for window in windows:
        # Compute on y_shifted so current day t is strictly excluded
        out[f"rolling_mean_{window}"] = y_shifted.rolling(window=window, min_periods=1).mean()
        out[f"rolling_std_{window}"] = y_shifted.rolling(window=window, min_periods=1).std().fillna(0.0)

    # Trend feature: linear day counter
    out["trend_index"] = np.arange(len(out), dtype=int)

    return out


def build_product_feature_matrix(
    df: pd.DataFrame,
    demand_col: str = "Demand",
    date_col: str = "Date",
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    windows: Tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    drop_initial_na: bool = True
) -> pd.DataFrame:
    """
    Construct a complete feature matrix combining calendar, lag, and rolling features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily product demand DataFrame.
    demand_col : str
        Target column name.
    date_col : str
        Date column name.
    lags : tuple of int
        Lag steps.
    windows : tuple of int
        Rolling window sizes.
    drop_initial_na : bool
        If True, drops early rows where max lag features are NaN.

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix ready for tabular machine learning models.
    """
    out = df.sort_values(by=date_col).copy()

    # 1. Add calendar features
    out = create_calendar_features(out, date_col=date_col)

    # 2. Add lag and rolling features
    out = create_lag_and_rolling_features(
        out,
        demand_col=demand_col,
        date_col=date_col,
        lags=lags,
        windows=windows
    )

    if drop_initial_na:
        max_lag = max(lags)
        # Drop rows where lag features are missing
        out = out.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)

    return out


def get_feature_column_names(
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    windows: Tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
) -> List[str]:
    """
    Return list of standard feature column names for model training.
    """
    feature_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "week_of_year",
        "day_of_year",
        "is_weekend",
        "is_saturday",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "trend_index"
    ]
    for lag in lags:
        feature_cols.append(f"lag_{lag}")
    for window in windows:
        feature_cols.append(f"rolling_mean_{window}")
        feature_cols.append(f"rolling_std_{window}")

    return feature_cols
