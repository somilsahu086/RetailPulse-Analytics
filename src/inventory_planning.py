"""
RetailPulse — Demand Forecasting Module
Inventory Planning & Stockout Risk Layer

Translates 30-day forecast demand into actionable inventory control parameters:
1. Safety Stock (SS) = Z * sigma_d * sqrt(Lead_Time)
2. Reorder Point (ROP) = (mu_d * Lead_Time) + SS
3. Suggested Order-Up-To Level (S) = mu_d * (Lead_Time + Review_Period) + SS
4. Multi-factor Stockout Risk Classification (LOW, MEDIUM, HIGH)

CRITICAL MODELING ASSUMPTIONS:
- Operational Lead Time is NOT provided in the retail sales dataset.
  Therefore, Lead Time (L) is exposed as a configurable parameter (default: 7 calendar days).
- Service level Z is derived from standard normal distribution quantiles (default: 95% -> Z=1.645).
- Demand variability sigma_d is computed directly from forecast profile dispersion.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Standard Z-score mapping fallback if scipy is unavailable
Z_SCORE_TABLE: Dict[float, float] = {
    0.80: 0.8416,
    0.85: 1.0364,
    0.90: 1.2816,
    0.95: 1.6449,
    0.98: 2.0537,
    0.99: 2.3263,
    0.999: 3.0902
}


def get_z_score(service_level: float = 0.95) -> float:
    """
    Compute standard normal inverse cumulative distribution value (Z-score)
    for a given target customer service level.

    Parameters
    ----------
    service_level : float
        Target cycle service level between 0.50 and 0.999 (default 0.95).

    Returns
    -------
    float
        Z-score value.
    """
    if not (0.50 <= service_level < 1.0):
        raise ValueError(f"Service level must be in [0.50, 0.999], got {service_level}")

    if HAS_SCIPY:
        return float(norm.ppf(service_level))
    
    # Analytical fallback lookup with interpolation
    sl_round = round(service_level, 2)
    if sl_round in Z_SCORE_TABLE:
        return Z_SCORE_TABLE[sl_round]
    # Default to 95% service level
    return 1.6449


def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95
) -> int:
    """
    Calculate buffer safety stock to protect against demand volatility during replenishment lead time:
        SS = ceil( Z * sigma_d * sqrt(L) )

    Parameters
    ----------
    daily_demand_std : float
        Standard deviation of daily demand (sigma_d).
    lead_time_days : int
        Replenishment lead time in days (default 7).
    service_level : float
        Target non-stockout probability (default 0.95).

    Returns
    -------
    int
        Safety stock in units (non-negative integer).
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1.")

    z = get_z_score(service_level)
    ss_val = z * float(daily_demand_std) * np.sqrt(float(lead_time_days))
    return int(np.ceil(max(0.0, ss_val)))


def calculate_reorder_point(
    daily_demand_mean: float,
    safety_stock: int,
    lead_time_days: int = 7
) -> int:
    """
    Calculate inventory level that triggers a replenishment purchase order:
        ROP = ceil( (mu_d * Lead_Time) + SS )

    Parameters
    ----------
    daily_demand_mean : float
        Average daily demand (mu_d).
    safety_stock : int
        Calculated safety stock units.
    lead_time_days : int
        Replenishment lead time in days.

    Returns
    -------
    int
        Reorder point threshold in units.
    """
    lead_time_demand = float(daily_demand_mean) * float(lead_time_days)
    rop_val = lead_time_demand + float(safety_stock)
    return int(np.ceil(max(0.0, rop_val)))


def calculate_order_up_to_level(
    daily_demand_mean: float,
    safety_stock: int,
    lead_time_days: int = 7,
    review_period_days: int = 14
) -> int:
    """
    Calculate suggested maximum target inventory / Order-Up-To Level (S):
        S = ceil( mu_d * (Lead_Time + Review_Period) + SS )

    Parameters
    ----------
    daily_demand_mean : float
        Average daily demand (mu_d).
    safety_stock : int
        Safety stock units.
    lead_time_days : int
        Replenishment lead time in days.
    review_period_days : int
        Replenishment review cycle in days (default 14).

    Returns
    -------
    int
        Order-up-to target level in units.
    """
    cycle_demand = float(daily_demand_mean) * float(lead_time_days + review_period_days)
    s_val = cycle_demand + float(safety_stock)
    return int(np.ceil(max(0.0, s_val)))


def assess_stockout_risk(
    historical_daily_mean: float,
    forecast_daily_mean: float,
    forecast_daily_std: float
) -> Dict[str, Union[str, float]]:
    """
    Evaluate product-level stockout risk using a multi-factor volatility and surge model:
    
    1. Demand Surge Factor (SF) = mu_forecast / (mu_history + eps)
    2. Coefficient of Variation (CV) = sigma_forecast / (mu_forecast + eps)

    Risk Classification Rules:
    - HIGH: Demand Surge >= 1.30 (>= 30% increase) OR CV >= 1.0 (highly volatile/intermittent)
    - MEDIUM: (1.10 <= Surge < 1.30) OR (0.50 <= CV < 1.0)
    - LOW: Surge < 1.10 AND CV < 0.50 (stable, predictable demand pattern)

    Parameters
    ----------
    historical_daily_mean : float
        Mean daily demand during pre-forecast period.
    forecast_daily_mean : float
        Projected daily demand over forecast horizon.
    forecast_daily_std : float
        Standard deviation of daily forecast demand.

    Returns
    -------
    dict
        {'stockout_risk': 'LOW'|'MEDIUM'|'HIGH', 'surge_factor': float, 'coeff_of_variation': float}
    """
    eps = 1e-4
    h_mean = max(0.0, float(historical_daily_mean))
    f_mean = max(0.0, float(forecast_daily_mean))
    f_std = max(0.0, float(forecast_daily_std))

    surge_factor = f_mean / (h_mean + eps) if h_mean > 0 else 1.0
    cv = f_std / (f_mean + eps) if f_mean > 0 else 0.0

    if surge_factor >= 1.30 or cv >= 1.0:
        risk_level = "HIGH"
    elif (1.10 <= surge_factor < 1.30) or (0.50 <= cv < 1.0):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "stockout_risk": risk_level,
        "surge_factor": round(float(surge_factor), 3),
        "coeff_of_variation": round(float(cv), 3)
    }


def generate_inventory_recommendations(
    forecast_series_dict: Dict[str, Union[np.ndarray, List[float]]],
    history_daily_means: Optional[Dict[str, float]] = None,
    product_descriptions: Optional[Dict[str, str]] = None,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    review_period_days: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate comprehensive inventory planning recommendations and stockout risk audits
    across a catalog of products based on 30-day demand forecasts.

    Parameters
    ----------
    forecast_series_dict : dict
        Mapping of StockCode -> array of 30-day daily predictions.
    history_daily_means : dict, optional
        Mapping of StockCode -> historical average daily demand.
    product_descriptions : dict, optional
        Mapping of StockCode -> product title.
    lead_time_days : int
        Replenishment lead time parameter.
    service_level : float
        Customer cycle service level (0.95 = 95%).
    review_period_days : int
        Periodic review cycle in days.

    Returns
    -------
    inventory_df : pd.DataFrame
        Table with inventory parameters per product.
    stockout_risk_df : pd.DataFrame
        Table with risk classification and surge factors per product.
    """
    inv_records = []
    risk_records = []

    history_means = history_daily_means or {}
    descriptions = product_descriptions or {}

    for stock_code, preds in forecast_series_dict.items():
        arr = np.maximum(0.0, np.asarray(preds, dtype=float))
        horizon = len(arr)

        total_forecast = float(np.sum(arr))
        daily_mean = float(np.mean(arr)) if horizon > 0 else 0.0
        daily_std = float(np.std(arr)) if horizon > 1 else 0.0

        # Inventory formulas
        ss = calculate_safety_stock(daily_std, lead_time_days=lead_time_days, service_level=service_level)
        rop = calculate_reorder_point(daily_mean, safety_stock=ss, lead_time_days=lead_time_days)
        max_level = calculate_order_up_to_level(daily_mean, safety_stock=ss, lead_time_days=lead_time_days, review_period_days=review_period_days)

        desc = descriptions.get(stock_code, "UNKNOWN PRODUCT")
        hist_mean = history_means.get(stock_code, daily_mean)

        risk_info = assess_stockout_risk(
            historical_daily_mean=hist_mean,
            forecast_daily_mean=daily_mean,
            forecast_daily_std=daily_std
        )

        inv_records.append({
            "StockCode": stock_code,
            "Description": desc,
            "Forecast_Horizon_Days": horizon,
            "Total_Forecast_Demand": round(total_forecast, 1),
            "Daily_Demand_Mean": round(daily_mean, 2),
            "Daily_Demand_Std": round(daily_std, 2),
            "Lead_Time_Days": lead_time_days,
            "Service_Level_Pct": int(service_level * 100),
            "Safety_Stock": ss,
            "Reorder_Point": rop,
            "Suggested_Max_Inventory": max_level
        })

        risk_records.append({
            "StockCode": stock_code,
            "Description": desc,
            "Stockout_Risk": risk_info["stockout_risk"],
            "Surge_Factor": risk_info["surge_factor"],
            "Coeff_of_Variation": risk_info["coeff_of_variation"],
            "Safety_Stock": ss,
            "Reorder_Point": rop
        })

    inventory_df = pd.DataFrame(inv_records).sort_values("Total_Forecast_Demand", ascending=False).reset_index(drop=True)
    stockout_risk_df = pd.DataFrame(risk_records).sort_values("Stockout_Risk", ascending=True).reset_index(drop=True)

    return inventory_df, stockout_risk_df
