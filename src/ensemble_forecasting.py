"""
RetailPulse — Demand Forecasting Module
Hybrid Prophet + PyTorch LSTM Ensemble Pipeline

Combines:
1. Prophet: Decomposable seasonal & trend modeling with UK bank holidays.
2. LSTM: Deep neural recurrent sequences capturing non-linear local autoregressive dynamics.
3. Multi-Model Ensembling: Inverse-error weighting and grid-search validation weighting.

Ensemble Weight Selection:
To prevent data leakage, ensemble weights are optimized on an internal
validation window (last 30 days of training history: 2011-10-11 to 2011-11-09)
without touching the 30-day out-of-sample holdout test set (2011-11-10 to 2011-12-09).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from evaluation import evaluate_forecast
from baseline_forecasting import moving_average_forecast
from forecasting import train_predict_prophet, train_predict_lightgbm
from lstm_forecasting import train_predict_lstm


def calculate_model_weights(
    val_errors: Dict[str, float],
    power: float = 1.0,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Calculate normalized ensemble weights inversely proportional to validation error:
        w_i = (1 / e_i)^p / sum_j (1 / e_j)^p

    Parameters
    ----------
    val_errors : dict
        Validation errors per model (e.g. {'Prophet': 10.0, 'LSTM': 12.0}).
    power : float
        Exponential power scaling (p=1.0 is standard inverse error).
    eps : float
        Small numerical stability term.

    Returns
    -------
    dict
        Normalized model weights summing to 1.0. Lower error yields strictly higher weight.
    """
    if not val_errors:
        return {}

    inv_errors = {}
    for model, err in val_errors.items():
        err_val = max(float(err), eps)
        inv_errors[model] = (1.0 / err_val) ** power

    total_inv = sum(inv_errors.values())
    if total_inv == 0.0:
        equal_w = 1.0 / len(val_errors)
        return {m: equal_w for m in val_errors}

    return {m: inv / total_inv for m, inv in inv_errors.items()}


def combine_forecasts(
    predictions: Dict[str, Union[np.ndarray, list]],
    weights: Dict[str, float]
) -> np.ndarray:
    """
    Combine multiple array predictions into a single ensemble array using normalized weights.
    Strictly clips combined demand at 0.0 to prevent negative retail predictions.

    Parameters
    ----------
    predictions : dict
        Dictionary of {model_name: np.ndarray}.
    weights : dict
        Dictionary of {model_name: float_weight}.

    Returns
    -------
    np.ndarray
        Combined non-negative demand predictions.
    """
    total_w = sum(weights.values())
    norm_weights = {m: w / total_w for m, w in weights.items()} if total_w > 0 else weights

    first_key = next(iter(predictions.keys()))
    length = len(predictions[first_key])
    combined = np.zeros(length, dtype=float)

    for m, preds in predictions.items():
        w = norm_weights.get(m, 0.0)
        combined += w * np.asarray(preds, dtype=float)

    return np.maximum(0.0, combined)


def optimize_ensemble_weights(
    y_val_true: np.ndarray,
    prophet_val_preds: np.ndarray,
    lstm_val_preds: np.ndarray,
    metric: str = "MAE",
    step: float = 0.05
) -> Tuple[float, float, Dict[str, float]]:
    """
    Grid search candidate ensemble weights w in [0.0, 1.0] on validation data:
        y_hat = w * y_prophet + (1 - w) * y_lstm

    Parameters
    ----------
    y_val_true : np.ndarray
        Ground truth validation demand.
    prophet_val_preds : np.ndarray
        Prophet validation predictions.
    lstm_val_preds : np.ndarray
        LSTM validation predictions.
    metric : str
        Optimization objective ('MAE', 'RMSE', or 'sMAPE').
    step : float
        Grid search step resolution.

    Returns
    -------
    best_w : float
        Optimal weight assigned to Prophet (1 - best_w assigned to LSTM).
    best_score : float
        Validation score at the optimal weight.
    search_log : dict
        Scores evaluated across candidate weights.
    """
    y_true = np.asarray(y_val_true, dtype=float)
    p_pred = np.asarray(prophet_val_preds, dtype=float)
    l_pred = np.asarray(lstm_val_preds, dtype=float)

    weights = np.round(np.arange(0.0, 1.0 + step, step), 3)
    best_w = 0.5
    best_score = float("inf")
    search_log: Dict[str, float] = {}

    for w in weights:
        combo = w * p_pred + (1.0 - w) * l_pred
        combo = np.maximum(0.0, combo)
        metrics = evaluate_forecast(y_true, combo)
        score = metrics.get(metric.upper(), metrics["MAE"])
        search_log[f"w_prophet_{w:.2f}"] = round(score, 4)

        if score < best_score:
            best_score = score
            best_w = float(w)

    return best_w, best_score, search_log


def combine_prophet_lstm_predictions(
    prophet_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
    prophet_weight: float = 0.5,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Merge and weight Prophet and LSTM forecast dataframes into an ensemble forecast.
    """
    p_sub = prophet_df.copy()
    l_sub = lstm_df.copy()

    p_sub[date_col] = pd.to_datetime(p_sub[date_col]).dt.floor("D")
    l_sub[date_col] = pd.to_datetime(l_sub[date_col]).dt.floor("D")

    merged = p_sub.merge(
        l_sub[[date_col, "Predicted_Demand"]],
        on=date_col,
        suffixes=("_prophet", "_lstm")
    )

    w = float(prophet_weight)
    ensemble_preds = w * merged["Predicted_Demand_prophet"] + (1.0 - w) * merged["Predicted_Demand_lstm"]
    ensemble_preds = np.maximum(0.0, ensemble_preds)

    out = pd.DataFrame({
        date_col: merged[date_col],
        "Predicted_Demand": ensemble_preds,
        "Model": f"Prophet+LSTM Ensemble (w={w:.2f})"
    })

    if "Lower_Bound" in merged.columns and "Upper_Bound" in merged.columns:
        out["Lower_Bound"] = np.maximum(0.0, merged["Lower_Bound"])
        out["Upper_Bound"] = np.maximum(0.0, merged["Upper_Bound"])

    return out


def train_predict_ensemble(
    history_df: pd.DataFrame,
    horizon: int = 30,
    weights: Optional[Dict[str, float]] = None,
    lookback: int = 30,
    date_col: str = "Date",
    demand_col: str = "Demand",
    prophet_params: Optional[dict] = None,
    lstm_params: Optional[dict] = None
) -> pd.DataFrame:
    """
    Train and combine component forecasting models (Prophet, LightGBM/LSTM, and Baseline).
    Produces an ensemble DataFrame compatible with both test suites and production dashboards.
    """
    if history_df.empty:
        raise ValueError("Cannot train ensemble on empty history DataFrame.")

    if weights is None:
        weights = {"Prophet": 0.5, "LSTM": 0.5}

    preds_dict = {}

    # 1. Prophet
    if "Prophet" in weights and weights["Prophet"] > 0.0:
        p_kwargs = prophet_params or {}
        p_df = train_predict_prophet(history_df, horizon=horizon, date_col=date_col, demand_col=demand_col, **p_kwargs)
        preds_dict["Prophet"] = p_df["Predicted_Demand"].values
        dates = p_df[date_col].values
        lower_bound = p_df["Lower_Bound"].values
        upper_bound = p_df["Upper_Bound"].values
    else:
        dates = pd.date_range(
            pd.to_datetime(history_df[date_col].iloc[-1]) + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )
        lower_bound = np.zeros(horizon)
        upper_bound = np.zeros(horizon)

    # 2. LSTM
    if "LSTM" in weights and weights["LSTM"] > 0.0:
        l_kwargs = {"lookback": lookback}
        if lstm_params:
            l_kwargs.update(lstm_params)
        l_df = train_predict_lstm(history_df, horizon=horizon, date_col=date_col, demand_col=demand_col, **l_kwargs)
        preds_dict["LSTM"] = l_df["Predicted_Demand"].values

    # 3. LightGBM
    if "LightGBM" in weights and weights["LightGBM"] > 0.0:
        lgb_df = train_predict_lightgbm(history_df, horizon=horizon, date_col=date_col, demand_col=demand_col)
        preds_dict["LightGBM"] = lgb_df["Predicted_Demand"].values

    # 4. Moving Average 28D Baseline
    if "Moving Average 28D" in weights and weights["Moving Average 28D"] > 0.0:
        ma_df = moving_average_forecast(history_df, window=28, horizon=horizon, date_col=date_col, demand_col=demand_col)
        preds_dict["Moving Average 28D"] = ma_df["Predicted_Demand"].values

    # Combine using combine_forecasts
    combined_arr = combine_forecasts(preds_dict, weights)

    out = pd.DataFrame({
        "Date": dates,
        "Predicted_Demand": combined_arr,
        "Model": "Ensemble"
    })

    # Add individual prediction columns if present
    if "Prophet" in preds_dict:
        out["Prophet_Prediction"] = preds_dict["Prophet"]
    if "LSTM" in preds_dict:
        out["LSTM_Prediction"] = preds_dict["LSTM"]
    if "LightGBM" in preds_dict:
        out["LightGBM_Prediction"] = preds_dict["LightGBM"]
    if "Moving Average 28D" in preds_dict:
        out["Baseline_Prediction"] = preds_dict["Moving Average 28D"]

    out["Ensemble_Prediction"] = combined_arr
    out["Lower_Bound"] = lower_bound
    out["Upper_Bound"] = upper_bound

    return out


def evaluate_product_phase5(
    train_history_df: pd.DataFrame,
    holdout_test_df: pd.DataFrame,
    stock_code: str,
    horizon: int = 30,
    val_days: int = 30,
    lookback: int = 30,
    lstm_epochs: int = 25,
    prophet_weight: Optional[float] = None,
    prophet_params: Optional[dict] = None,
    lstm_params: Optional[dict] = None
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame, float]:
    """
    Train, optimize weights, and evaluate Prophet, LSTM, and their Ensemble for one product.
    """
    train_sorted = train_history_df.sort_values("Date").copy()
    test_sorted = holdout_test_df.sort_values("Date").copy()

    p_kwargs = prophet_params or {}
    l_kwargs = {"lookback": lookback, "epochs": lstm_epochs}
    if lstm_params:
        l_kwargs.update(lstm_params)

    # Step 1: Internal validation weight tuning if weight not supplied
    if prophet_weight is None:
        if len(train_sorted) > (val_days + l_kwargs.get("lookback", lookback) + 10):
            val_cutoff = train_sorted["Date"].iloc[-val_days]
            pre_train = train_sorted[train_sorted["Date"] < val_cutoff].copy()
            val_df = train_sorted[train_sorted["Date"] >= val_cutoff].copy()

            val_prophet = train_predict_prophet(pre_train, horizon=len(val_df), **p_kwargs)
            val_lstm = train_predict_lstm(pre_train, horizon=len(val_df), **l_kwargs)

            y_val_true = val_df["Demand"].values
            opt_w, _, _ = optimize_ensemble_weights(
                y_val_true=y_val_true,
                prophet_val_preds=val_prophet["Predicted_Demand"].values,
                lstm_val_preds=val_lstm["Predicted_Demand"].values,
                metric="MAE"
            )
            selected_weight = opt_w
        else:
            selected_weight = 0.5
    else:
        selected_weight = float(prophet_weight)

    # Step 2: Fit on full training history
    p_preds = train_predict_prophet(train_sorted, horizon=horizon, **p_kwargs)
    l_preds = train_predict_lstm(train_sorted, horizon=horizon, **l_kwargs)
    e_preds = combine_prophet_lstm_predictions(p_preds, l_preds, prophet_weight=selected_weight)

    # Step 3: Evaluate on untouched holdout test ground truth
    y_test_true = test_sorted["Demand"].values

    m_prophet = evaluate_forecast(y_test_true, p_preds["Predicted_Demand"].values)
    m_lstm = evaluate_forecast(y_test_true, l_preds["Predicted_Demand"].values)
    m_ensemble = evaluate_forecast(y_test_true, e_preds["Predicted_Demand"].values)

    product_metrics = {
        "Prophet": m_prophet,
        "LSTM": m_lstm,
        "Ensemble": m_ensemble
    }

    # Format output predictions
    df_out_list = []
    for model_name, p_df in [("Prophet", p_preds), ("LSTM", l_preds), ("Ensemble", e_preds)]:
        frame = p_df.copy()
        frame["StockCode"] = stock_code
        frame["Actual_Demand"] = y_test_true
        frame["Model"] = model_name
        df_out_list.append(frame[["Date", "StockCode", "Actual_Demand", "Predicted_Demand", "Model"]])

    predictions_df = pd.concat(df_out_list, ignore_index=True)
    return product_metrics, predictions_df, selected_weight


def run_phase5_benchmark(
    daily_demand_df: pd.DataFrame,
    eligible_stock_codes: List[str],
    split_date: str = "2011-11-09",
    horizon: int = 30,
    lookback: int = 30,
    lstm_epochs: int = 25,
    fixed_weight: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run Phase 5 Prophet + LSTM Ensemble benchmark across multiple products.
    """
    subset_df = daily_demand_df[daily_demand_df["StockCode"].isin(eligible_stock_codes)].copy()
    subset_df["Date"] = pd.to_datetime(subset_df["Date"]).dt.floor("D")
    split_ts = pd.to_datetime(split_date).floor("D")

    train_data = subset_df[subset_df["Date"] <= split_ts].copy()
    test_data = subset_df[subset_df["Date"] > split_ts].copy()

    records = []
    prediction_frames = []
    weights_dict = {}

    for code in eligible_stock_codes:
        p_train = train_data[train_data["StockCode"] == code].sort_values("Date")
        p_test = test_data[test_data["StockCode"] == code].sort_values("Date")

        if len(p_train) == 0 or len(p_test) == 0:
            continue

        try:
            prod_metrics, preds_df, opt_w = evaluate_product_phase5(
                train_history_df=p_train,
                holdout_test_df=p_test,
                stock_code=code,
                horizon=horizon,
                lookback=lookback,
                lstm_epochs=lstm_epochs,
                prophet_weight=fixed_weight
            )

            weights_dict[code] = opt_w
            prediction_frames.append(preds_df)

            for m_name in ["Prophet", "LSTM", "Ensemble"]:
                m_vals = prod_metrics[m_name]
                records.append({
                    "StockCode": code,
                    "Model": m_name,
                    "MAE": m_vals["MAE"],
                    "RMSE": m_vals["RMSE"],
                    "MAPE": m_vals["MAPE"],
                    "sMAPE": m_vals["sMAPE"]
                })
        except Exception as e:
            continue

    eval_df = pd.DataFrame(records)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)

    summary_rows = []
    for m_name in ["Prophet", "LSTM", "Ensemble"]:
        sub = eval_df[eval_df["Model"] == m_name]
        if not sub.empty:
            summary_rows.append({
                "Model": m_name,
                "Product Count": sub["StockCode"].nunique(),
                "MAE": round(sub["MAE"].mean(), 4),
                "RMSE": round(sub["RMSE"].mean(), 4),
                "MAPE": round(sub["MAPE"].mean(), 2),
                "sMAPE": round(sub["sMAPE"].mean(), 2)
            })

    summary_df = pd.DataFrame(summary_rows).sort_values("MAE").reset_index(drop=True)
    return summary_df, all_predictions_df, weights_dict
