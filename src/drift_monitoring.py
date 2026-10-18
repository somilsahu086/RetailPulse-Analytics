"""
RetailPulse — Demand Forecasting Module
Phase 7: Production Drift Detection & Data Quality Monitoring with Evidently AI

Key Capabilities:
1. Reference vs Current Chronological Partitioning:
   - Explicit temporal separation preventing leakage between baseline and monitoring windows.
2. Evidently-Based Feature & Dataset Drift Detection:
   - Evaluates demand, lag features, and rolling statistics actually used by the forecasting pipeline.
   - Configurable statistical significance thresholds and dataset drift ratios.
3. Data Quality & Distribution Stability Auditing:
   - Tracks row counts, missing values, non-finite values, and statistical summary shifts.
4. Report & Artifact Generation:
   - Standalone offline HTML drift reports, structured JSON summaries, tabular CSV exports, and visual plots.
5. MLflow Monitoring Integration:
   - Records monitoring runs and drift artifacts into the local MLflow experiment tracking registry.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import evidently
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    HAS_EVIDENTLY = True
    EVIDENTLY_VERSION = getattr(evidently, "__version__", "unknown")
except ImportError:
    HAS_EVIDENTLY = False
    EVIDENTLY_VERSION = "not_installed"

from feature_engineering import create_lag_and_rolling_features


# Default monitoring features used in Phase 4-6 forecasting pipeline
CORE_MONITORED_FEATURES = [
    "Demand",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28"
]


def prepare_monitoring_datasets(
    daily_demand_df: pd.DataFrame,
    ref_start: str = "2011-01-01",
    ref_end: str = "2011-08-11",
    curr_start: str = "2011-08-12",
    curr_end: str = "2011-11-07",
    stock_codes: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    date_col: str = "Date",
    demand_col: str = "Demand",
    enforce_holdout_quarantine: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Extract and prepare chronological Reference and Current datasets with forecasting features.

    Strict Anti-Leakage Safeguard:
    - Enforces that ref_end is strictly earlier than curr_start.
    - Asserts that no dates in the Reference set overlap with or follow dates in the Current set.
    - Strict fail-safe quarantine validation occurs before loading or slicing any data.

    Parameters
    ----------
    daily_demand_df : pd.DataFrame
        Daily product demand dataset.
    ref_start : str
        Start date for reference baseline period.
    ref_end : str
        End date for reference baseline period.
    curr_start : str
        Start date for current monitoring period.
    curr_end : str
        End date for current monitoring period.
    stock_codes : Optional[List[str]]
        Subset of eligible products to monitor (default: None, uses all in dataset).
    feature_cols : Optional[List[str]]
        Specific numerical features to monitor.
    date_col : str
        Name of date column.
    demand_col : str
        Name of demand/quantity column.
    enforce_holdout_quarantine : bool
        If True (default), strictly forbids monitoring windows from extending into or
        accessing the official quarantined holdout period (2011-11-10 to 2011-12-09).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        (reference_df, current_df, available_features)
    """
    ref_start_ts = pd.to_datetime(ref_start).floor("D")
    ref_end_ts = pd.to_datetime(ref_end).floor("D")
    curr_start_ts = pd.to_datetime(curr_start).floor("D")
    curr_end_ts = pd.to_datetime(curr_end).floor("D")

    # Anti-leakage / chronological assertions
    if ref_end_ts >= curr_start_ts:
        raise ValueError(
            f"LEAKAGE DETECTED: Reference end date ({ref_end}) must strictly precede "
            f"Current start date ({curr_start})!"
        )

    if ref_start_ts > ref_end_ts:
        raise ValueError(f"Reference start date ({ref_start}) cannot be after reference end date ({ref_end}).")

    if curr_start_ts > curr_end_ts:
        raise ValueError(f"Current start date ({curr_start}) cannot be after current end date ({curr_end}).")

    # Strict Fail-Safe Holdout Quarantine Validation
    # Validates requested date boundaries before loading, slicing, or processing any data
    if enforce_holdout_quarantine:
        holdout_boundary = pd.Timestamp("2011-11-09")
        if (
            curr_end_ts > holdout_boundary
            or curr_start_ts > holdout_boundary
            or ref_end_ts > holdout_boundary
            or ref_start_ts > holdout_boundary
        ):
            raise ValueError(
                f"HOLDOUT QUARANTINE BREACH: Current monitoring window (ref: {ref_start} to {ref_end}, "
                f"curr: {curr_start} to {curr_end}) extends into the quarantined holdout period "
                f"(2011-11-10 to 2011-12-09). Drift monitoring must be performed strictly on "
                f"development data (t <= 2011-11-09)."
            )

    if daily_demand_df is None or getattr(daily_demand_df, "empty", False):
        raise ValueError("Cannot prepare monitoring datasets from empty DataFrame.")

    df = daily_demand_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.floor("D")

    if stock_codes is not None and len(stock_codes) > 0:
        df = df[df["StockCode"].isin(stock_codes)].copy()

    # Generate lag and rolling demand features per product to avoid cross-product contamination
    dfs_with_features = []
    if "StockCode" in df.columns:
        for _, group in df.groupby("StockCode"):
            grp_sorted = group.sort_values(by=date_col).copy()
            feat_grp = create_lag_and_rolling_features(grp_sorted, date_col=date_col, demand_col=demand_col)
            dfs_with_features.append(feat_grp)
        combined_df = pd.concat(dfs_with_features, ignore_index=True)
    else:
        combined_df = create_lag_and_rolling_features(df.sort_values(by=date_col), date_col=date_col, demand_col=demand_col)

    # Slice Reference and Current periods chronologically
    ref_mask = (combined_df[date_col] >= ref_start_ts) & (combined_df[date_col] <= ref_end_ts)
    curr_mask = (combined_df[date_col] >= curr_start_ts) & (combined_df[date_col] <= curr_end_ts)

    ref_df = combined_df[ref_mask].copy().reset_index(drop=True)
    curr_df = combined_df[curr_mask].copy().reset_index(drop=True)

    if ref_df.empty:
        raise ValueError(f"No records found in Reference period ({ref_start} to {ref_end}).")
    if curr_df.empty:
        raise ValueError(f"No records found in Current period ({curr_start} to {curr_end}).")

    # Anti-leakage verification
    assert ref_df[date_col].max() < curr_df[date_col].min(), (
        f"Temporal overlap detected: Reference max date {ref_df[date_col].max()} "
        f"is not strictly before Current min date {curr_df[date_col].min()}."
    )

    # Identify available numerical features to monitor
    target_candidates = feature_cols or CORE_MONITORED_FEATURES
    available_features = [c for c in target_candidates if c in combined_df.columns]

    if not available_features:
        raise ValueError("None of the requested monitoring features are present in the dataset.")

    # Clean non-finite values safely by replacing inf/-inf and filling initial warmups
    for col in available_features:
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        curr_df[col] = pd.to_numeric(curr_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ref_df, curr_df, available_features


def run_drift_monitoring(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    monitored_columns: Optional[List[str]] = None,
    drift_threshold: float = 0.05,
    drift_share: float = 0.5,
    output_dir: Optional[str] = None,
    save_html: bool = True,
    html_filename: str = "phase7_drift_report.html"
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Execute Evidently drift and data quality audit comparing Reference and Current datasets.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline historical dataset.
    current_df : pd.DataFrame
        Recent monitoring dataset.
    monitored_columns : Optional[List[str]]
        Specific numerical features to evaluate.
    drift_threshold : float
        Statistical test significance threshold (default: 0.05).
    drift_share : float
        Proportion of drifted features to declare dataset drift (default: 0.5).
    output_dir : Optional[str]
        Directory to save HTML report.
    save_html : bool
        Whether to generate the offline HTML report file.
    html_filename : str
        Filename for HTML report.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]
        (drift_summary_dict, data_quality_dict, feature_drift_df)
    """
    if not HAS_EVIDENTLY:
        raise ImportError("Evidently is not installed in the current environment.")

    cols = monitored_columns or [c for c in CORE_MONITORED_FEATURES if c in reference_df.columns and c in current_df.columns]

    ref_sub = reference_df[cols].copy()
    curr_sub = current_df[cols].copy()

    # Configure Evidently Report with configurable thresholds
    report = Report(metrics=[
        DataDriftPreset(drift_share=drift_share, stattest_threshold=drift_threshold),
        DataQualityPreset()
    ])

    report.run(reference_data=ref_sub, current_data=curr_sub)

    # Save HTML report if requested
    html_path = None
    if save_html and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, html_filename)
        report.save_html(html_path)

    # Parse JSON / dictionary metrics
    rep_dict = report.as_dict()

    drift_summary = {
        "dataset_drift_detected": False,
        "share_of_drifted_features": 0.0,
        "number_of_drifted_features": 0,
        "number_of_features": len(cols),
        "drift_threshold": float(drift_threshold),
        "drift_share_threshold": float(drift_share),
        "html_report_path": html_path,
        "features_monitored": cols
    }

    feature_drift_records = []
    data_quality_summary = {
        "reference_row_count": len(ref_sub),
        "current_row_count": len(curr_sub),
        "missing_values": {
            "reference_missing_count": int(ref_sub.isna().sum().sum()),
            "current_missing_count": int(curr_sub.isna().sum().sum()),
            "reference_missing_share": float(ref_sub.isna().mean().mean()),
            "current_missing_share": float(curr_sub.isna().mean().mean())
        },
        "non_finite_values": {
            "reference_inf_count": int(np.isinf(ref_sub.values).sum()),
            "current_inf_count": int(np.isinf(curr_sub.values).sum())
        },
        "feature_statistics": {}
    }

    for metric_entry in rep_dict.get("metrics", []):
        m_type = metric_entry.get("metric", "")
        res = metric_entry.get("result", {})

        # 1. Dataset Drift & Feature Drift Table
        if "drift_by_columns" in res:
            drift_summary["dataset_drift_detected"] = bool(res.get("dataset_drift", False))
            drift_summary["share_of_drifted_features"] = round(float(res.get("share_of_drifted_columns", 0.0)), 4)
            drift_summary["number_of_drifted_features"] = int(res.get("number_of_drifted_columns", 0))

            for col_name, c_res in res["drift_by_columns"].items():
                d_score = c_res.get("drift_score")
                if d_score is not None:
                    d_score = float(d_score)

                feature_drift_records.append({
                    "feature": col_name,
                    "drift_detected": bool(c_res.get("drift_detected", False)),
                    "drift_score": d_score,
                    "stattest_name": c_res.get("stattest_name", "K-S p_value"),
                    "threshold": float(c_res.get("stattest_threshold", drift_threshold))
                })

        # 2. Dataset Missing Values
        elif "current" in res and "reference" in res and "different_missing_values" in str(res):
            data_quality_summary["missing_values"]["reference_missing_count"] = res.get("reference", {}).get("number_of_missing_values", 0)
            data_quality_summary["missing_values"]["current_missing_count"] = res.get("current", {}).get("number_of_missing_values", 0)

    # Compute descriptive statistical shift per feature
    for col in cols:
        ref_mean = float(ref_sub[col].mean())
        curr_mean = float(curr_sub[col].mean())
        ref_std = float(ref_sub[col].std())
        curr_std = float(curr_sub[col].std())
        mean_abs_diff = abs(curr_mean - ref_mean)
        pct_mean_shift = (mean_abs_diff / (abs(ref_mean) + 1e-8)) * 100.0

        data_quality_summary["feature_statistics"][col] = {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(curr_mean, 4),
            "reference_std": round(ref_std, 4),
            "current_std": round(curr_std, 4),
            "mean_shift_percent": round(pct_mean_shift, 2)
        }

    feature_drift_df = pd.DataFrame(feature_drift_records)
    if not feature_drift_df.empty:
        feature_drift_df = feature_drift_df.sort_values(by=["drift_detected", "drift_score"], ascending=[False, True]).reset_index(drop=True)

    return drift_summary, data_quality_summary, feature_drift_df


def plot_drift_summary(
    feature_drift_df: pd.DataFrame,
    output_path: str,
    title: str = "RetailPulse Phase 7 — Feature Drift Significance (Evidently AI)"
) -> str:
    """
    Generate a visual summary plot of feature drift scores and drift flags.
    """
    if feature_drift_df.empty:
        return ""

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6))

    df_sorted = feature_drift_df.sort_values("drift_score", ascending=True).copy()

    # Colors: Red for drifted, Steelblue for non-drifted
    colors = ["#d62728" if d else "#1f77b4" for d in df_sorted["drift_detected"]]

    # Use log-scale for p-values to show very small values clearly
    scores = df_sorted["drift_score"].values
    y_pos = np.arange(len(df_sorted))

    bars = ax.barh(y_pos, scores, color=colors, height=0.65, alpha=0.88, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["feature"], fontsize=10, fontweight="bold")

    # Threshold marker and axis labeling
    threshold_val = df_sorted["threshold"].iloc[0] if "threshold" in df_sorted.columns else 0.05
    stattest_name = df_sorted["stattest_name"].iloc[0] if "stattest_name" in df_sorted.columns else ""
    is_wasserstein = "wasserstein" in str(stattest_name).lower()

    if is_wasserstein:
        thresh_label = f"Wasserstein Drift Threshold ({threshold_val})"
        xlabel = "Normalized Wasserstein Distance (Score > Threshold implies distribution drift)"
    else:
        thresh_label = f"Significance Threshold (alpha = {threshold_val})"
        xlabel = f"{stattest_name or 'p-value'} (Score <= alpha implies statistically significant drift)"

    ax.axvline(threshold_val, color="black", linestyle="--", linewidth=1.5, label=thresh_label)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path


def build_monitoring_metadata(
    reference_period: str,
    current_period: str,
    drift_summary: Dict[str, Any],
    data_quality_summary: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Construct a machine-readable governance metadata dictionary for Phase 7 monitoring.
    """
    from datetime import datetime

    metadata = {
        "metadata_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project": "RetailPulse Analytics",
        "module": "Demand Forecasting",
        "phase": "Phase 7 Drift Monitoring & Data Quality Audit",
        "evidently_version": EVIDENTLY_VERSION,
        "monitoring_scope": {
            "reference_period": reference_period,
            "current_period": current_period,
            "leakage_prevention": (
                "Reference period strictly precedes Current period. "
                "No lookahead features or future transactions are included in baseline reference distributions."
            )
        },
        "drift_summary": drift_summary,
        "data_quality_summary": data_quality_summary
    }

    if extra_metadata:
        metadata["extra_metadata"] = extra_metadata

    return metadata


def run_prediction_drift_monitoring(
    reference_preds: Union[pd.Series, np.ndarray, list],
    current_preds: Union[pd.Series, np.ndarray, list],
    drift_threshold: float = 0.05,
    prediction_col: str = "predicted_demand"
) -> Dict[str, Any]:
    """
    Evaluate prediction-distribution drift between Reference and Current forecast outputs.

    Distinguishes prediction drift from feature drift and performance degradation:
    - Prediction drift flags shifts in model output distribution (e.g. seasonal volume ramps).
    - It does NOT necessarily prove model degradation.

    Parameters
    ----------
    reference_preds : Series or array
        Historical baseline model predictions.
    current_preds : Series or array
        Recent monitoring window model predictions.
    drift_threshold : float
        P-value significance threshold (default: 0.05).
    prediction_col : str
        Name of the prediction metric.

    Returns
    -------
    Dict[str, Any]
        Prediction drift summary with KS statistic, p-value, and descriptive statistics.
    """
    ref_arr = np.asarray(reference_preds, dtype=float)
    curr_arr = np.asarray(current_preds, dtype=float)

    ref_arr = ref_arr[np.isfinite(ref_arr)]
    curr_arr = curr_arr[np.isfinite(curr_arr)]

    if len(ref_arr) == 0 or len(curr_arr) == 0:
        return {
            "prediction_drift_detected": False,
            "drift_score": 1.0,
            "p_value": 1.0,
            "ks_statistic": 0.0,
            "threshold": float(drift_threshold),
            "status": "INSUFFICIENT_DATA",
            "message": "Empty prediction arrays provided."
        }

    ks_res = stats.ks_2samp(ref_arr, curr_arr)
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    ref_mean = float(np.mean(ref_arr))
    curr_mean = float(np.mean(curr_arr))
    ref_std = float(np.std(ref_arr, ddof=1)) if len(ref_arr) > 1 else 0.0
    curr_std = float(np.std(curr_arr, ddof=1)) if len(curr_arr) > 1 else 0.0
    pct_mean_shift = (abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8)) * 100.0

    drift_detected = bool(p_val < drift_threshold)

    return {
        "prediction_metric": prediction_col,
        "prediction_drift_detected": drift_detected,
        "drift_status": "DRIFT_DETECTED" if drift_detected else "STABLE",
        "stattest_name": "Kolmogorov-Smirnov 2-sample test",
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_val, 6),
        "drift_threshold": float(drift_threshold),
        "reference_count": int(len(ref_arr)),
        "current_count": int(len(curr_arr)),
        "reference_mean": round(ref_mean, 4),
        "current_mean": round(curr_mean, 4),
        "reference_std": round(ref_std, 4),
        "current_std": round(curr_std, 4),
        "percent_mean_shift": round(pct_mean_shift, 2),
        "interpretation": (
            "Prediction drift measures shifts in the distribution of model outputs. "
            "A detected drift indicates changing forecast scale/variance (e.g. seasonal ramp), "
            "but does not alone prove model performance degradation."
        )
    }


def run_model_performance_monitoring(
    actual_demand: Union[pd.Series, np.ndarray, list],
    predicted_demand: Union[pd.Series, np.ndarray, list],
    mape_warning_threshold: float = 15.0,
    mape_critical_threshold: float = 20.0,
    bias_warning_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Evaluate actual-versus-forecast performance and bias where ground truth actuals are realized.

    Clearly distinguishes actual forecast error from input feature or prediction drift.
    """
    act_arr = np.asarray(actual_demand, dtype=float)
    pred_arr = np.asarray(predicted_demand, dtype=float)

    valid_mask = np.isfinite(act_arr) & np.isfinite(pred_arr)
    act_arr = act_arr[valid_mask]
    pred_arr = pred_arr[valid_mask]

    if len(act_arr) == 0:
        return {
            "performance_status": "INSUFFICIENT_DATA",
            "message": "Empty actual/predicted demand arrays."
        }

    act_sum = float(np.sum(act_arr))
    pred_sum = float(np.sum(pred_arr))

    if act_sum <= 0:
        raise ValueError("Sum of actual demand must be positive to calculate portfolio MAPE.")

    portfolio_mape_pct = (abs(pred_sum - act_sum) / act_sum) * 100.0
    signed_bias_pct = ((pred_sum - act_sum) / act_sum) * 100.0

    errors = pred_arr - act_arr
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    pos_mask = act_arr > 0
    pointwise_mape_pct = float(np.mean(abs_errors[pos_mask] / act_arr[pos_mask])) * 100.0 if np.any(pos_mask) else 0.0

    mad = float(np.mean(abs_errors))
    tracking_signal = float(np.sum(errors) / (mad + 1e-8))

    if portfolio_mape_pct > mape_critical_threshold:
        status = "PERFORMANCE_CRITICAL"
    elif portfolio_mape_pct > mape_warning_threshold or abs(signed_bias_pct) > bias_warning_threshold:
        status = "PERFORMANCE_DEGRADED"
    else:
        status = "PERFORMANCE_HEALTHY"

    return {
        "sample_count": int(len(act_arr)),
        "actual_total_units": round(act_sum, 2),
        "predicted_total_units": round(pred_sum, 2),
        "aggregate_volume_error_pct": round(portfolio_mape_pct, 4),
        "portfolio_mape_pct": round(portfolio_mape_pct, 4),
        "metric_scope_note": (
            "aggregate_volume_error_pct represents net volume discrepancy across the evaluated horizon. "
            "It is distinct from official mean weekly portfolio MAPE, where weekly errors do not cancel out."
        ),
        "pointwise_mape_pct": round(pointwise_mape_pct, 4),
        "signed_bias_pct": round(signed_bias_pct, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "tracking_signal": round(tracking_signal, 4),
        "performance_status": status,
        "mape_warning_threshold": float(mape_warning_threshold),
        "mape_critical_threshold": float(mape_critical_threshold),
        "bias_warning_threshold": float(bias_warning_threshold),
        "interpretation": (
            "Actual forecast error measures real predictive performance against ground truth. "
            "Note: Net aggregate volume error across an entire multi-week horizon allows positive errors in "
            "some weeks to offset negative errors in others. Acceptance criteria must be judged against the "
            "official mean weekly portfolio MAPE (14.81% on development backtests), not aggregate window volume."
        )
    }


def run_schema_and_quality_audit(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    max_missing_share: float = 0.0
) -> Dict[str, Any]:
    """
    Perform schema conformity and data quality integrity audits.
    """
    if reference_df.empty or current_df.empty:
        return {
            "schema_valid": False,
            "quality_pass": False,
            "status": "FAIL",
            "message": "Empty DataFrame passed to schema audit."
        }

    req_cols = required_columns or CORE_MONITORED_FEATURES
    ref_missing_cols = [c for c in req_cols if c not in reference_df.columns]
    curr_missing_cols = [c for c in req_cols if c not in current_df.columns]

    dtype_mismatches = {}
    for c in req_cols:
        if c in reference_df.columns and c in current_df.columns:
            if reference_df[c].dtype != current_df[c].dtype:
                dtype_mismatches[c] = (str(reference_df[c].dtype), str(current_df[c].dtype))

    common_cols = [c for c in req_cols if c in reference_df.columns and c in current_df.columns]
    ref_null_count = int(reference_df[common_cols].isna().sum().sum()) if common_cols else -1
    curr_null_count = int(current_df[common_cols].isna().sum().sum()) if common_cols else -1

    ref_null_share = float(reference_df[common_cols].isna().mean().mean()) if common_cols else 1.0
    curr_null_share = float(current_df[common_cols].isna().mean().mean()) if common_cols else 1.0

    schema_valid = (len(ref_missing_cols) == 0 and len(curr_missing_cols) == 0 and len(dtype_mismatches) == 0)
    quality_pass = (ref_null_share <= max_missing_share and curr_null_share <= max_missing_share)

    if schema_valid and quality_pass:
        status = "PASS"
    elif not schema_valid:
        status = "FAIL"
    else:
        status = "WARNING"

    return {
        "schema_valid": schema_valid,
        "quality_pass": quality_pass,
        "status": status,
        "reference_missing_columns": ref_missing_cols,
        "current_missing_columns": curr_missing_cols,
        "dtype_mismatches": dtype_mismatches,
        "reference_null_count": ref_null_count,
        "current_null_count": curr_null_count,
        "reference_null_share": round(ref_null_share, 4),
        "current_null_share": round(curr_null_share, 4),
        "max_missing_share_threshold": float(max_missing_share)
    }
