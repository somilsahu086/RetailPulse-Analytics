"""
RetailPulse Analytics — Demand Forecasting & Inventory Optimization Dashboard
Phase 9: Interactive Streamlit Web Application

Integrates:
- Phase 1 EDA & Catalog Selection
- Phase 2 Preprocessing & Daily Demand Series
- Phase 3 Baseline Benchmarks (Naive, Moving Average 28D)
- Phase 4 Advanced Models (Prophet, LightGBM)
- Phase 5 Hybrid Prophet + PyTorch LSTM Production Ensemble & Inventory Planning
- Phase 6 MLflow Experiment Tracking & Optuna Hyperparameter Governance
- Phase 7 Evidently AI Data Drift & Quality Continuous Monitoring
- Phase 8 Apache Airflow Automated Weekly Retraining Pipeline Status
"""

import os
import re
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import existing modular forecasting and evaluation utilities
try:
    from evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        mean_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
        weighted_mean_absolute_percentage_error,
        evaluate_forecast
    )
    from inventory_planning import (
        calculate_safety_stock,
        calculate_reorder_point,
        calculate_order_up_to_level,
        assess_stockout_risk,
        get_z_score
    )
except ImportError:
    sys.path.append(os.path.join(PROJECT_ROOT, "..", "src"))
    from evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        mean_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
        weighted_mean_absolute_percentage_error,
        evaluate_forecast
    )
    from inventory_planning import (
        calculate_safety_stock,
        calculate_reorder_point,
        calculate_order_up_to_level,
        assess_stockout_risk,
        get_z_score
    )


# =========================================================================
# Data Loading & Caching Helpers
# =========================================================================

@st.cache_data(show_spinner=False)
def load_catalog(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load product eligibility table with descriptions and volumes."""
    path = os.path.join(project_dir, "data", "product_eligibility.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_historical_series(stock_code: str, project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load historical clean daily demand series for a given product."""
    path = os.path.join(project_dir, "data", "daily_demand_clean.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["StockCode"] = df["StockCode"].astype(str)
    p_df = df[df["StockCode"] == str(stock_code)].copy()
    if not p_df.empty:
        p_df["Date"] = pd.to_datetime(p_df["Date"]).dt.floor("D")
        p_df = p_df.sort_values("Date").reset_index(drop=True)
    return p_df


@st.cache_data(show_spinner=False)
def load_all_predictions(project_dir: str = PROJECT_ROOT) -> Dict[str, pd.DataFrame]:
    """Load precomputed 30-day holdout predictions across all models."""
    preds_dict = {}
    base_dir = os.path.join(project_dir, "outputs", "forecasting")

    # 1. Phase 5 Ensemble (Production)
    ens_path = os.path.join(base_dir, "phase5", "ensemble_predictions.csv")
    if os.path.exists(ens_path):
        df_ens = pd.read_csv(ens_path)
        df_ens["StockCode"] = df_ens["StockCode"].astype(str)
        df_ens["Date"] = pd.to_datetime(df_ens["Date"]).dt.floor("D")
        preds_dict["Prophet + PyTorch LSTM Ensemble (Production)"] = df_ens

    # 2. Phase 5 Prophet Component
    prophet_path = os.path.join(base_dir, "phase5", "prophet_predictions.csv")
    if os.path.exists(prophet_path):
        df_p = pd.read_csv(prophet_path)
        df_p["StockCode"] = df_p["StockCode"].astype(str)
        df_p["Date"] = pd.to_datetime(df_p["Date"]).dt.floor("D")
        preds_dict["Prophet (Decomposable Trend + Seasonality)"] = df_p

    # 3. Phase 5 PyTorch LSTM Component
    lstm_path = os.path.join(base_dir, "phase5", "lstm_predictions.csv")
    if os.path.exists(lstm_path):
        df_l = pd.read_csv(lstm_path)
        df_l["StockCode"] = df_l["StockCode"].astype(str)
        df_l["Date"] = pd.to_datetime(df_l["Date"]).dt.floor("D")
        preds_dict["PyTorch LSTM (Deep Neural Sequence)"] = df_l

    # 4. Phase 4 LightGBM
    lgb_path = os.path.join(base_dir, "phase4", "lightgbm_predictions.csv")
    if os.path.exists(lgb_path):
        df_lgb = pd.read_csv(lgb_path)
        df_lgb["StockCode"] = df_lgb["StockCode"].astype(str)
        df_lgb["Date"] = pd.to_datetime(df_lgb["Date"]).dt.floor("D")
        preds_dict["LightGBM (Gradient Boosted Lags)"] = df_lgb

    # 5. Baseline MA28
    base_path = os.path.join(base_dir, "baseline_predictions_top50.csv")
    if os.path.exists(base_path):
        df_b = pd.read_csv(base_path)
        df_b["StockCode"] = df_b["StockCode"].astype(str)
        df_b["Date"] = pd.to_datetime(df_b["Date"]).dt.floor("D")
        ma28 = df_b[df_b["Model"] == "Moving Average (28D)"].copy()
        if not ma28.empty:
            preds_dict["Moving Average 28D (Baseline Benchmark)"] = ma28

    return preds_dict


@st.cache_data(show_spinner=False)
def load_inventory_catalog_recommendations(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load Phase 5 inventory planning recommendations table."""
    path = os.path.join(project_dir, "outputs", "forecasting", "phase5", "inventory_recommendations.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_stockout_risk_table(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load Phase 5 stockout risk classification table."""
    path = os.path.join(project_dir, "outputs", "forecasting", "phase5", "stockout_risk.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison_table(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load Phase 6 tuned model benchmark comparison table."""
    path = os.path.join(project_dir, "outputs", "forecasting", "phase6", "tuned_model_comparison.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_drift_monitoring_artifacts(project_dir: str = PROJECT_ROOT) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load Phase 7 Evidently AI drift summary and feature table."""
    base_dir = os.path.join(project_dir, "outputs", "forecasting", "phase7")
    summary_path = os.path.join(base_dir, "drift_summary.json")
    table_path = os.path.join(base_dir, "feature_drift.csv")

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    table_df = pd.DataFrame()
    if os.path.exists(table_path):
        table_df = pd.read_csv(table_path)

    return summary, table_df


@st.cache_data(show_spinner=False)
def load_retraining_artifacts(project_dir: str = PROJECT_ROOT) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Load Phase 8 Airflow orchestration summary, promotion decision, and metrics."""
    base_dir = os.path.join(project_dir, "outputs", "forecasting", "phase8")
    dag_path = os.path.join(base_dir, "dag_run_summary.json")
    decision_path = os.path.join(base_dir, "model_promotion_decision.json")
    metrics_path = os.path.join(base_dir, "retraining_metrics.csv")

    dag_summary, promotion_decision = {}, {}
    if os.path.exists(dag_path):
        with open(dag_path, "r", encoding="utf-8") as f:
            dag_summary = json.load(f)
    if os.path.exists(decision_path):
        with open(decision_path, "r", encoding="utf-8") as f:
            promotion_decision = json.load(f)

    metrics_df = pd.DataFrame()
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)

    return dag_summary, promotion_decision, metrics_df


# =========================================================================
# Streamlit contribution — Business Overview (feature branch:
# =========================================================================
# Business Overview Data Loading Helpers (Optimized for instant rendering)
# =========================================================================
@st.cache_data(show_spinner=False)
def load_business_summary(project_dir: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Load pre-aggregated sales overview metrics for instantaneous tab rendering."""
    summary_path = os.path.join(project_dir, "data", "business_overview_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data(show_spinner=False)
def load_business_transactions(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load cleaned retail transactions for the business overview tab (read-only)."""
    # 1. First priority: fast parquet storage (< 0.1s load time)
    parquet_path = os.path.join(project_dir, "data", "cleaned_data.parquet")
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(
                parquet_path,
                columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "Revenue"]
            )
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            return df.dropna(subset=["InvoiceDate"])
        except Exception:
            pass

    # 2. Fallback: Excel file (reads and concatenates all sheets to preserve all records)
    excel_path = os.path.join(project_dir, "data", "cleaned_data.xlsx")
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(
                excel_path,
                sheet_name=None,
                usecols=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "Revenue"],
            )
            if isinstance(df, dict):
                df = pd.concat(df.values(), ignore_index=True)
            if not df.empty:
                df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
                df = df.dropna(subset=["InvoiceDate"])
            return df
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_customer_segments(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load precomputed RFM customer segmentation table (read-only)."""
    path = os.path.join(project_dir, "data", "customer_segments.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_churn_artifacts(project_dir: str = PROJECT_ROOT):
    """Load saved churn models (read-only): scaler, kmeans, churn classifier."""
    try:
        scaler = joblib.load(os.path.join(project_dir, "model", "scaler.pkl"))
        kmeans = joblib.load(os.path.join(project_dir, "model", "kmeans.pkl"))
        churn_model = joblib.load(os.path.join(project_dir, "model", "churn_model.pkl"))
        return scaler, kmeans, churn_model
    except Exception:
        return None, None, None


# =========================================================================
# Main UI Dashboard Renderer
# =========================================================================

def render_dashboard():
    """Render the full interactive Streamlit dashboard."""
    # Configure Streamlit page layout and title
    st.set_page_config(
        page_title="RetailPulse | Demand Forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.35rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #4B5563;
            margin-bottom: 1.0rem;
            line-height: 1.5;
        }
        .metric-card {
            background: #F9FAFB;
            border-radius: 10px;
            padding: 16px;
            border: 1px solid #E5E7EB;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .metric-title {
            font-size: 0.8rem;
            color: #6B7280;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1F2937;
            margin: 4px 0;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #4B5563;
        }
        .badge-low {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .badge-med {
            background-color: #FEF08A;
            color: #713F12;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .badge-high {
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- Sidebar Controls ----------------
    st.sidebar.title("RetailPulse Control")
    st.sidebar.caption("Demand forecasting & inventory")

    catalog_df = load_catalog()
    if catalog_df.empty:
        st.error("Missing product eligibility catalog at `data/product_eligibility.csv`.")
        return

    filter_catalog = st.sidebar.radio(
        "Catalog Scope:",
        ["Top 50 High-Velocity SKUs", "All Eligible Products (1,269 SKUs)"],
        index=0
    )

    if filter_catalog == "Top 50 High-Velocity SKUs":
        active_catalog = catalog_df[catalog_df["eligible_for_forecasting"] == True].sort_values(
            by="TotalDemand", ascending=False
        ).head(50)
    else:
        active_catalog = catalog_df[catalog_df["eligible_for_forecasting"] == True].sort_values(
            by="TotalDemand", ascending=False
        )

    sku_options = []
    sku_mapping = {}
    for _, row in active_catalog.iterrows():
        label = f"{row['StockCode']} — {str(row['Description'])[:30]} ({row['TotalDemand']:,.0f} units)"
        sku_options.append(label)
        sku_mapping[label] = str(row['StockCode'])

    selected_label = st.sidebar.selectbox("Select Product / SKU:", sku_options, index=0)
    selected_sku = sku_mapping[selected_label]
    sku_metadata = catalog_df[catalog_df["StockCode"] == selected_sku].iloc[0]

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Data & Cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    # ---------------- Header ----------------
    st.markdown('<div class="main-header">📈 RetailPulse Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Demand forecasting, inventory planning, and model monitoring in one place.'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()

    # ---------------- Shared page state (widgets live on their pages) ----------------
    MODEL_OPTIONS = [
        "Prophet + PyTorch LSTM Ensemble (Production)",
        "Prophet (Decomposable Trend + Seasonality)",
        "PyTorch LSTM (Deep Neural Sequence)",
        "LightGBM (Gradient Boosted Lags)",
        "Moving Average 28D (Baseline Benchmark)"
    ]
    _PAGE_DEFAULTS = {
        "horizon": 30,
        "forecast_model": MODEL_OPTIONS[0],
        "lead_time_days": 7,
        "service_level": 0.95,
        "review_period_days": 14,
    }
    for _k, _v in _PAGE_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v
    horizon = st.session_state["horizon"]
    selected_model_name = st.session_state["forecast_model"]
    lead_time_days = st.session_state["lead_time_days"]
    service_level = st.session_state["service_level"]
    review_period_days = st.session_state["review_period_days"]

    # ---------------- Header navigation (website-style) ----------------
    page = st.segmented_control(
        "Navigate",
        options=["Overview", "Customers", "Forecast", "Inventory", "Models", "Drift", "Retraining"],
        default="Overview",
    )

    # ---------------- Predictions & Metrics ----------------
    predictions_store = load_all_predictions()
    historical_df = load_historical_series(selected_sku)

    has_predictions = (
        selected_model_name in predictions_store and
        selected_sku in predictions_store[selected_model_name]["StockCode"].values
    )

    if has_predictions:
        model_preds_df = predictions_store[selected_model_name]
        sku_preds = model_preds_df[model_preds_df["StockCode"] == selected_sku].sort_values("Date").head(horizon).copy()
        total_forecast = float(sku_preds["Predicted_Demand"].sum())
        mean_daily_forecast = float(sku_preds["Predicted_Demand"].mean())
        std_daily_forecast = float(sku_preds["Predicted_Demand"].std()) if len(sku_preds) > 1 else 1.0
        latest_forecast_date = str(sku_preds["Date"].max().strftime("%Y-%m-%d"))
        has_actuals = "Actual_Demand" in sku_preds.columns
    else:
        total_forecast = 0.0
        mean_daily_forecast = 0.0
        std_daily_forecast = 0.0
        latest_forecast_date = "N/A"
        has_actuals = False
        sku_preds = pd.DataFrame()

    # KPI Summary Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #1E3A8A;">
            <div class="metric-title">📦 Selected Product</div>
            <div class="metric-value">{selected_sku}</div>
            <div class="metric-sub">{str(sku_metadata['Description'])[:18]}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #2563EB;">
            <div class="metric-title">📅 Forecast Horizon</div>
            <div class="metric-value">{horizon} Days</div>
            <div class="metric-sub">Thru {latest_forecast_date}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #10B981;">
            <div class="metric-title">📊 Total Demand</div>
            <div class="metric-value">{total_forecast:,.0f}</div>
            <div class="metric-sub">Projected Units</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #8B5CF6;">
            <div class="metric-title">📈 Avg Daily Demand</div>
            <div class="metric-value">{mean_daily_forecast:,.1f}</div>
            <div class="metric-sub">Units / Day (±{std_daily_forecast:.1f})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        hist_mean = float(historical_df[historical_df["Date"] <= "2011-11-09"]["Demand"].mean()) if not historical_df.empty else mean_daily_forecast
        risk_info = assess_stockout_risk(
            historical_daily_mean=hist_mean,
            forecast_daily_mean=mean_daily_forecast,
            forecast_daily_std=std_daily_forecast
        )
        risk_class = risk_info["stockout_risk"]
        badge_style = "badge-low" if risk_class == "LOW" else ("badge-med" if risk_class == "MEDIUM" else "badge-high")
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #F59E0B;">
            <div class="metric-title">⚠️ Stockout Risk</div>
            <div class="metric-value"><span class="{badge_style}">{risk_class}</span></div>
            <div class="metric-sub">Surge Factor: {risk_info['surge_factor']}x</div>
        </div>
        """, unsafe_allow_html=True)

    # Compute inventory replenishment metrics in advance so they are globally in scope
    # for both Tab 2 (Inventory) and the Executive Business Summary card.
    ss_units = calculate_safety_stock(
        daily_demand_std=std_daily_forecast,
        lead_time_days=lead_time_days,
        service_level=service_level
    )
    rop_units = calculate_reorder_point(
        daily_demand_mean=mean_daily_forecast,
        safety_stock=ss_units,
        lead_time_days=lead_time_days
    )
    order_up_to = calculate_order_up_to_level(
        daily_demand_mean=mean_daily_forecast,
        safety_stock=ss_units,
        lead_time_days=lead_time_days,
        review_period_days=review_period_days
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Pages (one visible at a time via header nav) ----------------

    # --- PAGE: Business Overview ---
    if page == "Overview":
        st.subheader("📊 Business Overview — Sales at a Glance")
        st.caption("Revenue, orders and top products from cleaned sales.")
        st.divider()
        biz_summary = load_business_summary()
        if biz_summary:
            total_revenue = float(biz_summary["total_revenue"])
            total_orders = int(biz_summary["total_orders"])
            total_customers = int(biz_summary["total_customers"])
            avg_order_value = float(biz_summary["avg_order_value"])
            monthly_rev = pd.DataFrame(biz_summary["monthly_revenue"])
            monthly_rev["Month"] = pd.to_datetime(monthly_rev["Month"])
        else:
            with st.spinner("Loading cleaned sales transactions..."):
                biz_df = load_business_transactions()
            if not biz_df.empty:
                f1, f2 = st.columns(2)
                with f1:
                    if "Country" in biz_df.columns:
                        country_list = sorted(biz_df["Country"].dropna().unique().tolist())
                        selected_countries = st.multiselect(
                            "Filter by Country:",
                            options=country_list,
                            default=country_list,
                        )
                    else:
                        selected_countries = []
                with f2:
                    min_date = pd.to_datetime(biz_df["InvoiceDate"]).min().date()
                    max_date = pd.to_datetime(biz_df["InvoiceDate"]).max().date()
                    date_range = st.date_input(
                        "Filter by Date Range:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )
                filtered = biz_df.copy()
                if "Country" in filtered.columns:
                    if selected_countries:
                        filtered = filtered[filtered["Country"].isin(selected_countries)]
                    else:
                        filtered = filtered.iloc[0:0]
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    start_d = pd.to_datetime(date_range[0])
                    end_d = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    filtered = filtered[(filtered["InvoiceDate"] >= start_d) & (filtered["InvoiceDate"] <= end_d)]
                if filtered.empty:
                    total_revenue = None
                    monthly_rev = pd.DataFrame()
                else:
                    total_revenue = float(filtered["Revenue"].sum())
                    total_orders = int(filtered["Invoice"].nunique())
                    total_customers = int(filtered["Customer ID"].nunique())
                    avg_order_value = total_revenue / total_orders if total_orders else 0.0
                    monthly = filtered.copy()
                    monthly["Month"] = monthly["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
                    monthly_rev = monthly.groupby("Month", as_index=False)["Revenue"].sum().sort_values("Month")
                    st.caption(f"Showing {len(filtered):,} transactions across {total_orders:,} orders.")
            else:
                total_revenue = None
                monthly_rev = pd.DataFrame()

        if total_revenue is None:
            st.warning(
                "⚠️ No transactions match the current filters (or no sales dataset at `data/cleaned_data.parquet` or `data/cleaned_data.xlsx`). "
                "Widen the date range or select at least one country."
            )
        else:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #2563EB;">
                    <div class="metric-title">💰 Total Revenue</div>
                    <div class="metric-value">£{total_revenue:,.0f}</div>
                    <div class="metric-sub">Cleaned sales data</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #10B981;">
                    <div class="metric-title">🧾 Total Orders</div>
                    <div class="metric-value">{total_orders:,}</div>
                    <div class="metric-sub">Unique invoices</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #8B5CF6;">
                    <div class="metric-title">👥 Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-sub">Unique customer IDs</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #F59E0B;">
                    <div class="metric-title">🧮 Avg Order Value</div>
                    <div class="metric-value">£{avg_order_value:,.2f}</div>
                    <div class="metric-sub">Revenue / Orders</div>
                </div>
                """, unsafe_allow_html=True)

            rev_fig = px.line(
                monthly_rev,
                x="Month",
                y="Revenue",
                title="Monthly Revenue Trend",
                markers=True,
            )
            rev_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), template="plotly_white")
            st.plotly_chart(rev_fig, width="stretch")

            if not biz_summary and "filtered" in locals() and not filtered.empty and "Country" in filtered.columns:
                top_countries = (
                    filtered.groupby("Country", as_index=False)["Revenue"]
                    .sum()
                    .sort_values("Revenue", ascending=False)
                    .head(10)
                )
                country_fig = px.bar(
                    top_countries,
                    x="Country",
                    y="Revenue",
                    title="Top 10 Countries by Revenue (Filtered)",
                    color="Revenue",
                    color_continuous_scale="Blues",
                )
                country_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), xaxis_tickangle=-20, template="plotly_white")
                st.plotly_chart(country_fig, width="stretch")
                csv_monthly = monthly_rev.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Filtered Monthly Revenue CSV",
                    data=csv_monthly,
                    file_name="business_monthly_revenue_filtered.csv",
                    mime="text/csv",
                )

            if not biz_summary and "filtered" in locals() and not filtered.empty and "Description" in filtered.columns:
                top_products = (
                    filtered.groupby(["StockCode", "Description"], as_index=False)
                    .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"), Orders=("Invoice", "nunique"))
                    .sort_values("Revenue", ascending=False)
                    .head(10)
                )
                top_products["Product"] = (
                    top_products["StockCode"].astype(str)
                    + " — "
                    + top_products["Description"].astype(str).str.slice(0, 30)
                )
                prod_fig = px.bar(
                    top_products,
                    x="Product",
                    y="Revenue",
                    title="Top 10 Products by Revenue (Filtered)",
                    color="Revenue",
                    color_continuous_scale="Greens",
                )
                prod_fig.update_layout(height=340, margin=dict(l=40, r=40, t=30, b=40), xaxis_tickangle=-25, template="plotly_white")
                st.plotly_chart(prod_fig, width="stretch")
                prod_table = top_products[["Product", "Revenue", "Quantity", "Orders"]].copy()
                prod_table["Revenue"] = prod_table["Revenue"].round(2)
                st.dataframe(prod_table, width="stretch")
                csv_products = top_products.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Top Products CSV",
                    data=csv_products,
                    file_name="business_top_products_filtered.csv",
                    mime="text/csv",
                )

    # --- PAGE: Customers ---
    if page == "Customers":
        st.subheader("👥 Customers — Segmentation at a Glance")
        st.caption("Clusters, churn rate and per-customer search.")
        st.divider()
        cust_df = load_customer_segments()
        if cust_df.empty:
            st.warning(
                "⚠️ No customer segments found at `data/customer_segments.csv`. "
                "Ask your teammate to run the segmentation step."
            )
        else:
            total_cust = len(cust_df)
            churn_rate = float(cust_df["Churn"].mean() * 100.0) if "Churn" in cust_df.columns else 0.0
            avg_recency = float(cust_df["Recency"].mean()) if "Recency" in cust_df.columns else 0.0
            n_clusters = int(cust_df["Cluster"].nunique()) if "Cluster" in cust_df.columns else 0
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #8B5CF6;">
                    <div class="metric-title">👥 Total Customers</div>
                    <div class="metric-value">{total_cust:,}</div>
                    <div class="metric-sub">Segmented profiles</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #EF4444;">
                    <div class="metric-title">🚨 Churn Rate</div>
                    <div class="metric-value">{churn_rate:.1f}%</div>
                    <div class="metric-sub">Recency &gt; 90 days</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #2563EB;">
                    <div class="metric-title">🕒 Avg Recency</div>
                    <div class="metric-value">{avg_recency:.0f} days</div>
                    <div class="metric-sub">Since last purchase</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #10B981;">
                    <div class="metric-title">🧩 Clusters</div>
                    <div class="metric-value">{n_clusters}</div>
                    <div class="metric-sub">KMeans segments</div>
                </div>
                """, unsafe_allow_html=True)
            if "Cluster" in cust_df.columns:
                cluster_counts = (
                    cust_df.groupby("Cluster").size().reset_index(name="Customers").sort_values("Cluster")
                )
                cl_fig = px.bar(
                    cluster_counts,
                    x="Cluster",
                    y="Customers",
                    title="Customers per Cluster",
                    color="Customers",
                    color_continuous_scale="Purples",
                )
                cl_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), template="plotly_white")
                st.plotly_chart(cl_fig, width="stretch")
            st.markdown("---")
            st.markdown("##### 🔮 Predict Churn for New Customer")
            st.caption("Uses saved `model/scaler.pkl` + `model/kmeans.pkl` to compute Cluster, then `model/churn_model.pkl` to predict.")
            churn_scaler, churn_kmeans, churn_model = load_churn_artifacts()
            if churn_scaler is None or churn_kmeans is None or churn_model is None:
                st.warning("⚠️ Churn models not found in `model/` (need scaler.pkl, kmeans.pkl, churn_model.pkl).")
            else:
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    in_recency = st.number_input("Recency (days):", min_value=0, value=30, step=1)
                with p2:
                    in_freq = st.number_input("Frequency (orders):", min_value=1, value=3, step=1)
                with p3:
                    in_mon = st.number_input("Monetary (£):", min_value=0.0, value=450.0, step=10.0)
                with p4:
                    in_avg = st.number_input("Avg Order Value (£):", min_value=0.0, value=150.0, step=5.0)
                if st.button("Predict Churn", width="stretch", type="primary"):
                    try:
                        scaler_in = pd.DataFrame(
                            [[float(in_recency), float(in_freq), float(in_mon)]],
                            columns=["Recency", "Frequency", "Monetary"],
                        )
                        cluster = int(churn_kmeans.predict(churn_scaler.transform(scaler_in))[0])
                        churn_in = pd.DataFrame(
                            [[float(in_freq), float(in_mon), float(in_avg), cluster]],
                            columns=["Frequency", "Monetary", "Avg_Order_Value", "Cluster"],
                        )
                        pred = int(churn_model.predict(churn_in)[0])
                        proba = float(churn_model.predict_proba(churn_in)[0, 1])
                        if pred == 1:
                            st.error(f"🚨 Predicted: CHURN (Cluster {cluster}, {proba*100:.1f}% risk)")
                        else:
                            st.success(f"✅ Predicted: RETAINED (Cluster {cluster}, {proba*100:.1f}% churn risk)")
                    except Exception as e:
                        st.warning(f"⚠️ Prediction failed: {e}")
            search = st.text_input("Search by CustomerID:", value="")
            cust_view = cust_df.copy()
            if search.strip():
                cust_view = cust_view[cust_view["CustomerID"].astype(str).str.contains(search.strip())]
            if cust_view.empty:
                st.warning("⚠️ No customers match the current search.")
            else:
                st.caption(f"Showing {min(len(cust_view), 500):,} of {len(cust_view):,} matching customers.")
                st.dataframe(cust_view.head(500), width="stretch")
                csv_cust = cust_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Customers CSV",
                    data=csv_cust,
                    file_name="customer_segments_filtered.csv",
                    mime="text/csv",
                )

    # --- PAGE: Forecast ---
    if page == "Forecast":
        st.subheader(f"🔮 30-Day Demand Trajectory — {selected_sku}: {sku_metadata['Description']}")
        st.caption("Holdout forecast with 80% confidence interval.")
        st.divider()
        fc1, fc2 = st.columns(2)
        with fc1:
            st.slider("Forecast Horizon (Days):", min_value=7, max_value=30, step=7, key="horizon")
        with fc2:
            st.selectbox("Forecasting Architecture:", MODEL_OPTIONS, key="forecast_model")
        horizon = st.session_state["horizon"]
        selected_model_name = st.session_state["forecast_model"]

        if not sku_preds.empty and not historical_df.empty:
            cutoff_date = pd.Timestamp("2011-11-09")
            hist_display = historical_df[(historical_df["Date"] <= cutoff_date) & (historical_df["Date"] >= cutoff_date - pd.Timedelta(days=60))].copy()

            z_interval = 1.2816  # 80% CI
            upper_bound = sku_preds["Predicted_Demand"] + z_interval * max(std_daily_forecast, 2.0)
            lower_bound = np.maximum(0.0, sku_preds["Predicted_Demand"] - z_interval * max(std_daily_forecast, 2.0))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_display["Date"],
                y=hist_display["Demand"],
                mode="lines+markers",
                name="Observed Demand (History)",
                line=dict(color="#4B5563", width=2),
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=sku_preds["Date"],
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper Bound (80% CI)"
            ))
            fig.add_trace(go.Scatter(
                x=sku_preds["Date"],
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.15)",
                name="80% Confidence Interval",
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=sku_preds["Date"],
                y=sku_preds["Predicted_Demand"],
                mode="lines+markers",
                name=f"Forecast: {selected_model_name[:24]}",
                line=dict(color="#2563EB", width=3),
                marker=dict(size=6)
            ))
            if has_actuals and "Actual_Demand" in sku_preds.columns:
                fig.add_trace(go.Scatter(
                    x=sku_preds["Date"],
                    y=sku_preds["Actual_Demand"],
                    mode="lines+markers",
                    name="Holdout Ground Truth (Actual)",
                    line=dict(color="#10B981", width=2, dash="dot"),
                    marker=dict(size=5, symbol="diamond")
                ))

            fig.update_layout(
                height=420,
                hovermode="x unified",
                margin=dict(l=40, r=40, t=30, b=40),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Timeline (Date)", showgrid=True, gridcolor="#F3F4F6"),
                yaxis=dict(title="Sales Demand (Units)", showgrid=True, gridcolor="#F3F4F6", rangemode="nonnegative")
            )
            st.plotly_chart(fig, width="stretch")

            # Weekly Aggregation Summary
            sku_preds["Calendar_Week"] = sku_preds["Date"].dt.isocalendar().week
            _weekly_agg = {
                "Start_Date": ("Date", "min"),
                "End_Date": ("Date", "max"),
                "Days_Count": ("Date", "count"),
                "Weekly_Predicted_Demand": ("Predicted_Demand", "sum"),
            }
            if has_actuals and "Actual_Demand" in sku_preds.columns:
                _weekly_agg["Weekly_Actual_Demand"] = ("Actual_Demand", "sum")
            weekly_summary = sku_preds.groupby("Calendar_Week").agg(**_weekly_agg).reset_index()

            col_w1, col_w2 = st.columns([1, 1])
            with col_w1:
                st.markdown("##### 📅 Weekly Aggregated Operational Demand")
                display_cols = ["Calendar_Week", "Start_Date", "End_Date", "Weekly_Predicted_Demand"]
                if has_actuals and "Weekly_Actual_Demand" in weekly_summary.columns:
                    display_cols.append("Weekly_Actual_Demand")
                _fmt = {
                    "Weekly_Predicted_Demand": "{:,.1f}",
                    "Start_Date": lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else d,
                    "End_Date": lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else d,
                }
                if "Weekly_Actual_Demand" in display_cols:
                    _fmt["Weekly_Actual_Demand"] = "{:,.1f}"
                st.dataframe(
                    weekly_summary[display_cols].style.format(_fmt),
                    width="stretch"
                )
            with col_w2:
                st.markdown("##### 📥 Daily Forecast Export")
                csv_export = sku_preds[["Date", "StockCode", "Predicted_Demand"] + (["Actual_Demand"] if has_actuals else [])].to_csv(index=False).encode('utf-8')
                _safe_sku = re.sub(r"[^A-Za-z0-9_-]+", "_", str(selected_sku))
                _safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", str(selected_model_name[:8]))
                st.download_button(
                    label="⬇️ Download Daily Predictions CSV",
                    data=csv_export,
                    file_name=f"demand_forecast_{_safe_sku}_{_safe_model}.csv",
                    mime="text/csv",
                    width="stretch"
                )
                st.info(
                    f"**Forecasting Metadata:**\n"
                    f"- Pre-training Cutoff: `2011-11-09`\n"
                    f"- Holdout Evaluation Window: `2011-11-10` to `{latest_forecast_date}`\n"
                    f"- Zero-Demand Handling: Non-negative clipping strictly applied."
                )
        else:
            st.warning(
                f"⚠️ No precomputed forecast outputs or historical series available for SKU `{selected_sku}` under `{selected_model_name}`. "
                f"Please ensure Phase 2 data (`data/daily_demand_clean.parquet`) and Phase 5 prediction artifacts (`outputs/forecasting/phase5/ensemble_predictions.csv`) exist."
            )

    # --- PAGE: Inventory ---
    if page == "Inventory":
        st.subheader(f"📦 Inventory Replenishment & Stockout Control — {selected_sku}")
        st.caption("Safety stock, reorder point and order-up-to level.")
        st.divider()
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.slider("Replenishment Lead Time (Days):", min_value=1, max_value=30, key="lead_time_days")
        with ic2:
            st.selectbox(
                "Target Service Level:",
                [0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
                key="service_level",
                format_func=lambda x: f"{int(x*100)}% (Z={get_z_score(x):.2f})",
            )
        with ic3:
            st.slider("Review Cycle (Days):", min_value=1, max_value=30, key="review_period_days")
        # Uses single inventory calc from above (ss_units/rop_units/order_up_to_level).
        # Widget keys update session_state; values refresh on next rerun via top calc.
        lead_time_days = st.session_state["lead_time_days"]
        service_level = st.session_state["service_level"]
        review_period_days = st.session_state["review_period_days"]

        inv_c1, inv_c2, inv_c3, inv_c4 = st.columns(4)
        with inv_c1:
            st.metric("Safety Stock Buffer (SS)", f"{ss_units:,} units", f"Target: {int(service_level*100)}% Service")
        with inv_c2:
            st.metric("Reorder Point (ROP)", f"{rop_units:,} units", f"{lead_time_days}-Day Lead Time")
        with inv_c3:
            st.metric("Order-Up-To Level (S)", f"{order_up_to:,} units", f"{review_period_days}-Day Review Cycle")
        with inv_c4:
            st.metric("Stockout Risk Rating", risk_class, f"Surge: {risk_info['surge_factor']}x")

        st.markdown("---")
        st.markdown("#### 🧪 Interactive Warehouse Inventory Simulator")

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            current_stock = st.slider(
                "Current On-Hand Warehouse Inventory (Units):",
                min_value=0,
                max_value=max(int(order_up_to * 1.5), 100),
                value=int(rop_units * 0.85),
                step=10
            )

            if current_stock < rop_units:
                suggested_po = max(0, order_up_to - current_stock)
                st.error(
                    f"🚨 **REORDER REQUIRED IMMEDIATELY**\n\n"
                    f"Current stock ({current_stock:,} units) has breached the Reorder Point ({rop_units:,} units).\n\n"
                    f"**Recommended Purchase Order Quantity:** `{suggested_po:,} units` (to restore inventory to Order-Up-To level {order_up_to:,})."
                )
            else:
                buffer_above_rop = current_stock - rop_units
                st.success(
                    f"✅ **INVENTORY HEALTHY**\n\n"
                    f"Current stock ({current_stock:,} units) is safely above the Reorder Point ({rop_units:,} units).\n\n"
                    f"**Buffer above trigger:** `{buffer_above_rop:,} units`. Maintain normal review schedule."
                )

        with sim_col2:
            risk_df = load_stockout_risk_table()
            if not risk_df.empty:
                risk_counts = risk_df["Stockout_Risk"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Level", "Product_Count"]
                pie_fig = px.pie(
                    risk_counts,
                    names="Risk_Level",
                    values="Product_Count",
                    title="Catalog-Wide Stockout Risk Distribution (Top 50 SKUs)",
                    color="Risk_Level",
                    color_discrete_map={"LOW": "#10B981", "MEDIUM": "#FBBF24", "HIGH": "#EF4444"}
                )
                pie_fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), template="plotly_white")
                st.plotly_chart(pie_fig, width="stretch")
            else:
                st.info("ℹ️ Catalog-wide stockout risk assessment table not found at `outputs/forecasting/phase5/stockout_risk.csv`. Run Phase 5 inventory evaluation to populate risk ratings.")

    # --- PAGE: Models ---
    if page == "Models":
        st.subheader("🏆 Model Benchmarking & Holdout Evaluation")
        st.caption("Compare Prophet, LSTM, LightGBM and baselines on holdout MAE.")
        st.divider()

        comparison_df = load_model_comparison_table()
        if not comparison_df.empty:
            st.markdown("##### 🏆 Empirical Model Comparison (Top 50 Representative Products)")
            st.dataframe(comparison_df.style.highlight_min(subset=["Holdout MAE", "Holdout RMSE"], color="#DEF7EC"), width="stretch")

            bar_fig = px.bar(
                comparison_df,
                x="Model",
                y="Holdout MAE",
                title="Model Accuracy Comparison (Holdout MAE — Lower is Better)",
                color="Holdout MAE",
                color_continuous_scale="Blues_r"
            )
            bar_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), xaxis_tickangle=-15, template="plotly_white")
            st.plotly_chart(bar_fig, width="stretch")
        else:
            st.warning("⚠️ Model comparison benchmark table not found at `outputs/forecasting/phase6/tuned_model_comparison.csv`. Run Phase 6 evaluation to populate empirical model comparison results.")

        st.info(
            r"""
            ℹ️ **Operational Accuracy Reconciliation & The Low-Denominator MAPE Singularity:**
            - **Why is Daily SKU MAPE High (>300%)?**
              In retail daily transactions, demand is highly intermittent ($20.1\%$ zero days, and $25^{\text{th}}$ percentile is only **2 units**).
              When actual demand is $y_t = 2$ and the model predicts $\hat{y}_t = 30$, the absolute percentage error is $\\frac{|2 - 30|}{2} = \\mathbf{1,400\\%}$.
              Standard unweighted MAPE averages these large ratios equally across days, distorting business reality.
            - **Why WMAPE is the True Operational Metric:**
              **Volume-Weighted MAPE (WMAPE)** computes $\\frac{\\sum |y_t - \\hat{y}_t|}{\\sum y_t} = \\mathbf{94.04\\%}$, properly weighting high-volume revenue days.
            - **Weekly Aggregation:**
              Aggregating daily forecasts to weekly replenishment cycles reduces percentage errors dramatically, enabling reliable procurement planning.
            """
        )

    # --- PAGE: Drift ---
    if page == "Drift":
        st.subheader("🛰️ Evidently AI — Continuous Data & Feature Drift Monitoring")
        st.caption("Dataset stability and per-feature KS tests.")
        st.divider()

        drift_summary, feature_drift_df = load_drift_monitoring_artifacts()

        if drift_summary:
            drift_detected = drift_summary.get("dataset_drift_detected", False)
            share_drifted = drift_summary.get("share_of_drifted_features", 0.0)
            n_drifted = drift_summary.get("number_of_drifted_features", 0)
            total_feats = drift_summary.get("total_features_monitored", 12)

            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                status_text = "🚨 SIGNIFICANT DRIFT" if drift_detected else "✅ STABLE"
                st.metric("Dataset Drift Status", status_text, f"{n_drifted}/{total_feats} Features Drifted")
            with d_col2:
                st.metric("Drift Share Ratio", f"{share_drifted*100:.1f}%", f"Threshold: {drift_summary.get('drift_share_threshold', 0.5)*100:.0f}%")
            with d_col3:
                st.metric("Statistical Test", "Kolmogorov-Smirnov", "p-value threshold: 0.05")

            if drift_detected:
                st.warning(
                    f"⚠️ **Distribution Shift Detected:** Incoming demand and rolling feature distributions have shifted significantly "
                    f"relative to the reference baseline period ({drift_summary.get('reference_period', '2011-01-01 to 2011-10-10')}). "
                    f"This shift automatically informs Phase 8 Airflow retraining triggers."
                )

            if not feature_drift_df.empty:
                st.markdown("##### 📋 Monitored Feature Drift Breakdown")
                display_drift_df = feature_drift_df.rename(columns={
                    "feature": "Feature",
                    "drift_detected": "Drift Detected",
                    "drift_score": "p-value (KS)",
                    "stattest_name": "Statistical Test",
                    "threshold": "Threshold"
                })
                st.dataframe(
                    display_drift_df.style.map(
                        lambda v: "color: red; font-weight: bold;" if v is True else "color: green;",
                        subset=["Drift Detected"]
                    ),
                    width="stretch"
                )
            else:
                st.info("ℹ️ Feature drift breakdown table not found at `outputs/forecasting/phase7/feature_drift.csv`.")
        else:
            st.warning("⚠️ No drift monitoring summary found at `outputs/forecasting/phase7/drift_summary.json`.")

    # --- PAGE: Retraining ---
    if page == "Retraining":
        st.subheader("🔁 Apache Airflow Retraining Pipeline & Model Governance")
        st.caption("Weekly DAG status and promotion decision.")
        st.divider()

        dag_summary, promotion_decision, retraining_metrics_df = load_retraining_artifacts()

        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.metric("Airflow DAG ID", "demand_forecasting_retraining", "Schedule: @weekly")
        with r_col2:
            decision_label = promotion_decision.get("promotion_decision", "ACTIVE_PRODUCTION")
            decision_style = "PROMOTED" if "PROMOTED" in decision_label else "RETAINED"
            st.metric("Model Promotion Status", decision_style, f"Threshold: {promotion_decision.get('promotion_threshold_percent', 2.0)}% Holdout MAE")
        with r_col3:
            active_prod = promotion_decision.get("active_production_model", "Prophet + LSTM Ensemble")
            st.metric("Active Production Model", str(active_prod)[:24], "Zero-Leakage Retrained")

        st.markdown("---")
        st.markdown("##### 🔄 Airflow Workflow Execution DAG (8 Sequential Tasks)")
        st.markdown(
            """
            `1. data_validation` ➔ `2. preprocessing` ➔ `3. drift_monitoring` ➔ `4. hyperparameter_tuning` ➔  
            `5. model_training` ➔ `6. model_evaluation` ➔ `7. model_metadata` ➔ `8. mlflow_tracking`
            """
        )

        if not retraining_metrics_df.empty:
            st.markdown("##### 📈 Candidate vs. Production Model Retraining Performance")
            st.dataframe(retraining_metrics_df, width="stretch")
        else:
            st.info("ℹ️ Retraining performance metrics table not found at `outputs/forecasting/phase8/retraining_metrics.csv`.")

        st.markdown("##### 🏷️ MLOps Tracking & Governance Manifest")
        st.json({
            "Experiment Name": "RetailPulse-Demand-Forecasting",
            "MLflow Tracking URI": "outputs/forecasting/phase6/mlruns",
            "Deterministic Seed": 42,
            "Retraining Logic": "Drift-Gated Optuna Search with Conservative Holdout Promotion",
            "Software Dependencies": "Prophet 1.4.0, PyTorch 2.14.1, LightGBM 4.7.0, Evidently 0.7.23, Airflow 2.11.2"
        })

    # --- Executive Summary Card ---
    st.divider()
    st.markdown("### 📋 Executive Business Summary")
    with st.container(border=True):
        st.markdown(
            f"""
            - **Demand Projection:** SKU `{selected_sku}` ({sku_metadata['Description']}) is projected to consume **{total_forecast:,.0f} units** over the next **{horizon} days** (averaging **{mean_daily_forecast:,.1f} units/day**).
            - **Inventory Posture:** Calculated Safety Stock buffer is **{ss_units:,} units** with a replenishment Reorder Point at **{rop_units:,} units** under a **{lead_time_days}-day** lead time.
            - **Stockout Urgency:** Rated **{risk_class}** (Surge factor: {risk_info['surge_factor']}x).
            - **Model Governance:** Operating under the **Prophet + PyTorch LSTM Hybrid Ensemble**, continuously monitored via Evidently AI and automated by Apache Airflow.
            """
        )


# Only execute UI rendering when running under Streamlit runtime
if __name__ == "__main__" or (hasattr(st, "runtime") and st.runtime.exists()):
    render_dashboard()
