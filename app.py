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
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
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
# feat/streamlit-business-overview). Teammate code above untouched.
# =========================================================================
@st.cache_data(show_spinner=False)
def load_business_transactions(project_dir: str = PROJECT_ROOT) -> pd.DataFrame:
    """Load cleaned retail transactions for the business overview tab (read-only)."""
    path = os.path.join(project_dir, "data", "cleaned_data.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(
            path,
            sheet_name=None,
            usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Revenue"],
        )
        if isinstance(df, dict):
            df = pd.concat(df.values(), ignore_index=True)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    return df


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

    # Clean, simple styling
    st.markdown("""
    <style>
        .main-header { font-size: 1.8rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
        .sub-header { font-size: 0.95rem; color: #4B5563; margin-bottom: 1rem; }
        .metric-card { background: #FFFFFF; border-radius: 8px; padding: 12px; border: 1px solid #E5E7EB; text-align: center; }
        .metric-title { font-size: 0.75rem; color: #6B7280; text-transform: uppercase; font-weight: 600; }
        .metric-value { font-size: 1.4rem; font-weight: 700; color: #1F2937; margin: 4px 0; }
        .metric-sub { font-size: 0.75rem; color: #4B5563; }
        .badge-low { background-color: #DEF7EC; color: #03543F; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .badge-med { background-color: #FEF08A; color: #713F12; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .badge-high { background-color: #FDE8E8; color: #9B1C1C; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- Sidebar Navigation + Controls ----------------
    st.sidebar.title("RetailPulse")
    page = st.sidebar.radio(
        "Go to",
        ["Business Overview", "Forecast", "Inventory", "Benchmarks", "Drift", "Retraining"],
        index=0,
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("Product")

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

    horizon = st.sidebar.slider("Forecast Horizon (Days):", min_value=7, max_value=30, value=30, step=7)

    model_options = [
        "Prophet + PyTorch LSTM Ensemble (Production)",
        "Prophet (Decomposable Trend + Seasonality)",
        "PyTorch LSTM (Deep Neural Sequence)",
        "LightGBM (Gradient Boosted Lags)",
        "Moving Average 28D (Baseline Benchmark)"
    ]
    selected_model_name = st.sidebar.selectbox("Forecasting Architecture:", model_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inventory")
    lead_time_days = st.sidebar.slider("Lead Time (Days):", min_value=1, max_value=30, value=7)
    service_level = st.sidebar.selectbox(
        "Service Level:",
        [0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
        index=3,
        format_func=lambda x: f"{int(x*100)}% (Z={get_z_score(x):.2f})"
    )
    review_period_days = st.sidebar.slider("Review Cycle (Days):", min_value=1, max_value=30, value=14)

    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ---------------- Header ----------------
    st.markdown('<div class="main-header">RetailPulse Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Demand forecasting and inventory planning dashboard.</div>',
        unsafe_allow_html=True
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
        <div class="metric-card">
            <div class="metric-title">Selected Product</div>
            <div class="metric-value">{selected_sku}</div>
            <div class="metric-sub">{str(sku_metadata['Description'])[:18]}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Forecast Horizon</div>
            <div class="metric-value">{horizon} Days</div>
            <div class="metric-sub">Thru {latest_forecast_date}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Demand</div>
            <div class="metric-value">{total_forecast:,.0f}</div>
            <div class="metric-sub">Projected Units</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Daily Demand</div>
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
        <div class="metric-card">
            <div class="metric-title">Stockout Risk</div>
            <div class="metric-value"><span class="{badge_style}">{risk_class}</span></div>
            <div class="metric-sub">Surge Factor: {risk_info['surge_factor']}x</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Pages (sidebar navigation, no top tabs) ----------------
    # Streamlit contribution — sidebar page router. Page content below unchanged.
    if page == "Business Overview":
        st.subheader("Business Overview")
        biz_df = load_business_transactions()
        if biz_df.empty:
            st.warning("No business transactions found at `data/cleaned_data.xlsx`.")
        else:
            total_revenue = float(biz_df["Revenue"].sum())
            total_orders = int(biz_df["Invoice"].nunique())
            total_customers = int(biz_df["Customer ID"].nunique())
            avg_order_value = total_revenue / total_orders if total_orders else 0.0

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Revenue</div>
                    <div class="metric-value">£{total_revenue:,.0f}</div>
                    <div class="metric-sub">Cleaned sales data</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Orders</div>
                    <div class="metric-value">{total_orders:,}</div>
                    <div class="metric-sub">Unique invoices</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-sub">Unique customer IDs</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Order Value</div>
                    <div class="metric-value">£{avg_order_value:,.2f}</div>
                    <div class="metric-sub">Revenue / Orders</div>
                </div>
                """, unsafe_allow_html=True)

            monthly = biz_df.copy()
            monthly["Month"] = monthly["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
            monthly_rev = monthly.groupby("Month", as_index=False)["Revenue"].sum().sort_values("Month")
            rev_fig = px.line(
                monthly_rev,
                x="Month",
                y="Revenue",
                title="Monthly Revenue Trend",
                markers=True,
            )
            rev_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40))
            st.plotly_chart(rev_fig, use_container_width=True)

    elif page == "Forecast":
        st.subheader(f"Demand Trajectory — {selected_sku}: {sku_metadata['Description']}")

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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Timeline (Date)", showgrid=True, gridcolor="#F3F4F6"),
                yaxis=dict(title="Sales Demand (Units)", showgrid=True, gridcolor="#F3F4F6", rangemode="nonnegative")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Weekly Aggregation Summary
            sku_preds["Calendar_Week"] = sku_preds["Date"].dt.isocalendar().week
            weekly_summary = sku_preds.groupby("Calendar_Week").agg(
                Start_Date=("Date", "min"),
                End_Date=("Date", "max"),
                Days_Count=("Date", "count"),
                Weekly_Predicted_Demand=("Predicted_Demand", "sum"),
                Weekly_Actual_Demand=("Actual_Demand", "sum") if has_actuals else ("Predicted_Demand", "count")
            ).reset_index()

            col_w1, col_w2 = st.columns([1, 1])
            with col_w1:
                st.markdown("##### 📅 Weekly Aggregated Operational Demand")
                display_cols = ["Calendar_Week", "Start_Date", "End_Date", "Weekly_Predicted_Demand"]
                if has_actuals:
                    display_cols.append("Weekly_Actual_Demand")
                st.dataframe(
                    weekly_summary[display_cols].style.format({
                        "Weekly_Predicted_Demand": "{:,.1f}",
                        "Weekly_Actual_Demand": "{:,.1f}" if has_actuals else "{}",
                        "Start_Date": lambda d: d.strftime("%Y-%m-%d"),
                        "End_Date": lambda d: d.strftime("%Y-%m-%d")
                    }),
                    use_container_width=True
                )
            with col_w2:
                st.markdown("##### 📥 Daily Forecast Export")
                csv_export = sku_preds[["Date", "StockCode", "Predicted_Demand"] + (["Actual_Demand"] if has_actuals else [])].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Daily Predictions CSV",
                    data=csv_export,
                    file_name=f"demand_forecast_{selected_sku}_{selected_model_name[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.info(
                    f"**Forecasting Metadata:**\n"
                    f"- Pre-training Cutoff: `2011-11-09`\n"
                    f"- Holdout Evaluation Window: `2011-11-10` to `{latest_forecast_date}`\n"
                    f"- Zero-Demand Handling: Non-negative clipping strictly applied."
                )
        else:
            st.warning(f"No precomputed forecast outputs available for SKU `{selected_sku}` under `{selected_model_name}`.")

    elif page == "Inventory":
        st.subheader(f"Inventory — {selected_sku}")

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
                pie_fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(pie_fig, use_container_width=True)

    elif page == "Benchmarks":
        st.subheader("Model Benchmarks")

        comparison_df = load_model_comparison_table()
        if not comparison_df.empty:
            st.markdown("##### 🏆 Empirical Model Comparison (Top 50 Representative Products)")
            st.dataframe(comparison_df.style.highlight_min(subset=["Holdout MAE", "Holdout RMSE"], color="#DEF7EC"), use_container_width=True)

            bar_fig = px.bar(
                comparison_df,
                x="Model",
                y="Holdout MAE",
                title="Model Accuracy Comparison (Holdout MAE — Lower is Better)",
                color="Holdout MAE",
                color_continuous_scale="Blues_r"
            )
            bar_fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), xaxis_tickangle=-15)
            st.plotly_chart(bar_fig, use_container_width=True)

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

    elif page == "Drift":
        st.subheader("Data Drift")

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
                    use_container_width=True
                )
        else:
            st.warning("No drift monitoring summary found at `outputs/forecasting/phase7/drift_summary.json`.")

    elif page == "Retraining":
        st.subheader("Retraining Pipeline")

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
            st.dataframe(retraining_metrics_df, use_container_width=True)

        st.markdown("##### 🏷️ MLOps Tracking & Governance Manifest")
        st.json({
            "Experiment Name": "RetailPulse-Demand-Forecasting",
            "MLflow Tracking URI": "outputs/forecasting/phase6/mlruns",
            "Deterministic Seed": 42,
            "Retraining Logic": "Drift-Gated Optuna Search with Conservative Holdout Promotion",
            "Software Dependencies": "Prophet 1.4.0, PyTorch 2.14.1, LightGBM 4.7.0, Evidently 0.7.23, Airflow 2.11.2"
        })

    # --- Executive Summary Card (only where its inventory numbers exist) ---
    if page == "Inventory":
        st.markdown("---")
        st.markdown("### Executive Business Summary")
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
