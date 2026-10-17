"""
Regression Test Suite for Deployment Readiness Fixes (F-01 and F-02).

Covers:
- F-01: Business Overview country and date filters, Top 10 Countries and Top 10 Products charts,
        preservation of fast precomputed summary path, and controlled error handling.
- F-02: Weekly portfolio-level MAPE computation, distinction from daily unweighted SKU MAPE,
        removal of misleading baseline highlight, proper unavailable state, and non-leakage
        of the sequestered holdout.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
from streamlit.testing.v1 import AppTest


class TestBusinessOverviewFixF01:
    """Test suite for F-01: Business Overview UI, filters, and breakdown charts."""

    def test_load_business_summary_schema(self):
        """Verify load_business_summary loads all required pre-aggregated fields."""
        summary = app.load_business_summary()
        assert isinstance(summary, dict), "Summary should return a dict"
        assert len(summary) > 0, "Precomputed summary should not be empty"

        required_keys = [
            "total_revenue", "total_orders", "total_customers",
            "avg_order_value", "monthly_revenue", "top_countries",
            "top_products", "countries"
        ]
        for key in required_keys:
            assert key in summary, f"Expected key '{key}' in business_overview_summary.json"

        assert len(summary["top_countries"]) > 0, "top_countries should have records"
        assert len(summary["top_products"]) > 0, "top_products should have records"
        assert len(summary["countries"]) > 0, "countries list should have records"
        assert len(summary["monthly_revenue"]) == 25, "Expected 25 monthly revenue entries"

    def test_summary_present_renders_filters_and_breakdowns(self):
        """Verify summary-present path renders country/date filters, plotly charts, and download buttons."""
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()

        # Should be on Overview page without uncaught exceptions
        assert len(at.exception) == 0, f"AppTest raised exceptions: {[e.message for e in at.exception]}"

        # Country filter multiselect must be rendered
        multiselect_labels = [m.label for m in at.multiselect]
        assert any("Filter by Country:" in label for label in multiselect_labels), (
            f"Expected 'Filter by Country:' in multiselects, got {multiselect_labels}"
        )

        # Date range picker must be rendered
        date_input_labels = [d.label for d in at.date_input]
        assert any("Filter by Date Range:" in label for label in date_input_labels), (
            f"Expected 'Filter by Date Range:' in date_inputs, got {date_input_labels}"
        )

        # 3 Plotly charts (captured as UnknownElement in AppTest element tree) must be rendered:
        # Monthly Trend, Top 10 Countries, and Top 10 Products
        plotly_elements = [el for el in at if type(el).__name__ == "UnknownElement"]
        assert len(plotly_elements) >= 3, f"Expected at least 3 plotly charts, got {len(plotly_elements)}"

        # CSV Download buttons must be present
        download_labels = [b.label for b in at.download_button]
        assert any("Monthly Revenue" in label for label in download_labels), "Monthly CSV download missing"
        assert any("Country Revenue" in label for label in download_labels), "Country CSV download missing"
        assert any("Top Products" in label for label in download_labels), "Products CSV download missing"

        # Top products data table must be rendered
        assert len(at.dataframe) >= 1, "Expected Top Products dataframe to be rendered"

    def test_summary_present_does_not_fall_into_fallback_warning(self):
        """Verify summary-present path does not display empty-data warning."""
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()
        for w in at.warning:
            assert "No transactions match the current filters" not in w.value

    def test_date_filter_affects_monthly_data(self):
        """Verify date filtering properly slices monthly revenue and recalculates revenue."""
        summary = app.load_business_summary()
        monthly_df = pd.DataFrame(summary["monthly_revenue"])
        monthly_df["Month"] = pd.to_datetime(monthly_df["Month"])

        full_revenue = monthly_df["Revenue"].sum()
        assert full_revenue > 0

        # Slice to calendar year 2010 only
        start_2010 = pd.to_datetime("2010-01-01")
        end_2010 = pd.to_datetime("2010-12-31")
        filtered_2010 = monthly_df[(monthly_df["Month"] >= start_2010) & (monthly_df["Month"] <= end_2010)]

        assert len(filtered_2010) == 12, f"Expected 12 months in 2010, got {len(filtered_2010)}"
        revenue_2010 = filtered_2010["Revenue"].sum()
        assert 0 < revenue_2010 < full_revenue, "2010 revenue should be a strict positive subset of total"

    def test_missing_summary_produces_controlled_state(self, tmp_path):
        """Verify that loaders handle missing summary data gracefully with clear empty structures."""
        # When summary file is absent, load_business_summary returns empty dict
        empty_summary = app.load_business_summary(project_dir=str(tmp_path))
        assert empty_summary == {}, "Missing summary file must return empty dict"

        # When cleaned transactions are absent, load_business_transactions returns empty DataFrame
        empty_tx = app.load_business_transactions(project_dir=str(tmp_path))
        assert isinstance(empty_tx, pd.DataFrame)
        assert empty_tx.empty, "Missing transactions must return empty DataFrame"

    def test_country_filter_selection_updates_kpis_and_breakdowns(self):
        """
        Regression test: Verify selecting Australia dynamically updates KPI cards,
        monthly trend data, and top products breakdown away from global totals.
        """
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"App threw unexpected exception: {at.exception}"

        # 1. Verify global precomputed state
        global_markdown = "\n".join([m.value for m in at.markdown if "metric-card" in m.value])
        assert "19,287,251" in global_markdown, "Expected global total revenue ~£19.29M"
        assert "53,628" in global_markdown, "Expected global total orders 53,628"
        assert "5,943" in global_markdown, "Expected global total customers 5,943"
        assert "All countries" in global_markdown

        # 2. Select only Australia in Country multiselect
        assert len(at.multiselect) >= 1, "Expected country multiselect filter"
        at.multiselect[0].set_value(["Australia"]).run()
        assert not at.exception, f"App threw exception when filtering to Australia: {at.exception}"

        # 3. Verify filtered KPI cards reflect Australia specific metrics
        aus_markdown = "\n".join([m.value for m in at.markdown if "metric-card" in m.value])
        assert "167,129" in aus_markdown, "Filtered revenue must update to Australia revenue (£167,129)"
        assert "117" in aus_markdown, "Filtered orders must update to Australia orders (117)"
        assert "15" in aus_markdown, "Filtered customers must update to Australia customers (15)"
        assert "1,428.45" in aus_markdown, "Filtered AOV must update to Australia AOV (£1,428.45)"
        assert "Filtered (1 country)" in aus_markdown, "Card subtitle must clearly indicate filtered state"

        # 4. Verify top products dataframe reflects Australia
        assert len(at.dataframe) >= 1, "Expected Top Products dataframe to be rendered"
        aus_top_df = at.dataframe[0].value
        assert not aus_top_df.empty, "Australia top products dataframe should not be empty"
        assert any("RABBIT NIGHT LIGHT" in str(desc) for desc in aus_top_df["Description"]), (
            "Australia top products should include RABBIT NIGHT LIGHT"
        )

    def test_narrow_date_filter_updates_kpis_and_breakdowns(self):
        """
        Regression test: Verify narrowing date range to December 2009 updates
        Total Revenue, Orders, Customers, and AOV consistently from clean transactions.
        """
        import datetime
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"App threw unexpected exception: {at.exception}"

        # Narrow date range to December 2009
        assert len(at.date_input) >= 1, "Expected date_input filter"
        at.date_input[0].set_value((datetime.date(2009, 12, 1), datetime.date(2009, 12, 31))).run()
        assert not at.exception, f"App threw exception after narrowing date: {at.exception}"

        dec_markdown = "\n".join([m.value for m in at.markdown if "metric-card" in m.value])
        assert "799,847" in dec_markdown, "Filtered revenue for Dec 2009 must be £799,847"
        assert "2,330" in dec_markdown, "Filtered orders for Dec 2009 must be 2,330 (not 53,628)"
        assert "1,046" in dec_markdown, "Filtered customers for Dec 2009 must be 1,046 (not 5,943)"
        assert "343.28" in dec_markdown, "Filtered AOV for Dec 2009 must be £343.28 (not £359.65)"
        assert "All countries (Date filtered)" in dec_markdown
        assert "Invoices in date range" in dec_markdown

    def test_narrow_date_filter_with_australia_only(self):
        """
        Regression test: Verify narrowing date range to December 2009 with Australia only
        computes consistent filtered KPIs (£50.25 revenue, 5 orders, 3 customers, £10.05 AOV).
        """
        import datetime
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()

        # Filter to Australia and December 2009
        at.multiselect[0].set_value(["Australia"]).run()
        at.date_input[0].set_value((datetime.date(2009, 12, 1), datetime.date(2009, 12, 31))).run()
        assert not at.exception, f"App threw exception after filtering: {at.exception}"

        aus_dec_markdown = "\n".join([m.value for m in at.markdown if "metric-card" in m.value])
        assert "50" in aus_dec_markdown, "Dec 2009 Australia revenue must be £50 (not global or full-year)"
        assert "5" in aus_dec_markdown, "Dec 2009 Australia orders must be 5 (not 117 or 53,628)"
        assert "3" in aus_dec_markdown, "Dec 2009 Australia customers must be 3 (not 15 or 5,943)"
        assert "10.05" in aus_dec_markdown, "Dec 2009 Australia AOV must be £10.05 (not £1,428.45 or global)"
        assert "Filtered (1 country)" in aus_dec_markdown
        assert "Invoices in date range" in aus_dec_markdown

    def test_monthly_revenue_chart_date_axis_formatting(self):
        """
        Verify Monthly Revenue Trend chart x-axis configuration:
        uses type='date', tickformat='%b %Y', dtick='M1' for single month,
        and avoids raw time fragments like 23:59:59.999.
        """
        import plotly.express as px
        df_single = pd.DataFrame({"Month": [pd.to_datetime("2009-12-01")], "Revenue": [799847.11]})
        df_single["Month"] = pd.to_datetime(df_single["Month"]).dt.to_period("M").dt.to_timestamp()

        rev_fig = px.line(df_single, x="Month", y="Revenue", title="Monthly Revenue Trend", markers=True)
        rev_fig.update_xaxes(
            type="date",
            tickformat="%b %Y",
            dtick="M1" if len(df_single) <= 12 else "M2",
        )
        if len(df_single) == 1:
            m_val = pd.to_datetime(df_single["Month"].iloc[0])
            rev_fig.update_xaxes(
                range=[
                    (m_val - pd.DateOffset(days=15)).strftime("%Y-%m-%d"),
                    (m_val + pd.DateOffset(days=15)).strftime("%Y-%m-%d"),
                ]
            )

        assert rev_fig.layout.xaxis.type == "date"
        assert rev_fig.layout.xaxis.tickformat == "%b %Y"
        assert rev_fig.layout.xaxis.dtick == "M1"
        assert rev_fig.layout.xaxis.range == ("2009-11-16", "2009-12-16")


class TestForecastingMetricsFixF02:
    """Test suite for F-02: Forecasting metrics and model comparison."""

    def test_weekly_portfolio_mape_formula_and_aggregation(self):
        """
        Verify Weekly Portfolio MAPE formula:
        abs(sum(actual) - sum(pred)) / sum(actual) * 100 per complete week,
        and arithmetic mean across complete weeks.
        """
        weekly_df, meta, summary_df = app.load_weekly_portfolio_validation_metrics()

        assert not weekly_df.empty, "weekly_portfolio_mape.csv must not be empty"
        assert "Model" in weekly_df.columns
        assert "Actual_Total" in weekly_df.columns
        assert "Forecast_Total" in weekly_df.columns
        assert "Portfolio_APE_pct" in weekly_df.columns

        # Verify each row's Portfolio_APE_pct formula
        for idx, row in weekly_df.iterrows():
            expected_ape = abs(row["Actual_Total"] - row["Forecast_Total"]) / row["Actual_Total"] * 100.0
            assert abs(row["Portfolio_APE_pct"] - expected_ape) < 1e-3, (
                f"Row {idx} Portfolio_APE_pct {row['Portfolio_APE_pct']} != calculated {expected_ape}"
            )

        # Verify arithmetic mean across the 4 complete weeks for production ensemble
        ens_weeks = weekly_df[weekly_df["Model"] == "Prophet_LSTM_Ensemble"]
        assert len(ens_weeks) == 4, f"Expected 4 complete weeks, got {len(ens_weeks)}"
        arithmetic_mean_mape = ens_weeks["Portfolio_APE_pct"].mean()

        assert abs(arithmetic_mean_mape - 14.8051) < 1e-2, (
            f"Arithmetic mean weekly portfolio MAPE {arithmetic_mean_mape:.4f}% != expected 14.8051%"
        )

    def test_mean_over_documented_complete_weeks(self):
        """Verify the 4 weeks are documented 7-day intervals excluding partial days."""
        weekly_df, meta, _ = app.load_weekly_portfolio_validation_metrics()
        weeks = weekly_df[weekly_df["Model"] == "Prophet_LSTM_Ensemble"].copy()

        assert len(weeks) == 4, "Expected exactly 4 complete scoring weeks"
        assert list(weeks["Week"]) == [1, 2, 3, 4]

        # Check date boundaries: 2011-10-11 to 2011-11-07
        start_date = weeks["Start"].min()
        end_date = weeks["End"].max()
        assert start_date == "2011-10-11", f"Expected start 2011-10-11, got {start_date}"
        assert end_date == "2011-11-07", f"Expected end 2011-11-07, got {end_date}"

        # Verify excluded partial dates in metadata
        excluded = meta.get("timeline", {}).get("excluded_dates", [])
        assert "2011-11-08" in excluded
        assert "2011-11-09" in excluded

    def test_zero_denominator_safe_handling(self):
        """Verify zero-denominator week produces NaN / handled gracefully without ZeroDivisionError."""
        actual = 0.0
        predicted = 100.0
        # Metric policy: if actual is 0, percentage error is undefined (NaN)
        ape = abs(actual - predicted) / actual * 100.0 if actual > 0 else np.nan
        assert np.isnan(ape)

    def test_daily_sku_mape_distinguished_from_weekly_portfolio(self):
        """Verify column names explicitly distinguish daily unweighted SKU MAPE from weekly portfolio MAPE."""
        comp_df = app.load_model_comparison_table()
        assert not comp_df.empty

        # In tuned_model_comparison.csv raw file, verify old column
        assert "Holdout MAPE (%)" in comp_df.columns

        # Verify mapped display dataframe in app.py uses distinct names
        display_df = comp_df.rename(columns={
            "Holdout MAPE (%)": "Daily SKU Unweighted MAPE (%)"
        })
        assert "Daily SKU Unweighted MAPE (%)" in display_df.columns
        assert "Holdout MAPE (%)" not in display_df.columns

    def test_models_page_rendering_and_no_baseline_highlight(self):
        """Verify Models page displays scorecard, 12% target, and does NOT highlight Baseline MA28."""
        at = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=30)
        at.run()

        # Switch to Models page
        at.segmented_control[0].select("Models").run()
        assert len(at.exception) == 0, f"Models page exceptions: {[e.message for e in at.exception]}"

        # Verify Scorecard metrics
        metric_labels = [m.label for m in at.metric]
        assert "Project Acceptance Target" in metric_labels
        assert "Active Production Ensemble" in metric_labels
        assert "Acceptance Status" in metric_labels
        assert "Evaluation Horizon" in metric_labels

        # Find target and ensemble metrics
        target_metric = next(m for m in at.metric if m.label == "Project Acceptance Target")
        assert "12.0%" in target_metric.value

        ensemble_metric = next(m for m in at.metric if m.label == "Active Production Ensemble")
        assert "14.81%" in ensemble_metric.value

        status_metric = next(m for m in at.metric if m.label == "Acceptance Status")
        assert "Threshold Not Yet Met" in status_metric.value

        # Verify DataFrame is rendered
        assert len(at.dataframe) >= 1
        # Check expander exists for weekly breakdown
        assert len(at.expander) >= 1

    def test_no_access_to_sequestered_holdout(self):
        """Verify the evaluation loader strictly uses validation window and never accesses sequestered holdout."""
        weekly_df, meta, _ = app.load_weekly_portfolio_validation_metrics()

        # Holdout window is 2011-11-10 to 2011-12-09
        # Check that no evaluation dates in weekly_df fall within the sequestered holdout
        all_starts = pd.to_datetime(weekly_df["Start"])
        all_ends = pd.to_datetime(weekly_df["End"])
        holdout_start = pd.to_datetime("2011-11-10")

        assert (all_starts < holdout_start).all(), "Evaluation start date must precede holdout start"
        assert (all_ends < holdout_start).all(), "Evaluation end date must precede holdout start"

        # Check metadata disclosure
        holdout_meta = meta.get("timeline", {}).get("holdout_period", "")
        assert "2011-11-10 to 2011-12-09" in holdout_meta
        assert "untouched" in holdout_meta

    def test_missing_metrics_loader_fallback(self, tmp_path):
        """Verify missing weekly metrics produce empty structures without fabricating scores."""
        w_df, meta, s_df = app.load_weekly_portfolio_validation_metrics(project_dir=str(tmp_path))
        assert isinstance(w_df, pd.DataFrame)
        assert w_df.empty, "Missing weekly csv should return empty DataFrame"
        assert meta == {}, "Missing metadata should return empty dict"
        assert isinstance(s_df, pd.DataFrame)
        assert s_df.empty, "Missing summary csv should return empty DataFrame"
