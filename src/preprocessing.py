"""
RetailPulse — Demand Forecasting Module
Data Cleaning, Preprocessing & Time-Series Preparation Pipeline

This module provides modular, reusable functions to clean raw transaction data,
build continuous daily product demand series, analyze product eligibility,
and perform chronological train/holdout splits.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Known administrative, non-merchandise stock codes identified in Phase 1
NON_MERCHANDISE_CODES = {
    "POST",           # Postage
    "D",              # Discount
    "M",              # Manual entry
    "DOT",            # DOTCOM POSTAGE
    "BANK CHARGES",   # Bank transaction charges
    "C2",             # Carriage / Shipping
    "S",              # Samples
    "B",              # Bad debt adjustments
    "CRUK",           # Cancer Research UK fee
    "AMAZONFEE",      # Amazon platform fee
    "TEST",           # System test
    "TEST001",
    "TEST002",
    "ADJUST",         # Adjustment line
    "ADJUST2",
}


def clean_sales_data(
    df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Clean raw retail transaction dataset for demand forecasting.

    Preprocessing rules and business rationale:
    1. StockCode & Invoice Normalization: Strip whitespace, cast to uppercase string
       to prevent integer/string mismatch (e.g. 22423 vs '22423').
    2. Convert InvoiceDate: Parse to standard pandas datetime.
    3. Remove Exact Duplicates: Drop duplicate transaction rows logged identically.
    4. Remove Cancellations: Invoices starting with 'C' are customer returns.
       They do not represent new demand and would distort gross consumption.
    5. Remove Negative & Zero Quantities: Any remaining Quantity <= 0 corresponds
       to inventory adjustments, stock damage write-offs, or accounting fixes.
    6. Remove Zero & Negative Prices: Price <= 0 corresponds to free promotional
       samples, accounting write-offs, or debt adjustments.
    7. Remove Non-Merchandise Codes: Filter out shipping, postal fees, manual fees,
       discounts, and bank charges (e.g. POST, M, D, DOT, BANK CHARGES).
    8. Handle Missing Description: Impute missing descriptions from StockCode lookup.
    9. PRESERVE NULL Customer ID: 22.77% of valid sales are guest checkout purchases.
       Dropping them would artificially reduce demand by nearly a quarter.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame.
    verbose : bool
        If True, prints a summary of filtering steps and record counts.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned transaction DataFrame containing only valid product sales.
    audit_report : dict
        Detailed counts and percentages for each filtering rule.
    """
    initial_rows = len(df)
    audit_report: Dict[str, Union[int, float]] = {"initial_rows": initial_rows}

    # Make working copy to avoid side-effects on original DataFrame
    data = df.copy()

    # 1. Normalize identifiers
    data["StockCode"] = data["StockCode"].astype(str).str.strip().str.upper()
    data["Invoice"] = data["Invoice"].astype(str).str.strip().str.upper()

    # 2. Convert InvoiceDate
    if not pd.api.types.is_datetime64_any_dtype(data["InvoiceDate"]):
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    # 3. Exact Duplicates
    dup_mask = data.duplicated()
    exact_duplicates = int(dup_mask.sum())
    audit_report["exact_duplicates_removed"] = exact_duplicates
    data = data[~dup_mask].copy()

    # 4. Cancellations ('C' prefix in Invoice)
    cancellation_mask = data["Invoice"].str.startswith("C")
    cancellations_count = int(cancellation_mask.sum())
    audit_report["cancellations_removed"] = cancellations_count
    data = data[~cancellation_mask].copy()

    # 5. Non-positive Quantities
    non_pos_qty_mask = data["Quantity"] <= 0
    non_pos_qty_count = int(non_pos_qty_mask.sum())
    audit_report["non_positive_quantity_removed"] = non_pos_qty_count
    data = data[~non_pos_qty_mask].copy()

    # 6. Non-positive Prices
    non_pos_price_mask = data["Price"] <= 0
    non_pos_price_count = int(non_pos_price_mask.sum())
    audit_report["non_positive_price_removed"] = non_pos_price_count
    data = data[~non_pos_price_mask].copy()

    # 7. Non-merchandise StockCodes
    non_merch_mask = data["StockCode"].isin(NON_MERCHANDISE_CODES)
    non_merch_count = int(non_merch_mask.sum())
    audit_report["non_merchandise_codes_removed"] = non_merch_count
    data = data[~non_merch_mask].copy()

    # 8. Handle missing descriptions
    # Build lookup table of most frequent valid description per StockCode
    valid_desc = data.dropna(subset=["Description"])
    desc_lookup = (
        valid_desc.groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )
    missing_desc_before = int(data["Description"].isna().sum())
    data["Description"] = data["Description"].fillna(data["StockCode"].map(desc_lookup))
    data["Description"] = data["Description"].fillna("UNKNOWN PRODUCT")
    audit_report["missing_descriptions_imputed"] = missing_desc_before

    # Verify Customer ID preservation
    null_cust_count = int(data["Customer ID"].isna().sum())
    audit_report["preserved_null_customer_ids"] = null_cust_count

    final_rows = len(data)
    rows_removed = initial_rows - final_rows
    audit_report["final_rows"] = final_rows
    audit_report["total_rows_removed"] = rows_removed
    audit_report["retention_rate_pct"] = round(final_rows / initial_rows * 100, 2)

    if verbose:
        print("=== RetailPulse Sales Cleaning Audit ===")
        print(f"Initial raw transactions:       {initial_rows:,}")
        print(f"- Exact duplicates removed:      {exact_duplicates:,}")
        print(f"- Cancellations ('C') removed:   {cancellations_count:,}")
        print(f"- Quantity <= 0 removed:         {non_pos_qty_count:,}")
        print(f"- Price <= 0 removed:            {non_pos_price_count:,}")
        print(f"- Non-merchandise codes removed: {non_merch_count:,}")
        print(f"Total rows removed:             {rows_removed:,} ({rows_removed / initial_rows * 100:.2f}%)")
        print(f"Cleaned valid sales records:     {final_rows:,} ({audit_report['retention_rate_pct']}%)")
        print(f"Preserved null Customer ID rows: {null_cust_count:,} (Guest sales demand kept intact)")
        print("========================================")

    return data, audit_report


def build_daily_product_timeseries(
    df: pd.DataFrame,
    stock_code: str,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Build a continuous daily demand time series for a single product.

    Aggregates transaction-level quantities to daily totals, reindexes against
    a complete calendar date range, and fills missing dates with 0 demand.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales transaction DataFrame (output of clean_sales_data).
    stock_code : str
        The StockCode of the product to aggregate.
    start_date : str or pd.Timestamp, optional
        Start of the continuous calendar range. If None, uses min date in df.
    end_date : str or pd.Timestamp, optional
        End of the continuous calendar range. If None, uses max date in df.

    Returns
    -------
    pd.DataFrame
        Daily time series DataFrame with columns:
        ['Date', 'StockCode', 'Description', 'Demand']
    """
    code_norm = str(stock_code).strip().upper()
    prod_df = df[df["StockCode"] == code_norm].copy()

    # Determine product description fallback
    if len(prod_df) > 0 and prod_df["Description"].dropna().any():
        desc = prod_df["Description"].mode().iloc[0]
    else:
        desc = "UNKNOWN PRODUCT"

    # Derive daily calendar bounds
    df_date_series = pd.to_datetime(df["InvoiceDate"]).dt.floor("D")
    global_start = df_date_series.min() if start_date is None else pd.to_datetime(start_date).floor("D")
    global_end = df_date_series.max() if end_date is None else pd.to_datetime(end_date).floor("D")

    full_calendar = pd.date_range(global_start, global_end, freq="D", name="Date")

    if prod_df.empty:
        # Return all zero series across full calendar
        ts = pd.DataFrame({
            "Date": full_calendar,
            "StockCode": code_norm,
            "Description": desc,
            "Demand": 0
        })
        return ts

    # Aggregate Quantity by calendar date
    prod_df["Date"] = pd.to_datetime(prod_df["InvoiceDate"]).dt.floor("D")
    daily_agg = prod_df.groupby("Date")["Quantity"].sum()

    # Reindex against complete calendar and zero-fill
    reindexed = daily_agg.reindex(full_calendar, fill_value=0).reset_index()
    reindexed.rename(columns={"Quantity": "Demand"}, inplace=True)
    reindexed["StockCode"] = code_norm
    reindexed["Description"] = desc
    reindexed["Demand"] = reindexed["Demand"].astype(int)

    return reindexed[["Date", "StockCode", "Description", "Demand"]]


def analyze_product_eligibility(
    df: pd.DataFrame,
    min_active_days: int = 150
) -> pd.DataFrame:
    """
    Evaluate all products in the cleaned sales dataset for forecasting eligibility.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales transaction DataFrame.
    min_active_days : int
        Minimum number of unique trading days required for continuous modeling (default 150).

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        ['StockCode', 'Description', 'FirstDate', 'LastDate', 'ActiveDays',
         'TotalDemand', 'TransactionCount', 'eligible_for_forecasting']
    """
    data = df.copy()
    data["Date"] = pd.to_datetime(data["InvoiceDate"]).dt.floor("D")

    summary = (
        data.groupby("StockCode")
        .agg(
            Description=("Description", lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN"),
            FirstDate=("Date", "min"),
            LastDate=("Date", "max"),
            ActiveDays=("Date", "nunique"),
            TotalDemand=("Quantity", "sum"),
            TransactionCount=("Quantity", "count")
        )
        .reset_index()
    )

    summary["eligible_for_forecasting"] = summary["ActiveDays"] >= min_active_days
    summary = summary.sort_values(by=["eligible_for_forecasting", "TotalDemand"], ascending=[False, False])
    return summary


def build_all_daily_demand(
    df: pd.DataFrame,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    eligible_stock_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build daily aggregated demand for all (or a subset of) products across the full calendar.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales transaction DataFrame.
    start_date : str or pd.Timestamp, optional
        Start of calendar range.
    end_date : str or pd.Timestamp, optional
        End of calendar range.
    eligible_stock_codes : list of str, optional
        If provided, only aggregates products in this list.

    Returns
    -------
    pd.DataFrame
        Aggregated daily demand DataFrame with columns:
        ['Date', 'StockCode', 'Description', 'Demand']
    """
    data = df.copy()
    if eligible_stock_codes is not None:
        data = data[data["StockCode"].isin(eligible_stock_codes)]

    data["Date"] = pd.to_datetime(data["InvoiceDate"]).dt.floor("D")

    # Product Description mapping
    desc_map = (
        data.groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
        .to_dict()
    )

    # Global date range
    min_date = data["Date"].min() if start_date is None else pd.to_datetime(start_date).floor("D")
    max_date = data["Date"].max() if end_date is None else pd.to_datetime(end_date).floor("D")
    full_dates = pd.date_range(min_date, max_date, freq="D", name="Date")

    # Group by Date and StockCode
    daily_sums = (
        data.groupby(["Date", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "Demand"})
    )

    # Pivot to create continuous grid
    pivot_table = daily_sums.pivot(index="Date", columns="StockCode", values="Demand").fillna(0)
    pivot_reindexed = pivot_table.reindex(full_dates, fill_value=0)

    # Melt back to long format
    long_df = pivot_reindexed.reset_index().melt(
        id_vars=["Date"],
        var_name="StockCode",
        value_name="Demand"
    )
    long_df["Demand"] = long_df["Demand"].astype(int)
    long_df["Description"] = long_df["StockCode"].map(desc_map).fillna("UNKNOWN PRODUCT")

    # Order columns
    return long_df[["Date", "StockCode", "Description", "Demand"]].sort_values(
        by=["StockCode", "Date"]
    ).reset_index(drop=True)


def train_test_split_timeseries(
    df: pd.DataFrame,
    split_date: Union[str, pd.Timestamp] = "2011-11-09",
    date_col: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict chronological train/holdout test split on time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame containing a Date column.
    split_date : str or pd.Timestamp
        Last date of the training set (inclusive).
        Default is '2011-11-09' (creating a 30-day holdout: 2011-11-10 to 2011-12-09).
    date_col : str
        Name of the date column.

    Returns
    -------
    train_df : pd.DataFrame
        Training split (dates <= split_date).
    test_df : pd.DataFrame
        Holdout test split (dates > split_date).
    """
    split_ts = pd.to_datetime(split_date).floor("D")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col]).dt.floor("D")

    train_df = data[data[date_col] <= split_ts].copy()
    test_df = data[data[date_col] > split_ts].copy()

    # Temporal Leakage Validation
    if not train_df.empty and not test_df.empty:
        max_train = train_df[date_col].max()
        min_test = test_df[date_col].min()
        if max_train >= min_test:
            raise ValueError(
                f"Data Leakage Detected: Max train date ({max_train}) is >= Min test date ({min_test})!"
            )

    return train_df, test_df


def validate_daily_demand(
    df: pd.DataFrame,
    date_col: str = "Date",
    prod_col: str = "StockCode",
    demand_col: str = "Demand"
) -> Dict[str, Union[bool, int, str]]:
    """
    Validate data integrity and assumptions for the daily demand dataset.

    Checks:
    1. Null dates or demands
    2. Negative demand values
    3. Duplicate (Date + StockCode) rows
    4. Data types

    Parameters
    ----------
    df : pd.DataFrame
        Daily demand dataset to validate.

    Returns
    -------
    dict
        Validation report indicating status and metric checks.
    """
    report: Dict[str, Union[bool, int, str]] = {}

    null_dates = int(df[date_col].isna().sum())
    null_demand = int(df[demand_col].isna().sum())
    null_prods = int(df[prod_col].isna().sum())

    report["null_dates"] = null_dates
    report["null_demand"] = null_demand
    report["null_products"] = null_prods

    # Negative demands
    neg_demands = int((df[demand_col] < 0).sum())
    report["negative_demand_count"] = neg_demands

    # Duplicate Date + StockCode
    dup_keys = int(df.duplicated(subset=[date_col, prod_col]).sum())
    report["duplicate_date_product_count"] = dup_keys

    # Overall validation flag
    passed = (
        null_dates == 0 and
        null_demand == 0 and
        null_prods == 0 and
        neg_demands == 0 and
        dup_keys == 0
    )
    report["passed"] = passed

    if not passed:
        errors = []
        if null_dates > 0: errors.append(f"{null_dates} null dates")
        if null_demand > 0: errors.append(f"{null_demand} null demands")
        if neg_demands > 0: errors.append(f"{neg_demands} negative demands")
        if dup_keys > 0: errors.append(f"{dup_keys} duplicate Date+Product pairs")
        raise ValueError("Daily demand validation failed: " + "; ".join(errors))

    return report
