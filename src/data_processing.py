"""
Data processing utilities for marketing customer segmentation.
Handles data cleaning, feature engineering, and RFM computation.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the Online Retail II dataset.
    
    Steps:
    - Remove rows with missing CustomerID
    - Remove cancelled orders (InvoiceNo starting with 'C')
    - Remove rows with non-positive Quantity or Price
    - Parse dates
    """
    df = pd.read_csv(filepath, encoding="utf-8")

    # Standardize column names
    col_map = {
        "Invoice": "InvoiceNo",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
        "Country": "Country",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    print(f"Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Missing CustomerID: {df['CustomerID'].isna().sum():,}")

    # Drop missing customer
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove non-positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Compute revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"Cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")

    return df


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features per customer.
    
    Parameters
    ----------
    df : cleaned transaction dataframe
    reference_date : date to compute recency from (default: max date + 1 day)
    
    Returns
    -------
    DataFrame with CustomerID, Recency, Frequency, Monetary
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    # RFM scores (1-5 quantile-based)
    for col in ["Recency", "Frequency", "Monetary"]:
        ascending = col == "Recency"  # Lower recency = better
        rfm[f"{col}_Score"] = pd.qcut(
            rfm[col].rank(method="first", ascending=ascending),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

    rfm["RFM_Score"] = (
        rfm["Recency_Score"] + rfm["Frequency_Score"] + rfm["Monetary_Score"]
    )

    print(f"RFM table: {rfm.shape[0]:,} customers")
    print(f"\nRFM Statistics:")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))

    return rfm


def engineer_churn_features(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    churn_threshold_days: int = 90,
    reference_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Build features for churn prediction.
    
    Features:
    - RFM scores and raw values
    - Average order value
    - Purchase frequency (orders per active month)
    - Days since first purchase (tenure)
    - Number of unique products purchased
    - Average basket size
    - Revenue trend (last 3 months vs prior)
    - Weekend purchase ratio
    - Return rate
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    # Basic aggregations
    cust = (
        df.groupby("CustomerID")
        .agg(
            total_orders=("InvoiceNo", "nunique"),
            total_revenue=("Revenue", "sum"),
            total_items=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            avg_unit_price=("UnitPrice", "mean"),
        )
        .reset_index()
    )

    # Derived features
    cust["avg_order_value"] = cust["total_revenue"] / cust["total_orders"]
    cust["avg_basket_size"] = cust["total_items"] / cust["total_orders"]
    cust["tenure_days"] = (reference_date - cust["first_purchase"]).dt.days
    cust["recency_days"] = (reference_date - cust["last_purchase"]).dt.days
    cust["orders_per_month"] = cust["total_orders"] / (cust["tenure_days"] / 30 + 1)

    # Weekend purchase ratio
    df_copy = df.copy()
    df_copy["is_weekend"] = df_copy["InvoiceDate"].dt.dayofweek >= 5
    weekend_ratio = (
        df_copy.groupby("CustomerID")["is_weekend"]
        .mean()
        .reset_index()
        .rename(columns={"is_weekend": "weekend_ratio"})
    )
    cust = cust.merge(weekend_ratio, on="CustomerID", how="left")

    # Revenue trend: last 90 days vs prior
    cutoff = reference_date - timedelta(days=90)
    recent_rev = (
        df[df["InvoiceDate"] >= cutoff]
        .groupby("CustomerID")["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "recent_revenue"})
    )
    prior_rev = (
        df[df["InvoiceDate"] < cutoff]
        .groupby("CustomerID")["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "prior_revenue"})
    )
    cust = cust.merge(recent_rev, on="CustomerID", how="left")
    cust = cust.merge(prior_rev, on="CustomerID", how="left")
    cust["recent_revenue"] = cust["recent_revenue"].fillna(0)
    cust["prior_revenue"] = cust["prior_revenue"].fillna(0)
    cust["revenue_trend"] = cust["recent_revenue"] / (cust["prior_revenue"] + 1)

    # Merge RFM scores
    cust = cust.merge(
        rfm[["CustomerID", "Recency_Score", "Frequency_Score", "Monetary_Score", "RFM_Score"]],
        on="CustomerID",
        how="left",
    )

    # Churn label
    cust["is_churned"] = (cust["recency_days"] > churn_threshold_days).astype(int)

    # Drop date columns for modeling
    feature_cols = [
        "total_orders", "total_revenue", "total_items", "unique_products",
        "avg_unit_price", "avg_order_value", "avg_basket_size",
        "tenure_days", "recency_days", "orders_per_month", "weekend_ratio",
        "recent_revenue", "prior_revenue", "revenue_trend",
        "Recency_Score", "Frequency_Score", "Monetary_Score", "RFM_Score",
    ]

    print(f"Feature matrix: {cust.shape[0]:,} customers × {len(feature_cols)} features")
    print(f"Churn rate: {cust['is_churned'].mean()*100:.1f}%")

    return cust, feature_cols
