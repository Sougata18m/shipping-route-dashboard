"""
analytics.py — KPI computations and route analysis for the dashboard.
"""

import pandas as pd
import numpy as np


def get_kpi_cards(df: pd.DataFrame) -> dict:
    """Compute headline KPIs for the dashboard header."""
    total_orders = df["Order ID"].nunique()
    total_shipments = len(df)
    avg_lead_time = df["Shipping_Lead_Time"].mean()
    median_lead_time = df["Shipping_Lead_Time"].median()
    total_sales = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()
    unique_routes = df["Route"].nunique()
    unique_customers = df["Customer ID"].nunique()

    return {
        "Total Orders": f"{total_orders:,}",
        "Total Shipments": f"{total_shipments:,}",
        "Avg Lead Time": f"{avg_lead_time:.1f} days",
        "Median Lead Time": f"{median_lead_time:.1f} days",
        "Total Sales": f"${total_sales:,.0f}",
        "Gross Profit": f"${total_profit:,.0f}",
        "Unique Routes": f"{unique_routes:,}",
        "Unique Customers": f"{unique_customers:,}",
    }


def get_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per route (Factory → State)."""
    summary = (
        df.groupby("Route")
        .agg(
            Total_Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Median_Lead_Time=("Shipping_Lead_Time", "median"),
            Std_Lead_Time=("Shipping_Lead_Time", "std"),
            Min_Lead_Time=("Shipping_Lead_Time", "min"),
            Max_Lead_Time=("Shipping_Lead_Time", "max"),
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Total_Profit=("Gross Profit", "sum"),
        )
        .reset_index()
    )
    summary["Std_Lead_Time"] = summary["Std_Lead_Time"].fillna(0)

    # Efficiency score: 0–100 (lower lead time → higher score)
    max_lt = summary["Avg_Lead_Time"].max()
    min_lt = summary["Avg_Lead_Time"].min()
    if max_lt != min_lt:
        summary["Efficiency_Score"] = (
            100 * (max_lt - summary["Avg_Lead_Time"]) / (max_lt - min_lt)
        ).round(1)
    else:
        summary["Efficiency_Score"] = 100.0

    return summary.sort_values("Avg_Lead_Time")


def get_top_bottom_routes(summary: pd.DataFrame, n: int = 10):
    """Return top-N and bottom-N routes by efficiency score."""
    top = summary.nlargest(n, "Efficiency_Score")
    bottom = summary.nsmallest(n, "Efficiency_Score")
    return top, bottom


def get_regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance metrics grouped by customer Region."""
    summary = (
        df.groupby("Region")
        .agg(
            Total_Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Median_Lead_Time=("Shipping_Lead_Time", "median"),
            Std_Lead_Time=("Shipping_Lead_Time", "std"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
        )
        .reset_index()
    )
    summary["Std_Lead_Time"] = summary["Std_Lead_Time"].fillna(0)
    return summary.sort_values("Avg_Lead_Time")


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance metrics grouped by customer State/Province."""
    summary = (
        df.groupby("State/Province")
        .agg(
            Total_Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Median_Lead_Time=("Shipping_Lead_Time", "median"),
            Std_Lead_Time=("Shipping_Lead_Time", "std"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Lat=("State_Lat", "first"),
            Lon=("State_Lon", "first"),
        )
        .reset_index()
    )
    summary["Std_Lead_Time"] = summary["Std_Lead_Time"].fillna(0)
    return summary.sort_values("Avg_Lead_Time")


def get_ship_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance comparison by ship mode."""
    summary = (
        df.groupby("Ship Mode")
        .agg(
            Total_Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Median_Lead_Time=("Shipping_Lead_Time", "median"),
            Std_Lead_Time=("Shipping_Lead_Time", "std"),
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Total_Cost=("Cost", "sum"),
            Total_Profit=("Gross Profit", "sum"),
        )
        .reset_index()
    )
    summary["Std_Lead_Time"] = summary["Std_Lead_Time"].fillna(0)
    summary["Profit_Margin_%"] = (
        (summary["Total_Profit"] / summary["Total_Sales"]) * 100
    ).round(1)
    return summary.sort_values("Avg_Lead_Time")


def get_factory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance per factory."""
    summary = (
        df.groupby("Factory")
        .agg(
            Total_Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Median_Lead_Time=("Shipping_Lead_Time", "median"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Lat=("Factory_Lat", "first"),
            Lon=("Factory_Lon", "first"),
        )
        .reset_index()
    )
    return summary.sort_values("Avg_Lead_Time")


def get_delay_stats(df: pd.DataFrame, threshold: int = 180) -> pd.DataFrame:
    """Compute delay frequency per route given a threshold (days)."""
    route_agg = df.groupby("Route").agg(
        Total=("Row ID", "count"),
        Delayed=("Shipping_Lead_Time", lambda x: (x > threshold).sum()),
    ).reset_index()
    route_agg["Delay_%"] = ((route_agg["Delayed"] / route_agg["Total"]) * 100).round(1)
    return route_agg.sort_values("Delay_%", ascending=False)


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average lead time trend by month of order."""
    df_copy = df.copy()
    df_copy["Order_Month"] = df_copy["Order Date"].dt.to_period("M").dt.to_timestamp()
    trend = (
        df_copy.groupby("Order_Month")
        .agg(
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Total_Shipments=("Row ID", "count"),
            Total_Sales=("Sales", "sum"),
        )
        .reset_index()
    )
    return trend

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_delay_model(df: pd.DataFrame, threshold: int = 180) -> dict:
    """Train a Random Forest model to predict delays based on categorical features."""
    if df.empty or len(df) < 20: # Require minimum samples
        return None
    
    y = (df["Shipping_Lead_Time"] > threshold).astype(int)
    
    # Needs both classes to train
    if len(y.unique()) < 2:
        return None

    features = ["Factory", "Region", "State/Province", "Ship Mode"]
    X_raw = df[features]
    
    X = pd.DataFrame()
    encoders = {}
    for col in features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X_raw[col])
        encoders[col] = le
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    return {
        "model": clf,
        "features": features,
        "encoders": encoders,
        "accuracy": acc,
        "feature_importance": feat_imp
    }

def detect_anomalies(summary: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Flag routes experiencing lead times significantly worse than global average."""
    global_mean = df["Shipping_Lead_Time"].mean()
    global_std = df["Shipping_Lead_Time"].std()
    anomaly_threshold = global_mean + (1.5 * global_std)
    
    anomalies = summary[
        (summary["Avg_Lead_Time"] > anomaly_threshold) & 
        (summary["Total_Shipments"] >= 5)
    ].copy()
    
    anomalies["Deviation_from_Avg"] = (anomalies["Avg_Lead_Time"] - global_mean).round(1)
    return anomalies.sort_values("Deviation_from_Avg", ascending=False)

def get_customer_impact(df: pd.DataFrame, threshold: int = 180) -> pd.DataFrame:
    """Identify customers most affected by delays (high revenue at risk)."""
    if "Customer ID" not in df.columns:
         return pd.DataFrame()
         
    df_copy = df.copy()
    df_copy["Is_Delayed"] = df_copy["Shipping_Lead_Time"] > threshold
    
    impact = df_copy.groupby(["Customer ID"]).agg(
        Total_Orders=("Row ID", "count"),
        Delayed_Orders=("Is_Delayed", "sum"),
        Revenue_At_Risk=("Sales", lambda x: x[df_copy.loc[x.index, "Is_Delayed"]].sum()),
        Total_Revenue=("Sales", "sum")
    ).reset_index()
    
    # Filter to customers with at least 1 delay
    impact = impact[impact["Delayed_Orders"] > 0].copy()
    if not impact.empty:
        impact["Delay_Rate_%"] = ((impact["Delayed_Orders"] / impact["Total_Orders"]) * 100).round(1)
        return impact.sort_values("Revenue_At_Risk", ascending=False)
    return pd.DataFrame()
