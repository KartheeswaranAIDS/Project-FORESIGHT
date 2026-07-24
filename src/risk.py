"""
risk.py – Risk Scoring Engine for Project FORESIGHT
Implements transparent business rules for stockout / overstock detection
and computes financial impact in INR.
"""

import pandas as pd
import numpy as np


def run_risk_scoring(forecasts_df, inventory_df, sku_master_df):
    """
    Score every SKU for stockout & overstock risk and assign an action.

    Business Rules
    ──────────────
    Stockout HIGH  : lead_time_demand > available_stock
    Overstock HIGH : on_hand_units > 1.5 × demand_next_8_weeks

    Actions
    ───────
    REORDER NOW     – HIGH stockout risk
    MARKDOWN / CLEAR – HIGH overstock risk
    WATCH           – Available stock < 1.2 × lead_time_demand (borderline)
    HEALTHY         – Everything else

    Financial Impact (converted to ₹ using 1 USD ≈ 83 INR)
    ────────────────
    sales_at_risk_rupees  = lost_units × list_price × 83
    locked_capital_rupees = excess_units × unit_cost × 83

    Returns
    -------
    risk_df : pd.DataFrame
    health_score : int (0–100)
    """

    # Latest inventory snapshot per SKU
    inventory_df = inventory_df.copy()
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])
    latest_date = inventory_df["date"].max()
    current_inv = inventory_df[inventory_df["date"] == latest_date].copy()

    # Total 8-week forecast demand per SKU
    forecast_8w = (
        forecasts_df.groupby("sku_id")["forecast_demand"]
        .sum()
        .reset_index()
        .rename(columns={"forecast_demand": "demand_next_8_weeks"})
    )

    # Average weekly forecast (for lead-time demand calc)
    avg_weekly = (
        forecasts_df.groupby("sku_id")["forecast_demand"]
        .mean()
        .reset_index()
        .rename(columns={"forecast_demand": "avg_weekly_forecast"})
    )

    # Merge everything
    risk = current_inv.merge(forecast_8w, on="sku_id", how="left")
    risk = risk.merge(avg_weekly, on="sku_id", how="left")
    risk = risk.merge(
        sku_master_df[["sku_id", "product_name", "category", "subcategory",
                        "list_price", "unit_cost"]],
        on="sku_id", how="left",
    )

    risk["demand_next_8_weeks"] = risk["demand_next_8_weeks"].fillna(0)
    risk["avg_weekly_forecast"] = risk["avg_weekly_forecast"].fillna(0)

    # Derived columns
    risk["lead_time_demand"] = (
        (risk["lead_time_days"] / 7.0) * risk["avg_weekly_forecast"]
    ).round(0)
    risk["available_stock"] = risk["on_hand_units"] + risk["on_order_units"]

    # ── Risk classification ──────────────────────────────────────────
    risk["stockout_risk"] = "LOW"
    risk["overstock_risk"] = "LOW"
    risk["action"] = "HEALTHY"

    stockout_mask = risk["lead_time_demand"] > risk["available_stock"]
    risk.loc[stockout_mask, "stockout_risk"] = "HIGH"

    overstock_mask = risk["on_hand_units"] > (risk["demand_next_8_weeks"] * 1.5)
    risk.loc[overstock_mask, "overstock_risk"] = "HIGH"

    # Assign actions (stockout takes priority)
    risk.loc[overstock_mask, "action"] = "MARKDOWN / CLEAR"
    risk.loc[stockout_mask, "action"] = "REORDER NOW"

    watch_mask = (
        (risk["action"] == "HEALTHY")
        & (risk["available_stock"] < (risk["lead_time_demand"] * 1.2))
    )
    risk.loc[watch_mask, "action"] = "WATCH"

    # ── Financial impact ─────────────────────────────────────────────
    USD_TO_INR = 83.0

    risk["lost_units"] = np.maximum(
        0, risk["lead_time_demand"] - risk["available_stock"]
    )
    risk["sales_at_risk_rupees"] = (
        risk["lost_units"] * risk["list_price"] * USD_TO_INR
    ).round(0)

    risk["excess_units"] = np.maximum(
        0, risk["on_hand_units"] - risk["demand_next_8_weeks"]
    )
    risk["locked_capital_rupees"] = (
        risk["excess_units"] * risk["unit_cost"] * USD_TO_INR
    ).round(0)

    risk["lead_time_demand"] = risk["lead_time_demand"].round(0)

    # ── Inventory Health Score (0–100) ───────────────────────────────
    total = len(risk)
    healthy = len(risk[risk["action"] == "HEALTHY"])
    watch = len(risk[risk["action"] == "WATCH"])
    health_score = (
        int(((healthy * 1.0 + watch * 0.5) / total) * 100) if total > 0 else 0
    )

    return risk, health_score


if __name__ == "__main__":
    pass
