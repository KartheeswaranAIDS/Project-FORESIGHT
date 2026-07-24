"""
pipeline.py – Data Pipeline for Project FORESIGHT
Downloads or generates Walmart.csv and transforms it into the required project schema:
  sales_daily.csv, sku_master.csv, calendar.csv, inventory_snapshots.csv
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_FILE = os.path.join(DATA_DIR, "Walmart.csv")

CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/aditya-sundaram/Walmart-Sales-Forecasting/master/Walmart.csv",
    "https://raw.githubusercontent.com/khangtran311/Walmart-Sales-Forecasting/main/Walmart.csv",
    "https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv",
]

CATEGORIES = {
    "Grocery":     ["Snacks", "Beverages", "Dairy", "Bakery", "Frozen"],
    "Electronics": ["Accessories", "Audio", "Wearables", "Cables", "Gadgets"],
    "Apparel":     ["Men", "Women", "Kids", "Footwear", "Sportswear"],
    "Home":        ["Kitchen", "Decor", "Bedding", "Cleaning", "Storage"],
    "Sports":      ["Fitness", "Outdoor", "Team Sports", "Cycling", "Camping"],
}

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Fallback Synthetic Walmart Generator (if offline or download fails)
# ---------------------------------------------------------------------------

def _generate_synthetic_walmart():
    """Generates a standard 45-store, 143-week Walmart.csv schema dataset."""
    print("[Pipeline] Synthesizing Walmart dataset schema (45 stores, 143 weeks)...")
    stores = range(1, 46)
    dates = pd.date_range(start="2010-02-05", end="2012-10-26", freq="W-FRI")
    records = []
    
    for store in stores:
        base_sales = np.random.uniform(200000, 1800000)
        for date in dates:
            is_holiday = 1 if (date.month == 2 and date.day <= 15) or \
                              (date.month == 9 and date.day <= 10) or \
                              (date.month == 11 and date.day >= 20) or \
                              (date.month == 12 and date.day >= 25) else 0
            
            holiday_mult = 1.35 if is_holiday else 1.0
            trend = 1.0 + (date - dates[0]).days / 365.0 * 0.05
            season = 1.0 + 0.15 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            noise = np.random.normal(1.0, 0.08)
            
            weekly_sales = round(base_sales * holiday_mult * trend * season * noise, 2)
            temp = round(np.random.uniform(30, 90), 2)
            fuel_price = round(np.random.uniform(2.5, 4.2), 3)
            cpi = round(np.random.uniform(180, 225), 6)
            unemployment = round(np.random.uniform(5.5, 10.5), 3)
            
            records.append({
                "Store": store,
                "Date": date.strftime("%d-%m-%Y"),
                "Weekly_Sales": weekly_sales,
                "Holiday_Flag": is_holiday,
                "Temperature": temp,
                "Fuel_Price": fuel_price,
                "CPI": cpi,
                "Unemployment": unemployment
            })
            
    df = pd.DataFrame(records)
    return df


def _download_data():
    """Download Walmart.csv from candidates or use fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        print(f"[Pipeline] Using existing {RAW_FILE}")
        return pd.read_csv(RAW_FILE)

    # Check local Downloads folder fallback
    local_alt = r"c:\Users\DELL\Downloads\Project_FORESIGHT\data\train.csv"
    if os.path.exists(local_alt):
        print(f"[Pipeline] Found local dataset at {local_alt}")

    for url in CANDIDATE_URLS:
        try:
            print(f"[Pipeline] Attempting download from {url} ...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp, open(RAW_FILE, 'wb') as out_f:
                out_f.write(resp.read())
            print("[Pipeline] Download successful.")
            return pd.read_csv(RAW_FILE)
        except Exception as e:
            print(f"[Pipeline] Failed to download from {url}: {e}")

    # Fallback to synthesis
    df = _generate_synthetic_walmart()
    df.to_csv(RAW_FILE, index=False)
    print(f"[Pipeline] Saved synthetic Walmart dataset to {RAW_FILE}")
    return df


def _assign_sku_metadata(stores):
    """
    Assign each store a category-subcategory pair and a synthetic SKU ID
    so that the dataset has realistic SKU-level granularity.
    """
    records = []
    cat_list = list(CATEGORIES.keys())
    for idx, store in enumerate(sorted(stores)):
        cat = cat_list[idx % len(cat_list)]
        subcat = CATEGORIES[cat][idx % len(CATEGORIES[cat])]
        sku_id = f"SKU-{cat[:3].upper()}-{store:03d}"
        product_name = f"{subcat} {cat} – Store {store}"
        records.append({
            "store": store,
            "sku_id": sku_id,
            "category": cat,
            "subcategory": subcat,
            "product_name": product_name,
        })
    return pd.DataFrame(records)


def _get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def build_sku_master(meta_df, raw_df):
    """Create sku_master.csv."""
    avg_sales = raw_df.groupby("Store")["Weekly_Sales"].mean().reset_index()
    avg_sales.rename(columns={"Store": "store"}, inplace=True)

    sku_master = meta_df.merge(avg_sales, on="store")
    qty_factor = np.random.uniform(15, 60, size=len(sku_master))
    sku_master["list_price"] = (sku_master["Weekly_Sales"] / qty_factor).round(2)
    sku_master["unit_cost"] = (sku_master["list_price"] * np.random.uniform(0.40, 0.65, size=len(sku_master))).round(2)

    min_dates = raw_df.groupby("Store")["Date"].min().reset_index()
    min_dates.rename(columns={"Store": "store", "Date": "launch_date"}, inplace=True)
    sku_master = sku_master.merge(min_dates, on="store")
    sku_master["launch_date"] = pd.to_datetime(sku_master["launch_date"]).dt.strftime("%Y-%m-%d")

    sku_master = sku_master[["sku_id", "product_name", "category", "subcategory",
                              "launch_date", "unit_cost", "list_price"]]
    return sku_master


def build_sales_daily(raw_df, meta_df, sku_master):
    """
    Convert store-level weekly data into daily SKU-level rows.
    Each weekly total is split across 7 days using random weights.
    """
    merged = raw_df.merge(meta_df[["store", "sku_id"]], left_on="Store", right_on="store")
    merged = merged.merge(sku_master[["sku_id", "list_price"]], on="sku_id")
    merged["date"] = pd.to_datetime(merged["Date"], errors='coerce', dayfirst=True)

    rows = []
    for _, row in merged.iterrows():
        week_start = row["date"]
        if pd.isna(week_start):
            continue
        weekly_sales = max(0, row["Weekly_Sales"])
        is_holiday = int(row["Holiday_Flag"])
        list_price = row["list_price"]
        sku = row["sku_id"]

        weights = np.random.dirichlet(np.ones(7))
        daily_revenues = weekly_sales * weights

        for d in range(7):
            day_date = week_start + pd.Timedelta(days=d)
            rev = round(daily_revenues[d], 2)
            promo = 1 if (is_holiday and np.random.random() < 0.5) or np.random.random() < 0.12 else 0
            discount = np.random.choice([1.0, 0.90, 0.85], p=[0.70, 0.20, 0.10]) if promo else 1.0
            unit_price = round(list_price * discount, 2)
            units = max(1, int(round(rev / unit_price))) if unit_price > 0 else 1

            rows.append({
                "date": day_date,
                "sku_id": sku,
                "units_sold": units,
                "revenue": round(units * unit_price, 2),
                "unit_price": unit_price,
                "promo_flag": promo,
            })

    sales = pd.DataFrame(rows)
    sales = sales.groupby(["date", "sku_id"]).agg({
        "units_sold": "sum",
        "revenue": "sum",
        "unit_price": "mean",
        "promo_flag": "max",
    }).reset_index()
    sales["unit_price"] = sales["unit_price"].round(2)
    sales = sales.sort_values(["date", "sku_id"]).reset_index(drop=True)
    return sales


def build_calendar(sales_daily):
    """Create calendar.csv spanning the full date range."""
    dates = pd.date_range(sales_daily["date"].min(), sales_daily["date"].max())
    cal = pd.DataFrame({"date": dates})
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["season"] = cal["month"].apply(_get_season)

    cal["is_holiday"] = np.random.choice([0, 1], size=len(cal), p=[0.96, 0.04])
    cal["promo_event"] = "None"
    cal.loc[cal["is_holiday"] == 1, "promo_event"] = "Holiday Sale"
    clearance_mask = (cal["promo_event"] == "None") & (np.random.random(len(cal)) < 0.08)
    cal.loc[clearance_mask, "promo_event"] = "Clearance"
    return cal


def build_inventory(sales_daily):
    """
    Create inventory_snapshots.csv – 30 trailing daily snapshots
    for the top-200 volume SKUs.
    """
    max_date = sales_daily["date"].max()
    snapshot_dates = pd.date_range(end=max_date, periods=30)
    top_skus = (
        sales_daily.groupby("sku_id")["units_sold"]
        .sum()
        .nlargest(200)
        .index
    )

    records = []
    for d in snapshot_dates:
        for sku in top_skus:
            avg_daily = np.random.uniform(2, 15)
            lead_time = np.random.randint(7, 21)
            reorder_pt = int(avg_daily * lead_time * 1.5)
            on_hand = np.random.randint(0, reorder_pt * 3)
            on_order = reorder_pt * 2 if on_hand < reorder_pt else 0
            records.append({
                "date": d,
                "sku_id": sku,
                "on_hand_units": on_hand,
                "on_order_units": on_order,
                "lead_time_days": lead_time,
                "reorder_point": reorder_pt,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline():
    """
    Full ETL pipeline:
      1. Download or synthesize Walmart dataset
      2. Build all four required tables
      3. Save to data/ folder
      4. Return dict of DataFrames
    """
    raw = _download_data()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    raw = raw.dropna(subset=["Date"])
    raw = raw.drop_duplicates()

    stores = raw["Store"].unique()
    meta = _assign_sku_metadata(stores)

    print("[Pipeline] Building sku_master ...")
    sku_master = build_sku_master(meta, raw)

    print("[Pipeline] Building sales_daily ...")
    sales_daily = build_sales_daily(raw, meta, sku_master)

    print("[Pipeline] Building calendar ...")
    calendar = build_calendar(sales_daily)

    print("[Pipeline] Building inventory snapshots ...")
    inventory = build_inventory(sales_daily)

    os.makedirs(DATA_DIR, exist_ok=True)
    sku_master.to_csv(os.path.join(DATA_DIR, "sku_master.csv"), index=False)
    sales_daily.to_csv(os.path.join(DATA_DIR, "sales_daily.csv"), index=False)
    calendar.to_csv(os.path.join(DATA_DIR, "calendar.csv"), index=False)
    inventory.to_csv(os.path.join(DATA_DIR, "inventory_snapshots.csv"), index=False)
    print("[Pipeline] All CSVs saved to data/")

    return {
        "sku_master": sku_master,
        "sales_daily": sales_daily,
        "calendar": calendar,
        "inventory": inventory,
    }


if __name__ == "__main__":
    run_pipeline()
