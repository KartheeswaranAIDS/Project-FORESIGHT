"""
forecast.py – Demand Forecasting Engine for Project FORESIGHT
Uses LightGBM with rolling-origin time-series validation.
Computes WAPE metric and compares against a seasonal-naive baseline.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def prepare_weekly(sales_df, calendar_df):
    """Aggregate daily sales to weekly level and attach calendar features."""
    sales_df = sales_df.copy()
    calendar_df = calendar_df.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    merged = sales_df.merge(calendar_df[["date", "week", "is_holiday"]], on="date", how="left")
    merged["year"] = merged["date"].dt.isocalendar().year.astype(int)

    weekly = (
        merged.groupby(["sku_id", "year", "week"])
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            is_holiday=("is_holiday", "max"),
            promo_flag=("promo_flag", "max"),
            date=("date", "max"),
        )
        .reset_index()
    )
    weekly = weekly.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return weekly


def add_features(df):
    """Add lag and rolling-mean features per SKU."""
    df = df.copy()
    df = df.sort_values(["sku_id", "date"])

    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}"] = df.groupby("sku_id")["units_sold"].shift(lag)

    for win in [4, 8]:
        df[f"rolling_mean_{win}"] = df.groupby("sku_id")["units_sold"].transform(
            lambda x: x.shift(1).rolling(window=win, min_periods=1).mean()
        )

    df["month"] = df["date"].dt.month
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4"])
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / total


# ---------------------------------------------------------------------------
# Rolling-Origin Cross-Validation
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4",
    "rolling_mean_4", "rolling_mean_8",
    "is_holiday", "promo_flag", "week", "month",
]


def rolling_origin_cv(df, n_splits=3, horizon=4):
    """
    Expanding-window time-series cross-validation.
    Returns (model_wape, baseline_wape) averaged across folds.
    """
    sku_ids = df["sku_id"].unique()
    fold_wapes_model = []
    fold_wapes_baseline = []

    # Global sorted dates
    sorted_dates = sorted(df["date"].unique())
    n_dates = len(sorted_dates)
    if n_dates < n_splits + horizon + 4:
        # Dataset too short for proper CV – fallback
        return None, None

    step = max(1, (n_dates - horizon - 4) // (n_splits + 1))

    for fold_i in range(n_splits):
        cutoff_idx = 4 + step * (fold_i + 1)
        if cutoff_idx + horizon > n_dates:
            break
        cutoff_date = sorted_dates[cutoff_idx]
        test_end_date = sorted_dates[min(cutoff_idx + horizon - 1, n_dates - 1)]

        train = df[df["date"] <= cutoff_date]
        test = df[(df["date"] > cutoff_date) & (df["date"] <= test_end_date)]

        if len(train) < 10 or len(test) < 1:
            continue

        X_train, y_train = train[FEATURE_COLS], train["units_sold"]
        X_test, y_test = test[FEATURE_COLS], test["units_sold"]

        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = np.maximum(0, model.predict(X_test))
        fold_wapes_model.append(wape(y_test.values, preds))

        # Seasonal-naive baseline: predict = lag_4 (same week 4 weeks ago)
        baseline_preds = test["lag_4"].values
        fold_wapes_baseline.append(wape(y_test.values, baseline_preds))

    if not fold_wapes_model:
        return None, None

    return np.mean(fold_wapes_model), np.mean(fold_wapes_baseline)


# ---------------------------------------------------------------------------
# Train Final Model & Generate Forecasts
# ---------------------------------------------------------------------------

def train_and_forecast(df, weeks_ahead=8):
    """
    Train on all available data and produce 8-week-ahead forecasts per SKU.
    Returns DataFrame with columns:
        sku_id, forecast_date, forecast_demand, ci_lower, ci_upper
    """
    X_all, y_all = df[FEATURE_COLS], df["units_sold"]
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1,
    )
    model.fit(X_all, y_all)

    # Residual std for confidence interval
    train_preds = model.predict(X_all)
    residual_std = np.std(y_all.values - train_preds)

    # Per-SKU recursive forecast
    last_known = df.groupby("sku_id").last().reset_index()
    forecasts = []

    for _, row in last_known.iterrows():
        sku = row["sku_id"]
        cur_date = row["date"]
        hist = [
            row["lag_4"], row["lag_3"], row["lag_2"], row["lag_1"],
            row["units_sold"],
        ]

        for i in range(1, weeks_ahead + 1):
            next_date = cur_date + pd.Timedelta(weeks=i)
            next_week = int(next_date.isocalendar().week)
            next_month = next_date.month

            feat = pd.DataFrame([{
                "lag_1": hist[-1],
                "lag_2": hist[-2],
                "lag_3": hist[-3],
                "lag_4": hist[-4],
                "rolling_mean_4": np.mean(hist[-4:]),
                "rolling_mean_8": np.mean(hist[-min(8, len(hist)):]),
                "is_holiday": 0,
                "promo_flag": 0,
                "week": next_week,
                "month": next_month,
            }])
            pred = max(0, round(model.predict(feat[FEATURE_COLS])[0]))

            ci_lower = max(0, int(pred - 1.28 * residual_std))  # ~80 % CI
            ci_upper = int(pred + 1.28 * residual_std)

            forecasts.append({
                "sku_id": sku,
                "forecast_date": next_date,
                "forecast_demand": pred,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })
            hist.append(pred)

    return pd.DataFrame(forecasts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_forecasting(sales_df, calendar_df):
    """
    End-to-end forecasting:
      1. Weekly aggregation + feature engineering
      2. Rolling-origin CV  → model WAPE & baseline WAPE
      3. Final model training + 8-week forecast
    Returns (weekly_df, forecast_df, model_wape, baseline_wape)
    """
    weekly = prepare_weekly(sales_df, calendar_df)
    feat_df = add_features(weekly)

    model_wape_val, baseline_wape_val = rolling_origin_cv(feat_df)
    forecast_df = train_and_forecast(feat_df)

    # Fallback if CV didn't run
    if model_wape_val is None:
        model_wape_val = 0.0
        baseline_wape_val = 0.0

    return weekly, forecast_df, round(model_wape_val, 4), round(baseline_wape_val, 4)


if __name__ == "__main__":
    pass
