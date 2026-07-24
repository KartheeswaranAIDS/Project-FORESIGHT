# Project FORESIGHT – AI Demand Forecasting & Inventory Intelligence

> **Zidio Internship Project** – An AI-powered inventory planning system that forecasts SKU demand, detects stockout and overstock risks, and provides actionable reorder or markdown recommendations through an interactive Streamlit dashboard.

---

## Features

| Module | Description |
|--------|-------------|
| **Data Pipeline** | Auto-downloads [Walmart.csv](https://github.com/selva86/datasets/raw/master/Walmart.csv) and transforms it into `sales_daily.csv`, `sku_master.csv`, `calendar.csv`, `inventory_snapshots.csv` |
| **Demand Forecasting** | LightGBM Regressor with lag features, rolling means, calendar features, and rolling-origin time-series validation |
| **Risk Intelligence** | Transparent business rules for stockout/overstock detection with financial impact in ₹ |
| **Streamlit Dashboard** | 4-tab professional UI: Executive Overview, Forecast Explorer, Risk Intelligence, Action Center |

## Dataset

**Source:** [Walmart.csv](https://github.com/selva86/datasets/raw/master/Walmart.csv) (public retail dataset)

The pipeline automatically downloads and transforms this store-level weekly sales dataset into SKU-level daily data across 5 categories (Grocery, Electronics, Apparel, Home, Sports).

## Architecture

```
Project FORESIGHT/
├── app.py                  # Streamlit dashboard (4 tabs)
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Data download + ETL → 4 CSVs
│   ├── forecast.py         # LightGBM + WAPE + rolling-origin CV
│   └── risk.py             # Risk scoring + financial impact
├── data/                   # Auto-generated CSV files
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Internet connection (for first-time dataset download)

### Local Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd FORESIGHT

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app (everything runs automatically)
streamlit run app.py
```

The app will:
1. Download Walmart.csv from GitHub
2. Transform it into 4 CSV tables
3. Train LightGBM forecasting models
4. Compute risk scores
5. Launch the interactive dashboard

## Model Evaluation

| Metric | Value |
|--------|-------|
| **LightGBM WAPE** | Computed live on each run |
| **Seasonal-Naive Baseline** | 4-week lag baseline |
| **Validation** | 3-fold rolling-origin time-series cross-validation |

**WAPE** (Weighted Absolute Percentage Error) = Σ|actual − forecast| / Σ|actual|

The LightGBM model consistently outperforms the seasonal-naive baseline, demonstrating the value of ML-based demand forecasting over simple heuristics.

## Risk Scoring Rules

| Risk | Condition | Action |
|------|-----------|--------|
| **Stockout HIGH** | Lead-time demand > Available stock | REORDER NOW |
| **Overstock HIGH** | On-hand > 1.5 × 8-week forecast | MARKDOWN / CLEAR |
| **Borderline** | Available stock < 1.2 × lead-time demand | WATCH |
| **Normal** | No risk flags | HEALTHY |

### Financial Impact
- `sales_at_risk_rupees` = lost units × list price × 83 (USD→INR)
- `locked_capital_rupees` = excess units × unit cost × 83 (USD→INR)

## Deployment (Streamlit Community Cloud)

1. Push this entire project folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **"New app"**
4. Select your repository, branch, and set the main file path to `app.py`
5. Click **"Deploy"** – the application will be hosted publicly for free

No credit card required.

## Business Impact

The dashboard provides immediate visibility into critical inventory decisions:

- 📦 **Prevents Stockouts** – Pinpoints SKUs requiring immediate reorders to protect revenue
- 💰 **Frees Working Capital** – Identifies capital locked in overstocked items for targeted markdowns
- ⏱️ **Saves Time** – Automates demand forecasting that would take days of manual spreadsheet work
- 📊 **Data-Driven Decisions** – Replaces gut-feel inventory planning with ML-backed intelligence

---

*Built for the Zidio Project FORESIGHT Internship Evaluation*
