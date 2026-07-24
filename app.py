"""
app.py – Project FORESIGHT Dashboard (Pre-loaded Main UI)
AI Demand Forecasting & Inventory Intelligence
Zidio Internship – Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from src.pipeline import run_pipeline
from src.forecast import run_forecasting
from src.risk import run_risk_scoring

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG (No Sidebar, Main Dashboard Layout)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Project FORESIGHT – Inventory Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS – Modern Light SaaS Theme (Clean Main Layout)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Sidebar Completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Expand Main Container */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1350px !important;
    }

    /* Main Canvas Background */
    .stApp {
        background-color: #f4f6f9;
        color: #1e293b;
    }

    /* ── Top Header Navigation Bar ────────────────── */
    .top-header {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .brand-badge {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-icon {
        background: #165dfc;
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.1rem;
        padding: 8px 12px;
        border-radius: 8px;
    }
    .page-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .ai-search-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 8px 18px;
        font-size: 0.88rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 8px;
        width: 320px;
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .status-pill {
        background: #f1f5f9;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .progress-bar-fill {
        height: 6px;
        width: 80px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-bar-inner {
        height: 100%;
        background: #165dfc;
        border-radius: 3px;
    }
    .user-badge {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #165dfc;
        color: #ffffff;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
    }

    /* ── Stepper Navigation Tabs (Top Main View) ──── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        padding: 0 0 18px 0;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 30px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #64748b !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #cbd5e1 !important;
        color: #1e293b !important;
    }
    .stTabs [aria-selected="true"] {
        background: #165dfc !important;
        color: #ffffff !important;
        border-color: #165dfc !important;
        box-shadow: 0 4px 14px rgba(22, 93, 252, 0.25) !important;
    }

    /* ── Main Container White Cards ──────────────── */
    .main-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        margin-bottom: 24px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .card-subtitle {
        font-size: 0.86rem;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* ── KPI Metric Cards ───────────────────────── */
    .kpi-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .kpi-card {
        flex: 1;
        min-width: 170px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Status Accent Colors */
    .clr-blue   { color: #165dfc !important; }
    .clr-green  { color: #10b981 !important; }
    .clr-red    { color: #ef4444 !important; }
    .clr-orange { color: #f59e0b !important; }

    /* ── Recommendation Banner ───────────────────── */
    .rec-banner {
        background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
        border: 1px solid #bfdbfe;
        border-left: 5px solid #165dfc;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .rec-banner-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .rec-banner-body {
        font-size: 0.95rem;
        color: #1e40af;
        line-height: 1.55;
    }

    /* Select Inputs */
    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }
    .stSelectbox label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# AUTOMATIC DATA INITIALIZATION (ALREADY LOADED / CACHED)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all_data():
    """Auto-run pipeline, forecasting, and risk scoring on app start."""
    pipeline_data = run_pipeline()
    weekly, forecasts, m_wape, b_wape = run_forecasting(
        pipeline_data["sales_daily"], pipeline_data["calendar"]
    )
    risk_df, health = run_risk_scoring(
        forecasts, pipeline_data["inventory"], pipeline_data["sku_master"]
    )
    
    sales = pipeline_data["sales_daily"].merge(
        pipeline_data["sku_master"][["sku_id", "category", "subcategory", "product_name"]],
        on="sku_id", how="left",
    )
    forecasts = forecasts.merge(
        pipeline_data["sku_master"][["sku_id", "category", "subcategory"]],
        on="sku_id", how="left",
    )

    return {
        "sales": sales,
        "weekly": weekly,
        "forecasts": forecasts,
        "risk_df": risk_df,
        "health": health,
        "model_wape": m_wape,
        "baseline_wape": b_wape,
        "sku_master": pipeline_data["sku_master"],
        "total_revenue": sales["revenue"].sum(),
    }


# Automatically load data immediately on launch
data = load_all_data()

sales          = data["sales"]
forecasts      = data["forecasts"]
risk_df        = data["risk_df"]
health         = data["health"]
m_wape         = data["model_wape"]
b_wape         = data["baseline_wape"]
total_rev      = data["total_revenue"]
total_skus     = risk_df["sku_id"].nunique()
total_forecast = risk_df["demand_next_8_weeks"].sum()


# ═══════════════════════════════════════════════════════════════════
# TOP HEADER BAR (Main Area)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-header">
    <div class="header-left">
        <div class="brand-badge">
            <div class="brand-icon">FM</div>
            <div>
                <h1 class="page-title">Project FORESIGHT</h1>
                <div style="font-size:0.75rem; color:#64748b; font-weight:500;">AI Demand Forecasting & Inventory Intelligence</div>
            </div>
        </div>
        <div class="ai-search-box" style="margin-left: 20px;">
            <span>✨</span>
            <span>Ask AI: 'Show audition & inventory previews'</span>
        </div>
    </div>
    <div class="header-right">
        <div class="status-pill">
            <span>98% Model Health</span>
            <div class="progress-bar-fill">
                <div class="progress-bar-inner"></div>
            </div>
        </div>
        <div class="user-badge">
            <div class="user-avatar">LT</div>
            <div>
                <div style="font-size:0.85rem; font-weight:700; color:#0f172a;">Lillian Tom</div>
                <div style="font-size:0.72rem; color:#64748b;">Company Admin</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# STEPPER TAB NAVIGATION (Main View)
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Audition / Executive Info",
    "2. Demand Forecast Explorer",
    "3. Risk Intelligence Matrix",
    "4. Action Center"
])


# ───────────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE OVERVIEW
# ───────────────────────────────────────────────────────────────────
with tab1:
    high_stockout = len(risk_df[risk_df["stockout_risk"] == "HIGH"])
    high_overstock = len(risk_df[risk_df["overstock_risk"] == "HIGH"])

    # KPI Row
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value clr-blue">₹{total_rev:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">SKUs Tracked</div>
            <div class="kpi-value">{total_skus}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">8-Wk Demand</div>
            <div class="kpi-value">{total_forecast:,.0f} <span style="font-size:0.7em; color:#64748b">units</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Stockout Risk</div>
            <div class="kpi-value clr-red">{high_stockout}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Overstock Risk</div>
            <div class="kpi-value clr-orange">{high_overstock}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Health Score</div>
            <div class="kpi-value clr-green">{health}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation Banner
    rev_protected = risk_df[risk_df["action"] == "REORDER NOW"]["sales_at_risk_rupees"].sum()
    cash_locked = risk_df[risk_df["action"] == "MARKDOWN / CLEAR"]["locked_capital_rupees"].sum()
    reorder_n = len(risk_df[risk_df["action"] == "REORDER NOW"])
    markdown_n = len(risk_df[risk_df["action"] == "MARKDOWN / CLEAR"])

    st.markdown(f"""
    <div class="rec-banner">
        <div class="rec-banner-title">
            <span>✨</span>
            <span>AI Automated Recommendation</span>
        </div>
        <div class="rec-banner-body">
            Reorder inventory for <b>{reorder_n} high-demand SKUs</b> immediately to protect <b>₹{rev_protected / 100000:.2f} Lakh</b> in sales. 
            Initiate clearance markdowns for <b>{markdown_n} overstocked SKUs</b> to release <b>₹{cash_locked / 100000:.2f} Lakh</b> in working capital.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts Grid
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        <div class="main-card">
            <div class="card-title">Risk Distribution by Category</div>
            <div class="card-subtitle">Revenue at risk across major retail sub-categories</div>
        """, unsafe_allow_html=True)
        
        cat_rev = risk_df.groupby("category")["sales_at_risk_rupees"].sum().reset_index()
        fig_pie = px.pie(
            cat_rev, values="sales_at_risk_rupees", names="category",
            hole=0.5, color_discrete_sequence=["#165dfc", "#ef4444", "#10b981", "#8b5cf6", "#f59e0b"]
        )
        fig_pie.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div class="main-card">
            <div class="card-title">Model Accuracy Evaluation</div>
            <div class="card-subtitle">LightGBM Machine Learning vs Seasonal-Naive Baseline</div>
        """, unsafe_allow_html=True)

        st.write("")
        imp = round((b_wape - m_wape) / b_wape * 100, 1) if b_wape > 0 else 0
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("LightGBM WAPE", f"{m_wape:.4f}", delta=f"{imp}% improvement")
        with col_m2:
            st.metric("Baseline WAPE", f"{b_wape:.4f}")

        st.caption("Lower WAPE (Weighted Absolute Percentage Error) indicates higher demand forecasting precision.")
        st.markdown("</div>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# TAB 2: DEMAND FORECAST EXPLORER
# ───────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="main-card">
        <div class="card-title">SKU Demand Forecasting Explorer</div>
        <div class="card-subtitle">Historical sales trends vs 8-week AI forecast trajectory</div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([1, 3])

    with f_col1:
        cats = ["All"] + sorted(sales["category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category Filter", cats, key="tab2_cat")

        if sel_cat != "All":
            avail_skus = sales[sales["category"] == sel_cat]["sku_id"].unique()
        else:
            avail_skus = sales["sku_id"].unique()

        sel_sku = st.selectbox("Select SKU ID", sorted(avail_skus), key="tab2_sku")

        sku_info = risk_df[risk_df["sku_id"] == sel_sku]
        if not sku_info.empty:
            r = sku_info.iloc[0]
            st.write("---")
            st.write(f"**Product**: {r['product_name']}")
            st.write(f"**Current Stock**: {r['on_hand_units']:,.0f} units")
            st.write(f"**Lead Time**: {r['lead_time_days']:.0f} days")
            st.write(f"**Lead-Time Demand**: {r['lead_time_demand']:,.0f} units")
            st.write(f"**Action**: `{r['action']}`")

    with f_col2:
        sku_hist = (
            sales[sales["sku_id"] == sel_sku]
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .groupby(pd.Grouper(key="date", freq="W-SUN"))["units_sold"]
            .sum()
            .reset_index()
        )
        sku_fcst = forecasts[forecasts["sku_id"] == sel_sku].copy()

        fig_chart = go.Figure()

        if not sku_hist.empty:
            fig_chart.add_trace(go.Scatter(
                x=sku_hist["date"], y=sku_hist["units_sold"],
                mode="lines+markers", name="Historical Sales",
                line=dict(color="#165dfc", width=2.5),
                marker=dict(size=4)
            ))

        if not sku_fcst.empty:
            if not sku_hist.empty:
                lh = sku_hist.iloc[-1]
                ff = sku_fcst.iloc[0]
                fig_chart.add_trace(go.Scatter(
                    x=[lh["date"], ff["forecast_date"]],
                    y=[lh["units_sold"], ff["forecast_demand"]],
                    mode="lines", showlegend=False,
                    line=dict(color="#ef4444", width=2, dash="dash")
                ))

            fig_chart.add_trace(go.Scatter(
                x=sku_fcst["forecast_date"], y=sku_fcst["forecast_demand"],
                mode="lines+markers", name="AI Forecast",
                line=dict(color="#ef4444", width=3, dash="dash"),
                marker=dict(size=5)
            ))

            fig_chart.add_trace(go.Scatter(
                x=pd.concat([sku_fcst["forecast_date"], sku_fcst["forecast_date"][::-1]]),
                y=pd.concat([sku_fcst["ci_upper"], sku_fcst["ci_lower"][::-1]]),
                fill="toself", fillcolor="rgba(239, 68, 68, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% Confidence Band"
            ))

        fig_chart.update_layout(
            template="plotly_white",
            xaxis_title="", yaxis_title="Units Sold",
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# TAB 3: RISK INTELLIGENCE
# ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="main-card">
        <div class="card-title">Risk Matrix & Inventory Intelligence Table</div>
        <div class="card-subtitle">Automated classification of stockouts, overstock, and monetary impact</div>
    """, unsafe_allow_html=True)

    def _style_action(val):
        colors = {
            "REORDER NOW": "#ef4444",
            "MARKDOWN / CLEAR": "#8b5cf6",
            "WATCH": "#f59e0b",
            "HEALTHY": "#10b981"
        }
        return f"color: {colors.get(val, '#334155')}; font-weight: 700;"

    def _style_risk(val):
        return "color: #ef4444; font-weight: 700;" if val == "HIGH" else "color: #10b981;"

    display_cols = [
        "sku_id", "product_name", "on_hand_units", "demand_next_8_weeks",
        "stockout_risk", "overstock_risk", "action",
        "sales_at_risk_rupees", "locked_capital_rupees"
    ]
    renames = {
        "sku_id": "SKU Code",
        "product_name": "Product Name",
        "on_hand_units": "Stock On Hand",
        "demand_next_8_weeks": "Forecast 8W",
        "stockout_risk": "Stockout Risk",
        "overstock_risk": "Overstock Risk",
        "action": "Action Recommendation",
        "sales_at_risk_rupees": "Revenue at Risk (₹)",
        "locked_capital_rupees": "Capital Locked (₹)"
    }

    disp_df = risk_df[display_cols].copy().rename(columns=renames)

    styled_table = (
        disp_df.style
        .map(_style_action, subset=["Action Recommendation"])
        .map(_style_risk, subset=["Stockout Risk", "Overstock Risk"])
        .format({
            "Revenue at Risk (₹)": "₹{:,.0f}",
            "Capital Locked (₹)": "₹{:,.0f}",
            "Stock On Hand": "{:,.0f}",
            "Forecast 8W": "{:,.0f}"
        })
    )

    st.dataframe(styled_table, use_container_width=True, height=480)

    st.download_button(
        "📥 Download Full Risk Table (CSV)",
        data=risk_df.to_csv(index=False),
        file_name="risk_matrix.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# TAB 4: ACTION CENTER
# ───────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="main-card">
        <div class="card-title">Inventory Action Center</div>
        <div class="card-subtitle">Prioritized operational lists ordered by revenue impact</div>
    """, unsafe_allow_html=True)

    with st.expander("🔴 REORDER NOW (Immediate Purchase Orders)", expanded=True):
        reorders = risk_df[risk_df["action"] == "REORDER NOW"].sort_values("sales_at_risk_rupees", ascending=False)
        if reorders.empty:
            st.success("No SKUs currently require immediate reorders.")
        else:
            st.dataframe(
                reorders[["sku_id", "product_name", "on_hand_units", "demand_next_8_weeks", "sales_at_risk_rupees"]].rename(columns={
                    "sku_id": "SKU", "product_name": "Product", "on_hand_units": "Stock",
                    "demand_next_8_weeks": "Forecast 8W", "sales_at_risk_rupees": "Sales at Risk (₹)"
                }),
                use_container_width=True
            )
            st.download_button("Download Reorder List (CSV)", reorders.to_csv(index=False), "reorder_plan.csv", "text/csv")

    with st.expander("🟣 MARKDOWN / CLEAR (Clearance & Promotion Plan)"):
        markdowns = risk_df[risk_df["action"] == "MARKDOWN / CLEAR"].sort_values("locked_capital_rupees", ascending=False)
        if markdowns.empty:
            st.info("No SKUs currently flagged for clearance.")
        else:
            st.dataframe(
                markdowns[["sku_id", "product_name", "on_hand_units", "demand_next_8_weeks", "locked_capital_rupees"]].rename(columns={
                    "sku_id": "SKU", "product_name": "Product", "on_hand_units": "Stock",
                    "demand_next_8_weeks": "Forecast 8W", "locked_capital_rupees": "Capital Tied (₹)"
                }),
                use_container_width=True
            )
            st.download_button("Download Markdown List (CSV)", markdowns.to_csv(index=False), "markdown_plan.csv", "text/csv")

    with st.expander("🟠 WATCH LIST (Borderline Stock Levels)"):
        watches = risk_df[risk_df["action"] == "WATCH"].sort_values("sales_at_risk_rupees", ascending=False)
        st.dataframe(watches[["sku_id", "product_name", "on_hand_units", "demand_next_8_weeks"]], use_container_width=True)

    with st.expander("🟢 HEALTHY INVENTORY"):
        healthies = risk_df[risk_df["action"] == "HEALTHY"]
        st.dataframe(healthies[["sku_id", "product_name", "on_hand_units", "demand_next_8_weeks"]], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
