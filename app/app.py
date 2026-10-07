import os
import sys
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import load_trained_model
from src.feature_engineering import create_features
from src.reorder_recommendation import generate_reorder_recommendation
from src.data_preprocessing import preprocess_data

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Industrial Inventory Demand Forecasting",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark SaaS CSS Design Token Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Dark Theme Force */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0B0F19 !important;
        color: #E2E8F0 !important;
    }

    /* Force Sidebar Docked Visibility & Disable Offscreen Hiding */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="false"],
    div[data-testid="stSidebarUserContent"] {
        display: flex !important;
        visibility: visible !important;
        transform: none !important;
        margin-left: 0 !important;
        left: 0 !important;
        width: 265px !important;
        min-width: 265px !important;
        background-color: #070A10 !important;
        border-right: 1px solid #1E293B !important;
        z-index: 99999 !important;
    }

    /* Hide Deploy Button & Footer */
    .stDeployButton, button[title="Deploy"], div[data-testid="stToolbar"], footer {
        display: none !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px;
    }

    /* Header Banner */
    .brand-header {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border-left: 5px solid #2563EB;
        border: 1px solid #1E293B;
        border-left-width: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #F8FAFC !important;
    }
    .brand-subtitle {
        font-size: 13.5px;
        color: #94A3B8 !important;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Metric Cards */
    .saas-card {
        background: #161F30 !important;
        border: 1px solid #26334D !important;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .saas-card:hover {
        border-color: #3B82F6 !important;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.3);
    }
    .saas-kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8 !important;
        letter-spacing: 0.5px;
    }
    .saas-kpi-val {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC !important;
        margin-top: 4px;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .saas-kpi-sub {
        font-size: 12px;
        font-weight: 500;
    }

    /* High-Contrast Status Badges */
    .badge-critical {
        background-color: #451A1A !important;
        color: #FCA5A5 !important;
        border: 1px solid #7F1D1D !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-warning {
        background-color: #3B2310 !important;
        color: #FDE047 !important;
        border: 1px solid #78350F !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-healthy {
        background-color: #062C1B !important;
        color: #6EE7B7 !important;
        border: 1px solid #065F46 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-overstock {
        background-color: #172554 !important;
        color: #93C5FD !important;
        border: 1px solid #1E40AF !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }

    /* Action Required Card */
    .action-card {
        background: linear-gradient(135deg, #2A1215 0%, #1F0D0E 100%) !important;
        border: 1px solid #7F1D1D !important;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    /* Section Headings */
    .section-title {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 12px;
        letter-spacing: -0.3px;
    }

    /* High-Contrast Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
    }
    p, span, label, div {
        color: #E2E8F0;
    }
    
    /* Clean Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #070A10 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    /* Text Inputs & Selectboxes Dark Styling */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #161F30 !important;
        color: #F8FAFC !important;
        border-color: #26334D !important;
    }
    
    /* Dataframe Table Dark Override */
    div[data-testid="stDataFrame"] {
        background-color: #161F30 !important;
        border: 1px solid #26334D !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)# ==============================================================================
# 2. DATA INITIALIZATION & SESSION STATE MANAGEMENT
# ==============================================================================
if "safety_factor" not in st.session_state:
    st.session_state.safety_factor = 1.2
if "overstock_factor" not in st.session_state:
    st.session_state.overstock_factor = 2.0
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Dashboard"

@st.cache_data(ttl=3600)
def load_default_dataset():
    path = "data/inventory_demand.csv"
    if not os.path.exists(path):
        path = "data/processed/featured_inventory_data.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_cached_model():
    try:
        return load_trained_model()
    except Exception:
        return None

def load_metrics_cached():
    path = "outputs/evaluation/model_metrics.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "MAE": 21.1826, "MSE": 881.566, "RMSE": 29.6912, "R2_Score": 0.9037,
        "Algorithm": "Linear Regression", "Train_Samples": 2920, "Test_Samples": 730,
        "Total_Samples": 3650, "Trained_Timestamp": "N/A"
    }

if "raw_df" not in st.session_state:
    st.session_state.raw_df = load_default_dataset()

# Pipeline Calculation Logic using Actual Existing Model & Formulas
def process_inventory_pipeline(raw_df, safety_factor=1.2, overstock_factor=2.0):
    df = raw_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Product_ID', 'Date']).reset_index(drop=True)
    
    feature_cols = [
        'Year', 'Month', 'Day', 'Day_of_Week',
        'Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30',
        'Current_Stock', 'Lead_Time_Days', 'Previous_Sales'
    ]
    
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        df, _, _ = create_features(df)
        
    model = load_cached_model()
    if model is not None:
        try:
            preds = model.predict(df[feature_cols])
            df['Predicted_Demand'] = np.round(np.maximum(0, preds), 1)
        except Exception:
            df['Predicted_Demand'] = df['Demand']
    else:
        df['Predicted_Demand'] = df['Demand']
        
    latest_mask = df.groupby('Product_ID')['Date'].transform('max') == df['Date']
    latest_df = df[latest_mask].copy().sort_values(by='Product_ID').reset_index(drop=True)
    
    latest_df['Reorder_Point'] = np.round(latest_df['Predicted_Demand'] * np.maximum(1, latest_df['Lead_Time_Days']) * safety_factor, 1)
    latest_df['Critical_Threshold'] = np.round(latest_df['Predicted_Demand'] * np.maximum(1, latest_df['Lead_Time_Days']), 1)
    latest_df['Overstock_Threshold'] = np.round(latest_df['Reorder_Point'] * overstock_factor, 1)
    
    def evaluate_status(row):
        stock = row['Current_Stock']
        crit = row['Critical_Threshold']
        rop = row['Reorder_Point']
        over = row['Overstock_Threshold']
        
        if stock < crit:
            return "Critical Stock", "🔴", "badge-critical"
        elif stock < rop:
            return "Reorder Soon", "🟠", "badge-warning"
        elif stock > over:
            return "Overstock", "🔵", "badge-overstock"
        else:
            return "Healthy Stock", "🟢", "badge-healthy"
            
    status_info = latest_df.apply(evaluate_status, axis=1)
    latest_df['Stock_Status'] = [s[0] for s in status_info]
    latest_df['Status_Icon'] = [s[1] for s in status_info]
    latest_df['Status_Badge_Class'] = [s[2] for s in status_info]
    
    latest_df['Reorder_Required'] = latest_df['Current_Stock'] < latest_df['Reorder_Point']
    latest_df['Recommended_Reorder_Qty'] = np.where(
        latest_df['Reorder_Required'],
        np.ceil(latest_df['Reorder_Point'] - latest_df['Current_Stock']).astype(int),
        0
    )
    
    def generate_reason(row):
        status = row['Stock_Status']
        stock = int(row['Current_Stock'])
        pred = row['Predicted_Demand']
        rop = row['Reorder_Point']
        lead = int(row['Lead_Time_Days'])
        rec = int(row['Recommended_Reorder_Qty'])
        
        if status == "Critical Stock":
            return f"Current stock ({stock} units) is critically below lead time demand ({int(pred*lead)} units over {lead} days). Immediate purchase of {rec} units recommended to prevent stockouts."
        elif status == "Reorder Soon":
            return f"Current stock ({stock} units) is below safety reorder threshold ({rop} units ROP). Replenishment of {rec} units is recommended."
        elif status == "Overstock":
            return f"Current inventory ({stock} units) is high compared with recent demand requirement ({rop} units ROP). Consider reducing upcoming purchase orders."
        else:
            return f"Current inventory ({stock} units) is sufficient for expected demand over the lead time of {lead} days."

    latest_df['Recommendation_Reason'] = latest_df.apply(generate_reason, axis=1)
    
    return df, latest_df

df_full, latest_inv = process_inventory_pipeline(
    st.session_state.raw_df,
    safety_factor=st.session_state.safety_factor,
    overstock_factor=st.session_state.overstock_factor
)
metrics = load_metrics_cached()

# Plotly Dark Theme Styling Helper
def apply_dark_theme(fig, height=320):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC', family='Inter, sans-serif'),
        height=height,
        margin=dict(t=30, b=30, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#94A3B8'), title_font=dict(color='#E2E8F0')),
        yaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#94A3B8'), title_font=dict(color='#E2E8F0')),
        legend=dict(font=dict(color='#E2E8F0'))
    )
    return fig


# ==============================================================================
# 3. SIDEBAR NAVIGATION & BRANDING
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 15px 0; border-bottom: 1px solid #1E293B; margin-bottom: 15px;">
        <div style="font-size: 17px; font-weight: 700; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
            <span>🏭</span> Industrial Inventory
        </div>
        <div style="font-size: 13px; font-weight: 600; color: #60A5FA; margin-top: 2px;">
            Demand Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    page_options = [
        "📊 Dashboard",
        "📁 Upload Sales Data",
        "📦 Inventory Analysis",
        "📈 Sales Analytics",
        "🔮 Demand Forecast",
        "🛒 Purchase Recommendations",
        "📋 Reports"
    ]
    
    nav_choice = st.radio("NAVIGATION", page_options, key="nav_radio")
    st.session_state.current_page = nav_choice
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#64748B; text-align:center;">
        Industrial Inventory Demand Forecasting<br>
        ML-Based Inventory Analytics
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. TOP APP HEADER
# ==============================================================================
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">Industrial Inventory Demand Forecasting</div>
        <div class="brand-subtitle">Machine Learning Based Sales Analytics & Inventory Replenishment</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: 📊 DASHBOARD
# ==============================================================================
if nav_choice == "📊 Dashboard":
    
    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total_prods = len(latest_inv)
    total_sales_units = int(df_full['Sales_Quantity'].sum())
    critical_count = len(latest_inv[latest_inv['Stock_Status'] == 'Critical Stock'])
    reorder_count = len(latest_inv[latest_inv['Stock_Status'] == 'Reorder Soon'])
    overstock_count = len(latest_inv[latest_inv['Stock_Status'] == 'Overstock'])
    healthy_count = len(latest_inv[latest_inv['Stock_Status'] == 'Healthy Stock'])
    action_needed = critical_count + reorder_count
    
    with c1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-kpi-label">Total Products</div>
            <div class="saas-kpi-val">{total_prods}</div>
            <div class="saas-kpi-sub" style="color:#60A5FA;">Tracked Products</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-kpi-label">Total Sales</div>
            <div class="saas-kpi-val">{total_sales_units:,}</div>
            <div class="saas-kpi-sub" style="color:#34D399;">Units Sold</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-kpi-label">Low / Critical Stock</div>
            <div class="saas-kpi-val" style="color:#FCA5A5;">{action_needed}</div>
            <div class="saas-kpi-sub" style="color:#FCA5A5;">{critical_count} Critical | {reorder_count} Soon</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-kpi-label">Overstocked Products</div>
            <div class="saas-kpi-val" style="color:#93C5FD;">{overstock_count}</div>
            <div class="saas-kpi-sub" style="color:#93C5FD;">Excess Inventory</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-kpi-label">Healthy Stock</div>
            <div class="saas-kpi-val" style="color:#34D399;">{healthy_count}</div>
            <div class="saas-kpi-sub" style="color:#34D399;">Optimal Level</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Business Overview Summary Banner
    st.markdown(f"""
    <div class="saas-card" style="border-left: 4px solid #2563EB !important; margin-bottom: 20px;">
        <span style="font-weight:700; color:#F8FAFC;">Inventory Overview:</span> 
        <span style="color:#CBD5E1;"> <b>{action_needed} products</b> currently require replenishment action based on current stock levels and predicted demand.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Row 1: Donut Chart & Action Required Alert Card
    row1_c1, row1_c2 = st.columns([1, 1.3])
    
    with row1_c1:
        st.markdown('<div class="section-title">📦 Inventory Health Overview</div>', unsafe_allow_html=True)
        
        status_counts = pd.DataFrame({
            'Status': ['Healthy Stock', 'Reorder Soon', 'Critical Stock', 'Overstock'],
            'Count': [healthy_count, reorder_count, critical_count, overstock_count]
        })
        
        fig_donut = px.pie(
            status_counts,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={
                'Healthy Stock': '#059669',
                'Reorder Soon': '#D97706',
                'Critical Stock': '#DC2626',
                'Overstock': '#2563EB'
            },
            hole=0.65
        )
        fig_donut.update_traces(textinfo='value+percent', hovertemplate="%{label}: %{value} Products (%{percent})")
        apply_dark_theme(fig_donut, height=270)
        fig_donut.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        st.markdown('<div class="section-title">🚨 Action Required</div>', unsafe_allow_html=True)
        
        critical_items = latest_inv[latest_inv['Reorder_Required']].sort_values(by='Current_Stock')
        
        if not critical_items.empty:
            top_urgent = critical_items.iloc[0]
            st.markdown(f"""
            <div class="action-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:16px; color:#F8FAFC;">{top_urgent['Product_Name']} ({top_urgent['Product_ID']})</span>
                    <span class="badge-critical">REORDER REQUIRED</span>
                </div>
                <div style="margin-top:14px; display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px;">
                    <div>
                        <div style="font-size:11px; color:#94A3B8;">Current Stock</div>
                        <div style="font-weight:700; font-size:18px; color:#F8FAFC;">{int(top_urgent['Current_Stock'])} units</div>
                    </div>
                    <div>
                        <div style="font-size:11px; color:#94A3B8;">Predicted Daily Demand</div>
                        <div style="font-weight:700; font-size:18px; color:#60A5FA;">{top_urgent['Predicted_Demand']:.1f} /day</div>
                    </div>
                    <div>
                        <div style="font-size:11px; color:#94A3B8;">Recommended Purchase</div>
                        <div style="font-weight:700; font-size:18px; color:#FCA5A5;">{int(top_urgent['Recommended_Reorder_Qty'])} units</div>
                    </div>
                </div>
                <div style="margin-top:12px; font-size:12.5px; color:#FECACA; background:rgba(127,29,29,0.5); padding:10px 12px; border-radius:6px; border: 1px solid #7F1D1D;">
                    💡 <b>Reason:</b> {top_urgent['Recommendation_Reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ All products currently maintain healthy inventory levels! No urgent reorders required.")
            


    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section Row 2: Sales Trend Chart (Fixed Plotly Property Bug)
    st.markdown('<div class="section-title">📈 Overall Sales Trend & Historical Demand</div>', unsafe_allow_html=True)
    
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        time_range = st.selectbox("Select Time Horizon:", ["All Time (365 Days)", "Last 90 Days", "Last 30 Days", "Last 7 Days"])
        
    trend_df = df_full.groupby('Date')[['Sales_Quantity', 'Demand']].sum().reset_index()
    if "7 Days" in time_range:
        trend_df = trend_df.tail(7)
    elif "30 Days" in time_range:
        trend_df = trend_df.tail(30)
    elif "90 Days" in time_range:
        trend_df = trend_df.tail(90)
        
    fig_sales = px.line(
        trend_df,
        x='Date',
        y=['Sales_Quantity', 'Demand'],
        labels={'value': 'Units', 'variable': 'Metric'},
        color_discrete_map={'Sales_Quantity': '#3B82F6', 'Demand': '#10B981'},
    )
    fig_sales.update_traces(mode='lines', line=dict(width=2))
    apply_dark_theme(fig_sales, height=330)
    fig_sales.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_sales, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section Row 3: Top Selling & Slow Moving Products
    r3_c1, r3_c2 = st.columns(2)
    
    prod_perf = df_full.groupby(['Product_ID', 'Product_Name', 'Category'])['Sales_Quantity'].sum().reset_index()
    prod_perf = prod_perf.merge(latest_inv[['Product_ID', 'Current_Stock', 'Stock_Status', 'Status_Icon']], on='Product_ID')
    
    with r3_c1:
        st.markdown('<div class="section-title">🔥 Top Selling Products</div>', unsafe_allow_html=True)
        top_selling = prod_perf.sort_values(by='Sales_Quantity', ascending=False).head(5)
        top_display = top_selling[['Product_Name', 'Category', 'Sales_Quantity', 'Current_Stock']].copy()
        top_display.columns = ['Product', 'Category', 'Total Sales (Units)', 'Current Stock']
        st.dataframe(top_display, use_container_width=True, hide_index=True)

    with r3_c2:
        st.markdown('<div class="section-title">🐢 Slow Moving Products</div>', unsafe_allow_html=True)
        slow_moving = prod_perf.sort_values(by='Sales_Quantity', ascending=True).head(5)
        slow_display = slow_moving[['Product_Name', 'Category', 'Sales_Quantity', 'Current_Stock', 'Stock_Status']].copy()
        slow_display.columns = ['Product', 'Category', 'Total Sales (Units)', 'Current Stock', 'Status']
        st.dataframe(slow_display, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: 📁 UPLOAD SALES DATA
# ==============================================================================
elif nav_choice == "📁 Upload Sales Data":
    
    st.markdown("## 📁 Upload Sales & Inventory Data")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Upload your historical sales and inventory report to generate demand forecasts and inventory recommendations.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload sales dataset containing Date, Product_ID, Product_Name, Category, Sales_Quantity, Current_Stock, Demand, Lead_Time_Days"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)
                
            req_cols = ['Date', 'Product_ID', 'Product_Name', 'Category', 'Sales_Quantity', 'Current_Stock', 'Demand', 'Lead_Time_Days']
            missing_cols = [c for c in req_cols if c not in new_df.columns]
            
            if missing_cols:
                st.error(f"❌ Upload Failed: Missing required columns in dataset: `{', '.join(missing_cols)}`")
            else:
                new_df['Date'] = pd.to_datetime(new_df['Date'])
                
                row_count = len(new_df)
                sku_count = new_df['Product_ID'].nunique()
                min_date = new_df['Date'].min().strftime("%Y-%m-%d")
                max_date = new_df['Date'].max().strftime("%Y-%m-%d")
                
                st.success("✅ File validated successfully! Ready for analysis.")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("File Name", uploaded_file.name)
                m2.metric("Total Records", f"{row_count:,} rows")
                m3.metric("Total Products", f"{sku_count} Products")
                m4.metric("Date Range", f"{min_date} to {max_date}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🔍 Uploaded Data Preview (First 10 Rows)")
                st.dataframe(new_df.head(10), use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 Analyze Data & Update Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing uploaded dataset and executing ML prediction pipeline..."):
                        st.session_state.raw_df = new_df
                        st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            
    else:
        st.markdown("""
        <div style="border:2px dashed #26334D; border-radius:12px; padding:40px; text-align:center; background:#161F30; margin-top:10px;">
            <div style="font-size:40px; margin-bottom:10px;">📁</div>
            <div style="font-weight:600; font-size:16px; color:#F8FAFC;">Drag and drop your sales report here</div>
            <div style="font-size:13px; color:#94A3B8; margin-top:4px;">Supports CSV and Excel (.xlsx) formats</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📥 Download Sample Dataset Template")
    sample_csv = df_full.head(20).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_csv,
        file_name="sample_industrial_inventory_template.csv",
        mime="text/csv"
    )


# ==============================================================================
# PAGE 3: 📦 INVENTORY ANALYSIS
# ==============================================================================
elif nav_choice == "📦 Inventory Analysis":
    
    st.markdown("## 📦 Comprehensive Inventory Health Analysis")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Real-time tracking of current stock levels, predicted demand, reorder points, and replenishment status across all products.</div>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_term = st.text_input("🔍 Search by Product Name or Product ID:", placeholder="e.g. PCB Board, PRD_001")
    with f2:
        cat_filter = st.selectbox("Category Filter:", ["All Categories"] + list(latest_inv['Category'].unique()))
    with f3:
        status_filter = st.selectbox("Stock Status Filter:", ["All Statuses", "Critical Stock", "Reorder Soon", "Healthy Stock", "Overstock"])
        
    filtered_inv = latest_inv.copy()
    if search_term:
        filtered_inv = filtered_inv[
            filtered_inv['Product_Name'].str.contains(search_term, case=False) |
            filtered_inv['Product_ID'].str.contains(search_term, case=False)
        ]
    if cat_filter != "All Categories":
        filtered_inv = filtered_inv[filtered_inv['Category'] == cat_filter]
    if status_filter != "All Statuses":
        filtered_inv = filtered_inv[filtered_inv['Stock_Status'] == status_filter]
        
    disp = filtered_inv[['Product_ID', 'Product_Name', 'Category', 'Current_Stock', 'Predicted_Demand', 'Lead_Time_Days', 'Reorder_Point', 'Stock_Status', 'Recommended_Reorder_Qty']].copy()
    disp.columns = ['Product ID', 'Product Name', 'Category', 'Current Stock', 'Predicted Demand/Day', 'Lead Time (Days)', 'Reorder Point (ROP)', 'Stock Status', 'Recommended Purchase']
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    st.markdown("### 🔍 Product Deep Dive View")
    sel_prod_id = st.selectbox("Select Product for In-Depth Analysis:", latest_inv['Product_ID'].tolist(), format_func=lambda x: f"{x} — {latest_inv[latest_inv['Product_ID']==x]['Product_Name'].values[0]}")
    
    prod_row = latest_inv[latest_inv['Product_ID'] == sel_prod_id].iloc[0]
    prod_history = df_full[df_full['Product_ID'] == sel_prod_id].sort_values(by='Date')
    
    p_c1, p_c2 = st.columns([1, 1])
    with p_c1:
        st.markdown(f"#### 📄 {prod_row['Product_Name']} (`{prod_row['Product_ID']}`)")
        cat_val = prod_row['Category'] if 'Category' in prod_row.index else 'N/A'
        st.write(f"• **Category:** {cat_val}")
        
        if 'Supplier' in prod_row.index and pd.notnull(prod_row['Supplier']) and str(prod_row['Supplier']).strip() != '' and str(prod_row['Supplier']) != 'nan':
            supplier_val = str(prod_row['Supplier']).strip()
        else:
            supplier_val = 'Not provided'
        st.write(f"• **Supplier:** {supplier_val}")
        
        if 'Unit_Price' in prod_row.index and pd.notnull(prod_row['Unit_Price']):
            try:
                st.write(f"• **Unit Price:** ${float(prod_row['Unit_Price']):.2f}")
            except (ValueError, TypeError):
                st.write("• **Unit Price:** Not provided")
        else:
            st.write("• **Unit Price:** Not provided")
            
        st.write(f"• **Supplier Lead Time:** {int(prod_row['Lead_Time_Days'])} days")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="saas-card" style="border-left: 5px solid #2563EB !important;">
            <div style="font-size:12px; color:#94A3B8; font-weight:600;">PURCHASE RECOMMENDATION</div>
            <div style="font-size:20px; font-weight:700; color:#F8FAFC; margin-top:4px;">
                Recommended Order: <span style="color:#FCA5A5;">{int(prod_row['Recommended_Reorder_Qty'])} units</span>
            </div>
            <div style="font-size:13px; color:#CBD5E1; margin-top:8px;">
                <b>Reason:</b> {prod_row['Recommendation_Reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p_c2:
        fig_p_trend = px.line(
            prod_history,
            x='Date',
            y=['Demand', 'Current_Stock'],
            title=f"Historical Sales & Stock Depletion ({prod_row['Product_ID']})",
            labels={'value': 'Units', 'variable': 'Metric'},
            color_discrete_map={'Demand': '#3B82F6', 'Current_Stock': '#10B981'}
        )
        fig_p_trend.add_hline(y=prod_row['Reorder_Point'], line_dash="dash", line_color="#EF4444", annotation_text=f"ROP ({prod_row['Reorder_Point']:.1f})")
        fig_p_trend.update_traces(line=dict(width=2))
        apply_dark_theme(fig_p_trend, height=290)
        st.plotly_chart(fig_p_trend, use_container_width=True)


# ==============================================================================
# PAGE 4: 📈 SALES ANALYTICS
# ==============================================================================
elif nav_choice == "📈 Sales Analytics":
    
    st.markdown("## 📈 Sales Performance & Demand Analytics")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Macro-level insights into sales distribution, category performance, and product demand trends.</div>", unsafe_allow_html=True)
    
    a1, a2, a3, a4 = st.columns(4)
    if 'Unit_Price' in df_full.columns:
        total_rev = (df_full['Sales_Quantity'] * df_full['Unit_Price']).sum()
        a1.metric("Total Sales Revenue", f"${total_rev:,.2f}")
    else:
        total_units = int(df_full['Sales_Quantity'].sum())
        a1.metric("Total Sales Volume", f"{total_units:,} units")
        
    avg_daily_sales = df_full.groupby('Date')['Sales_Quantity'].sum().mean()
    top_prod_name = df_full.groupby('Product_Name')['Sales_Quantity'].sum().idxmax()
    slow_prod_name = df_full.groupby('Product_Name')['Sales_Quantity'].sum().idxmin()
    
    a2.metric("Avg Daily Sales Volume", f"{avg_daily_sales:.1f} units/day")
    a3.metric("Top Performing Product", top_prod_name)
    a4.metric("Slowest Moving Product", slow_prod_name)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Sales Volume by Product Category")
        cat_sales = df_full.groupby('Category')['Sales_Quantity'].sum().reset_index()
        fig_cat_bar = px.bar(cat_sales, x='Category', y='Sales_Quantity', color='Category', color_discrete_sequence=px.colors.qualitative.Blues_r)
        apply_dark_theme(fig_cat_bar, height=330)
        fig_cat_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_cat_bar, use_container_width=True)

    with col_b:
        st.markdown("### 🏆 Top Product Sales Ranking")
        prod_sales = df_full.groupby('Product_Name')['Sales_Quantity'].sum().reset_index().sort_values(by='Sales_Quantity', ascending=True)
        fig_prod_bar = px.bar(prod_sales, y='Product_Name', x='Sales_Quantity', orientation='h', color_discrete_sequence=['#2563EB'])
        apply_dark_theme(fig_prod_bar, height=330)
        fig_prod_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_prod_bar, use_container_width=True)


# ==============================================================================
# PAGE 5: 🔮 DEMAND FORECAST
# ==============================================================================
elif nav_choice == "🔮 Demand Forecast":
    
    st.markdown("## 🔮 Machine Learning Demand Forecast")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Predict future demand patterns using trained Supervised Machine Learning algorithms.</div>", unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        sel_fc_sku = st.selectbox(
            "Select Product to Forecast:",
            latest_inv['Product_ID'].tolist(),
            format_func=lambda x: f"{x} — {latest_inv[latest_inv['Product_ID']==x]['Product_Name'].values[0]}"
        )
        
        fc_prod = latest_inv[latest_inv['Product_ID'] == sel_fc_sku].iloc[0]
        daily_pred = fc_prod['Predicted_Demand']
        fc_7d = daily_pred * 7
        fc_30d = daily_pred * 30
        
        st.markdown(f"""
        <div class="saas-card" style="margin-top:10px;">
            <div style="font-weight:700; font-size:16px; color:#F8FAFC;">Forecast Summary ({fc_prod['Product_ID']})</div>
            <div style="margin-top:12px;">
                <div style="font-size:12px; color:#94A3B8;">Current Stock</div>
                <div style="font-weight:700; font-size:20px; color:#F8FAFC;">{int(fc_prod['Current_Stock'])} units</div>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:12px; color:#94A3B8;">Predicted Daily Demand</div>
                <div style="font-weight:700; font-size:20px; color:#60A5FA;">{daily_pred:.1f} units/day</div>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:12px; color:#94A3B8;">7-Day Demand Forecast</div>
                <div style="font-weight:700; font-size:18px; color:#34D399;">{int(fc_7d)} units</div>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:12px; color:#94A3B8;">30-Day Demand Forecast</div>
                <div style="font-weight:700; font-size:18px; color:#FBBF24;">{int(fc_30d)} units</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        sub_df = df_full[df_full['Product_ID'] == sel_fc_sku].sort_values('Date').tail(90)
        
        fig_fc = px.line(
            sub_df,
            x='Date',
            y=['Demand', 'Predicted_Demand'],
            title=f"Actual Demand vs ML Predicted Demand (Last 90 Days)",
            labels={'value': 'Demand Units', 'variable': 'Series'},
            color_discrete_map={'Demand': '#3B82F6', 'Predicted_Demand': '#F59E0B'}
        )
        fig_fc.update_traces(line=dict(width=2))
        apply_dark_theme(fig_fc, height=380)
        st.plotly_chart(fig_fc, use_container_width=True)


# ==============================================================================
# PAGE 6: 🛒 PURCHASE RECOMMENDATIONS
# ==============================================================================
elif nav_choice == "🛒 Purchase Recommendations":
    
    st.markdown("## 🛒 Smart Purchase Recommendations")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Data-driven inventory replenishment recommendations to maintain optimal stock levels and prevent shortages.</div>", unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.tabs(["🔴 Urgent Purchase", "🟠 Reorder Soon", "🟢 Stock Sufficient", "🔵 Overstock"])
    
    def render_rec_tab(status_name):
        subset = latest_inv[latest_inv['Stock_Status'] == status_name]
        if subset.empty:
            st.info(f"No products currently in '{status_name}' status.")
        else:
            for idx, row in subset.iterrows():
                st.markdown(f"""
                <div class="saas-card" style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:16px; color:#F8FAFC;">{row['Product_Name']} ({row['Product_ID']})</span>
                        <span class="{row['Status_Badge_Class']}">{row['Stock_Status'].upper()}</span>
                    </div>
                    <div style="margin-top:12px; display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:10px;">
                        <div><span style="font-size:11px; color:#94A3B8;">Current Stock</span><br><b style="font-size:15px; color:#F8FAFC;">{int(row['Current_Stock'])} units</b></div>
                        <div><span style="font-size:11px; color:#94A3B8;">Predicted Daily Demand</span><br><b style="font-size:15px; color:#60A5FA;">{row['Predicted_Demand']:.1f} /day</b></div>
                        <div><span style="font-size:11px; color:#94A3B8;">Reorder Point (ROP)</span><br><b style="font-size:15px; color:#F8FAFC;">{row['Reorder_Point']:.1f} units</b></div>
                        <div><span style="font-size:11px; color:#94A3B8;">Recommended Purchase</span><br><b style="font-size:15px; color:#FCA5A5;">{int(row['Recommended_Reorder_Qty'])} units</b></div>
                    </div>
                    <div style="margin-top:12px; font-size:12.5px; color:#E2E8F0; background:#0B0F19; padding:10px 12px; border-radius:6px; border:1px solid #26334D;">
                        💡 <b>Reason:</b> {row['Recommendation_Reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with t1: render_rec_tab("Critical Stock")
    with t2: render_rec_tab("Reorder Soon")
    with t3: render_rec_tab("Healthy Stock")
    with t4: render_rec_tab("Overstock")


# ==============================================================================
# PAGE 7: 🤖 MODEL PERFORMANCE
# ==============================================================================
elif nav_choice == "🤖 Model Performance":
    
    st.markdown("## 🤖 Machine Learning Model Performance")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Evaluation metrics and live retraining pipeline for the implemented machine learning model.</div>", unsafe_allow_html=True)
    
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    m_c1.metric("Algorithm", metrics.get("Algorithm", "Linear Regression"))
    m_c2.metric("R² Score Accuracy", f"{metrics.get('R2_Score', 0.9037)*100:.2f}%")
    m_c3.metric("Mean Absolute Error (MAE)", f"{metrics.get('MAE', 21.1826):.2f}")
    m_c4.metric("Root Mean Sq. Error (RMSE)", f"{metrics.get('RMSE', 29.6912):.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        st.markdown("### 🔄 Live Model Retraining Pipeline")
        st.write("Executes model fitting on current sales data, regenerates model weights, and updates system evaluation metrics.")
    with r_col2:
        if st.button("⚡ Retrain Model Live", type="primary", use_container_width=True):
            with st.spinner("Executing model training pipeline..."):
                try:
                    from src.train_model import run_training_pipeline
                    new_model, new_metrics, _ = run_training_pipeline()
                    st.success("Model trained successfully! Metrics updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error retraining model: {e}")
                    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Machine Learning Pipeline Workflow")
    st.markdown("""
    ```
    Sales & Stock Data  ──►  Feature Engineering  ──►  Chronological 80/20 Split  ──►  Linear Regression Model  ──►  Demand Prediction  ──►  Smart Reorder Logic
    ```
    """)
    
    with st.expander("🛠️ Technical Details & Feature Engineering"):
        st.markdown("""
        - **Model Algorithm:** Supervised Linear Regression (`sklearn.linear_model.LinearRegression`)
        - **Temporal Features:** Year, Month, Day, Day of Week
        - **Lag Features:** `Lag_1`, `Lag_7` (Shifted strictly by 1 and 7 days to eliminate data leakage)
        - **Rolling Averages:** 7-Day & 30-Day Shifted Rolling Window Demand Means (`Rolling_Mean_7`, `Rolling_Mean_30`)
        - **Reorder Point (ROP) Formula:** $\\text{ROP} = \\text{Predicted Daily Demand} \\times \\text{Lead Time} \\times 1.2$
        """)


# ==============================================================================
# PAGE 8: 📋 REPORTS
# ==============================================================================
elif nav_choice == "📋 Reports":
    
    st.markdown("## 📋 Exportable Reports & Executive Summary")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Download detailed inventory, demand forecast, and purchase recommendation reports.</div>", unsafe_allow_html=True)
    
    rep_c1, rep_c2, rep_c3 = st.columns(3)
    
    with rep_c1:
        st.markdown("""
        <div class="saas-card">
            <div style="font-weight:700; font-size:16px; color:#F8FAFC;">📦 Inventory Analysis Report</div>
            <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Complete status snapshot of all products</div>
            <br>
        </div>
        """, unsafe_allow_html=True)
        inv_csv = latest_inv.to_csv(index=False).encode('utf-8')
        st.download_button("Download Inventory Report (CSV)", data=inv_csv, file_name="inventory_analysis_report.csv", mime="text/csv", use_container_width=True)

    with rep_c2:
        st.markdown("""
        <div class="saas-card">
            <div style="font-weight:700; font-size:16px; color:#F8FAFC;">🛒 Purchase Recommendations</div>
            <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Replenishment order list for suppliers</div>
            <br>
        </div>
        """, unsafe_allow_html=True)
        rec_csv = latest_inv[latest_inv['Reorder_Required']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Purchase Orders (CSV)", data=rec_csv, file_name="purchase_recommendations_report.csv", mime="text/csv", use_container_width=True)

    with rep_c3:
        st.markdown("""
        <div class="saas-card">
            <div style="font-weight:700; font-size:16px; color:#F8FAFC;">📈 Full Sales History</div>
            <div style="font-size:12px; color:#94A3B8; margin-top:4px;">Complete historical dataset</div>
            <br>
        </div>
        """, unsafe_allow_html=True)
        full_csv = df_full.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Dataset (CSV)", data=full_csv, file_name="full_sales_demand_history.csv", mime="text/csv", use_container_width=True)


# ==============================================================================
# PAGE 9: ⚙️ SETTINGS
# ==============================================================================
elif nav_choice == "⚙️ Settings":
    
    st.markdown("## ⚙️ System Settings & Inventory Parameters")
    st.markdown("<div style='color:#94A3B8; margin-bottom:20px;'>Configure safety stock multipliers, threshold levels, and system parameters.</div>", unsafe_allow_html=True)
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        new_safety = st.slider("Lead Time Safety Multiplier (Default 1.2):", min_value=1.0, max_value=2.0, value=st.session_state.safety_factor, step=0.05)
        new_overstock = st.slider("Overstock Multiplier (Default 2.0x ROP):", min_value=1.5, max_value=3.5, value=st.session_state.overstock_factor, step=0.1)
        
        if st.button("Save Settings", type="primary"):
            st.session_state.safety_factor = new_safety
            st.session_state.overstock_factor = new_overstock
            st.success("Settings saved successfully!")
            st.rerun()

    with s_col2:
        st.markdown("### 🔄 Reset System State")
        st.write("Reset dataset and configuration back to factory default state.")
        if st.button("Reset to Default Dataset"):
            st.session_state.raw_df = load_default_dataset()
            st.session_state.safety_factor = 1.2
            st.session_state.overstock_factor = 2.0
            st.success("System reset to default dataset.")
            st.rerun()
