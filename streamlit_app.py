import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Product Volume Forecasting Dashboard", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-text {
        font-size: 1.1rem;
        color: #333;
        background-color: #e8f4f8;
        padding: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<p class="main-header">Product Volume Forecasting & Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides comprehensive analysis of product volumes over time, forecasts future volumes, 
and identifies key factors influencing volume changes. Upload your data to begin analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file containing historical product volume data", type=["csv"])

# Sample data option for demonstration
use_sample_data = st.checkbox("Use sample data for demonstration")

@st.cache_data
def generate_sample_data():
    # Generate dates for the past 2 years
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=i) for i in range(730)]
    
    # Create products
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    # Create features that might impact volume
    regions = ['North', 'South', 'East', 'West']
    channels = ['Online', 'Retail', 'Wholesale', 'Direct']
    promotions = [0, 1]  # 0 = no promotion, 1 = promotion
    
    # Create empty dataframe
    rows = []
    
    # Generate data with seasonal patterns and trends
    for date in dates:
        for product in products:
            # Base volume with product-specific baseline
            if product == 'Product A':
                base_volume = 1000
                seasonal_factor = 200 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                trend = date.timetuple().tm_yday / 30  # Upward trend
            elif product == 'Product B':
                base_volume = 800
                seasonal_factor = 150 * np.cos(2 * np.pi * date.timetuple().tm_yday / 365)
                trend = -date.timetuple().tm_yday / 40  # Downward trend
            elif product == 'Product C':
                base_volume = 1200
                seasonal_factor = 300 * np.sin(2 * np.pi * date.timetuple().tm_yday / 180)  # Bi-annual seasonality
                trend = date.timetuple().tm_yday / 50  # Slight upward trend
            else:  # Product D
                base_volume = 600
                seasonal_factor = 100 * np.sin(2 * np.pi * date.timetuple().tm_yday / 90)  # Quarterly seasonality
                trend = 0  # No trend
            
            # Random region, channel, and promotion
            region = np.random.choice(regions)
            channel = np.random.choice(channels)
            promotion = np.random.choice(promotions, p=[0.8, 0.2])  # 20% chance of promotion
            
            # Region impact
            if region == 'North':
                region_impact = 50
            elif region == 'South':
                region_impact = -30
            elif region == 'East':
                region_impact = 20
            else:  # West
                region_impact = 10
                
            # Channel impact
            if channel == 'Online':
                channel_impact = 100
            elif channel == 'Retail':
                channel_impact = 50
            elif channel == 'Wholesale':
                channel_impact = 200
            else:  # Direct
                channel_impact = -50
                
            # Promotion impact
            promo_impact = 300 if promotion == 1 else 0
            
            # Weather impact (simulated)
            weather_temp = 15 + 15 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)  # Temperature between 0-30
            weather_impact = (weather_temp - 15) * 5  # Higher temps generally increase volume
            
            # Competitor activity (simulated)
            competitor_activity = np.random.randint(0, 5)  # Scale 0-4
            competitor_impact = -50 * competitor_activity
            
            # Calculate final volume with some randomness
            volume = base_volume + seasonal_factor + trend + region_impact + channel_impact + promo_impact + weather_impact + competitor_impact
            volume = max(0, int(volume + np.random.normal(0, 50)))  # Add noise and ensure non-negative
            
            # Add row to data
            rows.append({
                'Date': date,
                'Product': product,
                'Region': region,
                'Channel': channel,
                'Promotion': promotion,
                'Weather_Temp': round(weather_temp, 1),
                'Competitor_Activity': competitor_activity,
                'Volume': volume
            })
    
    # Create dataframe
    df = pd.DataFrame(rows)
    return df

# Load data
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
elif use_sample_data:
    data = generate_sample_data()
    st.success("Sample data loaded for demonstration!")
else:
    st.info("Please upload a CSV file or use sample data to begin analysis.")
    st.stop()

# Display data preview in an expandable section
with st.expander("Data Preview"):
    st.dataframe(data.head(10))
    st.write(f"Dataset shape: {data.shape[0]} rows and {data.shape[1]} columns")

# Sidebar for configuration
st.sidebar.markdown('<p class="sub-header">Analysis Configuration</p>', unsafe_allow_html=True)

# Column selection
date_col = st.sidebar.selectbox("Select date column", 
                              [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()],
                              index=0 if any('date' in col.lower() for col in data.columns) else 0)

product_col = st.sidebar.selectbox("Select product column", 
                                 [col for col in data.columns if 'product' in col.lower() or 'item' in col.lower() or 'sku' in col.lower()],
                                 index=0 if any('product' in col.lower() for col in data.columns) else 0)

volume_col = st.sidebar.selectbox("Select volume column", 
                                [col for col in data.columns if 'volume' in col.lower() or 'sales' in col.lower() or 'quantity' in col.lower()],
                                index=0 if any('volume' in col.lower() for col in data.columns) else 0)

# Convert date column to datetime if it's not already
if data[date_col].dtype != 'datetime64[ns]':
    try:
        data[date_col] = pd.to_datetime(data[date_col])
        st.sidebar.info(f"Converted {date_col} to datetime format")
    except Exception as e:
        st.sidebar.error(f"Error converting {date_col} to datetime: {e}")
        st.stop()

# Time period selection
min_date = data[date_col].min()
max_date = data[date_col].max()
date_range = st.sidebar.date_input(
    "Select date range for analysis",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = data[(data[date_col] >= pd.Timestamp(start_date)) & 
                         (data[date_col] <= pd.Timestamp(end_date))]
else:
    filtered_data = data.copy()

# Product selection
available_products = filtered_data[product_col].unique()
selected_products = st.sidebar.multiselect(
    "Select products to analyze",
    options=available_products,
    default=available_products[:min(3, len(available_products))]
)

if selected_products:
    filtered_data = filtered_data[filtered_data[product_col].isin(selected_products)]
else:
    st.warning("Please select at least one product to analyze.")
    st.stop()

# Time granularity for analysis
time_granularity = st.sidebar.selectbox(
    "Select time granularity",
    options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
    index=2  # Default to Monthly
)

# Map selected granularity to pandas resample rule
granularity_map = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
    "Yearly": "Y"
}
resample_rule = granularity_map[time_granularity]

# Forecast configuration
st.sidebar.markdown('<p class="sub-header">Forecast Configuration</p>', unsafe_allow_html=True)
forecast_periods = st.sidebar.slider(
    f"Number of {time_granularity.lower()} periods to forecast", 
    1, 24, 6
)

# Feature selection for forecasting
feature_cols = [col for col in data.columns if col not in [date_col, product_col, volume_col]]
selected_features = st.sidebar.multiselect(
    "Select features for analysis and forecasting",
    options=feature_cols,
    default=feature_cols[:min(5, len(feature_cols))]
)

# Main dashboard
st.markdown('<p class="sub-header">Volume Analysis by Product</p>', unsafe_allow_html=True)

# Aggregate data by date and product
agg_data = filtered_data.groupby([pd.Grouper(key=date_col, freq=resample_rule), product_col])[volume_col].sum().reset_index()

# Create time series plot with Plotly
fig = px.line(
    agg_data, 
    x=date_col, 
    y=volume_col, 
    color=product_col,
    title=f"Product Volume Trends ({time_granularity})",
    template="plotly_white",
    markers=True
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    legend_title="Product",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Key metrics
st.markdown('<p class="sub-header">Key Performance Metrics</p>', unsafe_allow_html=True)

# Calculate metrics by product
metrics_cols = st.columns(len(selected_products))
for i, product in enumerate(selected_products):
    product_data = agg_data[agg_data[product_col] == product]
    
    # Calculate metrics
    total_volume = product_data[volume_col].sum()
    avg_volume = product_data[volume_col].mean()
    
    # Calculate growth
    if len(product_data) >= 2:
        first_period = product_data[volume_col].iloc[0]
        last_period = product_data[volume_col].iloc[-1]
        growth = ((last_period / first_period) - 1) * 100
        growth_text = f"{growth:.1f}%"
        growth_color = "green" if growth >= 0 else "red"
    else:
        growth_text = "N/A"
        growth_color = "gray"
    
    # Display metrics
    with metrics_cols[i]:
        st.markdown(f"<div class='metric-card'><h3>{product}</h3>", unsafe_allow_html=True)
        st.metric("Total Volume", f"{total_volume:,.0f}")
        st.metric("Average Volume", f"{avg_volume:,.0f}")
        st.markdown(f"<p>Period-over-Period Growth: <span style='color:{growth_color};font-weight:bold;'>{growth_text}</span></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Volume Distribution by Product
st.markdown('<p class="sub-header">Volume Distribution Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Product volume share (pie chart)
    product_totals = agg_data.groupby(product_col)[volume_col].sum().reset_index()
    fig = px.pie(
        product_totals, 
        values=volume_col, 
        names=product_col,
        title="Volume Share by Product",
        hole=0.4,
        template="plotly_white"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Volume distribution (box plot)
    fig = px.box(
        agg_data, 
        x=product_col, 
        y=volume_col,
        title="Volume Distribution by Product",
        template="plotly_white",
        color=product_col
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Seasonality Analysis
st.markdown('<p class="sub-header">Seasonality Analysis</p>', unsafe_allow_html=True)

# Extract month and year
agg_data['Month'] = agg_data[date_col].dt.month
agg_data['Year'] = agg_data[date_col].dt.year

# Create heatmap data
if time_granularity in ["Daily", "Weekly", "Monthly"]:
    # For monthly or finer granularity, show month-by-year heatmap
    heatmap_data = agg_data.pivot_table(
        index='Year', 
        columns='Month',
        values=volume_col,
        aggfunc='sum'
    )
    
    # Plot heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Year", color="Volume"),
        x=[f"{m}" for m in range(1, 13)],
        y=heatmap_data.index,
        title="Volume Heatmap by Month and Year",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal pattern insight
    monthly_avg = agg_data.groupby('Month')[volume_col].mean().reset_index()
    peak_month = monthly_avg.loc[monthly_avg[volume_col].idxmax(), 'Month']
    low_month = monthly_avg.loc[monthly_avg[volume_col].idxmin(), 'Month']
    
    month_names = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    
    st.markdown(f"""
    <div class='insight-text'>
        <strong>Seasonality Insight:</strong> Volume typically peaks in {month_names[peak_month-1]} and 
        is lowest in {month_names[low_month-1]}. Consider adjusting inventory and marketing strategies accordingly.
    </div>
    """, unsafe_allow_html=True)

# Feature Impact Analysis
if selected_features:
    st.markdown('<p class="sub-header">Feature Impact Analysis</p>', unsafe_allow_html=True)
    
    # Prepare data for modeling
    model_data = filtered_data.copy()
    
    # Handle categorical features
    cat_features = [col for col in selected_features if model_data[col].dtype == 'object' or model_data[col].nunique() < 10]
    num_features = [col for col in selected_features if col not in cat_features]
    
    # One-hot encode categorical features
    model_data = pd.get_dummies(model_data, columns=cat_features, drop_first=False)
    
    # Get all dummy columns
    dummy_cols = [col for col in model_data.columns if any(f"{cat}_" in col for cat in cat_features)]
    
    # Prepare feature matrix
    X_cols = num_features + dummy_cols
    X = model_data[X_cols]
    y = model_data[volume_col]
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    feature_imp = pd.DataFrame({
        'Feature': X_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot feature importance
        fig = px.bar(
            feature_imp.head(15), 
            x='Importance', 
            y='Feature',
            title="Top 15 Features Impacting Volume",
            template="plotly_white",
            orientation='h'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
            <strong>Feature Impact Insight:</strong><br>
            The chart shows the relative importance of different factors
