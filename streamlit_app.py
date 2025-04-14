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
elif use_sample_data
