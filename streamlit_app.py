import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

# App title and description
st.title("Time Series Forecasting Application")
st.write("Upload your time series data and generate forecasts using various methods.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file containing time series data", type=["csv"])

if uploaded_file is not None:
    # Load the data
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded!")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Column selection
        st.subheader("Select Columns")
        date_col = st.selectbox("Select date column", data.columns)
        value_col = st.selectbox("Select value column to forecast", 
                                [col for col in data.columns if col != date_col])
        
        # Convert date column to datetime if it's not already
        if data[date_col].dtype != 'datetime64[ns]':
            try:
                data[date_col] = pd.to_datetime(data[date_col])
                st.info(f"Converted {date_col} to datetime format")
            except Exception as e:
                st.error(f"Error converting {date_col} to datetime: {e}")
                st.stop()
        
        # Set date as index
        data = data.set_index(date_col)
        
        # Display time series plot
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data[value_col])
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(f'Time Series of {value_col}')
        ax.grid(True)
        st.pyplot(fig)
        
        # Forecasting options
        st.subheader("Forecasting Options")
        forecast_method = st.selectbox(
            "Select forecasting method",
            ["Holt-Winters Exponential Smoothing", "Simple Moving Average"]
        )
        
        forecast_periods = st.slider("Number of periods to forecast", 1, 365, 30)
        
        # Perform forecasting
        if st.button("Generate Forecast"):
            st.subheader("Forecasting Results")
            
            # Create train/test split
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size][value_col]
            test_data = data.iloc[train_size:][value_col]
            
            if forecast_method == "Holt-Winters Exponential Smoothing":
                # Check if we have enough data for seasonal decomposition
                if len(train_data) >= 2:
                    # Try to determine seasonality
                    try:
                        decomposition = seasonal_decompose(train_data, model='additive', period=12)
                        seasonal_period = 12
                    except:
                        seasonal_period = 1
                        st.warning("Could not determine seasonality, using non-seasonal model")
                    
                    # Fit model
                    model = ExponentialSmoothing(
                        train_data,
                        trend='add',
                        seasonal='add' if seasonal_period > 1 else None,
                        seasonal_periods=seasonal_period if seasonal_period > 1 else None
                    ).fit()
                    
                    # Generate forecast
                    forecast = model.forecast(len(test_data) + forecast_periods)
                    
                    # Calculate error metrics on test data
                    test_forecast = forecast[:len(test_data)]
                    mse = mean_squared_error(test_data, test_forecast)
                    rmse = np.sqrt(mse)
                    
                    st.write(f"Root Mean Square Error on test data: {rmse:.2f}")
                    
                    # Plot results
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(data.index, data[value_col], label='Historical Data')
                    ax.plot(forecast.index, forecast, label='Forecast', color='red')
                    ax.fill_between(
                        forecast.index, 
                        forecast - 1.96 * rmse, 
                        forecast + 1.96 * rmse, 
                        color='red', 
                        alpha=0.2, 
                        label='95% Confidence Interval'
                    )
                    ax.set_xlabel('Date')
                    ax.set_ylabel(value_col)
                    ax.set_title(f'Forecast of {value_col} using Holt-Winters Method')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Display forecast data
                    st.subheader("Forecast Data")
                    forecast_df = forecast.reset_index()
                    forecast_df.columns = ['Date', 'Forecast']
                    st.dataframe(forecast_df)
                    
                    # Option to download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name="forecast_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Not enough data for forecasting. Please provide a larger dataset.")
            
            elif forecast_method == "Simple Moving Average":
                # Calculate moving average
                window_size = st.slider("Select window size for moving average", 1, 30, 7)
                
                # Calculate moving average
                ma = data[value_col].rolling(window=window_size).mean()
                
                # Forecast using the last moving average value
                last_ma = ma.iloc[-1]
                forecast_index = pd.date_range(
                    start=data.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq='D'
                )
                forecast = pd.Series([last_ma] * forecast_periods, index=forecast_index)
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data[value_col], label='Historical Data')
                ax.plot(data.index, ma, label=f'{window_size}-Day Moving Average', color='orange')
                ax.plot(forecast.index, forecast, label='Forecast', color='red')
                ax.set_xlabel('Date')
                ax.set_ylabel(value_col)
                ax.set_title(f'Forecast of {value_col} using Simple Moving Average')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                # Display forecast data
                st.subheader("Forecast Data")
                forecast_df = forecast.reset_index()
                forecast_df.columns = ['Date', 'Forecast']
                st.dataframe(forecast_df)
                
                # Option to download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name="forecast_data.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin forecasting.")

# Add footer
st.markdown("---")
st.markdown("Developed by StephPendle")
