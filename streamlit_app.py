import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Set page configuration
st.set_page_config(page_title="Forecasting Performance Dashboard", layout="wide")

# App title and description
st.title("Forecasting Performance Dashboard")
st.write("Upload your data to visualize performance, generate forecasts, and analyze feature contributions.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file containing historical data", type=["csv"])

if uploaded_file is not None:
    # Load the data
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded!")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Column selection
        st.sidebar.subheader("Configure Analysis")
        date_col = st.sidebar.selectbox("Select date column", data.columns)
        target_col = st.sidebar.selectbox("Select target column to forecast", 
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
        data_indexed = data.set_index(date_col)
        
        # Performance to date visualization
        st.subheader("Performance to Date")
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data_indexed.index, data_indexed[target_col])
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col)
            ax.set_title(f'Historical {target_col} Over Time')
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            # Summary statistics
            st.write("Summary Statistics")
            st.dataframe(data_indexed[target_col].describe())
            
            # Monthly or quarterly aggregation
            if len(data_indexed) > 30:
                st.write("Monthly Aggregation")
                monthly_data = data_indexed[target_col].resample('M').sum()
                st.bar_chart(monthly_data)
        
        # Feature selection for forecasting
        feature_cols = st.sidebar.multiselect(
            "Select features for forecasting model",
            [col for col in data.columns if col not in [date_col, target_col]],
            default=[col for col in data.columns if col not in [date_col, target_col]][:min(5, len(data.columns)-2)]
        )
        
        # Forecast period
        forecast_periods = st.sidebar.slider("Number of periods to forecast", 1, 365, 30)
        
        # Train/test split ratio
        test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20) / 100
        
        if len(feature_cols) > 0 and st.sidebar.button("Generate Forecast"):
            st.subheader("Forecasting Results")
            
            # Prepare data for modeling
            X = data[feature_cols]
            y = data[target_col]
            
            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{mae:.2f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.2f}")
            
            # Create forecast dataframe
            test_dates = data.iloc[-len(y_test):][date_col].values
            forecast_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test.values,
                'Forecast': y_pred
            })
            
            # Display forecast vs actual comparison
            st.subheader("Forecast vs Actual Comparison")
            st.dataframe(forecast_df)
            
            # Plot forecast vs actual
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast_df['Date'], forecast_df['Actual'], label='Actual', marker='o')
            ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', marker='x', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col)
            ax.set_title(f'Forecast vs Actual {target_col}')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Feature importance analysis
            st.subheader("Feature Contribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Random Forest feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance[:10], ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)
            
            with col2:
                # SHAP values for more detailed feature contribution
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate SHAP values: {e}")
            
            # Generate future forecast
            st.subheader(f"Future Forecast (Next {forecast_periods} Periods)")
            
            # For simplicity, we'll use the last row's features repeated
            last_features = X.iloc[-1:].copy()
            future_X = pd.concat([last_features] * forecast_periods, ignore_index=True)
            
            # Generate predictions
            future_preds = model.predict(future_X)
            
            # Create future dates
            last_date = data[date_col].iloc[-1]
            if isinstance(last_date, pd.Timestamp):
                # Try to infer frequency
                if len(data) > 1:
                    freq = pd.infer_freq(data[date_col])
                    if freq is None:
                        # Default to daily if can't infer
                        freq = 'D'
                else:
                    freq = 'D'
                
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq=freq
                )
            else:
                # If not timestamp, just use integers
                future_dates = range(
                    int(data[date_col].iloc[-1]) + 1,
                    int(data[date_col].iloc[-1]) + forecast_periods + 1
                )
            
            # Create future forecast dataframe
            future_forecast = pd.DataFrame({
                'Date': future_dates,
                'Forecast': future_preds
            })
            
            # Display future forecast
            st.dataframe(future_forecast)
            
            # Plot future forecast
            fig, ax = plt.subplots(figsize=(12, 6))
            # Historical data
            ax.plot(data[date_col], data[target_col], label='Historical', color='blue')
            # Test predictions
            ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Test Predictions', color='green', linestyle='--')
            # Future forecast
            ax.plot(future_forecast['Date'], future_forecast['Forecast'], label='Future Forecast', color='red', marker='x')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col)
            ax.set_title(f'Complete Forecast of {target_col}')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Option to download forecast
            csv = future_forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name="forecast_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")

# Add footer
st.markdown("---")
st.markdown("Developed by StephPendle")
