#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up FastAPI app
app = FastAPI()

# Set page title
st.set_page_config(page_title='Energy Grid Congestion Forecasting', layout='wide')
st.title('Energy Grid Congestion Forecasting System')

# Weather API Key
WEATHER_API_KEY = st.secrets["WEATHER_KEY"]

# Function to fetch real-time weather data
def fetch_weather(city="Cologne"):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "Temperature": data["current"]["temp_c"],
            "Wind Speed": data["current"]["wind_kph"] / 3.6,  # Convert kph to m/s
            "Solar Radiation": np.random.normal(300, 100)  # Approximate value
        }
    else:
        return {"Temperature": 20, "Wind Speed": 10, "Solar Radiation": 300}  # Default values

# File Upload Section
st.sidebar.header("Upload Your Grid Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# Fetch real-time weather data
city = st.sidebar.text_input("Enter city for weather forecast", "Cologne")
weather_data = fetch_weather(city)
st.sidebar.write(f"**Temperature:** {weather_data['Temperature']}°C")
st.sidebar.write(f"**Wind Speed:** {weather_data['Wind Speed']} m/s")
st.sidebar.write(f"**Solar Radiation:** {weather_data['Solar Radiation']} W/m²")

# Sidebar for user input
st.sidebar.header('Forecasting Configuration')
prediction_days = st.sidebar.slider('Select Forecast Horizon (Hours)', min_value=1, max_value=48, value=12)

# Display Raw Data
st.subheader('Uploaded Energy Grid Data')
st.dataframe(data.head(10))

# Visualization
fig = px.line(data, x=data.columns[0], y=['Energy Demand', 'Grid Utilization'], title='Energy Demand vs Grid Utilization')
st.plotly_chart(fig)

# Feature Engineering
data['Hour'] = pd.to_datetime(data[data.columns[0]]).dt.hour
X = data[['Temperature', 'Wind Speed', 'Solar Radiation', 'Energy Demand', 'Hour']]
y = data['Grid Utilization']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
st.subheader('Model Evaluation')
st.write(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
st.write(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
st.write(f'R² Score: {r2_score(y_test, y_pred):.2f}')

# Forecasting Future Congestion
def forecast_future(prediction_days):
    future_dates = pd.date_range(start=pd.to_datetime(data[data.columns[0]]).max(), periods=prediction_days, freq='H')
    future_features = pd.DataFrame({'Temperature': np.full(prediction_days, weather_data['Temperature']),
                                    'Wind Speed': np.full(prediction_days, weather_data['Wind Speed']),
                                    'Solar Radiation': np.full(prediction_days, weather_data['Solar Radiation']),
                                    'Energy Demand': np.random.normal(500, 150, size=prediction_days),
                                    'Hour': future_dates.hour})
    future_forecast = model.predict(future_features)
    return pd.DataFrame({'Date': future_dates, 'Forecasted Grid Utilization': future_forecast})

forecast_data = forecast_future(prediction_days)
fig_forecast = px.line(forecast_data, x='Date', y='Forecasted Grid Utilization', title='Forecasted Grid Congestion')
st.plotly_chart(fig_forecast)

st.success('Forecast generated successfully! 🚀')

# API Endpoint to Get Forecasted Congestion Data
@app.get("/forecast/{prediction_days}")
def get_forecast(prediction_days: int):
    forecast_data = forecast_future(prediction_days)
    return forecast_data.to_dict(orient='records')

