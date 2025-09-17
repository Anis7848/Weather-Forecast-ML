import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from dotenv import load_dotenv
import os
import requests
import io
import streamlit as st

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌤 Weather Forecast Dashboard")

# User inputs
city = st.text_input("Enter City", "Delhi")
periods = st.slider("Forecast Days", 7, 30, 15)

# -----------------------------
# Fetch live weather
# -----------------------------
if API_KEY and city:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        st.subheader("Current Weather")
        st.write({
            "Temperature (°C)": data['main']['temp'],
            "Humidity (%)": data['main']['humidity'],
            "Pressure (mb)": data['main']['pressure'],
            "Wind Speed (km/h)": data['wind']['speed'],
            "Visibility (km)": data.get('visibility', 0)/1000
        })
    except Exception as e:
        st.error(f"Failed to fetch live weather: {e}")

# -----------------------------
# Load historical dataset
# -----------------------------
df = pd.read_csv("C:\\Users\\ANIS\\Downloads\\Weather forcast ML\\weather.csv")
df['date'] = pd.to_datetime(df['date'], utc=True, errors="coerce")
df = df[df['date'].notnull()]
df['date'] = df['date'].dt.tz_convert(None)

# -----------------------------
# Keep required columns and convert to numeric
# -----------------------------
df_main = df[['date', 'Temperature (C)', 'Humidity',
              'Wind Speed (km/h)', 'Visibility (km)',
              'Rainfall', 'Pressure (millibars)']].copy()

for col in df_main.columns[1:]:
    df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

df_main = df_main.dropna()

# -----------------------------
# Check if enough data exists
# -----------------------------
if df_main.shape[0] < 2:
    st.error("Not enough historical data to forecast. Please check your CSV.")
    st.stop()

# Downsample daily for speed
df_main.set_index('date', inplace=True)
df_main = df_main.resample('D').mean().reset_index()

# -----------------------------
# Forecast function with caching
# -----------------------------
@st.cache_data(show_spinner=False)
def forecast_feature(df, column, periods=15):
    df_temp = df[['date', column]].dropna().rename(columns={'date': 'ds', column: 'y'})
    if df_temp.shape[0] < 2:
        return None
    model = Prophet()
    model.fit(df_temp)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast['feature'] = column
    return forecast

# -----------------------------
# Display forecasts in 2x3 grid
# -----------------------------
st.subheader("📊 Forecasts for All Features")
features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
            'Visibility (km)', 'Rainfall', 'Pressure (millibars)']

all_forecasts = []
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        st.write(f"### 🔹 {feature}")
        forecast = forecast_feature(df_main, feature, periods=periods)
        if forecast is None:
            st.warning(f"Not enough data for {feature}. Skipping...")
            continue
        # Fast plotting
        fig, ax = plt.subplots()
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
        ax.set_title(f"{feature} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel(feature)
        ax.legend()
        st.pyplot(fig)
        all_forecasts.append(forecast)
    if (i + 1) % 3 == 0:
        cols = st.columns(3)

# -----------------------------
# Combine all forecasts for download
# -----------------------------
if all_forecasts:
    combined_df = pd.concat(all_forecasts, ignore_index=True)
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download All Forecasts (CSV)",
        data=csv_buffer.getvalue(),
        file_name="all_forecasts.csv",
        mime="text/csv"
    )
