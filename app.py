import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time
from datetime import datetime
import pymongo
from urllib.parse import quote_plus

# -----------------------------
# 1. Page Configuration & Setup
# -----------------------------
st.set_page_config(page_title="Hybrid Energy Prediction Dashboard", layout="wide")

st.title("☀️ Hybrid Energy Prediction and Optimization Dashboard")
st.markdown("This dashboard serves as the digital twin for our solar-thermal microgrid. It combines solar-side forecasting with **live TEG-side AI predictions** to estimate available power, monitor battery behavior, and determine the optimal operating mode.")

# -----------------------------
# 2. Data & Model Loaders
# -----------------------------
@st.cache_data
def load_future_data():
    if os.path.exists("Data/future_predictions_2026_2030.csv"):
        df = pd.read_csv("Data/future_predictions_2026_2030.csv")
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    return pd.DataFrame()

@st.cache_data
def load_teg_results():
    if os.path.exists("Data/teg_voltage_prediction_results.csv"):
        return pd.read_csv("Data/teg_voltage_prediction_results.csv")
    return pd.DataFrame()

@st.cache_resource
def load_teg_model():
    model_path = "Models/teg_voltage_rf_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

future_df = load_future_data()
teg_df = load_teg_results()
teg_model = load_teg_model()

# -----------------------------
# 2.5 Cloud Data Loader (MongoDB)
# -----------------------------
@st.cache_resource
def get_mongo_collection():
    password = quote_plus("Pasindi@2002.22")
    uri = f"mongodb+srv://en22198822_db_user:{password}@cluster0.zrezdhz.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(uri)
    db = client['FYP_Database']  
    return db['microgrid_data']  

cloud_collection = get_mongo_collection()

def get_latest_cloud_data():
    try:
        cursor = cloud_collection.find().sort("timestamp", -1).limit(50)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()

if future_df.empty:
    st.error("Future solar prediction CSV is empty or missing.")
    st.stop()

# System Status Banner
if teg_model is not None:
    st.success("✅ System Status: AI Prediction Model Loaded and Active")
else:
    st.error("🚨 System Status: AI Model Offline")

# -----------------------------
# 3. Sidebar: Control Panel
# -----------------------------
min_date = future_df["datetime"].min().date()
max_date = future_df["datetime"].max().date()

st.sidebar.header("🎛️ Control Panel")
st.sidebar.markdown("Use these settings to simulate different microgrid scenarios.")

selected_date = st.sidebar.date_input("Target Date", value=min_date, min_value=min_date, max_value=max_date)
selected_hour = st.sidebar.selectbox("Target Hour (24H)", list(range(24)), index=12)

with st.sidebar.expander("🔋 Battery & Load Configuration", expanded=True):
    battery_soc_now = st.number_input("Current SOC (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    load_demand_w = st.number_input("Load Demand (W)", min_value=0, max_value=1000, value=50, step=5)
    battery_capacity_wh = st.number_input("Battery Capacity (Wh)", min_value=0.1, max_value=5000.0, value=3.7, step=0.1, format="%.1f")

with st.sidebar.expander("🧠 Advanced AI Model Configuration", expanded=False):
    st.caption("These parameters adjust the inputs for the Random Forest TEG model.")
    time_s = st.number_input("Elapsed Test Time (s)", min_value=0, max_value=100000, value=600, step=10)
    sample_interval_s = st.number_input("Sampling Interval (s)", min_value=1, max_value=3600, value=3, step=1)
    temp_C = st.number_input("Current Temperature (°C)", min_value=0.0, max_value=200.0, value=55.0, step=0.1)
    temp_lag_1 = st.number_input("Temperature t-1 (°C)", min_value=0.0, max_value=200.0, value=54.5, step=0.1)
    temp_lag_2 = st.number_input("Temperature t-2 (°C)", min_value=0.0, max_value=200.0, value=54.0, step=0.1)
    voltage_lag_1 = st.number_input("Voltage t-1 (mV)", min_value=0.0, max_value=1000.0, value=60.0, step=0.1)
    voltage_lag_2 = st.number_input("Voltage t-2 (mV)", min_value=0.0, max_value=1000.0, value=58.0, step=0.1)
    heatsink_option = st.selectbox("Heatsink Installed", ["no", "yes"])
    reference_load_ohm = st.number_input("Ref. Load Resistance (Ω)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)

# -----------------------------
# 4. Main Dashboard Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📟 Live Monitoring", "📈 Scenario Optimizer", "🧠 AI Model Analytics"])

# --- TAB 1: LIVE MONITORING ---
with tab1:
    st.header("Real-Time Microgrid Telemetry")
    monitor_placeholder = st.empty() 

# --- TAB 2: HYBRID PREDICTIONS ---
with tab2:
    st.header("Scenario Optimizer")
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00")
    match = future_df[future_df["datetime"] == selected_datetime]

    if match.empty:
        st.warning("No solar prediction found for the selected date and time.")
    else:
        row = match.iloc[0]
        predicted_irradiance = float(row["predicted_irradiance"])
        predicted_pv_power = float(row["predicted_pv_power"])

        heatsink_bin = 1 if heatsink_option == "yes" else 0
        dtemp_dt = (temp_C - temp_lag_1) / sample_interval_s
        temp_rolling_3 = np.mean([temp_C, temp_lag_1, temp_lag_2])

        predicted_teg_power_W = 0.0
        predicted_teg_power_mW = 0.0

        if teg_model is not None:
            live_input = pd.DataFrame([{
                "time_s": time_s, "temp_C": temp_C, "heatsink_bin": heatsink_bin, 
                "dtemp_dt": dtemp_dt, "temp_rolling_3": temp_rolling_3, 
                "temp_lag_1": temp_lag_1, "temp_lag_2": temp_lag_2, 
                "voltage_lag_1": voltage_lag_1, "voltage_lag_2": voltage_lag_2
            }])
            predicted_teg_voltage_mV = max(float(teg_model.predict(live_input)[0]), 0)
            predicted_teg_voltage_V = predicted_teg_voltage_mV / 1000.0
            predicted_teg_power_W = (predicted_teg_voltage_V ** 2) / reference_load_ohm
            predicted_teg_power_mW = predicted_teg_power_W * 1000

        total_power = predicted_pv_power + predicted_teg_power_W
        net_power = total_power - load_demand_w
        soc_change = (net_power / battery_capacity_wh) * 100
        battery_soc_next = max(20, min(100, battery_soc_now + soc_change))

        def final_recommendation(total_power, soc_next, load_demand):
            if total_power >= 1.5 * load_demand and soc_next > 60:
                return "Run heavy load and charge battery"
            elif total_power >= load_demand and soc_next > 40:
                return "Run essential loads only"
            elif total_power < load_demand and predicted_teg_power_W > 0 and soc_next > 30:
                return "Use TEG-assisted operation"
            else:
                return "Battery saving mode / delay non-essential loads"

        recommendation = final_recommendation(total_power, battery_soc_next, load_demand_w)

        with st.container(border=True):
            st.subheader("Simulated Output based on Control Panel")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Solar Power (W)", f"{predicted_pv_power:.2f}")
            c2.metric("Predicted TEG Power (mW)", f"{predicted_teg_power_mW:.2f}")
            c3.metric("Total Available Power (W)", f"{total_power:.2f}")

        st.info(f"**System Recommendation:** {recommendation}")

        daily_df = future_df[future_df["datetime"].dt.date == selected_date].copy()
        daily_df["hour"] = daily_df["datetime"].dt.hour
        daily_df["daily_total_power"] = daily_df["predicted_pv_power"] + predicted_teg_power_W

        st.subheader("24-Hour Forecast")
        st.plotly_chart(px.line(daily_df, x="hour", y="daily_total_power", title="Predicted Total Power Capacity"), use_container_width=True)

# --- TAB 3: AI ANALYSIS ---
with tab3:
    st.header("🔄 Integrated Hybrid System Networks")
    st.markdown("Pulling live data securely via MongoDB from the remote hardware.")
    cloud_monitor_placeholder = st.empty() 

# -----------------------------
# 5. The Data Fetcher (Replaces the While Loop)
# -----------------------------
# Part A: Local Telemetry (Tab 1)
try:
    if os.path.exists("results.csv") and os.path.exists("battery.csv"):
        df_live = pd.read_csv("results.csv")
        df_bat = pd.read_csv("battery.csv")
        
        with monitor_placeholder.container():
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🌡️ Panel Temp", f"{float(df_live.iloc[-1]['Temperature_C']):.1f} °C")
                m2.metric("🔋 Raw Voltage", f"{float(df_bat.iloc[-1]['Raw_Volts']):.2f} V")
                m3.metric("⚡ Battery Level", f"{float(df_bat.iloc[-1]['SOC_Percentage']):.1f} %")
                m4.metric("🧠 AI Forecast", f"{float(df_bat.iloc[-1]['Predicted_TEG_Volts']):.3f} V")
except:
    monitor_placeholder.info("Awaiting live local data stream. Please verify `bridge.py` is running.")

# Part B: MongoDB Cloud Telemetry & Freshness Check (Tab 3)
cloud_df = get_latest_cloud_data()

with cloud_monitor_placeholder.container():
    st.subheader("📡 Live Node Telemetry")
    
    if not cloud_df.empty:
        latest_record = cloud_df.iloc[-1].to_dict()
        
        # Freshness check logic
        record_time = pd.to_datetime(latest_record.get('timestamp', datetime.utcnow()))
        record_time = record_time.tz_localize(None) # Ensure timezones match for math
        time_diff = (datetime.utcnow() - record_time).total_seconds()
        
        if time_diff > 60:
            st.error(f"⚠️ NODE OFFLINE: No new data received in {int(time_diff)} seconds. Showing last known state.")
        else:
            st.success("🟢 NODE ONLINE: Receiving live data stream.")
        
        # Display the data
        sensor_data = {k: v for k, v in latest_record.items() if k not in ['_id', 'timestamp']}
        cols = st.columns(len(sensor_data))
        
        for i, (key, value) in enumerate(sensor_data.items()):
            with cols[i]:
                with st.container(border=True):
                    clean_label = str(key).replace('_', ' ').title()
                    st.metric(label=clean_label, value=str(value))
                    
        st.caption("Raw Data Feed History")
        st.dataframe(cloud_df.drop(columns=['_id'], errors='ignore').tail(5), use_container_width=True)
        
    else:
        st.info("Awaiting live connection from ESP32 nodes...")

# This forces the app to refresh automatically every 5 seconds!
time.sleep(5)
st.rerun()
