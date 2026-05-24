import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime
import pymongo
from urllib.parse import quote_plus

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="TEG Energy Harvesting Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ TEG Hybrid Energy Harvesting Dashboard")
st.markdown("Live monitoring of solar-thermal and hot pot thermoelectric generation — SLIIT FYP G25-31")

# -----------------------------
# 2. MongoDB Atlas Connection
# -----------------------------
@st.cache_resource
def get_mongo_db():
    password = quote_plus("Pasindi@2002.22")
    uri = f"mongodb+srv://en22198822_db_user:{password}@cluster0.zrezdhz.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(uri)
    return client['FYP_Database']

db = get_mongo_db()
microgrid_col = db['microgrid_data']
battery_col   = db['battery_data']   # Pico/bridge.py writes here

# -----------------------------
# 3. Data Loaders
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

future_df  = load_future_data()
teg_df     = load_teg_results()
teg_model  = load_teg_model()

# -----------------------------
# 4. Live Cloud Data Fetchers
# -----------------------------
def get_microgrid_data(limit=50):
    """Fetch latest records from both ESP32s"""
    try:
        records = list(microgrid_col.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        if records:
            df = pd.DataFrame(records)
            df = df.iloc[::-1].reset_index(drop=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_battery_data(limit=50):
    """Fetch latest Pico/bridge.py battery readings"""
    try:
        records = list(battery_col.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        if records:
            df = pd.DataFrame(records)
            df = df.iloc[::-1].reset_index(drop=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_latest_values():
    """Get the single latest reading from each source"""
    try:
        solar  = microgrid_col.find_one({"source": "solar"},  {"_id": 0}, sort=[("timestamp", -1)]) or {}
        hotpot = microgrid_col.find_one({"source": "hotpot"}, {"_id": 0}, sort=[("timestamp", -1)]) or {}
        battery = battery_col.find_one({}, {"_id": 0}, sort=[("timestamp", -1)]) or {}
        return solar, hotpot, battery
    except:
        return {}, {}, {}

# -----------------------------
# 5. Sidebar Controls
# -----------------------------
st.sidebar.header("🎛️ Control Panel")

if not future_df.empty:
    min_date = future_df["datetime"].min().date()
    max_date = future_df["datetime"].max().date()
    selected_date = st.sidebar.date_input("Target Date", value=min_date, min_value=min_date, max_value=max_date)
    selected_hour = st.sidebar.selectbox("Target Hour (24H)", list(range(24)), index=12)

with st.sidebar.expander("🔋 Battery & Load Configuration", expanded=True):
    battery_capacity_wh = st.number_input("Battery Capacity (Wh)", min_value=0.1, max_value=5000.0, value=3.7, step=0.1)
    load_demand_w = st.number_input("Load Demand (W)", min_value=0, max_value=1000, value=50, step=5)

with st.sidebar.expander("🧠 AI Model Configuration", expanded=False):
    time_s           = st.number_input("Elapsed Time (s)",        value=600, step=10)
    sample_interval  = st.number_input("Sampling Interval (s)",   value=3,   step=1)
    temp_C           = st.number_input("Temperature (°C)",        value=55.0, step=0.1)
    temp_lag_1       = st.number_input("Temperature t-1 (°C)",   value=54.5, step=0.1)
    temp_lag_2       = st.number_input("Temperature t-2 (°C)",   value=54.0, step=0.1)
    voltage_lag_1    = st.number_input("Voltage t-1 (mV)",       value=60.0, step=0.1)
    voltage_lag_2    = st.number_input("Voltage t-2 (mV)",       value=58.0, step=0.1)
    heatsink_option  = st.selectbox("Heatsink Installed",        ["no", "yes"])
    ref_load_ohm     = st.number_input("Ref. Load Resistance (Ω)", value=10.0, step=0.1)

refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 3, 30, 5)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh"):
    st.rerun()

# Model status
if teg_model:
    st.sidebar.success("✅ AI Model Loaded")
else:
    st.sidebar.error("🚨 AI Model Offline")

# -----------------------------
# 6. Main Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📟 Live Monitoring", "📈 Scenario Optimizer", "🧠 AI Analytics"])

# ── TAB 1: LIVE MONITORING ──────────────────────────────────────
with tab1:
    st.header("Real-Time System Telemetry")

    solar_latest, hotpot_latest, battery_latest = get_latest_values()

    # --- Summary Metric Cards ---
    st.subheader("Current Readings")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        v = solar_latest.get('voltage', 0)
        st.metric("☀️ Solar TEG Voltage", f"{v:.3f} V")
    with c2:
        t = solar_latest.get('temperature', 0)
        st.metric("🌡️ Panel Temperature", f"{t:.1f} °C")
    with c3:
        v2 = hotpot_latest.get('voltage', 0)
        st.metric("🍲 Hot Pot TEG Voltage", f"{v2:.3f} V")
    with c4:
        t2 = hotpot_latest.get('temperature', 0)
        st.metric("🌡️ Pot Temperature", f"{t2:.1f} °C")
    with c5:
        soc = battery_latest.get('soc_percentage', 0)
        bv  = battery_latest.get('raw_volts', 0)
        st.metric("🔋 Battery SOC", f"{soc:.1f} %", delta=f"{bv:.2f}V")
    with c6:
        pteg = battery_latest.get('predicted_teg_volts', 0)
        st.metric("🧠 AI Predicted TEG", f"{pteg:.4f} V")

    st.markdown("---")

    # --- Live Charts ---
    mg_df  = get_microgrid_data(50)
    bat_df = get_battery_data(50)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("☀️ Member 1 — Solar TEG")
        if not mg_df.empty and 'source' in mg_df.columns:
            solar_df = mg_df[mg_df['source'] == 'solar']
            if not solar_df.empty and 'voltage' in solar_df.columns:
                fig = px.line(solar_df, x='timestamp', y='voltage',
                              title="Solar TEG Voltage (V)",
                              color_discrete_sequence=["#4299e1"])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                if 'temperature' in solar_df.columns:
                    fig2 = px.line(solar_df, x='timestamp', y='temperature',
                                   title="Panel Temperature (°C)",
                                   color_discrete_sequence=["#ed8936"])
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("⏳ Waiting for Member 1 (Solar) ESP32 data...")
        else:
            st.info("⏳ Waiting for Member 1 (Solar) ESP32 data...")

    with col2:
        st.subheader("🍲 Member 2 — Hot Pot TEG")
        if not mg_df.empty and 'source' in mg_df.columns:
            hotpot_df = mg_df[mg_df['source'] == 'hotpot']
            if not hotpot_df.empty and 'voltage' in hotpot_df.columns:
                fig3 = px.line(hotpot_df, x='timestamp', y='voltage',
                               title="Hot Pot TEG Voltage (V)",
                               color_discrete_sequence=["#48bb78"])
                fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig3, use_container_width=True)

                if 'temperature' in hotpot_df.columns:
                    fig4 = px.line(hotpot_df, x='timestamp', y='temperature',
                                   title="Pot Temperature (°C)",
                                   color_discrete_sequence=["#fc8181"])
                    fig4.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("⏳ Waiting for Member 2 (Hot Pot) ESP32 data...")
        else:
            st.info("⏳ Waiting for Member 2 (Hot Pot) ESP32 data...")

    st.markdown("---")
    st.subheader("🔋 Member 3 — Battery Management & AI Prediction")

    col3, col4 = st.columns(2)
    with col3:
        if not bat_df.empty and 'raw_volts' in bat_df.columns:
            fig5 = px.line(bat_df, x='timestamp', y='raw_volts',
                           title="Battery Voltage Over Time (V)",
                           color_discrete_sequence=["#4299e1"])
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("⏳ Waiting for Pico battery data...")

    with col4:
        if not bat_df.empty and 'soc_percentage' in bat_df.columns:
            fig6 = px.area(bat_df, x='timestamp', y='soc_percentage',
                           title="Battery SOC (%)",
                           color_discrete_sequence=["#48bb78"])
            fig6.add_hline(y=40, line_dash="dash", line_color="#ed8936",
                          annotation_text="Load ON threshold")
            fig6.add_hline(y=20, line_dash="dash", line_color="#fc8181",
                          annotation_text="Low battery warning")
            fig6.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig6, use_container_width=True)

    if not bat_df.empty and 'predicted_teg_volts' in bat_df.columns:
        fig7 = px.line(bat_df, x='timestamp', y='predicted_teg_volts',
                       title="AI Predicted TEG Voltage Over Time (V)",
                       color_discrete_sequence=["#9f7aea"])
        fig7.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig7, use_container_width=True)

    # Last updated timestamp
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshes every {refresh_rate}s")

# ── TAB 2: SCENARIO OPTIMIZER ────────────────────────────────────
with tab2:
    st.header("Scenario Optimizer")

    if future_df.empty:
        st.warning("Future solar prediction CSV not found.")
    else:
        selected_datetime = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00")
        match = future_df[future_df["datetime"] == selected_datetime]

        if match.empty:
            st.warning("No solar prediction found for selected date and time.")
        else:
            row = match.iloc[0]
            predicted_pv_power = float(row["predicted_pv_power"])

            heatsink_bin   = 1 if heatsink_option == "yes" else 0
            dtemp_dt       = (temp_C - temp_lag_1) / max(sample_interval, 1)
            temp_rolling_3 = np.mean([temp_C, temp_lag_1, temp_lag_2])

            predicted_teg_mV = 0.0
            predicted_teg_W  = 0.0

            if teg_model:
                live_input = pd.DataFrame([{
                    "time_s": time_s, "temp_C": temp_C, "heatsink_bin": heatsink_bin,
                    "dtemp_dt": dtemp_dt, "temp_rolling_3": temp_rolling_3,
                    "temp_lag_1": temp_lag_1, "temp_lag_2": temp_lag_2,
                    "voltage_lag_1": voltage_lag_1, "voltage_lag_2": voltage_lag_2
                }])
                predicted_teg_mV = max(float(teg_model.predict(live_input)[0]), 0)
                predicted_teg_V  = predicted_teg_mV / 1000.0
                predicted_teg_W  = (predicted_teg_V ** 2) / ref_load_ohm

            total_power = predicted_pv_power + predicted_teg_W
            net_power   = total_power - load_demand_w

            # Latest SOC from MongoDB
            _, _, battery_latest = get_latest_values()
            battery_soc_now = battery_latest.get('soc_percentage', 60.0)
            soc_change = (net_power / battery_capacity_wh) * 100
            battery_soc_next = max(20, min(100, battery_soc_now + soc_change))

            def recommendation(total_p, soc_next, demand):
                if total_p >= 1.5 * demand and soc_next > 60:
                    return "✅ Run heavy load and charge battery"
                elif total_p >= demand and soc_next > 40:
                    return "🟡 Run essential loads only"
                elif predicted_teg_W > 0 and soc_next > 30:
                    return "🟠 Use TEG-assisted operation"
                else:
                    return "🔴 Battery saving mode — delay non-essential loads"

            rec = recommendation(total_power, battery_soc_next, load_demand_w)

            with st.container(border=True):
                st.subheader("Predicted Output")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Solar Power (W)",    f"{predicted_pv_power:.2f}")
                c2.metric("TEG Power (mW)",     f"{predicted_teg_W*1000:.2f}")
                c3.metric("Total Power (W)",    f"{total_power:.2f}")
                c4.metric("Next Battery SOC",   f"{battery_soc_next:.1f}%")

            st.info(f"**AI Recommendation:** {rec}")

            daily_df = future_df[future_df["datetime"].dt.date == selected_date].copy()
            daily_df["hour"] = daily_df["datetime"].dt.hour
            daily_df["total_power"] = daily_df["predicted_pv_power"] + predicted_teg_W
            fig = px.line(daily_df, x="hour", y="total_power",
                          title="24-Hour Total Power Forecast (Solar + TEG)",
                          color_discrete_sequence=["#ed8936"])
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: AI ANALYTICS ──────────────────────────────────────────
with tab3:
    st.header("AI Model Analytics")

    if not teg_df.empty:
        st.subheader("TEG Voltage Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            if 'actual' in teg_df.columns and 'predicted' in teg_df.columns:
                fig = px.scatter(teg_df, x='actual', y='predicted',
                                 title="Actual vs Predicted Voltage",
                                 color_discrete_sequence=["#9f7aea"])
                fig.add_shape(type='line', x0=teg_df['actual'].min(), y0=teg_df['actual'].min(),
                             x1=teg_df['actual'].max(), y1=teg_df['actual'].max(),
                             line=dict(color='#fc8181', dash='dash'))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'actual' in teg_df.columns and 'predicted' in teg_df.columns:
                teg_df['error'] = teg_df['predicted'] - teg_df['actual']
                fig2 = px.histogram(teg_df, x='error', title="Prediction Error Distribution",
                                    color_discrete_sequence=["#38b2ac"])
                st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📡 All Live Cloud Records")
    mg_df = get_microgrid_data(100)
    if not mg_df.empty:
        st.dataframe(mg_df.drop(columns=['_id'], errors='ignore'), use_container_width=True)
    else:
        st.info("No cloud data yet.")

# -----------------------------
# 7. Auto-Refresh
# -----------------------------
time.sleep(refresh_rate)
st.rerun()
