import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Hybrid Energy Prediction Dashboard", layout="wide")

st.title("Hybrid Energy Prediction and Optimization Dashboard")
st.write(
    "This dashboard combines solar-side prediction and live TEG-side AI prediction to estimate "
    "available power, battery behavior, and the best operating mode for the hybrid system."
)

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_future_data():
    df = pd.read_csv("Data/future_predictions_2026_2030.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

@st.cache_data
def load_teg_results():
    df = pd.read_csv("Data/teg_voltage_prediction_results.csv")
    return df

@st.cache_resource
def load_teg_model():
    model_path = "Models/teg_voltage_rf_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

future_df = load_future_data()
teg_df = load_teg_results()
teg_model = load_teg_model()

if future_df.empty:
    st.error("Future solar prediction CSV is empty or missing.")
    st.stop()

# -----------------------------
# Sidebar Inputs
# -----------------------------
min_date = future_df["datetime"].min().date()
max_date = future_df["datetime"].max().date()

st.sidebar.header("User Inputs")

selected_date = st.sidebar.date_input(
    "Select Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

selected_hour = st.sidebar.selectbox("Select Hour", list(range(24)), index=12)

battery_soc_now = st.sidebar.number_input(
    "Current Battery SOC (%)",
    min_value=20,
    max_value=100,
    value=60,
    step=1
)

load_demand_w = st.sidebar.number_input(
    "Load Demand (W)",
    min_value=0,
    max_value=1000,
    value=50,
    step=5
)

battery_capacity_wh = st.sidebar.number_input(
    "Battery Capacity (Wh)",
    min_value=50,
    max_value=5000,
    value=500,
    step=50
)

st.sidebar.markdown("---")
st.sidebar.subheader("Live TEG AI Inputs")

time_s = st.sidebar.number_input(
    "Elapsed Time for TEG Test (s)",
    min_value=0,
    max_value=100000,
    value=600,
    step=10
)

sample_interval_s = st.sidebar.number_input(
    "Sampling Interval (s)",
    min_value=1,
    max_value=3600,
    value=3,
    step=1
)

temp_C = st.sidebar.number_input(
    "Current TEG Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=55.0,
    step=0.1
)

temp_lag_1 = st.sidebar.number_input(
    "Previous Temperature t-1 (°C)",
    min_value=0.0,
    max_value=200.0,
    value=54.5,
    step=0.1
)

temp_lag_2 = st.sidebar.number_input(
    "Previous Temperature t-2 (°C)",
    min_value=0.0,
    max_value=200.0,
    value=54.0,
    step=0.1
)

voltage_lag_1 = st.sidebar.number_input(
    "Previous Voltage t-1 (mV)",
    min_value=0.0,
    max_value=1000.0,
    value=60.0,
    step=0.1
)

voltage_lag_2 = st.sidebar.number_input(
    "Previous Voltage t-2 (mV)",
    min_value=0.0,
    max_value=1000.0,
    value=58.0,
    step=0.1
)

heatsink_option = st.sidebar.selectbox(
    "Heatsink Condition",
    ["no", "yes"]
)

reference_load_ohm = st.sidebar.number_input(
    "Reference Load Resistance for TEG Power (Ω)",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=0.1
)

# -----------------------------
# Solar Prediction Section
# -----------------------------
selected_datetime = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00")
match = future_df[future_df["datetime"] == selected_datetime]

if match.empty:
    st.warning("No solar prediction found for the selected date and time.")
    st.stop()

row = match.iloc[0]

predicted_irradiance = float(row["predicted_irradiance"])
predicted_pv_power = float(row["predicted_pv_power"])
base_solar_recommendation = str(row["base_solar_recommendation"])

# -----------------------------
# Live TEG Prediction
# -----------------------------
heatsink_bin = 1 if heatsink_option == "yes" else 0
dtemp_dt = (temp_C - temp_lag_1) / sample_interval_s
temp_rolling_3 = np.mean([temp_C, temp_lag_1, temp_lag_2])

predicted_teg_voltage_mV = 0.0
predicted_teg_power_W = 0.0
predicted_teg_power_mW = 0.0

if teg_model is not None:
    live_input = pd.DataFrame([{
        "time_s": time_s,
        "temp_C": temp_C,
        "heatsink_bin": heatsink_bin,
        "dtemp_dt": dtemp_dt,
        "temp_rolling_3": temp_rolling_3,
        "temp_lag_1": temp_lag_1,
        "temp_lag_2": temp_lag_2,
        "voltage_lag_1": voltage_lag_1,
        "voltage_lag_2": voltage_lag_2
    }])

    predicted_teg_voltage_mV = float(teg_model.predict(live_input)[0])
    predicted_teg_voltage_mV = max(predicted_teg_voltage_mV, 0)

    predicted_teg_voltage_V = predicted_teg_voltage_mV / 1000.0
    predicted_teg_power_W = (predicted_teg_voltage_V ** 2) / reference_load_ohm
    predicted_teg_power_mW = predicted_teg_power_W * 1000
else:
    st.warning("TEG model file not found. Live TEG prediction is disabled.")

# -----------------------------
# Hybrid Calculation
# -----------------------------
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

# -----------------------------
# Top Results
# -----------------------------
st.subheader("Hybrid Prediction for Selected Time")
st.caption(
    "Live TEG voltage is predicted using the trained Random Forest model from temperature, "
    "time, heatsink condition, and lag-based inputs."
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Predicted Irradiance (W/m²)", f"{predicted_irradiance:.2f}")
c2.metric("Predicted PV Power (W)", f"{predicted_pv_power:.2f}")
c3.metric("Predicted TEG Voltage (mV)", f"{predicted_teg_voltage_mV:.2f}")
c4.metric("Predicted TEG Power (mW)", f"{predicted_teg_power_mW:.4f}")
c5.metric("Total Available Power (W)", f"{total_power:.4f}")
c6.metric("Estimated Next SOC (%)", f"{battery_soc_next:.2f}")

st.info(f"Base Solar Recommendation: {base_solar_recommendation}")

if "heavy load" in recommendation.lower():
    st.success(recommendation)
elif "essential" in recommendation.lower() or "teg-assisted" in recommendation.lower():
    st.warning(recommendation)
else:
    st.error(recommendation)

# -----------------------------
# Selected Inputs Summary
# -----------------------------
st.subheader("Selected Input Summary")
s1, s2 = st.columns(2)

with s1:
    st.write(f"**Date:** {selected_date}")
    st.write(f"**Hour:** {selected_hour:02d}:00")
    st.write(f"**Battery SOC Now:** {battery_soc_now}%")
    st.write(f"**Load Demand:** {load_demand_w} W")
    st.write(f"**Battery Capacity:** {battery_capacity_wh} Wh")

with s2:
    st.write(f"**TEG Time:** {time_s} s")
    st.write(f"**Current Temperature:** {temp_C:.2f} °C")
    st.write(f"**Previous Temperature t-1:** {temp_lag_1:.2f} °C")
    st.write(f"**Previous Temperature t-2:** {temp_lag_2:.2f} °C")
    st.write(f"**Previous Voltage t-1:** {voltage_lag_1:.2f} mV")
    st.write(f"**Previous Voltage t-2:** {voltage_lag_2:.2f} mV")
    st.write(f"**Heatsink:** {heatsink_option}")
    st.write(f"**dT/dt:** {dtemp_dt:.4f} °C/s")
    st.write(f"**Reference Load:** {reference_load_ohm:.2f} Ω")

# -----------------------------
# Daily Solar + Hybrid Profile
# -----------------------------
daily_df = future_df[future_df["datetime"].dt.date == selected_date].copy()
daily_df["hour"] = daily_df["datetime"].dt.hour
daily_df["daily_total_power"] = daily_df["predicted_pv_power"] + predicted_teg_power_W

st.subheader("Daily Prediction Profile")

fig1 = px.line(
    daily_df,
    x="hour",
    y="predicted_irradiance",
    title="Predicted Irradiance Throughout the Day"
)
st.plotly_chart(fig1, width="stretch")

fig2 = px.line(
    daily_df,
    x="hour",
    y="predicted_pv_power",
    title="Predicted PV Power Throughout the Day"
)
st.plotly_chart(fig2, width="stretch")

fig3 = px.line(
    daily_df,
    x="hour",
    y="daily_total_power",
    title="Predicted Total Power Throughout the Day"
)
st.plotly_chart(fig3, width="stretch")

st.subheader("Selected Day Data Table")
st.dataframe(
    daily_df[["datetime", "predicted_irradiance", "predicted_pv_power", "daily_total_power"]],
    width="stretch"
)

# -----------------------------
# TEG AI Evaluation Section
# -----------------------------
st.subheader("TEG AI Prediction Module")

if not teg_df.empty:
    avg_actual = teg_df["voltage_mV"].mean()
    avg_pred = teg_df["rf_pred_mV"].mean()
    avg_error = (teg_df["voltage_mV"] - teg_df["rf_pred_mV"]).abs().mean()

    t1, t2, t3 = st.columns(3)
    t1.metric("Average Actual TEG Voltage (mV)", f"{avg_actual:.2f}")
    t2.metric("Average Predicted TEG Voltage (mV)", f"{avg_pred:.2f}")
    t3.metric("Average Absolute Error (mV)", f"{avg_error:.2f}")

    st.info("Best TEG model selected: Random Forest")

    teg_df["sample_index"] = range(len(teg_df))

    fig_teg_line = px.line(
        teg_df.head(120),
        x="sample_index",
        y=["voltage_mV", "rf_pred_mV"],
        title="Actual vs Predicted TEG Voltage"
    )
    st.plotly_chart(fig_teg_line, width="stretch")

    fig_teg_scatter = px.scatter(
        teg_df,
        x="voltage_mV",
        y="rf_pred_mV",
        color="heatsink",
        title="Actual vs Predicted TEG Voltage by Heatsink Condition",
        labels={
            "voltage_mV": "Actual Voltage (mV)",
            "rf_pred_mV": "Predicted Voltage (mV)"
        }
    )
    st.plotly_chart(fig_teg_scatter, width="stretch")

    heatsink_summary = teg_df.groupby("heatsink")[["voltage_mV", "rf_pred_mV"]].mean().reset_index()

    fig_heatsink = px.bar(
        heatsink_summary,
        x="heatsink",
        y=["voltage_mV", "rf_pred_mV"],
        barmode="group",
        title="Average TEG Voltage by Heatsink Condition"
    )
    st.plotly_chart(fig_heatsink, width="stretch")

    st.subheader("TEG Prediction Results Table")
    st.dataframe(
        teg_df[["test_id", "heatsink", "time_s", "temp_C", "voltage_mV", "rf_pred_mV"]].head(50),
        width="stretch"
    )
else:
    st.warning("TEG prediction results file not found or empty.")