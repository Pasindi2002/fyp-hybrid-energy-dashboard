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
# 1. Page Config & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="TEG Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Invisible JS refresh — no dimming, no flash
st.markdown("""
<script>
    setTimeout(function() { window.location.reload(); }, 10000);
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Suppress ANY overlay/dimming Streamlit adds during rerun */
    [data-testid="stAppViewBlockContainer"] > div[style*="opacity"] { opacity: 1 !important; }
    .stApp > div { opacity: 1 !important; transition: none !important; }
    iframe { pointer-events: none; }

    /* Main background */
    .stApp { background-color: #0a0e1a; }

    /* Custom metric card */
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1220 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7db3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .metric-unit {
        font-size: 13px;
        color: #6b7db3;
    }

    /* Status badges */
    .badge-online {
        background: #0d2d1f;
        color: #00e676;
        border: 1px solid #00e676;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-waiting {
        background: #2d1f0d;
        color: #ff9800;
        border: 1px solid #ff9800;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1220;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #1e2740;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #6b7db3;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2d50 !important;
        color: #ffffff !important;
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #6b7db3;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2740;
    }

    /* Dashboard header */
    .dash-header {
        background: linear-gradient(135deg, #0d1528 0%, #0a0e1a 100%);
        border: 1px solid #1e2740;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Member card header */
    .member-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1220 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Recommendation box */
    .rec-box {
        background: linear-gradient(135deg, #1a2a1a 0%, #0d1a0d 100%);
        border: 1px solid #00e676;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .rec-box-warn {
        background: linear-gradient(135deg, #2a1a0d 0%, #1a0d00 100%);
        border: 1px solid #ff9800;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .rec-box-danger {
        background: linear-gradient(135deg, #2a0d0d 0%, #1a0000 100%);
        border: 1px solid #f44336;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. MongoDB Atlas Connection
# -----------------------------
@st.cache_resource
def get_mongo_db():
    import os
    uri = st.secrets.get("MONGO_URI") or os.environ.get("MONGO_URI")
    client = pymongo.MongoClient(uri)
    return client['FYP_Database']

db            = get_mongo_db()
microgrid_col = db['microgrid_data']
battery_col   = db['battery_data']

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
    if os.path.exists("Models/teg_voltage_rf_model.pkl"):
        return joblib.load("Models/teg_voltage_rf_model.pkl")
    return None

future_df = load_future_data()
teg_df    = load_teg_results()
teg_model = load_teg_model()

# -----------------------------
# 4. Data Fetchers
# -----------------------------
def get_microgrid_data(limit=60):
    try:
        records = list(microgrid_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        if records:
            df = pd.DataFrame(records).iloc[::-1].reset_index(drop=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_battery_data(limit=60):
    try:
        records = list(battery_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        if records:
            df = pd.DataFrame(records).iloc[::-1].reset_index(drop=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_latest_values():
    try:
        solar   = microgrid_col.find_one({"source": "solar"},  {"_id": 0}, sort=[("timestamp", -1)]) or {}
        hotpot  = microgrid_col.find_one({"source": "hotpot"}, {"_id": 0}, sort=[("timestamp", -1)]) or {}
        battery = battery_col.find_one({}, {"_id": 0}, sort=[("timestamp", -1)]) or {}
        return solar, hotpot, battery
    except:
        return {}, {}, {}

# Chart layout helper
def chart_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#8899cc')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#8899cc',
        showlegend=False,
        margin=dict(l=0, r=0, t=35, b=0),
        xaxis=dict(gridcolor='#1a2040', showgrid=True, tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#1a2040', showgrid=True, tickfont=dict(size=10)),
        height=220
    )
    return fig

def metric_html(label, value, unit="", color="#4fc3f7"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}<span class="metric-unit"> {unit}</span></div>
    </div>
    """

# -----------------------------
# 5. Dashboard Header
# -----------------------------
solar_latest, hotpot_latest, battery_latest = get_latest_values()
mg_df  = get_microgrid_data(60)
bat_df = get_battery_data(60)

col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
with col_h1:
    st.markdown("## ⚡ TEG Hybrid Energy Dashboard")
    st.markdown("<span style='color:#6b7db3;font-size:13px;'>SLIIT FYP G25-31 — Live Thermoelectric Energy Monitoring</span>", unsafe_allow_html=True)
with col_h2:
    m1_status = "🟢 Online" if not mg_df.empty and 'source' in mg_df.columns and 'solar' in mg_df['source'].values else "🟡 Waiting"
    m2_status = "🟢 Online" if not mg_df.empty and 'source' in mg_df.columns and 'hotpot' in mg_df['source'].values else "🟡 Waiting"
    m3_status = "🟢 Online" if not bat_df.empty else "🟡 Waiting"
    st.markdown(f"""
    <div style='background:#0d1220;border:1px solid #1e2740;border-radius:10px;padding:12px 16px;font-size:12px;'>
    <div style='color:#6b7db3;margin-bottom:6px;font-weight:600;'>NODE STATUS</div>
    <div>☀️ Member 1 (Solar): <strong style='color:#fff'>{m1_status}</strong></div>
    <div>🍲 Member 2 (Hot Pot): <strong style='color:#fff'>{m2_status}</strong></div>
    <div>🔋 Member 3 (Battery): <strong style='color:#fff'>{m3_status}</strong></div>
    </div>
    """, unsafe_allow_html=True)
with col_h3:
    soc  = float(battery_latest.get('soc_percentage') or 0)
    bv   = float(battery_latest.get('raw_volts') or 0)
    pteg = float(battery_latest.get('predicted_teg_volts') or 0)
    soc_color = "#00e676" if soc > 40 else "#ff9800" if soc > 20 else "#f44336"
    st.markdown(f"""
    <div style='background:#0d1220;border:1px solid #1e2740;border-radius:10px;padding:12px 16px;font-size:12px;'>
    <div style='color:#6b7db3;margin-bottom:6px;font-weight:600;'>BATTERY STATUS</div>
    <div>SOC: <strong style='color:{soc_color}'>{soc:.1f}%</strong></div>
    <div>Voltage: <strong style='color:#fff'>{bv:.2f}V</strong></div>
    <div>AI TEG Pred: <strong style='color:#9c88ff'>{pteg:.4f}V</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:16px 0;border-top:1px solid #1e2740;'></div>", unsafe_allow_html=True)

# -----------------------------
# 6. Main Tabs
# -----------------------------
tab_overview, tab_m1, tab_m2, tab_m3, tab_ai, tab_optimizer = st.tabs([
    "📊 Overview",
    "☀️ Member 1 — Solar TEG",
    "🍲 Member 2 — Hot Pot TEG",
    "🔋 Member 3 — Battery & AI",
    "🧠 AI Analytics",
    "📈 Scenario Optimizer"
])

# ── OVERVIEW TAB ─────────────────────────────────────────────────
with tab_overview:
    st.markdown("<div class='section-header'>System Summary — All Members</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        v = float(solar_latest.get('voltage') or 0)
        st.markdown(metric_html("Solar TEG Voltage", f"{v:.3f}", "V", "#4fc3f7"), unsafe_allow_html=True)
    with c2:
        t = float(solar_latest.get('temperature') or 0)
        st.markdown(metric_html("Panel Temperature", f"{t:.1f}", "°C", "#ffb74d"), unsafe_allow_html=True)
    with c3:
        v2 = float(hotpot_latest.get('voltage') or 0)
        st.markdown(metric_html("Hot Pot TEG Voltage", f"{v2:.3f}", "V", "#66bb6a"), unsafe_allow_html=True)
    with c4:
        t2 = float(hotpot_latest.get('temperature') or 0)
        st.markdown(metric_html("Pot Temperature", f"{t2:.1f}", "°C", "#ef5350"), unsafe_allow_html=True)
    with c5:
        soc = float(battery_latest.get('soc_percentage') or 0)
        soc_color = "#00e676" if soc > 40 else "#ff9800" if soc > 20 else "#f44336"
        st.markdown(metric_html("Battery SOC", f"{soc:.1f}", "%", soc_color), unsafe_allow_html=True)
    with c6:
        pteg = float(battery_latest.get('predicted_teg_volts') or 0)
        st.markdown(metric_html("AI Predicted TEG", f"{pteg:.4f}", "V", "#ce93d8"), unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Live Data — All Sources</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**☀️ Solar TEG Voltage**")
        if not mg_df.empty and 'source' in mg_df.columns:
            s_df = mg_df[mg_df['source'] == 'solar']
            if not s_df.empty and 'voltage' in s_df.columns:
                fig = px.area(s_df, x='timestamp', y='voltage', color_discrete_sequence=["#4fc3f7"])
                st.plotly_chart(chart_layout(fig), use_container_width=True)
            else:
                st.info("⏳ Waiting for solar data...")
        else:
            st.info("⏳ Waiting for solar data...")

    with col_b:
        st.markdown("**🍲 Hot Pot TEG Voltage**")
        if not mg_df.empty and 'source' in mg_df.columns:
            h_df = mg_df[mg_df['source'] == 'hotpot']
            if not h_df.empty and 'voltage' in h_df.columns:
                fig2 = px.area(h_df, x='timestamp', y='voltage', color_discrete_sequence=["#66bb6a"])
                st.plotly_chart(chart_layout(fig2), use_container_width=True)
            else:
                st.info("⏳ Waiting for hotpot data...")
        else:
            st.info("⏳ Waiting for hotpot data...")

    with col_c:
        st.markdown("**🔋 Battery Voltage**")
        if not bat_df.empty and 'raw_volts' in bat_df.columns:
            fig3 = px.area(bat_df, x='timestamp', y='raw_volts', color_discrete_sequence=["#ce93d8"])
            st.plotly_chart(chart_layout(fig3), use_container_width=True)
        else:
            st.info("⏳ Waiting for battery data...")

    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} — auto-refreshes every few seconds")

# ── MEMBER 1 TAB ─────────────────────────────────────────────────
with tab_m1:
    st.markdown("""
    <div class='member-header'>
        <div style='font-size:18px;font-weight:700;color:#fff;'>☀️ Solar Panel TEG System (Chamath)</div>
        <div style='font-size:12px;color:#6b7db3;margin-top:4px;'>PV Panel → TEG → BQ25570 MPPT → LiPo Battery → Output</div>
    </div>
    """, unsafe_allow_html=True)

    if not mg_df.empty and 'source' in mg_df.columns:
        solar_df = mg_df[mg_df['source'] == 'solar']
        if not solar_df.empty:

            # ── Last seen status ─────────────────────────
            try:
                ts = pd.to_datetime(solar_df['timestamp'].iloc[-1])
                if ts.tzinfo is not None:
                    ts = ts.tz_convert('UTC').replace(tzinfo=None)
                secs_ago = abs((datetime.utcnow() - ts.to_pydatetime()).total_seconds())
            except Exception:
                secs_ago = 9999
            mins_ago = secs_ago / 60

            if mins_ago < 2:
                st.success(f"🟢 ESP32 Live — data received {int(secs_ago)}s ago")
            elif mins_ago < 10:
                st.warning(f"🟡 Disconnected — last seen {int(mins_ago)} mins ago")
            else:
                st.error(f"🔴 Offline — last seen {int(mins_ago)} mins ago (historical data)")

            # Helper: safe last value from solar_df
            def slast(col, decimals=3):
                return float(solar_df[col].iloc[-1]) if col in solar_df.columns else 0.0

            # ── CH1: TEG → BQ25570 ───────────────────────
            st.markdown("<div class='section-header'>CH1 — TEG → BQ25570 Input</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v_teg_val = slast('v_teg')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">V_TEG</div>
                    <div class="metric-value" style="color:#4fc3f7">{v_teg_val:.3f}<span class="metric-unit"> V</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                i_teg_val = slast('i_teg')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">I_TEG</div>
                    <div class="metric-value" style="color:#ffffff">{i_teg_val:.2f}<span class="metric-unit"> mA</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                p_teg_val = slast('p_teg')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P_TEG</div>
                    <div class="metric-value" style="color:#ffffff">{p_teg_val:.2f}<span class="metric-unit"> mW</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                e_teg_val = slast('energy_teg')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ENERGY FROM TEG</div>
                    <div class="metric-value" style="color:#ffffff">{e_teg_val:.3f}<span class="metric-unit"> J</span></div>
                </div>""", unsafe_allow_html=True)

            # ── CH2: BQ25570 → LiPo ─────────────────────
            st.markdown("<div class='section-header'>CH2 — BQ25570 → LiPo Charge</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v_bat_val = slast('v_bat')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">V_BAT</div>
                    <div class="metric-value" style="color:#ffffff">{v_bat_val:.3f}<span class="metric-unit"> V</span></div>
                    <div style="font-size:10px;color:#6b7db3;margin-top:4px;">Li-ion 3.0–4.2 V</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                i_charge_val = slast('i_charge')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">I_CHARGE</div>
                    <div class="metric-value" style="color:#ffffff">{i_charge_val:.2f}<span class="metric-unit"> mA</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                p_charge_val = slast('p_charge')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P_CHARGE</div>
                    <div class="metric-value" style="color:#ffffff">{p_charge_val:.2f}<span class="metric-unit"> mW</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                e_harv_val = slast('energy_harvested')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ENERGY HARVESTED</div>
                    <div class="metric-value" style="color:#ffffff">{e_harv_val:.3f}<span class="metric-unit"> J</span></div>
                </div>""", unsafe_allow_html=True)

            # ── CH3: LiPo → Output ───────────────────────
            st.markdown("<div class='section-header'>CH3 — LiPo → Output / Member 3</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v_out_val = slast('v_out')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">V_OUT</div>
                    <div class="metric-value" style="color:#ffffff">{v_out_val:.3f}<span class="metric-unit"> V</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                i_out_val = slast('i_out')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">I_OUT</div>
                    <div class="metric-value" style="color:#ffffff">{i_out_val:.2f}<span class="metric-unit"> mA</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                p_out_val = slast('p_out')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P_OUT</div>
                    <div class="metric-value" style="color:#ffffff">{p_out_val:.2f}<span class="metric-unit"> mW</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                # Last HTTP POST status — show seconds since last record
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">LAST HTTP POST</div>
                    <div class="metric-value" style="color:#00e676">201</div>
                    <div style="font-size:10px;color:#6b7db3;margin-top:4px;">{int(secs_ago)}s ago</div>
                </div>""", unsafe_allow_html=True)

            # ── THERMAL ──────────────────────────────────
            st.markdown("<div class='section-header'>Thermal — TEC ΔT</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                t_hot_val = slast('t_hot')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">T HOT</div>
                    <div class="metric-value" style="color:#ef5350">{t_hot_val:.2f}<span class="metric-unit"> °C</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                t_cold_val = slast('t_cold')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">T COLD</div>
                    <div class="metric-value" style="color:#4fc3f7">{t_cold_val:.2f}<span class="metric-unit"> °C</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                dt_val = slast('delta_t')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ΔT</div>
                    <div class="metric-value" style="color:#ffb74d">{dt_val:.2f}<span class="metric-unit"> °C</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                # Seebeck coefficient info card (matches Chamath's "A per module" card)
                seebeck = (v_teg_val * 1000 / dt_val) if dt_val > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">A (PER MODULE)</div>
                    <div class="metric-value" style="color:#ffffff">{seebeck:.1f}<span class="metric-unit"> mV/°C</span></div>
                    <div style="font-size:10px;color:#6b7db3;margin-top:4px;">TEC1-12706 nominal: ~50 mV/°C</div>
                </div>""", unsafe_allow_html=True)

            # ── CHARTS ───────────────────────────────────
            st.markdown("<div class='section-header'>Live Charts</div>", unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                if 'v_teg' in solar_df.columns:
                    fig = px.line(solar_df, x='timestamp', y='v_teg',
                                  color_discrete_sequence=["#4fc3f7"])
                    st.plotly_chart(chart_layout(fig, "TEG Voltage (V)"), use_container_width=True)
            with r2:
                if 't_hot' in solar_df.columns and 't_cold' in solar_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=solar_df['timestamp'], y=solar_df['t_hot'],
                        name='T Hot', line=dict(color='#ef5350', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=solar_df['timestamp'], y=solar_df['t_cold'],
                        name='T Cold', line=dict(color='#4fc3f7', width=2)
                    ))
                    fig.update_layout(showlegend=True, legend=dict(font=dict(color='#8899cc')))
                    st.plotly_chart(chart_layout(fig, "Temperature Hot vs Cold (°C)"), use_container_width=True)

            r3, r4 = st.columns(2)
            with r3:
                if 'p_teg' in solar_df.columns:
                    fig = px.area(solar_df, x='timestamp', y='p_teg',
                                  color_discrete_sequence=["#66bb6a"])
                    st.plotly_chart(chart_layout(fig, "TEG Power (mW)"), use_container_width=True)
            with r4:
                if 'v_bat' in solar_df.columns:
                    fig = px.area(solar_df, x='timestamp', y='v_bat',
                                  color_discrete_sequence=["#ce93d8"])
                    st.plotly_chart(chart_layout(fig, "Battery Voltage (V)"), use_container_width=True)

        else:
            st.info("⏳ Waiting for Member 1 (Solar) ESP32 data...")
    else:
        st.info("⏳ Waiting for Member 1 (Solar) ESP32 data...")

# ── MEMBER 2 TAB ─────────────────────────────────────────────────
with tab_m2:
    st.markdown("""
    <div class='member-header'>
        <div style='font-size:18px;font-weight:700;color:#fff;'>🍲 Hot Pot TEG System</div>
        <div style='font-size:12px;color:#6b7db3;margin-top:4px;'>Heat extracted from domestic hot pot → TEG → DC–DC Boost → Battery</div>
    </div>
    """, unsafe_allow_html=True)

    if not mg_df.empty and 'source' in mg_df.columns:
        hotpot_df = mg_df[mg_df['source'] == 'hotpot']
        if not hotpot_df.empty:
            hv  = float(hotpot_df['voltage'].iloc[-1])         if 'voltage'         in hotpot_df.columns else 0.0
            ht  = float(hotpot_df['temperature'].iloc[-1])     if 'temperature'     in hotpot_df.columns else 0.0
            hbv = float(hotpot_df['battery_voltage'].iloc[-1]) if 'battery_voltage' in hotpot_df.columns else 0.0
            hpw = float(hotpot_df['power_mW'].iloc[-1])        if 'power_mW'        in hotpot_df.columns else 0.0
            hma = float(hotpot_df['current_mA'].iloc[-1])      if 'current_mA'      in hotpot_df.columns else 0.0

            # Last seen for Member 2
            try:
                ts_hp = pd.to_datetime(hotpot_df['timestamp'].iloc[-1])
                if ts_hp.tzinfo is not None:
                    ts_hp = ts_hp.tz_convert('UTC').replace(tzinfo=None)
                secs_ago_hp = abs((datetime.utcnow() - ts_hp.to_pydatetime()).total_seconds())
            except Exception:
                secs_ago_hp = 9999
            mins_ago_hp = secs_ago_hp / 60

            if mins_ago_hp < 2:
                st.success(f"🟢 ESP32 Live — data received {int(secs_ago_hp)}s ago")
            elif mins_ago_hp < 10:
                st.warning(f"🟡 Disconnected — last seen {int(mins_ago_hp)} mins ago")
            else:
                st.error(f"🔴 Offline — last seen {int(mins_ago_hp)} mins ago (historical data)")

            # ── Main telemetry cards matching TEG Live Telemetry layout ──
            st.markdown("<div class='section-header'>Live Telemetry</div>", unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value" style="color:#00e5ff">{ht:.1f}<span class="metric-unit"> °C</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current</div>
                    <div class="metric-value" style="color:#00e5ff">{hma:.2f}<span class="metric-unit"> mA</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Voltage</div>
                    <div class="metric-value" style="color:#00e5ff">{hv:.3f}<span class="metric-unit"> V</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Power</div>
                    <div class="metric-value" style="color:#00e5ff">{hpw:.2f}<span class="metric-unit"> mW</span></div>
                </div>""", unsafe_allow_html=True)
            with c5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Battery Voltage</div>
                    <div class="metric-value" style="color:#00e5ff">{hbv:.3f}<span class="metric-unit"> V</span></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Hot Pot TEG Charts</div>", unsafe_allow_html=True)

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                if 'voltage' in hotpot_df.columns:
                    fig = px.line(hotpot_df, x='timestamp', y='voltage',
                                  color_discrete_sequence=["#66bb6a"])
                    st.plotly_chart(chart_layout(fig, "TEG Output Voltage (V)"), use_container_width=True)
            with r1c2:
                if 'temperature' in hotpot_df.columns:
                    fig = px.line(hotpot_df, x='timestamp', y='temperature',
                                  color_discrete_sequence=["#ef5350"])
                    st.plotly_chart(chart_layout(fig, "Pot Temperature (°C)"), use_container_width=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                if 'battery_voltage' in hotpot_df.columns:
                    fig = px.area(hotpot_df, x='timestamp', y='battery_voltage',
                                  color_discrete_sequence=["#4fc3f7"])
                    st.plotly_chart(chart_layout(fig, "Battery Voltage (V)"), use_container_width=True)
            with r2c2:
                if 'power_mW' in hotpot_df.columns:
                    fig = px.area(hotpot_df, x='timestamp', y='power_mW',
                                  color_discrete_sequence=["#ffb74d"])
                    st.plotly_chart(chart_layout(fig, "TEG Power Output (mW)"), use_container_width=True)

            if 'current_mA' in hotpot_df.columns:
                fig = px.line(hotpot_df, x='timestamp', y='current_mA',
                              color_discrete_sequence=["#ce93d8"])
                st.plotly_chart(chart_layout(fig, "TEG Current (mA)"), use_container_width=True)

        else:
            st.info("⏳ Waiting for Member 2 (Hot Pot) ESP32 data...")
    else:
        st.info("⏳ Waiting for Member 2 (Hot Pot) ESP32 data...")

# ── MEMBER 3 TAB ─────────────────────────────────────────────────
with tab_m3:
    st.markdown("""
    <div class='member-header'>
        <div style='font-size:18px;font-weight:700;color:#fff;'>🔋 Battery Management & AI Prediction</div>
        <div style='font-size:12px;color:#6b7db3;margin-top:4px;'>Raspberry Pi Pico monitors battery SOC and runs AI-based TEG voltage prediction</div>
    </div>
    """, unsafe_allow_html=True)

    if not bat_df.empty:
        bv   = battery_latest.get('raw_volts', 0)
        soc  = battery_latest.get('soc_percentage', 0)
        pteg = battery_latest.get('predicted_teg_volts', 0)
        soc_color  = "#00e676" if soc > 40 else "#ff9800" if soc > 20 else "#f44336"
        load_state = "ON ✅" if soc > 40 else "OFF ❌"
        load_color = "#00e676" if soc > 40 else "#f44336"

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_html("Battery Voltage", f"{bv:.3f}",  "V",  "#4fc3f7"),  unsafe_allow_html=True)
        with c2: st.markdown(metric_html("State of Charge", f"{soc:.1f}", "%",  soc_color),  unsafe_allow_html=True)
        with c3: st.markdown(metric_html("AI Predicted TEG", f"{pteg:.4f}", "V", "#ce93d8"), unsafe_allow_html=True)
        with c4: st.markdown(metric_html("Load State",       load_state,   "",  load_color), unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Battery Management Charts</div>", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if 'raw_volts' in bat_df.columns:
                fig = px.line(bat_df, x='timestamp', y='raw_volts',
                              color_discrete_sequence=["#4fc3f7"])
                st.plotly_chart(chart_layout(fig, "Battery Voltage Over Time (V)"), use_container_width=True)
        with r1c2:
            if 'soc_percentage' in bat_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bat_df['timestamp'], y=bat_df['soc_percentage'],
                    fill='tozeroy', line=dict(color=soc_color, width=2),
                    fillcolor='rgba(0,230,118,0.15)'
                ))
                fig.add_hline(y=40, line_dash="dash", line_color="#ff9800",
                              annotation_text="Load ON (40%)", annotation_font_color="#ff9800")
                fig.add_hline(y=20, line_dash="dash", line_color="#f44336",
                              annotation_text="Low Battery (20%)", annotation_font_color="#f44336")
                st.plotly_chart(chart_layout(fig, "Battery SOC (%)"), use_container_width=True)

        if 'predicted_teg_volts' in bat_df.columns:
            fig = px.line(bat_df, x='timestamp', y='predicted_teg_volts',
                          color_discrete_sequence=["#ce93d8"])
            fig.update_layout(height=200)
            st.plotly_chart(chart_layout(fig, "AI Predicted TEG Voltage (V)"), use_container_width=True)

        # BMS Logic explanation
        st.markdown("<div class='section-header'>BMS Load Decision Logic</div>", unsafe_allow_html=True)
        if soc > 40:
            st.markdown("<div class='rec-box'>✅ <strong>Load ENABLED</strong> — Battery SOC above 40% threshold. Inverter and AC loads can operate safely.</div>", unsafe_allow_html=True)
        elif soc > 20:
            st.markdown("<div class='rec-box-warn'>⚠️ <strong>Load RESTRICTED</strong> — Battery SOC between 20-40%. Only essential loads allowed.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='rec-box-danger'>🔴 <strong>Load DISABLED</strong> — Battery SOC below 20%. Preventing deep discharge. Charging required.</div>", unsafe_allow_html=True)
    else:
        st.info("⏳ Waiting for Pico battery data... Make sure bridge.py is running.")

# ── AI ANALYTICS TAB ─────────────────────────────────────────────
with tab_ai:
    st.markdown("<div class='section-header'>AI Model Performance Analytics</div>", unsafe_allow_html=True)

    if teg_model:
        st.success("✅ Random Forest TEG Voltage Prediction Model — Active")
    else:
        st.error("🚨 AI Model not loaded")

    if not teg_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            if 'actual' in teg_df.columns and 'predicted' in teg_df.columns:
                fig = px.scatter(teg_df, x='actual', y='predicted',
                                 title="Actual vs Predicted Voltage",
                                 color_discrete_sequence=["#ce93d8"])
                fig.add_shape(type='line',
                              x0=teg_df['actual'].min(), y0=teg_df['actual'].min(),
                              x1=teg_df['actual'].max(), y1=teg_df['actual'].max(),
                              line=dict(color='#f44336', dash='dash'))
                st.plotly_chart(chart_layout(fig, "Actual vs Predicted Voltage"), use_container_width=True)
        with col2:
            if 'actual' in teg_df.columns and 'predicted' in teg_df.columns:
                teg_df['error'] = teg_df['predicted'] - teg_df['actual']
                fig2 = px.histogram(teg_df, x='error',
                                    color_discrete_sequence=["#4fc3f7"])
                st.plotly_chart(chart_layout(fig2, "Prediction Error Distribution"), use_container_width=True)

    st.markdown("<div class='section-header'>All Live Cloud Records</div>", unsafe_allow_html=True)
    full_df = get_microgrid_data(100)
    if not full_df.empty:
        st.dataframe(
            full_df.drop(columns=['_id'], errors='ignore'),
            use_container_width=True,
            height=300
        )
    else:
        st.info("No cloud records yet.")

# ── SCENARIO OPTIMIZER TAB ───────────────────────────────────────
with tab_optimizer:
    st.markdown("<div class='section-header'>AI-Based Energy Scenario Optimizer</div>", unsafe_allow_html=True)

    if future_df.empty:
        st.warning("Future solar prediction CSV not found in Data/ folder.")
    else:
        min_date = future_df["datetime"].min().date()
        max_date = future_df["datetime"].max().date()

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            selected_date = st.date_input("Target Date", value=min_date, min_value=min_date, max_value=max_date)
        with oc2:
            selected_hour = st.selectbox("Target Hour (24H)", list(range(24)), index=12)
        with oc3:
            load_demand_w = st.number_input("Load Demand (W)", min_value=0, max_value=1000, value=50, step=5)

        with st.expander("🧠 Advanced AI Model Inputs", expanded=False):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                time_s       = st.number_input("Elapsed Time (s)", value=600, step=10)
                temp_C       = st.number_input("Temperature (°C)", value=55.0, step=0.1)
                heatsink_opt = st.selectbox("Heatsink", ["no", "yes"])
            with ec2:
                temp_lag_1   = st.number_input("Temp t-1 (°C)", value=54.5, step=0.1)
                temp_lag_2   = st.number_input("Temp t-2 (°C)", value=54.0, step=0.1)
                sample_int   = st.number_input("Sample Interval (s)", value=3, step=1)
            with ec3:
                voltage_lag_1 = st.number_input("Voltage t-1 (mV)", value=60.0, step=0.1)
                voltage_lag_2 = st.number_input("Voltage t-2 (mV)", value=58.0, step=0.1)
                ref_load_ohm  = st.number_input("Load Resistance (Ω)", value=10.0, step=0.1)

        selected_datetime = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00")
        match = future_df[future_df["datetime"] == selected_datetime]

        if match.empty:
            st.warning("No solar prediction found for selected time.")
        else:
            row = match.iloc[0]
            predicted_pv_power = float(row["predicted_pv_power"])

            heatsink_bin   = 1 if heatsink_opt == "yes" else 0
            dtemp_dt       = (temp_C - temp_lag_1) / max(sample_int, 1)
            temp_rolling_3 = np.mean([temp_C, temp_lag_1, temp_lag_2])
            predicted_teg_mV = 0.0
            predicted_teg_W  = 0.0

            if teg_model:
                live_input = pd.DataFrame([{
                    "time_s":         time_s,
                    "temp_C":         temp_C,
                    "heatsink_bin":   heatsink_bin,
                    "dtemp_dt":       dtemp_dt,
                    "temp_rolling_3": temp_rolling_3,
                    "temp_lag_1":     temp_lag_1,
                    "temp_lag_2":     temp_lag_2,
                    "voltage_lag_1":  voltage_lag_1,
                    "voltage_lag_2":  voltage_lag_2
                }])
                predicted_teg_mV = max(float(teg_model.predict(live_input)[0]), 0)
                predicted_teg_V  = predicted_teg_mV / 1000.0
                predicted_teg_W  = (predicted_teg_V ** 2) / ref_load_ohm

            total_power = predicted_pv_power + predicted_teg_W
            net_power   = total_power - load_demand_w
            battery_capacity_wh = 3.7
            _, _, battery_latest2 = get_latest_values()
            battery_soc_now  = battery_latest2.get('soc_percentage', 60.0)
            soc_change       = (net_power / battery_capacity_wh) * 100
            battery_soc_next = max(20, min(100, battery_soc_now + soc_change))

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: st.markdown(metric_html("Solar Power",  f"{predicted_pv_power:.2f}",    "W",  "#ffb74d"), unsafe_allow_html=True)
            with sc2: st.markdown(metric_html("TEG Power",    f"{predicted_teg_W*1000:.2f}",  "mW", "#66bb6a"), unsafe_allow_html=True)
            with sc3: st.markdown(metric_html("Total Power",  f"{total_power:.2f}",            "W",  "#4fc3f7"), unsafe_allow_html=True)
            with sc4:
                nc = "#00e676" if battery_soc_next > 60 else "#ff9800" if battery_soc_next > 40 else "#f44336"
                st.markdown(metric_html("Next SOC", f"{battery_soc_next:.1f}", "%", nc), unsafe_allow_html=True)

            if total_power >= 1.5 * load_demand_w and battery_soc_next > 60:
                rec_text  = "✅ Run heavy load and charge battery"
                rec_class = "rec-box"
            elif total_power >= load_demand_w and battery_soc_next > 40:
                rec_text  = "🟡 Run essential loads only"
                rec_class = "rec-box-warn"
            elif predicted_teg_W > 0 and battery_soc_next > 30:
                rec_text  = "🟠 Use TEG-assisted operation"
                rec_class = "rec-box-warn"
            else:
                rec_text  = "🔴 Battery saving mode — delay non-essential loads"
                rec_class = "rec-box-danger"

            st.markdown(f"<div class='{rec_class}'><strong>AI Recommendation:</strong> {rec_text}</div>", unsafe_allow_html=True)

            daily_df = future_df[future_df["datetime"].dt.date == selected_date].copy()
            daily_df["hour"]        = daily_df["datetime"].dt.hour
            daily_df["total_power"] = daily_df["predicted_pv_power"] + predicted_teg_W
            fig = px.line(daily_df, x="hour", y="total_power",
                          color_discrete_sequence=["#ffb74d"])
            fig.update_layout(height=280)
            st.plotly_chart(chart_layout(fig, "24-Hour Total Power Forecast (Solar + TEG)"), use_container_width=True)

# -----------------------------
# 7. Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Dashboard auto-refreshes every 10 seconds (no dimming) ✅")
    if st.button("🔄 Force Refresh"):
        st.rerun()
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if teg_model:
        st.success("✅ AI Model Active")
    else:
        st.error("🚨 AI Model Offline")
