import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Manufacturing Quality Planner",
    page_icon="🏭",
    layout="wide"
)

# ---------- PREMIUM DARK FACTORY DASHBOARD THEME ----------
st.markdown("""
<style>
:root {
    --bg: #050d1b;
    --panel: #0b1930;
    --panel2: #10233e;
    --cyan: #35d8f2;
    --orange: #ff9b3d;
    --muted: #9bb1c9;
}
.stApp {
    background:
      radial-gradient(circle at 50% 18%, rgba(19,94,145,.28), transparent 34%),
      linear-gradient(135deg, #030916 0%, #071326 55%, #091b31 100%);
    color: #edf6ff;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1930, #071222);
    border-right: 1px solid rgba(53,216,242,.25);
}
[data-testid="stSidebar"] * { color: #eaf5ff !important; }
h1, h2, h3, h4, p, label { color: #edf6ff; }
h1 { letter-spacing: -.7px; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
.hero {
    background: linear-gradient(110deg, rgba(8,25,49,.98), rgba(13,48,77,.88));
    border: 1px solid rgba(53,216,242,.35);
    border-radius: 20px;
    padding: 24px 28px;
    box-shadow: 0 0 32px rgba(0,175,255,.12);
    margin-bottom: 18px;
}
.hero-kicker { color: #45e3f4; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; }
.hero-title { color: #f4f9ff; font-size: 30px; font-weight: 800; margin: 7px 0; }
.hero-sub { color: #a9c3dc; font-size: 14px; }
.factory-scene {
    border-radius: 18px;
    min-height: 190px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background:
      linear-gradient(90deg, rgba(5,13,27,.18), rgba(5,13,27,.38)),
      radial-gradient(circle at 50% 48%, rgba(35,198,244,.32), transparent 35%),
      linear-gradient(135deg, #102a49, #071326 65%, #173a55);
    border: 1px solid rgba(53,216,242,.22);
    box-shadow: inset 0 0 45px rgba(24,160,220,.12);
    margin-bottom: 18px;
}
.factory-icon { font-size: 66px; filter: drop-shadow(0 0 14px rgba(53,216,242,.55)); }
.factory-caption { color: #b7d7ee; font-size: 12px; letter-spacing: 1.2px; margin-top: 5px; }
.section-card {
    background: linear-gradient(145deg, rgba(14,32,57,.96), rgba(8,21,39,.96));
    border: 1px solid rgba(91,151,195,.24);
    border-radius: 16px;
    padding: 16px 18px;
    margin: 8px 0 16px 0;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102743, #0a172b);
    border: 1px solid rgba(53,216,242,.28);
    border-radius: 15px;
    padding: 15px 16px;
    box-shadow: 0 5px 22px rgba(0,0,0,.18);
}
div[data-testid="stMetric"] label { color: #9bb9d3 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f2f8ff !important; }
.stButton > button {
    background: linear-gradient(90deg, #ff8b32, #ffb34f);
    color: #101827;
    font-weight: 800;
    border: 0;
    border-radius: 10px;
    min-height: 44px;
}
.stButton > button:hover { background: linear-gradient(90deg, #ffad50, #ffd078); color: #071326; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: #10233e; border-color: #315477; color: white;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(53,216,242,.22);
    border-radius: 12px;
    overflow: hidden;
}
hr { border-color: rgba(92,148,190,.25); }
.small-muted { color: #9bb1c9; font-size: 12px; }
.status-pass { color: #45e09a; font-weight: 800; }
.status-review { color: #ffb34f; font-weight: 800; }
.status-hold { color: #ffd166; font-weight: 800; }
.status-reject { color: #ff667d; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ---------- MODEL BASELINE ----------
rng = np.random.default_rng(42)
normal = pd.DataFrame({
    "surface_score": np.clip(rng.normal(92, 3, 300), 75, 100),
    "dimension_deviation_mm": np.abs(rng.normal(0.08, 0.04, 300)),
    "temperature_c": rng.normal(68, 3, 300),
    "vibration_mm_s": np.abs(rng.normal(1.8, 0.45, 300)),
})
model = IsolationForest(contamination=0.08, random_state=42)
model.fit(normal)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">CSE276 | INTELLIGENT MANUFACTURING SYSTEM</div>
  <div class="hero-title">🏭 AI-Based Manufacturing Quality Inspection Planner</div>
  <div class="hero-sub">Explainable quality inspection • Machine-learning anomaly detection • Rule-based decisions • State-space planning</div>
</div>
""", unsafe_allow_html=True)

scene_col, note_col = st.columns([1.55, 1])
with scene_col:
    st.markdown("""
    <div class="factory-scene">
      <div>
        <div class="factory-icon">🏭 🤖 ⚙️</div>
        <div class="factory-caption">SMART FACTORY • LIVE QUALITY CONTROL • AI INSPECTION</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with note_col:
    st.markdown("""
    <div class="section-card" style="min-height:190px">
      <div class="hero-kicker">SYSTEM OVERVIEW</div>
      <h3 style="margin:8px 0;color:#f4f9ff">Intelligent inspection console</h3>
      <p style="color:#a9c3dc;font-size:14px">Enter product measurements, run the inspection planner, and review the recommended action with its explanation and inspection route.</p>
      <p style="color:#ffb34f;font-size:12px">● Demo environment | Synthetic training baseline</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- DATASET AND KPIs ----------
st.markdown("### 📊 Manufacturing Quality Analytics")
try:
    df = pd.read_csv("manufacturing_data.csv")
    required = {"product_id", "surface_score", "dimension_deviation_mm",
                "temperature_c", "vibration_mm_s", "quality_status"}
    if not required.issubset(df.columns):
        st.warning("Dataset columns do not match the expected manufacturing format.")
        df = pd.DataFrame()
except FileNotFoundError:
    df = pd.DataFrame()
    st.warning("manufacturing_data.csv was not found. The live inspection tool still works.")

if not df.empty:
    total = len(df)
    passed = int((df["quality_status"].astype(str).str.upper() == "PASS").sum())
    reinspect = int((df["quality_status"].astype(str).str.upper() == "REINSPECT").sum())
    hold = int((df["quality_status"].astype(str).str.upper() == "HOLD").sum())
    reject = int((df["quality_status"].astype(str).str.upper() == "REJECT").sum())
else:
    total = passed = reinspect = hold = reject = 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Products", total)
k2.metric("PASS", passed)
k3.metric("REINSPECT", reinspect)
k4.metric("HOLD", hold)
k5.metric("REJECT", reject)

left_chart, right_chart = st.columns([1.05, 1.4])
with left_chart:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Quality status distribution")
    if not df.empty:
        counts = df["quality_status"].astype(str).str.upper().value_counts()
        st.bar_chart(counts, color="#35d8f2")
    else:
        st.info("Add manufacturing_data.csv to display dataset analytics.")
    st.markdown("</div>", unsafe_allow_html=True)
with right_chart:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Manufacturing dataset")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("Dataset table will appear here when the CSV file is available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- SIDEBAR INPUT ----------
with st.sidebar:
    st.markdown("## 🧪 Product inspection")
    st.caption("Enter the current product measurements")
    product_id = st.text_input("Product ID", "UNIT-104")
    surface = st.slider("Surface quality score (0–100)", 0, 100, 72)
    deviation = st.number_input("Dimension deviation (mm)", 0.0, 2.0, 0.32, 0.01)
    temp = st.number_input("Machine temperature (°C)", 30.0, 120.0, 76.0, 0.5)
    vibration = st.number_input("Machine vibration (mm/s)", 0.0, 10.0, 3.4, 0.1)
    urgency = st.selectbox("Production urgency", ["Normal", "High", "Critical"])
    submitted = st.button("▶ Run AI inspection plan", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Inspection signals**")
    st.caption("Surface • Dimensions • Temperature • Vibration • ML anomaly")

# ---------- STATE-SPACE PLANNER ----------
TRANSITIONS = {
    "Received": ["Visual_Check"],
    "Visual_Check": ["Measurement_Check", "Hold"],
    "Measurement_Check": ["Machine_Check", "Rework", "Pass"],
    "Machine_Check": ["Pass", "Rework", "Hold"],
    "Rework": ["Reinspection"],
    "Reinspection": ["Pass", "Hold"],
    "Hold": ["Supervisor_Review"],
    "Supervisor_Review": ["Rework", "Pass", "Reject"],
    "Pass": [], "Reject": []
}

def bfs_plan(start, goal):
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for nxt in TRANSITIONS.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return [start, "No valid route"]

def decide(surface, deviation, temp, vibration, anomaly, urgency):
    reasons = []
    if surface < 70:
        return "Reject", "Critical surface defect: score below 70.", "Critical"
    if deviation > 0.50:
        return "Hold", "Dimensional deviation exceeds 0.50 mm; supervisor review required.", "High"
    if surface < 85:
        reasons.append("surface score is below the 85-point quality target")
    if deviation > 0.20:
        reasons.append("dimensional deviation is above the 0.20 mm alert threshold")
    if temp > 75:
        reasons.append("machine temperature is above the 75°C alert threshold")
    if vibration > 3.0:
        reasons.append("machine vibration is above the 3.0 mm/s alert threshold")
    if anomaly:
        reasons.append("ML anomaly detector flagged an unusual measurement pattern")
    if reasons or urgency in ["High", "Critical"]:
        return "Reinspect", "; ".join(reasons) if reasons else f"Production urgency is {urgency}.", "High"
    return "Pass", "All measured features are within the demo acceptance limits.", "Low"

# ---------- INSPECTION RESULT ----------
st.markdown("### 🎯 AI Inspection Console")
if submitted:
    row = pd.DataFrame([{
        "surface_score": surface,
        "dimension_deviation_mm": deviation,
        "temperature_c": temp,
        "vibration_mm_s": vibration
    }])
    anomaly = model.predict(row)[0] == -1
    outcome, reason, priority = decide(surface, deviation, temp, vibration, anomaly, urgency)

    if outcome == "Pass":
        path = bfs_plan("Received", "Pass")
        final_status = "PASS"
    elif outcome == "Reject":
        path = ["Received", "Visual_Check", "Measurement_Check", "Machine_Check",
                "Hold", "Supervisor_Review", "Reject"]
        final_status = "REJECT"
    elif outcome == "Hold":
        path = ["Received", "Visual_Check", "Hold", "Supervisor_Review"]
        final_status = "HOLD"
    else:
        path = bfs_plan("Received", "Supervisor_Review")
        final_status = "REINSPECT"

    status_class = {
        "PASS": "status-pass", "REINSPECT": "status-review",
        "HOLD": "status-hold", "REJECT": "status-reject"
    }[final_status]
    st.markdown(f"""
    <div class="section-card">
      <div class="hero-kicker">INSPECTION RESULT | {product_id}</div>
      <h2 style="margin:8px 0"><span class="{status_class}">{outcome.upper()}</span></h2>
      <p style="color:#b3c8dd;margin:0">{reason}</p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Decision", outcome)
    r2.metric("Priority", priority)
    r3.metric("ML anomaly", "Detected" if anomaly else "Not detected")

    st.markdown("#### 🧭 State-space inspection route")
    st.code("  →  ".join(path), language="text")

    st.markdown("#### 🤖 Intelligent agent contributions")
    agent_data = pd.DataFrame([
        ["Measurement Agent", f"Recorded surface={surface}, deviation={deviation:.2f} mm"],
        ["ML Anomaly Agent", "Unusual pattern detected" if anomaly else "Pattern is close to learned baseline"],
        ["Rule Engine Agent", f"Applied production rules and selected {outcome}"],
        ["Planning Agent", "Generated a valid sequence of inspection states"],
        ["Explanation Agent", "Converted decision factors into a human-readable reason"]
    ], columns=["Agent", "Output"])
    st.dataframe(agent_data, hide_index=True, use_container_width=True)
else:
    st.markdown("""
    <div class="section-card">
      <div class="hero-kicker">READY FOR INSPECTION</div>
      <h3 style="color:#f4f9ff">Waiting for product measurements</h3>
      <p style="color:#a9c3dc">Use the inspection panel on the left and select <b>Run AI inspection plan</b>. The decision, anomaly flag, explanation and route will appear here.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Prototype limitation: advisory decisions use illustrative thresholds and synthetic training data. Real deployment requires validated factory sensor data, domain-approved limits, human sign-off and model monitoring.")
