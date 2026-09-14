import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import plotly.graph_objects as go

# ---------- Page Config ----------
st.set_page_config(
    page_title="Motor Insurance Pricing Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        color: #1E3A5F;
    }
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2C5282;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model Artifacts ----------
@st.cache_resource
def load_artifacts():
    model = sm.load('frequency_model.pickle')
    with open('pricing_artifacts.pkl', 'rb') as f:
        pricing = pickle.load(f)
    return model, pricing

nb_model, pricing = load_artifacts()

# ---------- Header ----------
st.markdown('<p class="main-header">🚗 Motor Insurance Pricing Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Actuarial pricing model — Frequency-Severity methodology (Negative Binomial GLM), '
    'trained on 678,000+ real motor insurance policies (freMTPL2 dataset)</p>',
    unsafe_allow_html=True
)
st.divider()

# ---------- Sidebar Inputs ----------
st.sidebar.header("📋 Policyholder Profile")

driv_age = st.sidebar.slider("Driver Age", 18, 100, 35)
veh_age = st.sidebar.slider("Vehicle Age (years)", 0, 30, 3)
veh_power = st.sidebar.slider("Vehicle Power", 1, 15, 6)
bonus_malus = st.sidebar.slider(
    "Bonus-Malus Score", 50, 150, 55,
    help="French no-claims discount system. 50 = best (no claims history)."
)
area = st.sidebar.selectbox("Area Density Zone", ['A', 'B', 'C', 'D', 'E', 'F'], index=3)
veh_gas = st.sidebar.radio("Fuel Type", ['Regular', 'Diesel'])
exposure = st.sidebar.slider("Policy Exposure (fraction of year)", 0.1, 1.0, 1.0)

calculate = st.sidebar.button("🔍 Calculate Premium", use_container_width=True)

# ---------- Helper Functions ----------
def get_age_band(age):
    if age <= 25: return '18-25'
    elif age <= 35: return '26-35'
    elif age <= 45: return '36-45'
    elif age <= 55: return '46-55'
    elif age <= 65: return '56-65'
    else: return '65+'

def get_vehage_band(age):
    if age <= 2: return '0-2'
    elif age <= 5: return '3-5'
    elif age <= 10: return '6-10'
    else: return '10+'

area_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

# ---------- Calculation ----------
drivage_band = get_age_band(driv_age)
vehage_band = get_vehage_band(veh_age)

input_df = pd.DataFrame({
    'VehPower': [veh_power],
    'BonusMalus': [bonus_malus],
    'Area_encoded': [area_map[area]],
    'VehGas_encoded': [1 if veh_gas == 'Diesel' else 0],
    'DrivAge_band': [drivage_band],
    'VehAge_band': [vehage_band],
    'Exposure': [exposure]
})

predicted_freq = nb_model.predict(input_df, offset=np.log(input_df['Exposure']))[0]
expected_severity = pricing['severity_by_vehage'].get(
    vehage_band, np.mean(list(pricing['severity_by_vehage'].values()))
)
pure_premium = predicted_freq * expected_severity

loading_factor = 1 / (1 - pricing['expense_ratio'] - pricing['profit_margin'])
commercial_premium = pure_premium * loading_factor * (1 + pricing['reinsurance_loading'])

expense_amount = pure_premium * loading_factor * pricing['expense_ratio']
profit_amount = pure_premium * loading_factor * pricing['profit_margin']
reinsurance_amount = commercial_premium - (pure_premium * loading_factor)

# ---------- Metrics Row ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Frequency", f"{predicted_freq:.3f}",
              help="Expected number of claims per policy-year")
with col2:
    st.metric("Expected Severity", f"€{expected_severity:,.0f}",
              help="Average cost per claim, based on vehicle age")
with col3:
    st.metric("Pure Premium", f"€{pure_premium:,.2f}",
              help="Frequency × Severity — theoretical baseline cost")
with col4:
    st.metric("💰 Commercial Premium", f"€{commercial_premium:,.2f}",
              delta=f"+{(commercial_premium/pure_premium-1)*100:.0f}% loading")

st.divider()

col_left, col_right = st.columns([1.3, 1])

# ---------- Waterfall Chart ----------
with col_left:
    st.subheader("Premium Build-up")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Pure Premium", "Expenses", "Profit Margin", "Reinsurance", "Commercial Premium"],
        y=[pure_premium, expense_amount, profit_amount, reinsurance_amount, 0],
        connector={"line": {"color": "#CBD5E1"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#3B82F6"}},
        totals={"marker": {"color": "#1E3A5F"}}
    ))
    fig.update_layout(height=420, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Risk Profile Panel ----------
with col_right:
    st.subheader("Risk Profile Summary")
    st.write(f"**Driver Age Band:** {drivage_band}")
    st.write(f"**Vehicle Age Band:** {vehage_band}")
    st.write(f"**Fuel Type:** {veh_gas}")
    st.write(f"**Area Zone:** {area}")
    st.write(f"**Bonus-Malus:** {bonus_malus}")

    risk_level = (
        "🔴 High" if predicted_freq > 0.08
        else "🟡 Medium" if predicted_freq > 0.05
        else "🟢 Low"
    )
    st.metric("Risk Level", risk_level)

    st.info(
        "💡 This premium reflects the policyholder's risk profile, "
        "estimated using a Negative Binomial GLM (frequency) combined "
        "with segmented average severity by vehicle age."
    )

st.divider()
st.caption(
    "Built with Streamlit | Model: Negative Binomial GLM (Frequency) + "
    "Segmented Average Severity | Data: freMTPL2 (CASdatasets)"
)