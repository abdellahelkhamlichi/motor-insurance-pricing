import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Motor Insurance Pricing Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        font-size: 1.6rem;
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
    .stButton>button:hover { background-color: #2C5282; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
import json
from pathlib import Path

# Resolve paths relative to this script's location, regardless of
# the working directory Streamlit Cloud runs from (repo root, not dashboard/)
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_COEFFICIENTS = {
    'Intercept': -2.85,
    'VehPower': 0.045,
    'BonusMalus': 0.023,
    'Area_encoded': 0.12,
    'VehGas_encoded': 0.05,
    'C(DrivAge_band)[T.26-35]': -0.16,
    'C(DrivAge_band)[T.36-45]': 0.11,
    'C(DrivAge_band)[T.46-55]': 0.25,
    'C(DrivAge_band)[T.56-65]': 0.13,
    'C(DrivAge_band)[T.65+]': 0.22,
    'C(VehAge_band)[T.3-5]': 0.03,
    'C(VehAge_band)[T.6-10]': 0.08,
    'C(VehAge_band)[T.10+]': 0.18,
}

DEFAULT_PRICING = {
    'severity_by_vehage': {'0-2': 1300, '3-5': 1500, '6-10': 1700, '10+': 1900},
    'expense_ratio': 0.12,
    'profit_margin': 0.08,
    'reinsurance_loading': 0.03,
}

@st.cache_resource
def load_artifacts():
    coef_path = BASE_DIR / 'model_coefficients.json'
    artifact_path = BASE_DIR / 'pricing_artifacts.pkl'

    if coef_path.exists():
        with open(coef_path, 'r') as f:
            coefs = json.load(f)
    else:
        st.warning(
            "Fichier de coefficients absent: utilisation d'un jeu de valeurs de secours pour éviter un crash. "
            "Pour une précision maximale, générez le fichier model_coefficients.json depuis le notebook."
        )
        coefs = DEFAULT_COEFFICIENTS

    if artifact_path.exists():
        with open(artifact_path, 'rb') as f:
            pricing = pickle.load(f)
    else:
        st.warning(
            "Fichier pricing_artifacts.pkl absent: utilisation des paramètres de tarification par défaut."
        )
        pricing = DEFAULT_PRICING

    return coefs, pricing

coefs, pricing = load_artifacts()

def predict_frequency(veh_power, bonus_malus, area_encoded, veh_gas_encoded,
                       drivage_band, vehage_band, exposure):
    """Manually reconstruct the GLM log-linear predictor from saved coefficients.
    Equivalent to nb_model.predict() but without needing the statsmodels object."""
    lp = coefs['Intercept']
    lp += coefs.get(f'C(DrivAge_band)[T.{drivage_band}]', 0.0)
    lp += coefs.get(f'C(VehAge_band)[T.{vehage_band}]', 0.0)
    lp += coefs['VehPower'] * veh_power
    lp += coefs['BonusMalus'] * bonus_malus
    lp += coefs['Area_encoded'] * area_encoded
    lp += coefs['VehGas_encoded'] * veh_gas_encoded
    lp += np.log(exposure)
    return np.exp(lp)

AREA_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
DRIVAGE_BANDS = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
VEHAGE_BANDS = ['0-2', '3-5', '6-10', '10+']

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

def compute_premium(driv_age, veh_age, veh_power, bonus_malus, area, veh_gas, exposure=1.0):
    """Compute full pricing breakdown for a given policyholder profile."""
    drivage_band = get_age_band(driv_age)
    vehage_band = get_vehage_band(veh_age)

    predicted_freq = predict_frequency(
        veh_power=veh_power,
        bonus_malus=bonus_malus,
        area_encoded=AREA_MAP[area],
        veh_gas_encoded=1 if veh_gas == 'Diesel' else 0,
        drivage_band=drivage_band,
        vehage_band=vehage_band,
        exposure=exposure
    )
    expected_severity = pricing['severity_by_vehage'].get(
        vehage_band, np.mean(list(pricing['severity_by_vehage'].values()))
    )
    pure_premium = predicted_freq * expected_severity

    loading_factor = 1 / (1 - pricing['expense_ratio'] - pricing['profit_margin'])
    commercial_premium = pure_premium * loading_factor * (1 + pricing['reinsurance_loading'])

    expense_amount = pure_premium * loading_factor * pricing['expense_ratio']
    profit_amount = pure_premium * loading_factor * pricing['profit_margin']
    reinsurance_amount = commercial_premium - (pure_premium * loading_factor)

    return {
        'drivage_band': drivage_band, 'vehage_band': vehage_band,
        'predicted_freq': predicted_freq, 'expected_severity': expected_severity,
        'pure_premium': pure_premium, 'commercial_premium': commercial_premium,
        'expense_amount': expense_amount, 'profit_amount': profit_amount,
        'reinsurance_amount': reinsurance_amount
    }

def risk_gauge(freq, title="Claim Frequency Risk"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=freq,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 0.20]},
            'bar': {'color': "#1E3A5F"},
            'steps': [
                {'range': [0, 0.05], 'color': "#D1FAE5"},
                {'range': [0.05, 0.08], 'color': "#FEF3C7"},
                {'range': [0.08, 0.20], 'color': "#FEE2E2"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🚗 Motor Insurance Pricing Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Actuarial pricing model — Frequency-Severity methodology (Negative Binomial GLM), '
    'trained on 678,000+ real motor insurance policies (freMTPL2 dataset)</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["💶 Quote Calculator", "📊 Risk Factor Insights", "⚖️ Compare Profiles"])

# ============================================================
# TAB 1 — QUOTE CALCULATOR
# ============================================================
with tab1:
    st.sidebar.header("📋 Policyholder Profile")
    driv_age = st.sidebar.slider("Driver Age", 18, 100, 35, key="q_driv")
    veh_age = st.sidebar.slider("Vehicle Age (years)", 0, 30, 3, key="q_veh")
    veh_power = st.sidebar.slider("Vehicle Power", 1, 15, 6, key="q_power")
    bonus_malus = st.sidebar.slider("Bonus-Malus Score", 50, 150, 55, key="q_bm",
                                     help="French no-claims discount system. 50 = best.")
    area = st.sidebar.selectbox("Area Density Zone", ['A','B','C','D','E','F'], index=3, key="q_area")
    veh_gas = st.sidebar.radio("Fuel Type", ['Regular', 'Diesel'], key="q_gas")
    exposure = st.sidebar.slider("Policy Exposure (fraction of year)", 0.1, 1.0, 1.0, key="q_exp")

    r = compute_premium(driv_age, veh_age, veh_power, bonus_malus, area, veh_gas, exposure)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Frequency", f"{r['predicted_freq']:.3f}")
    col2.metric("Expected Severity", f"€{r['expected_severity']:,.0f}")
    col3.metric("Pure Premium", f"€{r['pure_premium']:,.2f}")
    col4.metric("💰 Commercial Premium", f"€{r['commercial_premium']:,.2f}",
                delta=f"+{(r['commercial_premium']/r['pure_premium']-1)*100:.0f}% loading")

    st.divider()
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Premium Build-up")
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Pure Premium", "Expenses", "Profit Margin", "Reinsurance", "Commercial Premium"],
            y=[r['pure_premium'], r['expense_amount'], r['profit_amount'], r['reinsurance_amount'], 0],
            connector={"line": {"color": "#CBD5E1"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#3B82F6"}},
            totals={"marker": {"color": "#1E3A5F"}}
        ))
        fig.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Gauge")
        st.plotly_chart(risk_gauge(r['predicted_freq']), use_container_width=True)
        st.write(f"**Driver Age Band:** {r['drivage_band']}")
        st.write(f"**Vehicle Age Band:** {r['vehage_band']}")
        st.write(f"**Fuel Type:** {veh_gas}  |  **Area:** {area}")

    st.download_button(
        "📄 Download Quote Summary",
        data=(
            f"MOTOR INSURANCE QUOTE\n{'='*30}\n"
            f"Driver Age Band: {r['drivage_band']}\nVehicle Age Band: {r['vehage_band']}\n"
            f"Vehicle Power: {veh_power}\nBonus-Malus: {bonus_malus}\nArea: {area}\nFuel: {veh_gas}\n\n"
            f"Predicted Frequency: {r['predicted_freq']:.4f}\n"
            f"Expected Severity: €{r['expected_severity']:,.2f}\n"
            f"Pure Premium: €{r['pure_premium']:,.2f}\n"
            f"Commercial Premium: €{r['commercial_premium']:,.2f}\n"
        ),
        file_name="insurance_quote.txt"
    )

# ============================================================
# TAB 2 — RISK FACTOR INSIGHTS (Interpretability)
# ============================================================
with tab2:
    st.subheader("How Each Risk Factor Affects the Premium")
    st.caption(
        "Multiplicative effect on claim frequency relative to the baseline category "
        "(Driver 18-25, Vehicle 0-2 years), derived directly from the GLM coefficients — "
        "this is the transparency regulators expect from actuarial pricing models."
    )

    params = coefs
    rows = []
    for band in DRIVAGE_BANDS[1:]:
        key = f"C(DrivAge_band)[T.{band}]"
        if key in params:
            rows.append({'Factor': f"Driver Age: {band}", 'Multiplier': np.exp(params[key])})
    for band in VEHAGE_BANDS[1:]:
        key = f"C(VehAge_band)[T.{band}]"
        if key in params:
            rows.append({'Factor': f"Vehicle Age: {band}", 'Multiplier': np.exp(params[key])})

    factor_df = pd.DataFrame(rows).sort_values('Multiplier', ascending=True)
    factor_df['Effect'] = factor_df['Multiplier'].apply(lambda x: f"{(x-1)*100:+.1f}%")

    fig2 = go.Figure(go.Bar(
        x=factor_df['Multiplier'] - 1,
        y=factor_df['Factor'],
        orientation='h',
        marker_color=np.where(factor_df['Multiplier'] >= 1, '#EF4444', '#22C55E'),
        text=factor_df['Effect'], textposition='outside'
    ))
    fig2.update_layout(
        title="Risk Multiplier vs. Baseline (Driver 18-25, Vehicle 0-2y)",
        xaxis_title="Change in Claim Frequency", height=450,
        xaxis_tickformat='+.0%', margin=dict(l=10, r=60, t=60, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Reading this chart**: a bar at +25% means that risk factor increases expected "
        "claim frequency by 25% compared to the youngest driver / newest vehicle baseline. "
        "Negative bars represent lower-risk categories."
    )

    st.subheader("Continuous Variable Effects")
    c1, c2 = st.columns(2)
    with c1:
        bm_effect = np.exp(params.get('BonusMalus', 0))
        st.metric("Bonus-Malus (+1 point)", f"{(bm_effect-1)*100:+.2f}% frequency")
    with c2:
        power_effect = np.exp(params.get('VehPower', 0))
        st.metric("Vehicle Power (+1 unit)", f"{(power_effect-1)*100:+.2f}% frequency")

# ============================================================
# TAB 3 — COMPARE PROFILES
# ============================================================
with tab3:
    st.subheader("Compare Two Policyholder Profiles Side by Side")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Profile A**")
        a_driv = st.slider("Driver Age (A)", 18, 100, 22, key="a_driv")
        a_veh = st.slider("Vehicle Age (A)", 0, 30, 1, key="a_veh")
        a_power = st.slider("Vehicle Power (A)", 1, 15, 8, key="a_power")
        a_bm = st.slider("Bonus-Malus (A)", 50, 150, 60, key="a_bm")
        a_area = st.selectbox("Area (A)", ['A','B','C','D','E','F'], index=3, key="a_area")
        a_gas = st.radio("Fuel (A)", ['Regular', 'Diesel'], key="a_gas")

    with colB:
        st.markdown("**Profile B**")
        b_driv = st.slider("Driver Age (B)", 18, 100, 45, key="b_driv")
        b_veh = st.slider("Vehicle Age (B)", 0, 30, 8, key="b_veh")
        b_power = st.slider("Vehicle Power (B)", 1, 15, 6, key="b_power")
        b_bm = st.slider("Bonus-Malus (B)", 50, 150, 50, key="b_bm")
        b_area = st.selectbox("Area (B)", ['A','B','C','D','E','F'], index=3, key="b_area")
        b_gas = st.radio("Fuel (B)", ['Regular', 'Diesel'], key="b_gas")

    ra = compute_premium(a_driv, a_veh, a_power, a_bm, a_area, a_gas)
    rb = compute_premium(b_driv, b_veh, b_power, b_bm, b_area, b_gas)

    st.divider()
    comp_df = pd.DataFrame({
        'Metric': ['Predicted Frequency', 'Expected Severity (€)', 'Pure Premium (€)', 'Commercial Premium (€)'],
        'Profile A': [f"{ra['predicted_freq']:.3f}", f"{ra['expected_severity']:,.0f}",
                      f"{ra['pure_premium']:,.2f}", f"{ra['commercial_premium']:,.2f}"],
        'Profile B': [f"{rb['predicted_freq']:.3f}", f"{rb['expected_severity']:,.0f}",
                      f"{rb['pure_premium']:,.2f}", f"{rb['commercial_premium']:,.2f}"],
    })
    st.table(comp_df.set_index('Metric'))

    diff_pct = (rb['commercial_premium'] / ra['commercial_premium'] - 1) * 100
    if diff_pct > 0:
        st.success(f"Profile B pays **{diff_pct:.1f}% more** than Profile A.")
    else:
        st.success(f"Profile B pays **{abs(diff_pct):.1f}% less** than Profile A.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Built with Streamlit | Model: Negative Binomial GLM (Frequency) + Segmented Average Severity "
    "| Benchmark: XGBoost | Data: freMTPL2 (CASdatasets)"
)