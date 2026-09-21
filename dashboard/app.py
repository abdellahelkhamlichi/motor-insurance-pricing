import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import datetime
from pathlib import Path

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Motor Insurance Pricing Engine | Enterprise Edition",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --ink: #18252b;
        --muted: #68777c;
        --paper: #f6f7f2;
        --panel: #ffffff;
        --line: #dfe5df;
        --lime: #c9e86b;
        --coral: #ef775f;
        --teal: #2b7772;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #eef2e9;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    .block-container { max-width: 1440px; padding: 2.5rem 4rem 2rem; }
    h1, h2, h3, h4, p, label, button, input, textarea, select {
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    h2, h3 { color: var(--ink); letter-spacing: 0; }
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 2rem;
        padding: 1.5rem 0 2.2rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }
    .main-header {
        font-family: Georgia, serif;
        font-size: clamp(2.4rem, 4.5vw, 4.2rem);
        line-height: .94;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0;
        letter-spacing: -0.03em;
    }
    .sub-header {
        max-width: 750px;
        font-size: 1rem;
        line-height: 1.55;
        color: var(--muted);
        margin: .8rem 0 0;
    }
    .eyebrow {
        display: inline-block;
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }
    .hero-mark {
        display: grid;
        place-items: center;
        width: 72px;
        height: 72px;
        flex: 0 0 72px;
        border-radius: 50%;
        background: var(--lime);
        color: var(--ink);
        font-family: Georgia, serif;
        font-size: 2.5rem;
        transform: rotate(-8deg);
    }
    div[data-testid="stMetricValue"] {
        font-family: Georgia, serif;
        font-size: 1.7rem;
        color: var(--ink);
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 4px solid var(--lime);
        padding: 1rem 1.1rem;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(24, 37, 43, .05);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .section-label {
        color: var(--teal);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .stButton>button {
        background-color: var(--ink);
        color: var(--lime);
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: var(--teal); color: white; }
    .stTabs [data-baseweb="tab"] { font-size: .95rem; font-weight: 700; }
    .stTabs [aria-selected="true"] { color: var(--teal); }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--coral); }
    div[data-testid="stExpander"] { border-color: var(--line); background: var(--panel); }
    .badge-approved { background-color: #d1fae5; color: #065f46; padding: 6px 14px; border-radius: 6px; font-weight: 700; display: inline-block; }
    .badge-referred { background-color: #fef3c7; color: #92400e; padding: 6px 14px; border-radius: 6px; font-weight: 700; display: inline-block; }
    @media (max-width: 800px) {
        .block-container { padding: 1.5rem 1rem 2rem; }
        .hero { align-items: flex-start; }
        .hero-mark { display: none; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS (Robust Architecture)
# ============================================================
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

    coefs = DEFAULT_COEFFICIENTS
    if coef_path.exists():
        try:
            with open(coef_path, 'r') as f:
                coefs = json.load(f)
        except Exception:
            pass

    pricing = DEFAULT_PRICING
    if artifact_path.exists():
        try:
            with open(artifact_path, 'rb') as f:
                pricing = pickle.load(f)
        except Exception:
            pass

    return coefs, pricing

coefs, pricing = load_artifacts()

# ============================================================
# ACTUARIAL CORE ENGINE
# ============================================================
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

def predict_frequency(veh_power, bonus_malus, area_encoded, veh_gas_encoded,
                      drivage_band, vehage_band, exposure):
    lp = coefs.get('Intercept', -2.85)
    lp += coefs.get(f'C(DrivAge_band)[T.{drivage_band}]', 0.0)
    lp += coefs.get(f'C(VehAge_band)[T.{vehage_band}]', 0.0)
    lp += coefs.get('VehPower', 0.045) * veh_power
    lp += coefs.get('BonusMalus', 0.023) * bonus_malus
    lp += coefs.get('Area_encoded', 0.12) * area_encoded
    lp += coefs.get('VehGas_encoded', 0.05) * veh_gas_encoded
    lp += np.log(exposure)
    return float(np.exp(lp))

def compute_premium(driv_age, veh_age, veh_power, bonus_malus, area, veh_gas, exposure=1.0):
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
    
    sev_dict = pricing.get('severity_by_vehage', {'0-2': 1300, '3-5': 1500, '6-10': 1700, '10+': 1900})
    expected_severity = sev_dict.get(vehage_band, np.mean(list(sev_dict.values())))
    pure_premium = predicted_freq * expected_severity

    exp_ratio = pricing.get('expense_ratio', 0.12)
    prof_margin = pricing.get('profit_margin', 0.08)
    reins_loading = pricing.get('reinsurance_loading', 0.03)

    loading_factor = 1 / (1 - exp_ratio - prof_margin)
    commercial_premium = pure_premium * loading_factor * (1 + reins_loading)

    expense_amount = pure_premium * loading_factor * exp_ratio
    profit_amount = pure_premium * loading_factor * prof_margin
    reinsurance_amount = commercial_premium - (pure_premium * loading_factor)

    # Underwriting referral rule (e.g. high frequency or bad bonus-malus)
    referred = predicted_freq > 0.12 or bonus_malus > 125

    return {
        'drivage_band': drivage_band, 'vehage_band': vehage_band,
        'predicted_freq': predicted_freq, 'expected_severity': expected_severity,
        'pure_premium': pure_premium, 'commercial_premium': commercial_premium,
        'expense_amount': expense_amount, 'profit_amount': profit_amount,
        'reinsurance_amount': reinsurance_amount, 'referred': referred
    }

def risk_gauge(freq, title="Claim Frequency Risk"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=freq,
        number={'format': ".3f", 'font': {'color': '#18252b', 'size': 24}},
        title={'text': title, 'font': {'size': 14, 'color': '#68777c'}},
        gauge={
            'axis': {'range': [0, 0.20], 'tickwidth': 1, 'tickcolor': "#dfe5df"},
            'bar': {'color': "#18252b"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#dfe5df",
            'steps': [
                {'range': [0, 0.05], 'color': "#d1fae5"},
                {'range': [0.05, 0.10], 'color': "#fef3c7"},
                {'range': [0.10, 0.20], 'color': "#fee2e2"},
            ],
            'threshold': {
                'line': {'color': "#ef775f", 'width': 4},
                'thickness': 0.75,
                'value': 0.12
            }
        }
    ))
    fig.update_layout(height=230, margin=dict(t=30, b=10, l=20, r=20))
    return fig

# ============================================================
# HEADER & SIDEBAR GOVERNANCE
# ============================================================
st.markdown(
    '<div class="hero"><div><div class="eyebrow">Actuarial decision system • Enterprise Edition</div>'
    '<p class="main-header">Motor insurance<br>pricing engine</p>'
    '<p class="sub-header">A transparent frequency-severity GLM framework for turning raw driver profiles into audit-ready, compliant commercial premiums.</p>'
    '</div><div class="hero-mark">✦</div></div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 🏛️ Governance & Context")
    actuary_role = st.selectbox("Underwriting Role", ["Chief Actuary", "Senior Pricing Analyst", "Compliance Auditor"], index=0)
    currency_symbol = st.selectbox("Currency Unit", ["EUR (€)", "USD ($)", "GBP (£)"], index=0).split("(")[1].replace(")", "")
    
    st.divider()
    st.markdown("### 📋 Policyholder Profile")
    driv_age = st.slider("Driver Age", 18, 100, 32, key="q_driv")
    veh_age = st.slider("Vehicle Age (years)", 0, 30, 4, key="q_veh")
    veh_power = st.slider("Vehicle Power Rating", 1, 15, 7, key="q_power")
    bonus_malus = st.slider("Bonus-Malus Index", 50, 150, 68, key="q_bm",
                            help="No-claims discount scale. 50 = max bonus, 150 = max malus.")
    area = st.selectbox("Area Density Code", ['A','B','C','D','E','F'], index=2, key="q_area")
    veh_gas = st.radio("Fuel Engine Type", ['Regular', 'Diesel'], key="q_gas")
    exposure = st.slider("Policy Exposure Term (Year Fraction)", 0.1, 1.0, 1.0, key="q_exp")

tab1, tab2, tab3, tab4 = st.tabs(["💶 Quote Calculator", "📊 Risk Factor Insights", "⚖️ Compare Profiles", "📑 Compliance & Audit Trail"])

r = compute_premium(driv_age, veh_age, veh_power, bonus_malus, area, veh_gas, exposure)

# ============================================================
# TAB 1 — QUOTE CALCULATOR
# ============================================================
with tab1:
    if r['referred']:
        st.markdown('<div class="badge-referred" style="width: 100%; text-align: center; margin-bottom: 20px; font-size: 1.05rem;">⚠️ UNDERWRITING REFERRAL REQUIRED — Risk parameters breach standard automated appetite bounds. Manual sign-off needed.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-approved" style="width: 100%; text-align: center; margin-bottom: 20px; font-size: 1.05rem;">✅ AUTOMATED CLEARANCE — Standard risk profile processed successfully under GLM rules.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Frequency", f"{r['predicted_freq']:.3f}")
    col2.metric("Expected Severity", f"{currency_symbol}{r['expected_severity']:,.0f}")
    col3.metric("Pure Technical Premium", f"{currency_symbol}{r['pure_premium']:,.2f}")
    col4.metric("💰 Commercial Premium", f"{currency_symbol}{r['commercial_premium']:,.2f}",
                delta=f"+{(r['commercial_premium']/r['pure_premium']-1)*100:.0f}% total load")

    st.divider()
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Actuarial Premium Build-up Waterfall")
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Pure Premium", "Expenses", "Profit Margin", "Reinsurance", "Commercial Premium"],
            y=[r['pure_premium'], r['expense_amount'], r['profit_amount'], r['reinsurance_amount'], 0],
            connector={"line": {"color": "#dfe5df"}},
            decreasing={"marker": {"color": "#ef775f"}},
            increasing={"marker": {"color": "#2b7772"}},
            totals={"marker": {"color": "#18252b"}}
        ))
        fig.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False, yaxis_title=f"Amount ({currency_symbol})")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Frequency Risk Gauge")
        st.plotly_chart(risk_gauge(r['predicted_freq']), use_container_width=True)
        st.markdown(f"""
        * **Driver Segment:** Age Band {r['drivage_band']}
        * **Vehicle Segment:** Class {r['vehage_band']}
        * **Engine & Geography:** {veh_gas} • Zone {area}
        * **Signed By Role:** {actuary_role}
        """)

    # Professional Audit Report download
    quote_report = f"""MOTOR INSURANCE ACTUARIAL RATING REPORT
=====================================================
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Governance Role: {actuary_role}

POLICYHOLDER METRICS:
- Driver Age: {driv_age} (Band: {r['drivage_band']})
- Vehicle Age: {veh_age} yrs (Band: {r['vehage_band']})
- Vehicle Power: {veh_power} | Bonus-Malus: {bonus_malus}
- Area Code: {area} | Fuel Spec: {veh_gas}
- Policy Exposure: {exposure * 100}%

PRICING MODEL OUTPUTS:
- Expected Claim Frequency: {r['predicted_freq']:.4f}
- Expected Severity Loss: {currency_symbol}{r['expected_severity']:,.2f}
- Pure Technical Premium: {currency_symbol}{r['pure_premium']:,.2f}
- Final Commercial Premium: {currency_symbol}{r['commercial_premium']:,.2f}
- Underwriting Status: {'REFERRED' if r['referred'] else 'APPROVED'}
=====================================================
CONFIDENTIAL - ACTUARIAL PRICING ENGINE VAULT
"""
    st.download_button(
        "📄 Download Actuarial Dossier (.txt)",
        data=quote_report,
        file_name=f"Actuarial_Quote_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

# ============================================================
# TAB 2 — RISK FACTOR INSIGHTS (Interpretability)
# ============================================================
with tab2:
    st.subheader("GLM Risk Factor Multiplier Attribution")
    st.caption(
        "Multiplicative impact on claim frequency relative to the benchmark category "
        "(Driver 18-25, Vehicle 0-2 years), derived directly from the negative binomial log-linear coefficients."
    )

    rows = []
    for band in DRIVAGE_BANDS[1:]:
        key = f"C(DrivAge_band)[T.{band}]"
        if key in coefs:
            rows.append({'Factor': f"Driver Age: {band}", 'Multiplier': np.exp(coefs[key])})
    for band in VEHAGE_BANDS[1:]:
        key = f"C(VehAge_band)[T.{band}]"
        if key in coefs:
            rows.append({'Factor': f"Vehicle Age: {band}", 'Multiplier': np.exp(coefs[key])})

    factor_df = pd.DataFrame(rows).sort_values('Multiplier', ascending=True)
    factor_df['Effect'] = factor_df['Multiplier'].apply(lambda x: f"{(x-1)*100:+.1f}%")

    fig2 = go.Figure(go.Bar(
        x=factor_df['Multiplier'] - 1,
        y=factor_df['Factor'],
        orientation='h',
        marker_color=np.where(factor_df['Multiplier'] >= 1, '#ef775f', '#2b7772'),
        text=factor_df['Effect'], textposition='outside'
    ))
    fig2.update_layout(
        title="Relative Risk Multiplier vs Baseline",
        xaxis_title="Frequency Shift (%)", height=420,
        xaxis_tickformat='+.0%', margin=dict(l=10, r=60, t=50, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Actuarial Note**: Bars in teal represent protective risk characteristics reducing expected "
        "claim frequency, whereas coral bars represent premium-loading risk categories."
    )

    st.subheader("Continuous Sensitivities")
    c1, c2 = st.columns(2)
    with c1:
        bm_eff = np.exp(coefs.get('BonusMalus', 0.023))
        st.metric("Bonus-Malus Elasticity (+1 Index)", f"{(bm_eff-1)*100:+.2f}% frequency")
    with c2:
        pow_eff = np.exp(coefs.get('VehPower', 0.045))
        st.metric("Engine Power Elasticity (+1 Unit)", f"{(pow_eff-1)*100:+.2f}% frequency")

# ============================================================
# TAB 3 — COMPARE PROFILES
# ============================================================
with tab3:
    st.subheader("Side-by-Side Portfolio Risk Simulation")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Profile A (Reference)**")
        a_driv = st.slider("Driver Age (A)", 18, 100, 24, key="a_d")
        a_veh = st.slider("Vehicle Age (A)", 0, 30, 1, key="a_v")
        a_power = st.slider("Vehicle Power (A)", 1, 15, 6, key="a_p")
        a_bm = st.slider("Bonus-Malus (A)", 50, 150, 55, key="a_b")
        a_area = st.selectbox("Area (A)", ['A','B','C','D','E','F'], index=1, key="a_ar")
        a_gas = st.radio("Fuel (A)", ['Regular', 'Diesel'], key="a_g")

    with colB:
        st.markdown("**Profile B (Target Scenario)**")
        b_driv = st.slider("Driver Age (B)", 18, 100, 48, key="b_d")
        b_veh = st.slider("Vehicle Age (B)", 0, 30, 7, key="b_v")
        b_power = st.slider("Vehicle Power (B)", 1, 15, 9, key="b_p")
        b_bm = st.slider("Bonus-Malus (B)", 50, 150, 90, key="b_b")
        b_area = st.selectbox("Area (B)", ['A','B','C','D','E','F'], index=4, key="b_ar")
        b_gas = st.radio("Fuel (B)", ['Regular', 'Diesel'], key="b_g")

    ra = compute_premium(a_driv, a_veh, a_power, a_bm, a_area, a_gas)
    rb = compute_premium(b_driv, b_veh, b_power, b_bm, b_area, b_gas)

    st.divider()
    comp_df = pd.DataFrame({
        'Actuarial Metric': ['Predicted Frequency', f'Expected Severity ({currency_symbol})', f'Pure Premium ({currency_symbol})', f'Commercial Premium ({currency_symbol})', 'Underwriting Status'],
        'Profile A': [f"{ra['predicted_freq']:.3f}", f"{currency_symbol}{ra['expected_severity']:,.0f}",
                      f"{currency_symbol}{ra['pure_premium']:,.2f}", f"{currency_symbol}{ra['commercial_premium']:,.2f}",
                      "REFERRED" if ra['referred'] else "APPROVED"],
        'Profile B': [f"{rb['predicted_freq']:.3f}", f"{currency_symbol}{rb['expected_severity']:,.0f}",
                      f"{currency_symbol}{rb['pure_premium']:,.2f}", f"{currency_symbol}{rb['commercial_premium']:,.2f}",
                      "REFERRED" if rb['referred'] else "APPROVED"],
    })
    st.table(comp_df.set_index('Actuarial Metric'))

    diff_pct = (rb['commercial_premium'] / ra['commercial_premium'] - 1) * 100
    if diff_pct > 0:
        st.success(f"Profile B yields a **{diff_pct:.1f}% higher** premium rate than Profile A.")
    else:
        st.success(f"Profile B yields a **{abs(diff_pct):.1f}% lower** premium rate than Profile A.")

# ============================================================
# TAB 4 — COMPLIANCE & AUDIT TRAIL
# ============================================================
with tab4:
    st.subheader("Model Governance & Regulatory Parameters")
    st.markdown("""
    * **Engine Framework:** Negative Binomial GLM (Frequency) + Segmented Severity Table.
    * **Data Origin Benchmark:** Trained on standardized European motor portfolios (`freMTPL2` specifications).
    * **Solvency II Alignment:** Technical provisions incorporate explicit expense, risk margin, and reinsurance safety loadings.
    """)
    st.json({
        "intercept_coefficient": coefs.get('Intercept'),
        "monitored_coefficients_count": len(coefs),
        "pricing_loadings": pricing
    })

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Motor Insurance Pricing Engine v3.2-Enterprise | Powered by Actuarial GLM Analytics & Streamlit"
)
