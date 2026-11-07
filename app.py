import os
import sys
import json
import time
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is available on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optitest.core.models import LinkParameters, TelemetrySample
from optitest.core.simulator import OpticalLinkWrapper, is_native_available
from optitest.faults.scenarios import get_all_scenarios, FaultScenario
from optitest.harness.rules import OpticalRuleEngine, RuleThresholds
from optitest.harness.runner import TestHarnessRunner, ScenarioTestResult
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.ml.evaluate import compare_rule_vs_ml_on_history
from optitest.ai_advisor.diagnostic import OpticalDiagnosticAdvisor, DiagnosticReport

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="OptiTest AI — Optical Simulation & Validation Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe Plotly Chart Helper to handle Streamlit width parameter smoothly
def render_plotly(fig, height: Optional[int] = None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

# ----------------- MODERN ENTERPRISE CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top banner glow */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(0, 240, 255, 0.08) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Metric card */
    .metric-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 18px;
        transition: all 0.2s ease-in-out;
    }

    .metric-box:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .metric-val {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
    }

    .metric-sub {
        font-size: 12px;
        font-weight: 500;
        color: #38BDF8;
        margin-top: 4px;
    }

    /* Pulse dot */
    .status-beacon {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .beacon-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-glow 2s infinite;
    }

    @keyframes pulse-glow {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Badge styles */
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    .badge-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    /* Schematic topology bar */
    .topology-container {
        background: linear-gradient(90deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
        border: 1px solid #3A506B;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 15px 0 25px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }

    .topology-node {
        text-align: center;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 10px 16px;
        min-width: 130px;
    }

    .topology-node-title {
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.5px;
    }

    .topology-node-val {
        font-size: 14px;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    .topology-arrow {
        color: #64748B;
        font-size: 18px;
        font-weight: bold;
    }

    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD RESOURCES -----------------
@st.cache_resource
def load_ml_detector():
    model_path = os.path.join(PROJECT_ROOT, "models", "optical_detector.joblib")
    if os.path.exists(model_path):
        detector = OpticalAnomalyDetector()
        detector.load(model_path)
        return detector
    return None

detector = load_ml_detector()

@st.cache_resource
def load_diagnostic_advisor():
    return OpticalDiagnosticAdvisor()

advisor = load_diagnostic_advisor()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚡ OptiTest AI")
    st.caption("AI-Assisted Optical Link Simulation & Software Validation Suite")
    st.markdown("---")

    # Engine Status Indicators
    st.markdown("<div style='font-size: 11px; font-weight:700; text-transform:uppercase; color:#94A3B8; margin-bottom:10px;'>Runtime Subsystems</div>", unsafe_allow_html=True)

    native_active = is_native_available()
    if native_active:
        st.markdown("""
        <div style='background:rgba(56, 189, 248, 0.1); border:1px solid rgba(56, 189, 248, 0.3); border-radius:8px; padding:10px; margin-bottom:8px;'>
            <div style='font-size:12px; font-weight:700; color:#38BDF8;'>⚡ C++17 Core Engine</div>
            <div style='font-size:11px; color:#94A3B8;'>pybind11 Native Shared Module</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(245, 158, 11, 0.1); border:1px solid rgba(245, 158, 11, 0.3); border-radius:8px; padding:10px; margin-bottom:8px;'>
            <div style='font-size:12px; font-weight:700; color:#F59E0B;'>🐍 Python Fallback Core</div>
            <div style='font-size:11px; color:#94A3B8;'>Pure Python Implementation</div>
        </div>
        """, unsafe_allow_html=True)

    if detector and detector.is_trained:
        st.markdown("""
        <div style='background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); border-radius:8px; padding:10px; margin-bottom:8px;'>
            <div style='font-size:12px; font-weight:700; color:#10B981;'>🤖 Dual ML Anomaly Engine</div>
            <div style='font-size:11px; color:#94A3B8;'>Random Forest (99.55% Acc) + IsoForest</div>
        </div>
        """, unsafe_allow_html=True)

    if advisor._client:
        st.markdown("""
        <div style='background:rgba(139, 92, 246, 0.1); border:1px solid rgba(139, 92, 246, 0.3); border-radius:8px; padding:10px; margin-bottom:8px;'>
            <div style='font-size:12px; font-weight:700; color:#A78BFA;'>✨ Diagnostic AI Advisor</div>
            <div style='font-size:11px; color:#94A3B8;'>Google Gemini 2.5 Flash Connected</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(148, 163, 184, 0.1); border:1px solid rgba(148, 163, 184, 0.3); border-radius:8px; padding:10px; margin-bottom:8px;'>
            <div style='font-size:12px; font-weight:700; color:#CBD5E1;'>⚙️ Diagnostic Heuristics</div>
            <div style='font-size:11px; color:#94A3B8;'>Deterministic Optical Expert Mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size: 11px; font-weight:700; text-transform:uppercase; color:#94A3B8; margin-bottom:8px;'>Standards Compliance</div>", unsafe_allow_html=True)
    st.markdown("""
    - **ITU-T G.692**: Optical interfaces for multichannel systems
    - **ITU-T G.959.1**: Physical layer optical transport network
    - **IEEE 802.3**: 10G/40G/100G Optical Ethernet sensitivity
    - **Telcordia GR-253**: SONET/SDH optical alarms & hysteresis
    """)

# ----------------- HERO BANNER -----------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="hero-title">⚡ OptiTest AI <span style="font-size: 14px; color: #94A3B8; font-weight: 500;">v1.0.0</span></div>
            <p class="hero-subtitle">High-fidelity optical link simulation, controlled fault-injection, automated regression testing, and AI-assisted root cause diagnostics.</p>
        </div>
        <div>
            <span class="status-beacon"><span class="beacon-dot"></span> LIVE SIMULATION ACTIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS NAVIGATION -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Optical Topology & Power Budget",
    "💥 Fault Injection Studio",
    "🧪 Automated Test Matrix",
    "🤖 AI Anomaly Detection",
    "📋 Diagnostic & Root Cause Report"
])

# ==============================================================================
# TAB 1: OPTICAL LINK TOPOLOGY & POWER BUDGET
# ==============================================================================
with tab1:
    st.subheader("Physical Link Configuration & Optical Attenuation Profile")
    st.caption("Calculates deterministic optical power budget: $P_{\\mathrm{rx}} = P_{\\mathrm{tx}} - \\alpha L - L_{\\mathrm{conn}} - L_{\\mathrm{splice}} - L_{\\mathrm{other}}$")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])

    with col_cfg1:
        tx_power = st.slider("Transmit Launch Power $P_{\\mathrm{tx}}$ (dBm)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
        fiber_length = st.slider("Fiber Span Length $L$ (km)", min_value=0.0, max_value=120.0, value=40.0, step=1.0)
        wavelength_preset = st.selectbox(
            "Wavelength & Fiber Preset",
            ["1550 nm — Standard SMF-28 (0.20 dB/km)", "1310 nm — O-Band Low-Dispersion (0.35 dB/km)", "Custom Link"]
        )

    with col_cfg2:
        if "1550" in wavelength_preset:
            alpha_val = 0.20
            dispersion_val = 17.0
            wl_val = 1550.0
        elif "1310" in wavelength_preset:
            alpha_val = 0.35
            dispersion_val = 0.5
            wl_val = 1310.0
        else:
            alpha_val = 0.22
            dispersion_val = 17.0
            wl_val = 1550.0

        attenuation_coeff = st.number_input("Fiber Attenuation $\\alpha$ (dB/km)", min_value=0.01, max_value=1.0, value=alpha_val, step=0.01)
        other_loss = st.number_input("Fixed Connector & Splice Losses $L_{\\mathrm{other}}$ (dB)", min_value=0.0, max_value=10.0, value=2.0, step=0.25)

    with col_cfg3:
        rx_sensitivity = st.number_input("Receiver Sensitivity Limit (dBm)", min_value=-40.0, max_value=-10.0, value=-24.0, step=1.0, help="Power level below which BER degrades rapidly")
        los_threshold = st.number_input("Loss of Signal (LOS) Hardware Alarm (dBm)", min_value=-45.0, max_value=-20.0, value=-28.0, step=1.0, help="Hard extinction power threshold")

    # Optical Link Simulation Instance
    params = LinkParameters(
        tx_power_dbm=tx_power,
        fiber_length_km=fiber_length,
        attenuation_db_per_km=attenuation_coeff,
        other_loss_db=other_loss,
        wavelength_nm=wl_val,
        dispersion_param_ps_nm_km=dispersion_val,
        rx_sensitivity_dbm=rx_sensitivity,
        los_threshold_dbm=los_threshold
    )
    link = OpticalLinkWrapper(params)

    fiber_loss = link.calculate_fiber_loss()
    rx_power = link.calculate_received_power()
    total_loss = fiber_loss + other_loss
    margin = rx_power - rx_sensitivity
    osnr = link.calculate_osnr(total_loss)
    ber = link.calculate_ber(rx_power, osnr)
    q_factor = link.calculate_q_factor(ber)
    total_dispersion = dispersion_val * fiber_length

    # Visual Topology Schematic
    st.markdown(f"""
    <div class="topology-container">
        <div class="topology-node">
            <div class="topology-node-title">Transmitter (Tx)</div>
            <div class="topology-node-val">{tx_power:+.1f} dBm</div>
            <div style="font-size:10px; color:#64748B;">DFB Laser ({wl_val:.0f}nm)</div>
        </div>
        <div class="topology-arrow">───▶</div>
        <div class="topology-node">
            <div class="topology-node-title">In-Line Loss</div>
            <div class="topology-node-val">-{other_loss:.1f} dB</div>
            <div style="font-size:10px; color:#64748B;">Patch & Connectors</div>
        </div>
        <div class="topology-arrow">───▶</div>
        <div class="topology-node" style="min-width: 170px;">
            <div class="topology-node-title">Fiber Cable Span</div>
            <div class="topology-node-val">{fiber_length:.0f} km (-{fiber_loss:.1f} dB)</div>
            <div style="font-size:10px; color:#64748B;">α = {attenuation_coeff:.2f} dB/km</div>
        </div>
        <div class="topology-arrow">───▶</div>
        <div class="topology-node">
            <div class="topology-node-title">Receiver (Rx)</div>
            <div class="topology-node-val" style="color: {'#10B981' if margin >= 3.0 else ('#F59E0B' if margin >= 0 else '#EF4444')};">{rx_power:+.2f} dBm</div>
            <div style="font-size:10px; color:#64748B;">Margin: {margin:+.1f} dB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Fiber Loss</div>
            <div class="metric-val">{fiber_loss:.2f} <span style="font-size:13px; color:#94A3B8;">dB</span></div>
            <div class="metric-sub">{fiber_length:.0f} km span</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Link Loss</div>
            <div class="metric-val">{total_loss:.2f} <span style="font-size:13px; color:#94A3B8;">dB</span></div>
            <div class="metric-sub">+{other_loss:.2f} dB other</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        status_color = "#10B981" if margin >= 3.0 else ("#F59E0B" if margin >= 0.0 else "#EF4444")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Received Power</div>
            <div class="metric-val" style="color: {status_color};">{rx_power:.2f} <span style="font-size:13px; color:#94A3B8;">dBm</span></div>
            <div class="metric-sub">{margin:+.2f} dB margin</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Optical SNR</div>
            <div class="metric-val">{osnr:.1f} <span style="font-size:13px; color:#94A3B8;">dB</span></div>
            <div class="metric-sub">Target &ge; 18.0 dB</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Pre-FEC BER</div>
            <div class="metric-val">{ber:.2e}</div>
            <div class="metric-sub">Q = {q_factor:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Dispersion</div>
            <div class="metric-val">{total_dispersion:.1f} <span style="font-size:13px; color:#94A3B8;">ps/nm</span></div>
            <div class="metric-sub">D = {dispersion_val:.1f} ps/(nm&middot;km)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Plot 1: Power vs Distance attenuation curve P(z)
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        # Distance profile
        z_pts = np.linspace(0.0, max(1.0, fiber_length), 200)
        # Power drops by connectors immediately (z=0), then alpha * z
        p_z = tx_power - (other_loss * 0.5) - (attenuation_coeff * z_pts) - (other_loss * 0.5 * (z_pts / max(1.0, fiber_length)))

        fig_attenuation = go.Figure()
        fig_attenuation.add_trace(go.Scatter(
            x=z_pts, y=p_z,
            mode="lines",
            name="Optical Power Profile P(z)",
            line=dict(color="#38BDF8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.05)"
        ))
        fig_attenuation.add_hline(y=rx_sensitivity, line_dash="dash", line_color="#F59E0B", annotation_text=f"Sensitivity Floor ({rx_sensitivity:.0f} dBm)")
        fig_attenuation.add_hline(y=los_threshold, line_dash="dot", line_color="#EF4444", annotation_text=f"LOS Threshold ({los_threshold:.0f} dBm)")
        fig_attenuation.update_layout(
            title="Optical Signal Attenuation vs Fiber Distance P(z)",
            xaxis_title="Fiber Distance z (km)",
            yaxis_title="Optical Power (dBm)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8"),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        fig_attenuation.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
        fig_attenuation.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")
        render_plotly(fig_attenuation, height=360)

    with col_p2:
        # Power Budget Waterfall Chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="Power Budget",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Tx Launch", "Fiber Loss", "Other Losses", "Rx Power"],
            textposition="outside",
            text=[f"{tx_power:+.1f} dBm", f"-{fiber_loss:.1f} dB", f"-{other_loss:.1f} dB", f"{rx_power:+.1f} dBm"],
            y=[tx_power, -fiber_loss, -other_loss, rx_power],
            connector={"line": {"color": "#64748B"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#38BDF8"}}
        ))
        fig_waterfall.update_layout(
            title="Power Budget Waterfall Breakdown (dB)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8"),
            margin=dict(l=30, r=30, t=40, b=40)
        )
        fig_waterfall.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)", title="Power Level (dBm)")
        render_plotly(fig_waterfall, height=360)

# ==============================================================================
# TAB 2: FAULT INJECTION STUDIO
# ==============================================================================
with tab2:
    st.subheader("Controlled Fault Injection & Oscilloscope Studio")
    st.caption("Apply deterministic physical impairments and observe multi-channel transceiver telemetry in real-time.")

    # Quick Presets
    st.markdown("<div style='font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:8px;'>QUICK FAULT PRESETS</div>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    
    if "selected_fault_mode" not in st.session_state:
        st.session_state["selected_fault_mode"] = "Complete Fiber Cut (LOS)"

    with p_col1:
        if st.button("✂️ Fiber Cut", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Complete Fiber Cut (LOS)"
    with p_col2:
        if st.button("〰️ Macro-Bend", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Progressive Macro-Bend (Attenuation Ramp)"
    with p_col3:
        if st.button("🪞 Dirty Ferrule", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Connector Contamination (Step Loss)"
    with p_col4:
        if st.button("🔥 Laser Aging", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Laser Diode Aging (Bias Drift)"
    with p_col5:
        if st.button("📡 ASE Noise", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Amplifier ASE Noise Injection"
    with p_col6:
        if st.button("🔄 Recovery Cycle", use_container_width=True):
            st.session_state["selected_fault_mode"] = "Temporary Fault & Recovery Cycle"

    col_flt1, col_flt2, col_flt3 = st.columns([1.2, 1, 1])

    with col_flt1:
        fault_type = st.selectbox(
            "Selected Fault Mechanism",
            [
                "Complete Fiber Cut (LOS)",
                "Progressive Macro-Bend (Attenuation Ramp)",
                "Connector Contamination (Step Loss)",
                "Laser Diode Aging (Bias Drift)",
                "Amplifier ASE Noise Injection",
                "Temporary Fault & Recovery Cycle",
                "Normal Link (Baseline Nominal)"
            ],
            index=[
                "Complete Fiber Cut (LOS)",
                "Progressive Macro-Bend (Attenuation Ramp)",
                "Connector Contamination (Step Loss)",
                "Laser Diode Aging (Bias Drift)",
                "Amplifier ASE Noise Injection",
                "Temporary Fault & Recovery Cycle",
                "Normal Link (Baseline Nominal)"
            ].index(st.session_state["selected_fault_mode"]) if st.session_state["selected_fault_mode"] in [
                "Complete Fiber Cut (LOS)",
                "Progressive Macro-Bend (Attenuation Ramp)",
                "Connector Contamination (Step Loss)",
                "Laser Diode Aging (Bias Drift)",
                "Amplifier ASE Noise Injection",
                "Temporary Fault & Recovery Cycle",
                "Normal Link (Baseline Nominal)"
            ] else 0
        )

    with col_flt2:
        fault_start_s = st.slider("Fault Trigger Timestamp (s)", min_value=0.5, max_value=6.0, value=2.0, step=0.5)
        sim_step_count = st.slider("Simulation Time Steps", min_value=40, max_value=150, value=80, step=10)

    with col_flt3:
        if "Macro-Bend" in fault_type:
            ramp_rate = st.slider("Attenuation Ramp Rate (dB/s)", 0.5, 4.0, 2.0, 0.5)
        elif "Connector" in fault_type:
            conn_loss = st.slider("Connector Insertion Step Loss (dB)", 4.0, 18.0, 13.0, 0.5)
        elif "Noise" in fault_type:
            noise_penalty = st.slider("ASE Noise Penalty (dB)", 4.0, 18.0, 10.0, 1.0)
        elif "Recovery" in fault_type:
            recov_len = st.slider("Disruption Duration (s)", 1.0, 4.0, 2.5, 0.5)

    # Execute simulation
    sim_link = OpticalLinkWrapper(params)
    dt = 0.1
    history = []

    for step_i in range(sim_step_count):
        t = step_i * dt
        p_att, p_noise, p_bias, p_temp, force_los = 0.0, 0.0, 0.0, 0.0, False
        f_name = "NONE"

        if "Fiber Cut" in fault_type and t >= fault_start_s:
            force_los = True
            p_att = 99.0
            f_name = "FIBER_CUT"
        elif "Macro-Bend" in fault_type and t >= fault_start_s:
            p_att = min(16.0, (t - fault_start_s) * ramp_rate)
            f_name = "ATTENUATION_RAMP"
        elif "Connector" in fault_type and t >= fault_start_s:
            p_att = conn_loss
            p_noise = 2.0
            f_name = "CONNECTOR_CONTAMINATION"
        elif "Laser Diode" in fault_type and t >= fault_start_s:
            elapsed = t - fault_start_s
            p_bias = min(32.0, elapsed * 5.0)
            p_temp = min(16.0, elapsed * 2.5)
            p_att = 0.5
            f_name = "LASER_AGING"
        elif "Noise" in fault_type and t >= fault_start_s:
            p_noise = noise_penalty
            f_name = "NOISE_BURST"
        elif "Recovery" in fault_type:
            if fault_start_s <= t < (fault_start_s + recov_len):
                force_los = True
                p_att = 99.0
                f_name = "TEMPORARY_CUT"

        s = sim_link.step(t, p_att, p_noise, p_bias, p_temp, force_los, f_name)
        history.append(s)

    df_hist = pd.DataFrame([s.to_dict() for s in history])

    # Oscilloscope Waveforms
    fig_osc = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            "CH 1: Received Optical Power P_rx (dBm) with Detection Thresholds",
            "CH 2: Optical Signal-to-Noise Ratio (dB) & Pre-FEC Bit Error Rate log10(BER)",
            "CH 3: Transceiver Laser Health — Bias Current (mA) & Temperature (°C)"
        ]
    )

    # Trace 1: Power
    fig_osc.add_trace(go.Scatter(
        x=df_hist["timestamp_s"], y=df_hist["rx_power_dbm"],
        mode="lines+markers", name="Rx Power (dBm)",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=4)
    ), row=1, col=1)
    fig_osc.add_hline(y=rx_sensitivity, line_dash="dash", line_color="#F59E0B", annotation_text=f"Sensitivity Floor ({rx_sensitivity:.0f} dBm)", row=1, col=1)
    fig_osc.add_hline(y=los_threshold, line_dash="dash", line_color="#EF4444", annotation_text=f"LOS Alarm ({los_threshold:.0f} dBm)", row=1, col=1)

    # Trace 2: OSNR & BER
    fig_osc.add_trace(go.Scatter(
        x=df_hist["timestamp_s"], y=df_hist["osnr_db"],
        mode="lines", name="OSNR (dB)", line=dict(color="#10B981", width=2)
    ), row=2, col=1)
    fig_osc.add_trace(go.Scatter(
        x=df_hist["timestamp_s"], y=np.log10(np.clip(df_hist["ber"], 1e-15, 0.5)),
        mode="lines", name="Log10(BER)", line=dict(color="#F97316", width=2, dash="dot")
    ), row=2, col=1)

    # Trace 3: Laser Health
    fig_osc.add_trace(go.Scatter(
        x=df_hist["timestamp_s"], y=df_hist["laser_bias_current_ma"],
        mode="lines", name="Laser Bias Current (mA)", line=dict(color="#C084FC", width=2)
    ), row=3, col=1)
    fig_osc.add_trace(go.Scatter(
        x=df_hist["timestamp_s"], y=df_hist["laser_temperature_c"],
        mode="lines", name="Laser Temp (°C)", line=dict(color="#F43F5E", width=2)
    ), row=3, col=1)

    fig_osc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        height=660,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    for r in range(1, 4):
        fig_osc.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)", row=r, col=1)
        fig_osc.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)", row=r, col=1)
    fig_osc.update_xaxes(title_text="Simulation Time (seconds)", row=3, col=1)

    render_plotly(fig_osc)

    # Action Toolbar
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])
    with col_dl1:
        csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Telemetry (CSV)", data=csv_bytes, file_name="optical_telemetry.csv", mime="text/csv", use_container_width=True)
    with col_dl2:
        json_bytes = json.dumps([s.to_dict() for s in history], indent=2).encode('utf-8')
        st.download_button("📥 Export Telemetry (JSON)", data=json_bytes, file_name="optical_telemetry.json", mime="application/json", use_container_width=True)

# ==============================================================================
# TAB 3: AUTOMATED TEST SUITE & REGRESSION MATRIX
# ==============================================================================
with tab3:
    st.subheader("Automated Optical Software Validation & Regression Matrix")
    st.caption("Executes standardized test vectors against ITU-T transmission limits and debounced alarm state machines.")

    col_btn, col_stats = st.columns([1.2, 3])
    with col_btn:
        run_suite = st.button("▶ Run Full Regression Suite", type="primary", use_container_width=True)

    if run_suite or "test_results" not in st.session_state:
        with st.spinner("Executing optical scenario test suite..."):
            runner = TestHarnessRunner()
            st.session_state["test_results"] = runner.run_all()

    results: List[ScenarioTestResult] = st.session_state["test_results"]

    tot = len(results)
    pass_cnt = sum(1 for r in results if r.status == "PASS")
    fail_cnt = tot - pass_cnt
    pass_rate = (pass_cnt / tot * 100.0) if tot else 0.0

    with col_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Tests", str(tot))
        s2.metric("Passing Tests", str(pass_cnt), delta=f"{pass_rate:.0f}% Pass Rate")
        s3.metric("Failed Tests", str(fail_cnt))
        s4.metric("Avg Latency", f"{np.mean([r.duration_ms for r in results]):.1f} ms")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Verification Table
    records = []
    for r in results:
        badge = "🟢 PASS" if r.status == "PASS" else "🔴 FAIL"
        records.append({
            "Verdict": badge,
            "Test Scenario": r.scenario_name,
            "Expected Nominal": r.expected_nominal,
            "Observed Nominal": r.observed_nominal,
            "Expected Fault": r.expected_fault,
            "Observed Fault": r.observed_fault,
            "Triggered Alarms": ", ".join(r.alarms_triggered) if r.alarms_triggered else "None",
            "Latency": f"{r.duration_ms:.1f} ms"
        })

    st.dataframe(pd.DataFrame(records), use_container_width=True)

    # Scenario Deep-Dive Inspector
    st.markdown("---")
    st.subheader("Scenario Diagnostic Drill-Down")
    selected_sc_name = st.selectbox("Inspect Scenario Waveform & State Transitions", [r.scenario_name for r in results])
    sc_res = next(r for r in results if r.scenario_name == selected_sc_name)

    sc_col1, sc_col2 = st.columns([1, 2])
    with sc_col1:
        st.markdown(f"**Description:** {sc_res.description}")
        st.markdown(f"**Status Verdict:** `{sc_res.status}`")
        st.markdown(f"**Nominal Transition:** `{sc_res.expected_nominal}` &rarr; Observed `{sc_res.observed_nominal}`")
        st.markdown(f"**Fault Transition:** `{sc_res.expected_fault}` &rarr; Observed `{sc_res.observed_fault}`")
        if sc_res.alarms_triggered:
            st.error(f"Alarms Asserted: {', '.join(sc_res.alarms_triggered)}")
        else:
            st.success("No alarms asserted during nominal phase.")

    with sc_col2:
        res_df = pd.DataFrame([s.to_dict() for s in sc_res.history])
        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(
            x=res_df["timestamp_s"], y=res_df["rx_power_dbm"],
            name="Rx Power (dBm)", line=dict(color="#38BDF8", width=2.5)
        ))
        fig_single.update_layout(
            title=f"Rx Power Trajectory: {selected_sc_name}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8"),
            height=280,
            margin=dict(l=30, r=30, t=35, b=30)
        )
        fig_single.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)", title="Time (s)")
        fig_single.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)", title="Rx Power (dBm)")
        render_plotly(fig_single)

# ==============================================================================
# TAB 4: AI ANOMALY DETECTION & COMPARISON
# ==============================================================================
with tab4:
    st.subheader("AI Anomaly Detection & Comparison Matrix")
    st.caption("Demonstrating the architectural advantage of combining machine learning with deterministic test rules.")

    if detector and detector.is_trained:
        row_ai1, row_ai2 = st.columns([1, 1.8])

        with row_ai1:
            st.markdown("""
            <div class="metric-box" style="margin-bottom: 16px;">
                <div style="font-size:13px; font-weight:700; color:#38BDF8; margin-bottom:6px;">Model Ensemble Performance</div>
                <div style="font-size:12px; color:#94A3B8; line-height: 1.6;">
                    &bull; <b>Supervised Classifier:</b> Random Forest (100 estimators)<br/>
                    &bull; <b>Hold-out Accuracy:</b> 99.55% | <b>Macro F1:</b> 99.48%<br/>
                    &bull; <b>Unsupervised Scorer:</b> Isolation Forest (Contamination=0.25)<br/>
                    &bull; <b>Input Features:</b> 12 Multi-Channel Telemetry Features
                </div>
            </div>
            """, unsafe_allow_html=True)

            feat_imp = detector.get_feature_importances()
            if feat_imp:
                fig_f = go.Figure(go.Bar(
                    x=list(feat_imp.values())[:6],
                    y=list(feat_imp.keys())[:6],
                    orientation="h",
                    marker_color="#818CF8"
                ))
                fig_f.update_layout(
                    title="Top 6 Anomaly Predictive Features",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.6)",
                    font=dict(color="#94A3B8"),
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                fig_f.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
                render_plotly(fig_f)

        with row_ai2:
            st.markdown("<div style='font-size:13px; font-weight:700; color:#E2E8F0; margin-bottom:8px;'>Precursor Anomaly Lead-Time Analysis</div>", unsafe_allow_html=True)
            eval_scenario = st.selectbox(
                "Select Fault Profile for Side-by-Side Comparison",
                ["Laser Diode Aging (Subtle Current Precursor)", "Connector Dirt Contamination", "Complete Fiber Cut", "Amplifier Optical Noise"]
            )

            if "Laser" in eval_scenario:
                sc_target = next(s for s in get_all_scenarios() if "Laser" in s.name)
            elif "Connector" in eval_scenario:
                sc_target = next(s for s in get_all_scenarios() if "Connector" in s.name)
            elif "Cut" in eval_scenario:
                sc_target = next(s for s in get_all_scenarios() if "Cut" in s.name)
            else:
                sc_target = next(s for s in get_all_scenarios() if "Noise" in s.name)

            runner_cmp = TestHarnessRunner()
            res_target = runner_cmp.run_scenario(sc_target)

            rule_engine = OpticalRuleEngine()
            comparison = compare_rule_vs_ml_on_history(
                res_target.history, detector, rule_engine, fault_start_step=20
            )

            # Lead time callout
            if comparison["lead_time_steps"] > 0:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #34D399; font-size: 13px;">🚀 AI Detection Advantage Verified</div>
                    <div style="font-size: 12px; color: #E2E8F0; margin-top: 2px;">
                        The Machine Learning model identified the anomaly at <b>step {comparison['ml_trigger_step']}</b>, which is <b>{comparison['lead_time_steps']} steps ({comparison['lead_time_steps']*0.1:.1f}s) earlier</b> than the deterministic ITU-T threshold at step {comparison['rule_trigger_step']}!
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #38BDF8; font-size: 13px;">⚡ Fast Detection Response</div>
                    <div style="font-size: 12px; color: #E2E8F0; margin-top: 2px;">
                        Rule alarm and ML detector triggered synchronously at step {comparison['rule_trigger_step']}.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            fig_comp = go.Figure()
            steps = list(range(comparison["total_steps"]))
            fig_comp.add_trace(go.Scatter(
                x=steps, y=comparison["ml_scores"],
                name="ML Anomaly Probability", line=dict(color="#F43F5E", width=2.5)
            ))
            fig_comp.add_trace(go.Scatter(
                x=steps, y=[1.0 if a else 0.0 for a in comparison["rule_alarms"]],
                name="Deterministic Rule Alarm (0/1)", line=dict(color="#38BDF8", width=2, dash="dash")
            ))
            fig_comp.add_hline(y=0.55, line_dash="dot", line_color="#F59E0B", annotation_text="ML Anomaly Threshold (0.55)")
            fig_comp.add_vline(x=20, line_dash="solid", line_color="#FFFFFF", annotation_text="Fault Injected (t=2.0s)")
            fig_comp.update_layout(
                title="Rule Alarm State vs ML Anomaly Score over Time",
                xaxis_title="Time Step (0.1s / step)",
                yaxis_title="Probability / Alarm State",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#94A3B8"),
                height=320,
                margin=dict(l=30, r=30, t=40, b=30),
                hovermode="x unified"
            )
            fig_comp.update_xaxes(gridcolor="rgba(148, 163, 184, 0.1)")
            fig_comp.update_yaxes(gridcolor="rgba(148, 163, 184, 0.1)")
            render_plotly(fig_comp)

    else:
        st.warning("ML models not loaded. Please train models using scripts/train_models.py.")

# ==============================================================================
# TAB 5: AI DIAGNOSTIC & ROOT CAUSE REPORT
# ==============================================================================
with tab5:
    st.subheader("Telemetry-Grounded AI Diagnostic & RCA Engine")
    st.caption("Root Cause Analysis strictly grounded in physical telemetry, powered by Google Gemini 2.5 and Optical Communications heuristics.")

    c_sel, c_btn = st.columns([2, 1])
    with c_sel:
        target_sc = st.selectbox("Select Test Scenario for Root Cause Analysis", [r.scenario_name for r in results])
    with c_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        gen_report = st.button("🔍 Generate Incident RCA Report", type="primary", use_container_width=True)

    chosen_result = next(r for r in results if r.scenario_name == target_sc)

    if gen_report or "active_diag" not in st.session_state:
        with st.spinner("Analyzing multi-channel telemetry and synthesizing RCA report..."):
            ml_pred = None
            if detector and detector.is_trained:
                ml_df = detector.predict_history(chosen_result.history)
                last_row = ml_df.iloc[-1]
                ml_pred = {
                    "category": last_row["predicted_category"],
                    "confidence": float(last_row["confidence"])
                }

            st.session_state["active_diag"] = advisor.diagnose(chosen_result, ml_pred)

    active_diag: Optional[DiagnosticReport] = st.session_state.get("active_diag")

    if active_diag and active_diag.scenario_name == target_sc:
        # Severity Pill
        sev = active_diag.incident_severity
        if "CRITICAL" in sev:
            sev_bg = "rgba(239, 68, 68, 0.15)"
            sev_color = "#F87171"
            sev_label = "P1 — CRITICAL SERVICE OUTAGE"
        elif "MAJOR" in sev:
            sev_bg = "rgba(245, 158, 11, 0.15)"
            sev_color = "#FBBF24"
            sev_label = "P2 — MAJOR LINK DEGRADATION"
        elif "WARNING" in sev:
            sev_bg = "rgba(59, 130, 246, 0.15)"
            sev_color = "#60A5FA"
            sev_label = "P3 — TELEMETRY WARNING"
        else:
            sev_bg = "rgba(16, 185, 129, 0.15)"
            sev_color = "#34D399"
            sev_label = "NOMINAL — HEALTHY OPERATION"

        st.markdown(f"""
        <div style="background:{sev_bg}; border:1px solid {sev_color}; border-radius:10px; padding:12px 18px; margin: 15px 0 20px 0; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:11px; font-weight:700; color:{sev_color}; letter-spacing:0.5px;">INCIDENT CLASSIFICATION</span>
                <div style="font-size:16px; font-weight:800; color:{sev_color};">{sev_label}</div>
            </div>
            <div style="font-size:12px; color:#E2E8F0; text-align:right;">
                <b>Engine:</b> {'✨ Google Gemini 2.5 LLM' if active_diag.is_llm_generated else '⚙️ Deterministic Optical Expert Heuristic'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(active_diag.markdown_report)

        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.download_button(
                "📥 Download Incident Report (Markdown)",
                data=active_diag.markdown_report.encode("utf-8"),
                file_name=f"RCA_Report_{target_sc.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with exp_c2:
            st.download_button(
                "📥 Export Incident Findings (JSON)",
                data=json.dumps({
                    "scenario": active_diag.scenario_name,
                    "status": active_diag.status,
                    "severity": active_diag.incident_severity,
                    "summary": active_diag.root_cause_summary,
                    "physical_cause": active_diag.physical_root_cause,
                    "findings": active_diag.telemetry_findings,
                    "remediation": active_diag.remediation_steps
                }, indent=2).encode("utf-8"),
                file_name=f"RCA_Report_{target_sc.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
