"""
app.py — Entry point Streamlit
DSS Prediksi Triage IGD RSU Aulia
XGBoost + SHAP + SATS-TEWS

Run: streamlit run app/app.py
"""

import os
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="DSS Triage IGD — RSU Aulia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "**Sistem Prediksi Triage IGD** v2.0\n\nRSU Aulia · SATS-TEWS · XGBoost + SHAP\n\nPrototype penelitian — bukan pengganti keputusan medis."
    }
)

# ─── Load custom CSS ───────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# ─── Load model artifacts ─────────────────────────────────────────────────
from backend.predictor import load_artifacts
artifacts = load_artifacts()

model_active = artifacts is not None
model_version = artifacts.get('model_version', 'v2.0') if artifacts else '-'

# ─── Global Header (Topbar) ──────────────────────────────────────────────
status_dot   = 'active' if model_active else 'inactive'
status_text  = 'Model Aktif' if model_active else 'Tidak Tersedia'

st.markdown(f"""
<div class="dss-header">
    <div class="dss-header-left">
        <div>
            <h1>Sistem Prediksi Triage IGD</h1>
            <p>RSU Aulia &nbsp;·&nbsp; Standar SATS-TEWS &nbsp;·&nbsp; Model XGBoost + SHAP</p>
        </div>
    </div>
    <div class="dss-header-badges">
        <span class="badge badge-active">
            <span class="badge-dot {status_dot}"></span>
            {status_text}
        </span>
        <span class="badge badge-version">{model_version} · Final</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Legend strip (always-visible SATS color reference) ────────────────────
st.markdown("""
<div class="legend-strip">
    <div class="legend-chip"><span class="legend-swatch" style="background:var(--sats-red)"></span><span class="legend-text"><span class="legend-label">Merah — Resusitasi</span><span class="legend-time">0 menit</span></span></div>
    <div class="legend-chip"><span class="legend-swatch" style="background:var(--sats-orange)"></span><span class="legend-text"><span class="legend-label">Oranye — Emergent</span><span class="legend-time">&lt; 10 menit</span></span></div>
    <div class="legend-chip"><span class="legend-swatch" style="background:var(--sats-yellow)"></span><span class="legend-text"><span class="legend-label">Kuning — Urgent</span><span class="legend-time">&lt; 60 menit</span></span></div>
    <div class="legend-chip"><span class="legend-swatch" style="background:var(--sats-green)"></span><span class="legend-text"><span class="legend-label">Hijau — Less Urgent</span><span class="legend-time">&lt; 4 jam</span></span></div>
    <div class="legend-chip"><span class="legend-swatch" style="background:var(--sats-blue)"></span><span class="legend-text"><span class="legend-label">Biru — Not Urgent</span><span class="legend-time">&lt; 6 jam</span></span></div>
</div>
""", unsafe_allow_html=True)

# ─── Model unavailable warning ────────────────────────────────────────────
if not model_active:
    st.error(
        "⚠️ **Model tidak dapat dimuat.** Pastikan file `.pkl` tersedia di direktori `model/artifacts/`. "
        "Jalankan `scripts/retrain.py` atau notebook untuk men-generate artefak."
    )
    st.stop()

# ─── 3 Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬 Input Pasien & Hasil Prediksi",
    "🧮 Kalkulator TEWS",
    "📖 Panduan SATS",
])

with tab1:
    from ui.tab_prediction import render_tab_prediction
    render_tab_prediction(artifacts)

with tab2:
    from ui.tab_tews import render_tab_tews
    render_tab_tews()

with tab3:
    from ui.tab_guidelines import render_tab_guidelines
    render_tab_guidelines()
