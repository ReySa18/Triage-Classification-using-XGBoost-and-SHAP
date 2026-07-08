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

# ─── Theme state ───────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

theme = st.session_state['theme']

# ─── Load custom CSS ───────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# ─── Apply theme variables and data attributes ─────────────────────────────
theme_override_css = ""
if theme == 'dark':
    theme_override_css = """
    <style>
    :root {
        --bg-app: #111318;
        --bg-card: #1a1d24;
        --bg-surface: #22252d;
        --bg-surface-high: #2a2e38;
        --bg-surface-low: #191c22;
        --border-main: #3a3f4b;
        --border-light: #2e3240;
        --text-primary: #e4e5eb;
        --text-secondary: #b0b4c0;
        --text-tertiary: #8a8f9e;
        --accent-primary: #6aadff;
        --accent-primary-dark: #4d94e6;
        --accent-primary-light: #8ec4ff;

        --input-bg: #22252d;
        --input-border: #3a3f4b;
        --input-focus-border: #6aadff;
        --input-focus-shadow: rgba(106, 173, 255, 0.2);

        --metric-bg: #22252d;
        --metric-value: #e4e5eb;
        --metric-label: #b0b4c0;
        --metric-sub: #8a8f9e;

        --gcs-bg: linear-gradient(135deg, #22252d, #2a2e38);
        --gcs-border: #3a3f4b;
        --gcs-value: #6aadff;
        --gcs-label: #b0b4c0;

        --disclaimer-bg: #1a1d24;
        --disclaimer-border: #3a3f4b;
        --disclaimer-text: #b0b4c0;

        --placeholder-bg: #1a1d24;
        --placeholder-border: #3a3f4b;
        --placeholder-text: #8a8f9e;

        --table-header-bg: #22252d;
        --table-header-color: #b0b4c0;
        --table-border: #3a3f4b;
        --table-row-border: #2e3240;
        --table-row-hover: #22252d;
        --table-cell-color: #e4e5eb;

        --tab-bg: #22252d;
        --tab-text: #b0b4c0;
        --tab-active-bg: #2a2e38;
        --tab-active-text: #6aadff;
        --tab-active-shadow: rgba(0,0,0,0.3);

        --flowchart-bg: #22252d;
        --flowchart-border: #3a3f4b;
        --flowchart-step-label: #e4e5eb;
        --flowchart-step-num: #8a8f9e;
        --flowchart-arrow: #8a8f9e;

        --ref-pill-bg: #22252d;
        --ref-pill-border: #3a3f4b;
        --ref-pill-title: #6aadff;
        --ref-pill-text: #b0b4c0;

        --prob-track: #2a2e38;
        --prob-label: #e4e5eb;
        --prob-pct: #b0b4c0;

        --shap-track-bg: #22252d;
        --shap-center: #3a3f4b;
        --shap-label: #b0b4c0;

        --vital-hint: #8a8f9e;

        --toggle-bg: #22252d;
        --toggle-border: #3a3f4b;
        --toggle-text: #b0b4c0;
        --toggle-icon-bg: #2a2e38;

        --legend-bg: #22252d;

        --sats-red-bg: #3d1f1f;
        --sats-red-border: #7f1d1d;
        --sats-red-text: #fca5a5;
        --sats-red-dot: #ef4444;

        --sats-orange-bg: #3d2e10;
        --sats-orange-border: #78350f;
        --sats-orange-text: #fcd34d;
        --sats-orange-dot: #f59e0b;

        --sats-yellow-bg: #3d3000;
        --sats-yellow-border: #856404;
        --sats-yellow-text: #ffe066;
        --sats-yellow-dot: #f59e0b;

        --sats-green-bg: #1f3d1f;
        --sats-green-border: #14532d;
        --sats-green-text: #86efac;
        --sats-green-dot: #22c55e;

        --sats-blue-bg: #1e293b;
        --sats-blue-border: #1e3a8a;
        --sats-blue-text: #93c5fd;
        --sats-blue-dot: #3b82f6;

        --sats-time-badge-bg: rgba(255,255,255,0.15);
        --sats-card-opacity: 0.92;
    }
    </style>
    """
st.markdown(theme_override_css, unsafe_allow_html=True)

# Set the data-theme attribute via iframe component as a backup for third-party elements
import streamlit.components.v1 as components
theme_attr = 'dark' if theme == 'dark' else 'light'
components.html(f"""
<script>
    try {{
        const parentDoc = window.parent.document;
        if (parentDoc) {{
            parentDoc.documentElement.setAttribute('data-theme', '{theme_attr}');
            const stApp = parentDoc.querySelector('.stApp');
            if (stApp) stApp.setAttribute('data-theme', '{theme_attr}');
        }}
    }} catch (e) {{
        console.warn('Cross-origin iframe sandboxing prevented setting data-theme attribute on parent document.');
    }}
</script>
""", height=0)

# Custom styles are loaded entirely from assets/style.css which supports themes via CSS variables.

# ─── Load model artifacts ─────────────────────────────────────────────────
from backend.predictor import load_artifacts
artifacts = load_artifacts()

model_active = artifacts is not None
model_version = artifacts.get('model_version', 'v2.0') if artifacts else '-'

# ─── Global Header with theme toggle ──────────────────────────────────────
status_class = 'badge-active' if model_active else 'badge-inactive'
status_dot   = 'active' if model_active else 'inactive'
status_text  = 'Aktif' if model_active else 'Tidak Tersedia'

# Header row with toggle
hdr_col1, hdr_col2 = st.columns([11, 1])

with hdr_col1:
    st.markdown(f"""
    <div class="dss-header">
        <div class="dss-header-left">
            <h1>🏥 Sistem Prediksi Triage IGD</h1>
            <p>RSU Aulia &nbsp;·&nbsp; Standar SATS-TEWS &nbsp;·&nbsp; Model XGBoost + SHAP</p>
        </div>
        <div class="dss-header-badges">
            <span class="badge badge-version">🤖 Model {model_version}</span>
            <span class="badge {status_class}">
                <span class="badge-dot {status_dot}"></span>
                {status_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hdr_col2:
    # Theme toggle button
    is_dark = theme == 'dark'
    toggle_icon = '☀️' if is_dark else '🌙'
    toggle_label = 'Light' if is_dark else 'Dark'
    if st.button(f"{toggle_icon} {toggle_label}", key="theme_toggle", use_container_width=True):
        st.session_state['theme'] = 'light' if is_dark else 'dark'
        st.rerun()

# ─── Model unavailable warning ────────────────────────────────────────────
if not model_active:
    st.error(
        "⚠️ **Model tidak dapat dimuat.** Pastikan file `.pkl` tersedia di direktori `model/artifacts/`. "
        "Jalankan `scripts/retrain.py` atau notebook untuk men-generate artefak."
    )
    st.stop()

# TEWS copy banner has been moved inside Tab 1 to prevent global layout shifting.

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
