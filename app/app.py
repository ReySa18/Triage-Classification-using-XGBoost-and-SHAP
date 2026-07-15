"""
app.py — Entry point Streamlit
DSS Prediksi Triage IGD RSU Aulia
XGBoost + SHAP + SATS-TEWS

Run: streamlit run app/app.py
"""

from pathlib import Path
import sys

import streamlit as st

# Streamlit and its test runner initialize sys.path differently. Resolve all
# local imports from this file so startup does not depend on the terminal cwd.
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from deployment import (
    RetrainingError,
    artifacts_available,
    invalid_artifacts,
    run_retraining,
)

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
css_path = APP_DIR / 'assets' / 'style.css'
if css_path.is_file():
    with css_path.open('r', encoding='utf-8') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# ─── Ensure model artifacts exist before loading them ─────────────────────
if not artifacts_available():
    with st.status(
        "Model belum tersedia. Sedang melakukan retraining...",
        expanded=True,
    ) as status:
        try:
            run_retraining()
            remaining = invalid_artifacts()
            if remaining:
                missing_names = ", ".join(path.name for path in remaining)
                raise RetrainingError(
                    "Retraining selesai, tetapi satu atau lebih artefak model tidak terbentuk.",
                    f"Artefak tidak valid: {missing_names}",
                )

            status.update(
                label="Retraining selesai. Model siap digunakan.",
                state="complete",
                expanded=False,
            )
        except RetrainingError as error:
            status.update(label="Retraining model gagal.", state="error")
            st.error(str(error))
            if error.details:
                st.code(error.details, language="text")
            st.stop()
        except Exception as error:
            status.update(label="Retraining model gagal.", state="error")
            st.error("Terjadi kesalahan tidak terduga saat menyiapkan model.")
            st.code(str(error), language="text")
            st.stop()

# ─── Load model artifacts only after successful validation ────────────────
from backend.predictor import load_artifacts
artifacts = load_artifacts()

if artifacts is None:
    load_artifacts.clear()
    st.error("Model tidak dapat dimuat meskipun seluruh file artefak tersedia.")
    st.stop()

model_active = True
model_version = artifacts.get('model_version', 'v2.0')

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
