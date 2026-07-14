"""
ui/tab_tews.py
Tab 2: Kalkulator TEWS (real-time, tanpa tombol submit)
"""

import streamlit as st
from backend.tews_calculator import (
    compute_tews_subscores, check_override,
    get_sats_zone_from_tews, get_active_range_label
)
from backend.feature_engineering import SATS_COLORS
from ui.components import score_badge_html, triage_badge_html


TEWS_REKOMENDASI = {
    0: "🚨 Resusitasi segera. Panggil tim resusitasi. Monitor kontinu.",
    1: "⚡ Tangani dalam 10 menit. Pasang IV line, monitor vital signs.",
    2: "⏰ Evaluasi dalam 60 menit. Pasang monitoring, pantau perubahan.",
    3: "🟢 Tangani dalam 4 jam. Triage ulang jika kondisi berubah.",
    4: "🔵 Tangani dalam 6 jam. Pertimbangkan rujuk ke fasilitas primer.",
}

def render_score_badge(score: int) -> str:
    return score_badge_html(score)


def render_tews_breakdown(subscores: dict, rr, spo2, sbp, hr, temp, gcs_total):
    params = [
        ('rr',   'Laju Pernafasan', f'{rr:.0f} x/mnt',   subscores['TEWS_rr_score']),
        ('spo2', 'SpO₂',            f'{spo2:.0f}%',       subscores['TEWS_spo2_score']),
        ('bp',   'Tekanan Darah',   f'{sbp:.0f} mmHg',    subscores['TEWS_bp_score']),
        ('hr',   'Denyut Jantung',  f'{hr:.0f} bpm',      subscores['TEWS_hr_score']),
        ('temp', 'Suhu Tubuh',      f'{temp:.1f}°C',      subscores['TEWS_temp_score']),
        ('gcs',  'GCS Total',       f'{gcs_total}',       subscores['TEWS_gcs_score']),
    ]

    rows_html = ''
    for key, label, value_str, score in params:
        range_label = get_active_range_label(key, None, score)
        badge_html = render_score_badge(score)
        rows_html += f"""
        <tr>
            <td style="font-weight:600;">{label}</td>
            <td>{value_str}</td>
            <td style="text-align:center;">{badge_html}</td>
            <td style="font-size:0.75rem;color:var(--text-tertiary);font-style:italic;">{range_label}</td>
        </tr>"""

    st.markdown(f"""
    <table class="session-table">
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Nilai</th>
                <th style="text-align:center;">Skor</th>
                <th>Rentang Aktif</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


def render_reference_table():
    rows = [
        ('Laju Pernafasan', [
            ('< 9', 3), ('9–11', 1), ('12–20', 0), ('21–29', 2), ('≥ 30', 3)
        ]),
        ('SpO₂', [
            ('< 90%', 3), ('90–94%', 2), ('95–96%', 1), ('≥ 97%', 0)
        ]),
        ('Tekanan Darah Sistole', [
            ('< 70', 3), ('70–89', 2), ('90–109', 1), ('110–149', 0), ('150–179', 1), ('≥ 180', 2)
        ]),
        ('Denyut Jantung', [
            ('< 40', 3), ('40–50', 1), ('51–100', 0), ('101–110', 1), ('111–129', 2), ('≥ 130', 3)
        ]),
        ('Suhu Tubuh', [
            ('< 35°C', 2), ('35.0–38.4°C', 0), ('≥ 38.5°C', 1)
        ]),
        ('GCS Total', [
            ('≤ 8', 3), ('9–12', 2), ('13–14', 1), ('15', 0)
        ]),
    ]

    html = '<table class="session-table" style="font-size:0.78rem;">'
    html += '<thead><tr>'
    html += '<th>Parameter</th>'
    html += '<th style="text-align:center;">Rentang → Skor</th>'
    html += '</tr></thead><tbody>'
    for param, ranges in rows:
        ranges_html = ' &nbsp;|&nbsp; '.join(
            [f'<b>{r}</b>→{render_score_badge(s)}' for r, s in ranges]
        )
        html += f'<tr><td style="font-weight:600;">{param}</td><td>{ranges_html}</td></tr>'
    html += '</tbody></table>'

    html += '<div style="margin-top:0.5rem;font-size:0.72rem;color:var(--text-tertiary);">'
    html += '<b>Ket:</b> Klik "Salin ke Tab Input" untuk mengirim data ke form prediksi.'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def reset_tews_inputs():
    defaults = {
        'tews_rr': 16,
        'tews_spo2': 98,
        'tews_sbp': 120,
        'tews_hr': 80,
        'tews_temp': 36.5,
        'tews_gcs': 15,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def render_tab_tews():
    st.markdown('<p style="font-size:0.85rem;color:var(--text-tertiary);margin-bottom:1rem;">Kalkulator TEWS mandiri — skor diperbarui secara real-time setiap kali nilai berubah.</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 5], gap="large")

    # ══════════════════════════════════════════════════════
    # PANEL KIRI — Input Parameter
    # ══════════════════════════════════════════════════════
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">📋 Parameter Tanda Vital</div>', unsafe_allow_html=True)

            t1, t2 = st.columns(2)
            with t1:
                t_rr = st.number_input("Laju Pernafasan (x/mnt)", min_value=0, max_value=80,
                                        value=16, step=1, key="tews_rr")
            with t2:
                t_spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100,
                                          value=98, step=1, key="tews_spo2")

            t3, t4 = st.columns(2)
            with t3:
                t_sbp = st.number_input("Tekanan Darah Sistole (mmHg)", min_value=0, max_value=300,
                                         value=120, step=1, key="tews_sbp")
            with t4:
                t_hr = st.number_input("Denyut Jantung (bpm)", min_value=0, max_value=300,
                                        value=80, step=1, key="tews_hr")

            t5, t6 = st.columns(2)
            with t5:
                t_temp = st.number_input("Suhu Tubuh (°C)", min_value=30.0, max_value=45.0,
                                          value=36.5, step=0.1, format="%.1f", key="tews_temp")
            with t6:
                t_gcs = st.number_input("GCS Total", min_value=3, max_value=15,
                                         value=15, step=1, key="tews_gcs")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.button(
                    "🔄 Reset",
                    use_container_width=True,
                    key="tews_reset",
                    on_click=reset_tews_inputs,
                )
            with col_btn2:
                copy_to_tab1 = st.button("📋 Salin ke Tab Input", type="primary", use_container_width=True, key="tews_copy")

            if copy_to_tab1:
                st.session_state['copy_from_tews'] = {
                    'laju_pernafasan': t_rr,
                    'SpO2': t_spo2,
                    'sistole': t_sbp,
                    'denyut_jantung': t_hr,
                    'suhu_tubuh': t_temp,
                    'gcs_total': t_gcs,
                }
                st.success("✅ Data disalin! Buka Tab 'Input Pasien' dan klik 'Muat dari Kalkulator TEWS'.")

        # Reference table (accordion)
        with st.expander("📖 Lihat Tabel Referensi Scoring Lengkap", expanded=False):
            render_reference_table()

    # ══════════════════════════════════════════════════════
    # PANEL KANAN — Output Real-time
    # ══════════════════════════════════════════════════════
    with col_right:
        # Compute real-time
        subscores = compute_tews_subscores(t_rr, t_spo2, t_sbp, t_hr, t_temp, t_gcs)
        tews_total = subscores['TEWS_total']
        overrides = check_override(t_gcs, t_sbp, t_spo2, t_rr)
        sats_zone = get_sats_zone_from_tews(tews_total, overrides)

        # ── TEWS Total Display ──
        with st.container(border=True):
            st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)

            # Zone colors
            if tews_total <= 2:
                tews_color = 'var(--sats-blue-dot)'
                zone_label = 'Zona Biru'
            elif tews_total <= 4:
                tews_color = 'var(--sats-green-dot)'
                zone_label = 'Zona Hijau'
            elif tews_total <= 6:
                tews_color = 'var(--sats-yellow-dot)'
                zone_label = 'Zona Kuning'
            else:
                tews_color = 'var(--sats-red-dot)'
                zone_label = 'Zona Oranye+'

            if overrides:
                tews_color = 'var(--sats-red-dot)'
                zone_label = 'OVERRIDE → Merah'

            sats_c = SATS_COLORS[sats_zone]
            st.markdown(f'<div style="padding:0.5rem 0;"><div style="font-size:0.75rem;color:var(--text-tertiary);font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">TEWS Total</div><div class="tews-score-big" style="color:{tews_color};">{tews_total}</div></div>', unsafe_allow_html=True)
            st.markdown(triage_badge_html(sats_zone), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Override Warning ──
        if overrides:
            override_items = ''.join([f'<div class="override-alert-item">• {cond}</div>' for cond in overrides])
            st.markdown(f"""
            <div class="override-alert">
                <div class="override-alert-title">🚨 OVERRIDE MERAH TERDETEKSI</div>
                <div class="override-alert-item" style="margin-bottom:0.25rem;">Kondisi berikut memerlukan resusitasi segera:</div>
                {override_items}
            </div>""", unsafe_allow_html=True)

        # ── Sub-score Breakdown ──
        with st.container(border=True):
            st.markdown('<div class="section-title">📊 Breakdown Sub-skor</div>', unsafe_allow_html=True)
            render_tews_breakdown(subscores, t_rr, t_spo2, t_sbp, t_hr, t_temp, t_gcs)

        # ── Rekomendasi ──
        with st.container(border=True):
            st.markdown('<div class="section-title">💊 Rekomendasi Tindakan</div>', unsafe_allow_html=True)
            rekomendasi = TEWS_REKOMENDASI[sats_zone]
            rek_color = sats_c['text']
            rek_bg = sats_c['bg']
            st.markdown(f'<div style="background:{rek_bg};border:1.5px solid {sats_c["border"]};border-radius:8px;padding:0.75rem 1rem;"><div style="font-size:0.85rem;color:{rek_color};font-weight:600;">{rekomendasi}</div></div>', unsafe_allow_html=True)
