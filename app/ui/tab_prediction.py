"""
ui/tab_prediction.py
Tab 1: Input Pasien & Hasil Prediksi
Redesigned — matching triage-redesign.html + ID Petugas feature
"""

import streamlit as st
import pandas as pd
import datetime
from backend.feature_engineering import SATS_COLORS
from ui.components import LABEL_TO_CLASS, TRIAGE_ORDER, triage_result_hero_html


# ─── Vital sign thresholds for real-time color flags ───────────────────────
VITAL_THRESHOLDS = {
    'sistole':          {'warn_low': 90, 'warn_high': 180, 'danger_low': 90, 'danger_high': 200, 'normal': '110–149 mmHg'},
    'diastole':         {'warn_low': 50, 'warn_high': 100, 'danger_low': 50, 'danger_high': 110, 'normal': '60–90 mmHg'},
    'denyut_jantung':   {'warn_low': 40, 'warn_high': 110, 'danger_low': 40, 'danger_high': 130, 'normal': '51–100 bpm'},
    'laju_pernafasan':  {'warn_low': 9,  'warn_high': 30,  'danger_low': 9,  'danger_high': 30,  'normal': '12–20 x/mnt'},
    'suhu_tubuh':       {'warn_low': 35.0,'warn_high': 38.5,'danger_low': 35.0,'danger_high': 39.5,'normal': '36.0–37.5 °C'},
    'SpO2':             {'warn_low': 95,  'warn_high': 100, 'danger_low': 90,  'danger_high': 100, 'normal': '97–100 %'},
}

def vital_status(field: str, value) -> str:
    if value is None: return 'normal'
    t = VITAL_THRESHOLDS.get(field, {})
    dl = t.get('danger_low'); dh = t.get('danger_high')
    wl = t.get('warn_low'); wh = t.get('warn_high')
    if dl and value < dl: return 'danger'
    if dh and value >= dh: return 'danger'
    if wl and value < wl: return 'warning'
    if wh and value >= wh: return 'warning'
    return 'normal'


def render_vital_hint(field: str, value):
    status = vital_status(field, value)
    hint = VITAL_THRESHOLDS.get(field, {}).get('normal', '')
    if status == 'danger':
        st.markdown(f'<div class="vital-danger-text">⚠️ Nilai kritis!</div><div class="vital-normal-hint">Normal: {hint}</div>', unsafe_allow_html=True)
    elif status == 'warning':
        st.markdown(f'<div class="vital-warning-text">⚡ Di luar rentang normal</div><div class="vital-normal-hint">Normal: {hint}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="vital-normal-hint">Normal: {hint}</div>', unsafe_allow_html=True)


def render_sats_badge(predicted_class: int):
    st.markdown(triage_result_hero_html(predicted_class), unsafe_allow_html=True)


def render_probability_bars(probabilities: dict):
    html = '<div class="prob-bar-container">'
    for cls in TRIAGE_ORDER:
        label = next((name for name, label_cls in LABEL_TO_CLASS.items() if label_cls == cls), "")
        prob = probabilities.get(label, 0)
        color = SATS_COLORS[cls]['dot']
        pct = prob * 100
        html += f'<div class="prob-bar-row"><span class="prob-bar-label">{label}</span><div class="prob-bar-track"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div><span class="prob-bar-pct">{pct:.1f}%</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_flag_chips(active_flags: dict):
    if not active_flags:
        st.markdown('<div class="flag-chip-empty">✓ Tidak ada flag klinis abnormal</div>', unsafe_allow_html=True)
        return
    html = '<div class="flags-container">'
    for key, (label, severity) in active_flags.items():
        html += f'<span class="flag-chip {severity}">{label}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_clinical_metrics(clinical_scores: dict):
    si = clinical_scores['shock_index']
    mews = clinical_scores['mews_approx']
    tews = clinical_scores['tews_total']
    cr = clinical_scores['cardiorespiratory_score']

    si_label = 'Normal' if si < 0.9 else ('Perhatian' if si <= 1.0 else 'Kritis')
    mews_label = 'Rendah' if mews <= 2 else ('Sedang' if mews <= 4 else 'Tinggi')
    tews_label = 'Biru' if tews <= 2 else ('Hijau' if tews <= 4 else ('Kuning' if tews <= 6 else 'Oranye+'))
    cr_label = 'Normal' if cr == 0 else ('Perhatian' if cr <= 2 else 'Distres berat')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-value">{si:.2f}</div>
            <div class="metric-card-label">Shock Index</div>
            <div class="metric-card-sub">{si_label}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-value">{mews}</div>
            <div class="metric-card-label">MEWS Approx</div>
            <div class="metric-card-sub">{mews_label}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-value">{tews}</div>
            <div class="metric-card-label">TEWS Total</div>
            <div class="metric-card-sub">Zona {tews_label}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-value">{cr}</div>
            <div class="metric-card-label">Cardiorespiratory</div>
            <div class="metric-card-sub">{cr_label}</div>
        </div>""", unsafe_allow_html=True)


def render_shap_panel(shap_top_features: list, predicted_class: int):
    """SHAP waterfall bar chart — redesign style with center-line."""

    max_abs = max(abs(sv) for _, sv, _ in shap_top_features) if shap_top_features else 1.0
    if max_abs == 0: max_abs = 1.0

    st.markdown("""
    <div style="font-size:10.5px;color:var(--text-faint);margin-bottom:0.75rem;display:flex;gap:1.5rem;">
        <span>🔴 Mendorong ke kelas lebih kritis</span>
        <span>🔵 Menekan ke kelas lebih aman</span>
    </div>""", unsafe_allow_html=True)

    html = ''
    for feat_name, shap_val, display_name in shap_top_features:
        is_pos = shap_val >= 0
        bar_pct = min(abs(shap_val) / max_abs * 45, 45)
        val_str = f'+{shap_val:.3f}' if is_pos else f'{shap_val:.3f}'
        val_color = 'var(--sats-red)' if is_pos else 'var(--sats-blue)'
        fill_class = 'pos' if is_pos else 'neg'

        html += f'''<div class="shap-row">
<span class="shap-label">{display_name}</span>
<div class="shap-track"><div class="shap-mid"></div><div class="shap-fill {fill_class}" style="width:{bar_pct}%"></div></div>
<span class="shap-val" style="color:{val_color}">{val_str}</span>
</div>'''

    st.markdown(html, unsafe_allow_html=True)


def render_session_history():
    history = st.session_state.get('prediction_history', [])
    if not history:
        st.markdown('<div style="font-size:12px;color:var(--text-faint);font-style:italic;text-align:center;padding:1rem;">Belum ada riwayat prediksi dalam sesi ini.</div>', unsafe_allow_html=True)
        return

    sats_class_map = {'Merah': 0, 'Oranye': 1, 'Kuning': 2, 'Hijau': 3, 'Biru': 4}

    rows_html = ''
    for i, entry in enumerate(reversed(history[-20:])):
        cls = sats_class_map.get(entry['prediksi'], 2)
        dot_color = SATS_COLORS[cls]['dot']
        id_petugas = entry.get('id_petugas', '-')
        rows_html += f"""
        <tr>
            <td>{len(history) - i}</td>
            <td>{entry['waktu']}</td>
            <td>{entry.get('nama', '-')}</td>
            <td>{entry.get('usia', '-')} th</td>
            <td><span style="display:inline-flex;align-items:center;gap:0.4rem;">
                <span class="triage-number">L{cls + 1}</span>
                <span style="width:9px;height:9px;border-radius:50%;background:{dot_color};flex-shrink:0;"></span>
                <strong>{entry['prediksi']}</strong>
            </span></td>
            <td>{entry.get('confidence', '-')}%</td>
            <td>{id_petugas}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="session-table">
        <thead><tr>
            <th>#</th><th>Waktu</th><th>Nama</th><th>Usia</th><th>Prediksi</th><th>Confidence</th><th>ID Petugas</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ─── ID Petugas input (required before export) ───
    st.markdown('<hr style="margin:0.75rem 0;border:none;border-top:1px solid var(--border);">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="border-bottom:none;margin-bottom:0.5rem;padding-bottom:0;">📥 Export Data Riwayat</div>', unsafe_allow_html=True)

    id_petugas_export = st.text_input(
        "ID Petugas *",
        placeholder="Masukkan ID petugas sebelum export (wajib)",
        key="export_id_petugas",
        max_chars=50,
    )

    if not id_petugas_export:
        st.markdown('<div class="petugas-required">⚠️ ID Petugas wajib diisi untuk melakukan export data</div>', unsafe_allow_html=True)

    # Export CSV — disabled if no ID Petugas
    df_hist = pd.DataFrame(history)
    # Add export ID petugas column to all rows
    df_hist['id_petugas_export'] = id_petugas_export if id_petugas_export else ''
    csv = df_hist.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 Export CSV",
        data=csv.encode('utf-8-sig'),
        file_name=f"riwayat_triage_{datetime.date.today()}.csv",
        mime='text/csv',
        use_container_width=True,
        disabled=not bool(id_petugas_export),
    )


def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-bar">
        <div class="disclaimer-icon">ℹ️</div>
        <div class="disclaimer-text">
            <strong>Disclaimer DSS:</strong> Hasil ini adalah rekomendasi sistem pendukung keputusan (DSS).
            Keputusan triage final adalah wewenang eksklusif tenaga medis berlisensi. Sistem ini tidak menggantikan
            penilaian klinis profesional.
        </div>
    </div>""", unsafe_allow_html=True)


def map_gcs_total_to_emv(total: int):
    # Map GCS total to plausible (E, M, V)
    if total == 15: return 4, 6, 5
    if total == 14: return 4, 5, 5
    if total == 13: return 3, 5, 5
    if total == 12: return 3, 5, 4
    if total == 11: return 3, 4, 4
    if total == 10: return 2, 4, 4
    if total == 9:  return 2, 4, 3
    if total == 8:  return 2, 3, 3
    if total == 7:  return 2, 3, 2
    if total == 6:  return 1, 3, 2
    if total == 5:  return 1, 2, 2
    if total == 4:  return 1, 2, 1
    return 1, 1, 1


def render_tab_prediction(artifacts):
    # ─── Load from TEWS handler ───
    if 'copy_from_tews' in st.session_state:
        copy_data = st.session_state['copy_from_tews']
        col_banner, col_btn = st.columns([9, 2])
        with col_banner:
            st.info(
                f"📋 **Data dari Kalkulator TEWS tersedia** — "
                f"RR={copy_data['laju_pernafasan']}, SpO₂={copy_data['SpO2']}%, "
                f"Sistole={copy_data['sistole']} mmHg, HR={copy_data['denyut_jantung']} bpm, "
                f"Suhu={copy_data['suhu_tubuh']}°C, GCS={copy_data['gcs_total']}."
            )
        with col_btn:
            st.markdown('<div style="margin-top: 1.15rem;"></div>', unsafe_allow_html=True)
            if st.button("📥 Muat Data", use_container_width=True, key="btn_load_tews"):
                st.session_state['inp_rr'] = float(copy_data['laju_pernafasan'])
                st.session_state['inp_spo2'] = int(copy_data['SpO2'])
                st.session_state['inp_sistole'] = int(copy_data['sistole'])
                st.session_state['inp_hr'] = int(copy_data['denyut_jantung'])
                st.session_state['inp_suhu'] = float(copy_data['suhu_tubuh'])
                
                ge, gm, gv = map_gcs_total_to_emv(copy_data['gcs_total'])
                st.session_state['inp_gcs_e'] = ge
                st.session_state['inp_gcs_m'] = gm
                st.session_state['inp_gcs_v'] = gv
                
                del st.session_state['copy_from_tews']
                st.rerun()

    # ─── 2-column layout ───────────────────────────────────────────────────────
    col_input, col_output = st.columns([7, 5], gap="large")

    # ══════════════════════════════════════════════════════
    # PANEL KIRI — INPUT PASIEN
    # ══════════════════════════════════════════════════════
    with col_input:

        # ── SEKSI A: Identitas & Demografis ──
        with st.container(border=True):
            st.markdown('<div class="section-title">👤 Identitas & Demografis</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                nama = st.text_input("Nama Pasien", placeholder="Opsional (max 100 karakter)",
                                      max_chars=100, key="inp_nama")
            with c2:
                usia = st.number_input("Usia (tahun) *", min_value=0, max_value=120, value=None,
                                        placeholder="0–120", key="inp_usia")

            c3, c4 = st.columns(2)
            with c3:
                jenis_kelamin = st.radio("Jenis Kelamin *", ["Laki-laki", "Perempuan"],
                                          horizontal=True, key="inp_jk")
            with c4:
                cara_datang = st.selectbox("Cara Datang *",
                    ["Sendiri", "KLL (Kecelakaan)", "Rujukan Puskesmas", "Rujukan Dokter"],
                    key="inp_cara")

        # ── SEKSI B: Tanda Vital ──
        with st.container(border=True):
            st.markdown('<div class="section-title">❤️ Tanda Vital</div>', unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                sistole = st.number_input("Sistole (mmHg) *", min_value=40, max_value=300,
                                           value=None, placeholder="110–149", key="inp_sistole")
                render_vital_hint('sistole', sistole)
            with b2:
                diastole = st.number_input("Diastole (mmHg) *", min_value=20, max_value=200,
                                            value=None, placeholder="60–90", key="inp_diastole")
                render_vital_hint('diastole', diastole)

            b3, b4, b5 = st.columns(3)
            with b3:
                hr = st.number_input("Denyut Jantung (bpm) *", min_value=10, max_value=300,
                                      value=None, placeholder="51–100", key="inp_hr")
                render_vital_hint('denyut_jantung', hr)
            with b4:
                rr = st.number_input("Laju Pernafasan (x/mnt) *", min_value=0, max_value=80,
                                      value=None, placeholder="12–20", key="inp_rr")
                render_vital_hint('laju_pernafasan', rr)
            with b5:
                suhu = st.number_input("Suhu Tubuh (°C) *", min_value=30.0, max_value=45.0,
                                        step=0.1, value=None, placeholder="36.0–37.5", key="inp_suhu")
                render_vital_hint('suhu_tubuh', suhu)

            spo2 = st.number_input("SpO₂ (%) *", min_value=0, max_value=100,
                                     value=None, placeholder="97–100", key="inp_spo2")
            render_vital_hint('SpO2', spo2)

        # ── SEKSI C: Neurologis (GCS) ──
        with st.container(border=True):
            st.markdown('<div class="section-title">🧠 Neurologis <span style="color:var(--text-faint);font-weight:500;font-size:11.5px;margin-left:6px;">Glasgow Coma Scale</span></div>', unsafe_allow_html=True)

            gcs_e_opts = {4: '4 — Spontan', 3: '3 — Respons suara', 2: '2 — Respons nyeri', 1: '1 — Tidak ada'}
            gcs_m_opts = {6: '6 — Ikuti perintah', 5: '5 — Lokasi nyeri', 4: '4 — Fleksi normal',
                          3: '3 — Fleksi abnormal', 2: '2 — Ekstensi', 1: '1 — Tidak ada'}
            gcs_v_opts = {5: '5 — Orientasi baik', 4: '4 — Bingung', 3: '3 — Hanya kata-kata',
                          2: '2 — Hanya suara', 1: '1 — Tidak ada'}

            gc1, gc2, gc3, gc4 = st.columns([3, 3, 3, 2])
            with gc1:
                gcs_e = st.selectbox("Eye (E)", options=list(gcs_e_opts.keys()),
                                      format_func=lambda x: gcs_e_opts[x], key="inp_gcs_e")
            with gc2:
                gcs_m = st.selectbox("Motor (M)", options=list(gcs_m_opts.keys()),
                                      format_func=lambda x: gcs_m_opts[x], key="inp_gcs_m")
            with gc3:
                gcs_v = st.selectbox("Verbal (V)", options=list(gcs_v_opts.keys()),
                                      format_func=lambda x: gcs_v_opts[x], key="inp_gcs_v")
            with gc4:
                gcs_total = gcs_e + gcs_m + gcs_v
                if gcs_total <= 8:
                    gcs_interp = 'Koma berat'
                    gcs_color = 'var(--sats-red)'
                elif gcs_total <= 12:
                    gcs_interp = 'Sedang'
                    gcs_color = 'var(--sats-yellow)'
                elif gcs_total <= 14:
                    gcs_interp = 'Ringan'
                    gcs_color = 'var(--sats-orange)'
                else:
                    gcs_interp = 'Normal'
                    gcs_color = 'var(--sats-green)'

                st.markdown(f'<div class="gcs-display" style="margin-top:1.5rem;"><div class="gcs-total-value">{gcs_total}</div><div class="gcs-total-label">GCS Total</div><div class="gcs-interpretation" style="color:{gcs_color};">{gcs_interp}</div></div>', unsafe_allow_html=True)

        # FEAT-007: skala_nyeri dihapus dari UI sepenuhnya

        # ── SEKSI D: Tombol Aksi ──
        required_fields = [usia, sistole, diastole, hr, rr, suhu, spo2]
        all_filled = all(v is not None for v in required_fields)

        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            predict_btn = st.button(
                "🔍 Prediksi Triage",
                type="primary",
                use_container_width=True,
                disabled=not all_filled,
                key="btn_predict"
            )
        with btn_col2:
            reset_btn = st.button("🔄 Reset", use_container_width=True, key="btn_reset")

        if not all_filled:
            st.caption("⚠️ Lengkapi semua field wajib (*) untuk mengaktifkan prediksi")

        if reset_btn:
            # Clear relevant session state keys
            for k in ['prediction_result', 'last_input']:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # ══════════════════════════════════════════════════════
    # PANEL KANAN — HASIL PREDIKSI
    # ══════════════════════════════════════════════════════
    with col_output:

        # ── Handle prediction ──
        if predict_btn and all_filled:
            cara_clean = {'Sendiri': 'Sendiri', 'KLL (Kecelakaan)': 'KLL',
                          'Rujukan Puskesmas': 'Puskesmas', 'Rujukan Dokter': 'Dokter'}.get(cara_datang, 'Sendiri')

            input_data = {
                'usia_tahun': int(usia),
                'jenis_kelamin': jenis_kelamin,
                'cara_datang': cara_clean,
                'GCS_E': gcs_e, 'GCS_M': gcs_m, 'GCS_V': gcs_v,
                # skala_nyeri dihapus — FEAT-007
                'sistole': sistole, 'diastole': diastole,
                'denyut_jantung': hr, 'laju_pernafasan': rr,
                'suhu_tubuh': suhu, 'SpO2': spo2,
            }

            from backend.predictor import predict_triage_v3
            with st.spinner("Memproses prediksi..."):
                try:
                    result = predict_triage_v3(input_data, artifacts)
                    st.session_state['prediction_result'] = result
                    st.session_state['last_input'] = input_data

                    # Add to history (include id_petugas from session)
                    if 'prediction_history' not in st.session_state:
                        st.session_state['prediction_history'] = []
                    confidence = round(result['probabilities'].get(result['predicted_label'], 0) * 100, 1)

                    # Get current ID petugas from session state
                    current_id_petugas = st.session_state.get('inp_id_petugas', '')

                    st.session_state['prediction_history'].append({
                        'waktu': datetime.datetime.now().strftime('%H:%M:%S'),
                        'nama': nama or '-',
                        'usia': usia,
                        'prediksi': result['predicted_label'],
                        'confidence': confidence,
                        'tews': result['clinical_scores']['tews_total'],
                        'gcs': result['gcs_total'],
                        'id_petugas': current_id_petugas or '-',
                    })
                except Exception as e:
                    st.error(f"❌ Error prediksi: {e}")

        result = st.session_state.get('prediction_result')

        if result is None:
            st.markdown("""
            <div class="placeholder-panel">
                <div class="placeholder-icon">🏥</div>
                <div class="placeholder-text">Isi data pasien dan tekan Prediksi</div>
                <div class="placeholder-sub">Hasil klasifikasi SATS dan penjelasan SHAP akan muncul di sini</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── A: SATS Badge (solid color banner) ──
            render_sats_badge(result['predicted_class'])

            # ── B: Probabilitas ──
            with st.container(border=True):
                st.markdown('<div class="section-title" style="border-bottom:none;padding-bottom:0;margin-bottom:0.5rem;">Distribusi Probabilitas</div>', unsafe_allow_html=True)
                render_probability_bars(result['probabilities'])

            # ── C: Clinical Flags ──
            with st.container(border=True):
                st.markdown('<div class="section-title" style="border-bottom:none;padding-bottom:0;margin-bottom:0.5rem;">🚨 Clinical Flags Aktif</div>', unsafe_allow_html=True)
                render_flag_chips(result['active_flags'])

            # ── D: SHAP Panel ──
            with st.container(border=True):
                st.markdown('<div class="section-title" style="border-bottom:none;padding-bottom:0;margin-bottom:0.5rem;">🔍 Alasan Prediksi — Top Fitur (SHAP)</div>', unsafe_allow_html=True)
                render_shap_panel(result['shap_top_features'], result['predicted_class'])

            # ── E: Clinical Metrics ──
            with st.expander("📊 Skor Klinis Turunan", expanded=False):
                render_clinical_metrics(result['clinical_scores'])

            # ── F: Session History ──
            with st.expander("📋 Riwayat Sesi", expanded=False):
                render_session_history()

            # ── G: Disclaimer (permanent) ──
            render_disclaimer()
