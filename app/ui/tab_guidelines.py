"""
ui/tab_guidelines.py
Tab 3: Panduan SATS — referensi klinis statis
"""

import streamlit as st
from backend.feature_engineering import SATS_COLORS


SATS_GUIDE_DATA = [
    {
        'level': 0, 'color_name': 'Merah', 'en_name': 'Resuscitation',
        'time': 'Segera — 0 mnt',
        'definition': 'Pasien dalam kondisi mengancam jiwa. Tanda vital sangat tidak stabil atau tidak ada.',
        'examples': 'Cardiac arrest, henti napas, GCS ≤ 8, syok berat (TD < 70), obstruksi jalan napas total.',
    },
    {
        'level': 1, 'color_name': 'Oranye', 'en_name': 'Emergent',
        'time': '< 10 menit',
        'definition': 'Kondisi berpotensi mengancam jiwa jika tidak segera ditangani. Tanda vital tidak stabil.',
        'examples': 'Nyeri dada akut, distres napas berat, SpO₂ < 90%, penurunan kesadaran tiba-tiba, trauma mayor.',
    },
    {
        'level': 2, 'color_name': 'Kuning', 'en_name': 'Urgent',
        'time': '< 60 menit',
        'definition': 'Kondisi yang memerlukan perhatian segera namun tidak langsung mengancam jiwa.',
        'examples': 'Fraktur tertutup, nyeri sedang-berat, demam tinggi, muntah persisten, perdarahan terkontrol.',
    },
    {
        'level': 3, 'color_name': 'Hijau', 'en_name': 'Less Urgent',
        'time': '< 4 jam',
        'definition': 'Kondisi tidak mengancam jiwa. Pasien stabil namun memerlukan perawatan medis.',
        'examples': 'Luka ringan, nyeri ringan, infeksi superfisial, keluhan kronis yang tidak eksaserbasi.',
    },
    {
        'level': 4, 'color_name': 'Biru', 'en_name': 'Not Urgent',
        'time': '< 6 jam',
        'definition': 'Kondisi tidak memerlukan perawatan darurat. Dapat ditangani di fasilitas primer.',
        'examples': 'Keluhan ringan, pemeriksaan rutin, kontrol post-rawat, kondisi stabil kronik.',
    },
]

TEWS_SCORING_TABLE = [
    {
        'param': 'Laju Pernafasan (x/mnt)',
        'ranges': [
            ('< 9', 3, 'Sangat rendah/tinggi — hipoventilasi berat atau takipnea berat'),
            ('9–11', 1, 'Pernafasan lambat — perlu monitoring'),
            ('12–20', 0, 'Normal'),
            ('21–29', 2, 'Takipnea ringan-sedang'),
            ('≥ 30', 3, 'Takipnea berat — risiko kegagalan napas'),
        ]
    },
    {
        'param': 'SpO₂ (%)',
        'ranges': [
            ('< 90', 3, 'Hipoksia berat — ancaman jiwa'),
            ('90–94', 2, 'Hipoksia sedang'),
            ('95–96', 1, 'Di batas bawah normal'),
            ('≥ 97', 0, 'Normal'),
        ]
    },
    {
        'param': 'Tekanan Darah Sistole (mmHg)',
        'ranges': [
            ('< 70', 3, 'Syok berat — perfusi organ tidak adekuat'),
            ('70–89', 2, 'Hipotensi — risiko syok'),
            ('90–109', 1, 'Rendah-normal'),
            ('110–149', 0, 'Normal'),
            ('150–179', 1, 'Hipertensi ringan'),
            ('≥ 180', 2, 'Hipertensi berat'),
        ]
    },
    {
        'param': 'Denyut Jantung (bpm)',
        'ranges': [
            ('< 40', 3, 'Bradikardia berat — risiko sinkop/syok'),
            ('40–50', 1, 'Bradikardia'),
            ('51–100', 0, 'Normal'),
            ('101–110', 1, 'Takikardia ringan'),
            ('111–129', 2, 'Takikardia sedang'),
            ('≥ 130', 3, 'Takikardia berat'),
        ]
    },
    {
        'param': 'Suhu Tubuh (°C)',
        'ranges': [
            ('< 35.0', 2, 'Hipotermi — risiko aritmia dan koagulopati'),
            ('35.0–38.4', 0, 'Normal'),
            ('≥ 38.5', 1, 'Demam — infeksi/inflamasi'),
        ]
    },
    {
        'param': 'GCS Total',
        'ranges': [
            ('≤ 8', 3, 'Koma berat — airway tidak terlindungi'),
            ('9–12', 2, 'Penurunan kesadaran sedang'),
            ('13–14', 1, 'Penurunan kesadaran ringan'),
            ('15', 0, 'Sadar penuh — normal'),
        ]
    },
]


def render_sats_cards():
    cols = st.columns(5)
    for i, (col, data) in enumerate(zip(cols, SATS_GUIDE_DATA)):
        c = SATS_COLORS[data['level']]
        with col:
            st.markdown(f"""
            <div class="sats-guide-card" style="background:{c['bg']};border-color:{c['border']};">
                <div style="display:flex;align-items:center;margin-bottom:0.35rem;">
                    <span class="sats-level-dot" style="background:{c['dot']};"></span>
                    <span class="sats-level-name" style="color:{c['text']};">{data['color_name']}</span>
                </div>
                <div style="font-size:0.72rem;color:{c['text']};opacity:0.75;font-style:italic;">{data['en_name']}</div>
                <div class="sats-time-badge" style="color:{c['text']};">⏱ {data['time']}</div>
                <div class="sats-definition" style="color:{c['text']};">{data['definition']}</div>
                <div class="sats-examples" style="color:{c['text']};">{data['examples']}</div>
            </div>""", unsafe_allow_html=True)


def render_tews_table():
    SCORE_COLORS = {
        0: 'var(--sats-green-dot)',
        1: 'var(--sats-yellow-dot)',
        2: 'var(--sats-orange-dot)',
        3: 'var(--sats-red-dot)',
    }
    SCORE_BG = {
        0: 'var(--sats-green-bg)',
        1: 'var(--sats-yellow-bg)',
        2: 'var(--sats-orange-bg)',
        3: 'var(--sats-red-bg)',
    }

    # Table header
    hdr = st.columns([3, 2, 1, 5])
    with hdr[0]:
        st.markdown('<div style="font-weight:700;font-size:0.82rem;color:var(--table-header-color);padding:0.4rem 0;border-bottom:2px solid var(--table-border);">Parameter</div>', unsafe_allow_html=True)
    with hdr[1]:
        st.markdown('<div style="font-weight:700;font-size:0.82rem;color:var(--table-header-color);text-align:center;padding:0.4rem 0;border-bottom:2px solid var(--table-border);">Rentang Nilai</div>', unsafe_allow_html=True)
    with hdr[2]:
        st.markdown('<div style="font-weight:700;font-size:0.82rem;color:var(--table-header-color);text-align:center;padding:0.4rem 0;border-bottom:2px solid var(--table-border);">Skor</div>', unsafe_allow_html=True)
    with hdr[3]:
        st.markdown('<div style="font-weight:700;font-size:0.82rem;color:var(--table-header-color);padding:0.4rem 0;border-bottom:2px solid var(--table-border);">Keterangan Klinis</div>', unsafe_allow_html=True)

    # Table rows
    for i, param_data in enumerate(TEWS_SCORING_TABLE):
        for j, (range_val, score, keterangan) in enumerate(param_data['ranges']):
            sc = SCORE_COLORS[min(score, 3)]
            sb = SCORE_BG[min(score, 3)]
            row = st.columns([3, 2, 1, 5])
            with row[0]:
                if j == 0:
                    st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:var(--text-primary);padding:0.3rem 0;">{param_data["param"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="padding:0.3rem 0;">&nbsp;</div>', unsafe_allow_html=True)
            with row[1]:
                st.markdown(f'<div style="font-size:0.8rem;font-weight:600;text-align:center;color:var(--text-primary);padding:0.3rem 0;">{range_val}</div>', unsafe_allow_html=True)
            with row[2]:
                st.markdown(f'<div style="text-align:center;padding:0.3rem 0;"><span style="display:inline-block;padding:0.1rem 0.5rem;border-radius:999px;background:{sb};color:{sc};font-weight:700;font-size:0.8rem;border:1px solid {sc}40;">{score}</span></div>', unsafe_allow_html=True)
            with row[3]:
                st.markdown(f'<div style="font-size:0.78rem;color:var(--text-secondary);padding:0.3rem 0;">{keterangan}</div>', unsafe_allow_html=True)
        # Separator between parameter groups
        st.markdown('<hr style="margin:0.2rem 0;border:none;border-top:1px solid var(--table-row-border);">', unsafe_allow_html=True)

    # Legend
    st.markdown("""<div style="margin-top:0.5rem;padding:0.5rem 0.75rem;background:var(--legend-bg);border-radius:6px;font-size:0.73rem;color:var(--text-secondary);"><strong>Legenda Skor:</strong> &nbsp;<span style="color:var(--sats-green-dot);">●</span> 0 = Normal &nbsp;|&nbsp;<span style="color:var(--sats-yellow-dot);">●</span> 1 = Perhatian &nbsp;|&nbsp;<span style="color:var(--sats-orange-dot);">●</span> 2 = Waspada &nbsp;|&nbsp;<span style="color:var(--sats-red-dot);">●</span> 3 = Kritis &nbsp;&nbsp;<strong>TEWS Total</strong> = Jumlah semua sub-skor (range 0–18)</div>""", unsafe_allow_html=True)


def render_override_cards():
    overrides = [
        {
            'icon': '🧠',
            'title': 'GCS Total ≤ 8',
            'desc': 'Penurunan kesadaran berat. Airway tidak terlindungi, risiko aspirasi tinggi.',
            'action': 'Airway management segera. Pertimbangkan intubasi.',
        },
        {
            'icon': '💉',
            'title': 'Sistole < 70 mmHg',
            'desc': 'Perfusi organ tidak adekuat. Syok berat dengan risiko kegagalan multi-organ.',
            'action': 'Resusitasi cairan agresif, vasopressor, akses IV double-lumen.',
        },
        {
            'icon': '🫁',
            'title': 'SpO₂ < 90% + (RR < 9 ATAU RR ≥ 30)',
            'desc': 'Kombinasi hipoksia dengan gangguan frekuensi napas — kegagalan ventilasi.',
            'action': 'Oksigen high-flow, pertimbangkan ventilasi mekanis segera.',
        },
    ]
    cols = st.columns(3)
    for col, ovr in zip(cols, overrides):
        with col:
            st.markdown(f"""
            <div class="override-alert" style="height:100%; display:flex; flex-direction:column; margin:0;">
                <div style="font-size:1.5rem;margin-bottom:0.4rem;">{ovr['icon']}</div>
                <div class="override-alert-title" style="margin-bottom:0.4rem;">{ovr['title']}</div>
                <div class="override-alert-item" style="flex:1; margin:0 0 0.5rem 0; line-height:1.5;">{ovr['desc']}</div>
                <div class="override-alert-item" style="font-size:0.72rem; font-style:italic; border-top:1px solid currentColor; padding-top:0.4rem; margin:0;">
                    ⚡ {ovr['action']}
                </div>
            </div>""", unsafe_allow_html=True)


def render_flowchart():
    steps = [
        {'num': 'Langkah 1', 'label': 'Pasien tiba → Ukur Tanda Vital & GCS'},
        {'num': 'Langkah 2', 'label': 'Cek Override Criteria (3 kondisi)'},
        {'num': 'Langkah 3', 'label': 'Hitung TEWS Total'},
        {'num': 'Langkah 4', 'label': 'Verifikasi Dokter/Perawat → Tindakan'},
    ]

    outcomes = {
        1: [('Override terpenuhi', '#E24B4A', '#FCEBEB', '#F09595', 'MERAH — Resusitasi Segera')],
        2: [
            ('TEWS ≥ 7', '#BA7517', '#FAEEDA', '#FAC775', 'ORANYE — < 10 mnt'),
            ('TEWS 5–6', '#EF9F27', '#FAEEDA', '#EF9F27', 'KUNING — < 60 mnt'),
            ('TEWS 3–4', '#639922', '#EAF3DE', '#C0DD97', 'HIJAU — < 4 jam'),
            ('TEWS 0–2', '#378ADD', '#E6F1FB', '#85B7EB', 'BIRU — < 6 jam'),
        ],
    }

    st.markdown('<div style="display:flex;align-items:stretch;gap:0.5rem;overflow-x:auto;padding:0.5rem 0;">', unsafe_allow_html=True)

    for i, step in enumerate(steps):
        col_html = f'''
        <div style="min-width:160px;flex:1;">
            <div class="flowchart-step" style="height:70px;display:flex;flex-direction:column;justify-content:center;">
                <div class="step-num">{step["num"]}</div>
                <div class="step-label">{step["label"]}</div>
            </div>'''

        # Outcomes for step 2 (index 1) and step 3 (index 2)
        if i == 1 and 1 in outcomes:
            for cond, tc, bg, bc, label in outcomes[1]:
                col_html += f'''
                <div style="margin-top:0.25rem;background:{bg};border:1px solid {bc};border-radius:8px;padding:0.35rem 0.5rem;font-size:0.7rem;">
                    <div style="color:var(--text-tertiary);font-size:0.65rem;">↳ {cond}</div>
                    <div style="color:{tc};font-weight:700;">{label}</div>
                </div>'''

        if i == 2 and 2 in outcomes:
            for cond, tc, bg, bc, label in outcomes[2]:
                col_html += f'''
                <div style="margin-top:0.25rem;background:{bg};border:1px solid {bc};border-radius:8px;padding:0.35rem 0.5rem;font-size:0.7rem;">
                    <div style="color:var(--text-tertiary);font-size:0.65rem;">↳ {cond}</div>
                    <div style="color:{tc};font-weight:700;">{label}</div>
                </div>'''

        col_html += '</div>'
        if i < len(steps) - 1:
            col_html += '<div style="display:flex;align-items:flex-start;padding-top:22px;color:var(--text-tertiary);font-size:1.1rem;">→</div>'

        st.markdown(col_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_references():
    refs = [
        ('SATS', 'South African Triage Scale (SATS), 2nd Edition, 2012'),
        ('TEWS', 'Twomey M. et al. — Emerg Med J, 2007'),
        ('MEWS', 'Subbe CP. et al. — QJM, 2001'),
        ('Shock Index', 'Birkhahn RH. et al. — Am J Emerg Med, 2005'),
    ]
    cols = st.columns(4)
    for col, (title, text) in zip(cols, refs):
        with col:
            st.markdown(f"""
            <div class="ref-pill">
                <div class="ref-pill-title">{title}</div>
                <div class="ref-pill-text">{text}</div>
            </div>""", unsafe_allow_html=True)


def render_tab_guidelines():
    # ── Seksi A: 5 Level SATS ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏥 5 Level Triage SATS</div>', unsafe_allow_html=True)
    render_sats_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Seksi B: Tabel TEWS ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Tabel Scoring TEWS Lengkap</div>', unsafe_allow_html=True)
    render_tews_table()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Seksi C: Override Criteria ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚨 Kondisi Override — Langsung MERAH</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:var(--text-tertiary);margin-bottom:0.75rem;">Jika salah satu kondisi berikut terpenuhi, pasien langsung dikategorikan MERAH (Resusitasi) terlepas dari skor TEWS total.</p>', unsafe_allow_html=True)
    render_override_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Seksi D: Flowchart ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔄 Alur Keputusan Triage</div>', unsafe_allow_html=True)
    render_flowchart()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Seksi E: Referensi ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📚 Referensi Ilmiah</div>', unsafe_allow_html=True)
    render_references()
    st.markdown('</div>', unsafe_allow_html=True)
