"""
ui/tab_guidelines.py
Tab 3: Panduan SATS — referensi klinis statis
Redesigned — matching triage-redesign.html level-grid layout
"""

import streamlit as st
from ui.components import score_badge_html, triage_badge_html


SATS_GUIDE_DATA = [
    {
        'level': 0, 'color_name': 'Merah', 'en_name': 'Resuscitation',
        'time': 'Segera — 0 mnt', 'slug': 'merah',
        'definition': 'Pasien dalam kondisi mengancam jiwa. Tanda vital sangat tidak stabil atau tidak ada.',
        'examples': 'Cardiac arrest, henti napas, GCS ≤ 8, syok berat (TD < 70), obstruksi jalan napas total.',
    },
    {
        'level': 1, 'color_name': 'Oranye', 'en_name': 'Emergent',
        'time': '< 10 menit', 'slug': 'oranye',
        'definition': 'Kondisi berpotensi mengancam jiwa jika tidak segera ditangani. Tanda vital tidak stabil.',
        'examples': 'Nyeri dada akut, distres napas berat, SpO₂ < 90%, penurunan kesadaran tiba-tiba, trauma mayor.',
    },
    {
        'level': 2, 'color_name': 'Kuning', 'en_name': 'Urgent',
        'time': '< 60 menit', 'slug': 'kuning',
        'definition': 'Kondisi yang memerlukan perhatian segera namun tidak langsung mengancam jiwa.',
        'examples': 'Fraktur tertutup, nyeri sedang-berat, demam tinggi, muntah persisten, perdarahan terkontrol.',
    },
    {
        'level': 3, 'color_name': 'Hijau', 'en_name': 'Less Urgent',
        'time': '< 4 jam', 'slug': 'hijau',
        'definition': 'Kondisi tidak mengancam jiwa. Pasien stabil namun memerlukan perawatan medis.',
        'examples': 'Luka ringan, nyeri ringan, infeksi superfisial, keluhan kronis yang tidak eksaserbasi.',
    },
    {
        'level': 4, 'color_name': 'Biru', 'en_name': 'Not Urgent',
        'time': '< 6 jam', 'slug': 'biru',
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
    """Render 5 SATS level cards in a grid — matching triage-redesign.html level-grid."""
    slugs_css = ['red', 'orange', 'yellow', 'green', 'blue']
    cols = st.columns(5)
    for col, data in zip(cols, SATS_GUIDE_DATA):
        with col:
            slug = data['slug']
            css_slug = slugs_css[data['level']]
            level_num = data['level'] + 1

            card_html = f"""
            <div class="sats-guide-card triage-{css_slug}">
                <div class="level-head">
                    <div class="level-badge {slug}">L{level_num}</div>
                    <div>
                        <div class="level-title">{data['color_name']}</div>
                        <div class="level-time {slug}">{data['time']}</div>
                    </div>
                </div>
                <div class="sats-definition">{data['definition']}</div>
                <div class="sats-examples">{data['examples']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


def render_tews_table():
    """TEWS scoring table — HTML-based for consistent styling."""
    rows_html = []
    for param_data in TEWS_SCORING_TABLE:
        for j, (range_val, score, keterangan) in enumerate(param_data['ranges']):
            param_cell = f'<td rowspan="{len(param_data["ranges"])}" style="font-weight:600;vertical-align:top;">{param_data["param"]}</td>' if j == 0 else ''
            rows_html.append(
                '<tr>'
                f'{param_cell}'
                f'<td>{range_val}</td>'
                f'<td style="text-align:center;">{score_badge_html(score)}</td>'
                f'<td style="font-size:11px;color:var(--text-soft);">{keterangan}</td>'
                '</tr>'
            )

    # Keep the table as one continuous, left-aligned HTML block. Indented table
    # rows can be parsed by Streamlit Markdown as a code block instead of HTML.
    table_html = (
        '<table class="session-table">'
        '<thead><tr>'
        '<th>Parameter</th>'
        '<th>Rentang</th>'
        '<th style="text-align:center;">Skor</th>'
        '<th>Keterangan Klinis</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style="margin-top:0.5rem;padding:0.5rem 0.75rem;background:var(--bg);border-radius:6px;font-size:10.5px;color:var(--text-soft);">
        <strong>Legenda Skor:</strong> &nbsp;
        <span style="color:var(--sats-green);">●</span> 0 = Normal &nbsp;|&nbsp;
        <span style="color:var(--sats-yellow);">●</span> 1 = Perhatian &nbsp;|&nbsp;
        <span style="color:var(--sats-orange);">●</span> 2 = Waspada &nbsp;|&nbsp;
        <span style="color:var(--sats-red);">●</span> 3 = Kritis &nbsp;&nbsp;
        <strong>TEWS Total</strong> = Jumlah semua sub-skor (range 0–18)
    </div>""", unsafe_allow_html=True)


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
                <div class="override-alert-item" style="font-size:10.5px; font-style:italic; border-top:1px solid currentColor; padding-top:0.4rem; margin:0;">
                    ⚡ {ovr['action']}
                </div>
            </div>""", unsafe_allow_html=True)


def render_flowchart():
    st.markdown("""
    <div class="decision-flow" aria-label="Alur keputusan triage SATS TEWS">
        <div class="decision-node">
            <span class="decision-step-num">1</span>
            <div>
                <div class="decision-title">Pasien tiba</div>
                <div class="decision-text">Ukur tanda vital lengkap, SpO2, dan GCS.</div>
            </div>
        </div>
        <div class="decision-connector">&rarr;</div>
        <div class="decision-branch">
            <div class="decision-node">
                <span class="decision-step-num">2</span>
                <div>
                    <div class="decision-title">Cek override merah</div>
                    <div class="decision-text">GCS <= 8, sistole < 70, atau hipoksia berat + RR abnormal.</div>
                </div>
            </div>
            <div class="decision-path triage-red">
                <strong>Ya: langsung Level 1 Merah</strong>
                Resusitasi segera tanpa menunggu total TEWS.
            </div>
            <div class="decision-path">
                <strong>Tidak: lanjut jalur normal</strong>
                Hitung TEWS total dari 6 parameter.
            </div>
        </div>
        <div class="decision-connector">&rarr;</div>
        <div class="decision-outcome">
            <div class="decision-node" style="margin-bottom:0.55rem;">
                <span class="decision-step-num">3</span>
                <div>
                    <div class="decision-title">Tentukan level dari TEWS</div>
                    <div class="decision-text">Urutan dibaca dari paling kritis ke paling rendah.</div>
                </div>
            </div>
            <div class="decision-path-list">
                <div class="decision-path triage-orange"><strong>TEWS >= 7</strong>Level 2 Oranye, < 10 menit</div>
                <div class="decision-path triage-yellow"><strong>TEWS 5-6</strong>Level 3 Kuning, < 60 menit</div>
                <div class="decision-path triage-green"><strong>TEWS 3-4</strong>Level 4 Hijau, < 4 jam</div>
                <div class="decision-path triage-blue"><strong>TEWS 0-2</strong>Level 5 Biru, < 6 jam</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
    with st.container(border=True):
        st.markdown('<div class="section-title">🏥 5 Level Triage SATS</div>', unsafe_allow_html=True)
        render_sats_cards()

    # ── Seksi B: Tabel TEWS ──
    with st.container(border=True):
        st.markdown('<div class="section-title">📊 Tabel Scoring TEWS Lengkap</div>', unsafe_allow_html=True)
        render_tews_table()

    # ── Seksi C: Override Criteria ──
    with st.container(border=True):
        st.markdown('<div class="section-title">🚨 Kondisi Override — Langsung MERAH</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px;color:var(--text-faint);margin-bottom:0.75rem;">Jika salah satu kondisi berikut terpenuhi, pasien langsung dikategorikan MERAH (Resusitasi) terlepas dari skor TEWS total.</p>', unsafe_allow_html=True)
        render_override_cards()

    # ── Seksi D: Flowchart ──
    with st.container(border=True):
        st.markdown('<div class="section-title">🔄 Alur Keputusan Triage</div>', unsafe_allow_html=True)
        render_flowchart()

    # ── Seksi E: Referensi ──
    with st.container(border=True):
        st.markdown('<div class="section-title">📚 Referensi Ilmiah</div>', unsafe_allow_html=True)
        render_references()
