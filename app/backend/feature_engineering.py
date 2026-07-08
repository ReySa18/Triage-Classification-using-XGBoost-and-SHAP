"""
backend/feature_engineering.py
Feature engineering pipeline — sesuai Step 8 notebook V5
FEAT-007: skala_nyeri dihapus dari FEATURE_COLS dan build_feature_vector()
"""

import numpy as np
import pandas as pd
from .tews_calculator import compute_tews_subscores


FEATURE_COLS = [
    'usia_tahun', 'kelompok_usia', 'jenis_kelamin_enc',
    'GCS_E', 'GCS_M', 'GCS_V', 'GCS_total',
    # skala_nyeri dihapus — FEAT-007: zero information gain terhadap label SATS-TEWS
    'sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2',
    'MAP',
    'flag_takikardia', 'flag_bradikardia', 'flag_hipotensi', 'flag_hipertensi',
    'flag_takipnea', 'flag_hipoksia', 'flag_demam', 'flag_hipotermi',
    'n_vital_abnormal',
    'shock_index', 'cardiorespiratory_score', 'pulse_pressure', 'mews_approx',
    'flag_hypoxic_shock',
    'cara_datang_KLL', 'cara_datang_Puskesmas', 'cara_datang_Sendiri',
]

SATS_LABELS = {
    0: 'Merah',
    1: 'Oranye',
    2: 'Kuning',
    3: 'Hijau',
    4: 'Biru',
}

SATS_FULL_LABELS = {
    0: 'Merah — Resusitasi',
    1: 'Oranye — Emergent',
    2: 'Kuning — Urgent',
    3: 'Hijau — Less Urgent',
    4: 'Biru — Not Urgent',
}

SATS_WAKTU = {
    0: 'Penanganan segera — 0 menit',
    1: '< 10 menit',
    2: '< 60 menit',
    3: '< 4 jam',
    4: '< 6 jam',
}

SATS_COLORS = {
    0: {'bg': 'var(--sats-red-bg)', 'border': 'var(--sats-red-border)', 'text': 'var(--sats-red-text)', 'dot': 'var(--sats-red-dot)', 'hex': 'var(--sats-red-dot)'},
    1: {'bg': 'var(--sats-orange-bg)', 'border': 'var(--sats-orange-border)', 'text': 'var(--sats-orange-text)', 'dot': 'var(--sats-orange-dot)', 'hex': 'var(--sats-orange-dot)'},
    2: {'bg': 'var(--sats-yellow-bg)', 'border': 'var(--sats-yellow-border)', 'text': 'var(--sats-yellow-text)', 'dot': 'var(--sats-yellow-dot)', 'hex': 'var(--sats-yellow-dot)'},
    3: {'bg': 'var(--sats-green-bg)', 'border': 'var(--sats-green-border)', 'text': 'var(--sats-green-text)', 'dot': 'var(--sats-green-dot)', 'hex': 'var(--sats-green-dot)'},
    4: {'bg': 'var(--sats-blue-bg)', 'border': 'var(--sats-blue-border)', 'text': 'var(--sats-blue-text)', 'dot': 'var(--sats-blue-dot)', 'hex': 'var(--sats-blue-dot)'},
}


def get_kelompok_usia(usia: int) -> int:
    if usia < 1: return 0
    elif usia < 12: return 1
    elif usia < 18: return 2
    elif usia < 45: return 3
    elif usia < 60: return 4
    elif usia < 75: return 5
    else: return 6


def clean_cara_datang(val: str) -> str:
    if not val: return 'Sendiri'
    val = str(val).upper()
    if 'SENDIRI' in val or 'MANDIRI' in val: return 'Sendiri'
    if 'PUSKESMAS' in val or 'RUJUKAN' in val: return 'Puskesmas'
    if 'KECELAKAAN' in val or 'KLL' in val: return 'KLL'
    if 'DOKTER' in val: return 'Dokter'
    return 'Sendiri'


def compute_cardiorespiratory_score(rr: float, spo2: float) -> int:
    """Composite cardiorespiratory distress score (0-4)."""
    score = 0
    if rr < 9 or rr >= 30: score += 2
    elif rr <= 11 or rr >= 21: score += 1
    if spo2 < 90: score += 2
    elif spo2 <= 94: score += 1
    return min(score, 4)


def compute_mews_approx(sbp: float, hr: float, rr: float, temp: float, gcs_total: int) -> int:
    """Modified Early Warning Score approximation."""
    score = 0
    # BP component
    if sbp < 70: score += 3
    elif sbp <= 89: score += 2
    elif sbp <= 100: score += 1
    # HR component
    if hr < 40 or hr >= 130: score += 3
    elif hr < 51 or hr >= 111: score += 2
    elif hr >= 101: score += 1
    # RR component
    if rr < 9 or rr >= 30: score += 3
    elif rr < 15 or rr >= 21: score += 1
    # Temp component
    if temp < 35.0 or temp >= 39.0: score += 2
    elif temp >= 38.5: score += 1
    # GCS component (AVPU approximation)
    if gcs_total <= 8: score += 3
    elif gcs_total <= 12: score += 2
    elif gcs_total <= 14: score += 1
    return score


def build_feature_vector(input_data: dict) -> pd.DataFrame:
    """
    Build full 31-feature vector dari raw input dict.

    FEAT-007: skala_nyeri dihapus dari input — fitur ini tidak digunakan dalam
    konstruksi label SATS-TEWS sehingga memiliki zero information gain.

    Input dict keys:
        usia_tahun, jenis_kelamin ('Laki-laki'/'Perempuan'),
        cara_datang, GCS_E, GCS_M, GCS_V,
        sistole, diastole, denyut_jantung, laju_pernafasan, suhu_tubuh, SpO2
    """
    # Raw values
    usia = float(input_data.get('usia_tahun', 30))
    jk = input_data.get('jenis_kelamin', 'Laki-laki')
    jk_enc = 1 if 'Laki' in str(jk) else 0
    cara = clean_cara_datang(input_data.get('cara_datang', 'Sendiri'))

    gcs_e = float(input_data.get('GCS_E', 4))
    gcs_m = float(input_data.get('GCS_M', 6))
    gcs_v = float(input_data.get('GCS_V', 5))
    gcs_total = int(gcs_e + gcs_m + gcs_v)

    # FEAT-007: skala_nyeri tidak lagi di-ekstrak dari input_data
    sistole = float(input_data.get('sistole', 120))
    diastole = float(input_data.get('diastole', 80))
    hr = float(input_data.get('denyut_jantung', 80))
    rr = float(input_data.get('laju_pernafasan', 16))
    temp = float(input_data.get('suhu_tubuh', 36.5))
    spo2 = float(input_data.get('SpO2', 98))

    # Derived features
    kelompok_usia = get_kelompok_usia(int(usia))
    MAP = diastole + (sistole - diastole) / 3
    shock_index = hr / sistole if sistole > 0 else 0
    pulse_pressure = sistole - diastole
    cardiorespiratory_score = compute_cardiorespiratory_score(rr, spo2)
    mews_approx = compute_mews_approx(sistole, hr, rr, temp, gcs_total)

    # Binary flags
    flag_takikardia = int(hr > 100)
    flag_bradikardia = int(hr < 60)
    flag_hipotensi = int(sistole < 90)
    flag_hipertensi = int(sistole > 160)
    flag_takipnea = int(rr > 20)
    flag_hipoksia = int(spo2 < 95)
    flag_demam = int(temp > 37.5)
    flag_hipotermi = int(temp < 36.0)
    flag_hypoxic_shock = int(spo2 < 90 and sistole < 90)

    flags = [flag_takikardia, flag_bradikardia, flag_hipotensi, flag_hipertensi,
             flag_takipnea, flag_hipoksia, flag_demam, flag_hipotermi, flag_hypoxic_shock]
    n_vital_abnormal = sum(flags[:-1])  # excluding hypoxic_shock

    # TEWS subscores
    tews = compute_tews_subscores(rr, spo2, sistole, hr, temp, gcs_total)

    # Cara datang one-hot
    cara_kll = int(cara == 'KLL')
    cara_puskesmas = int(cara == 'Puskesmas')
    cara_sendiri = int(cara == 'Sendiri')

    feature_dict = {
        'usia_tahun': usia,
        'kelompok_usia': kelompok_usia,
        'jenis_kelamin_enc': jk_enc,
        'GCS_E': gcs_e,
        'GCS_M': gcs_m,
        'GCS_V': gcs_v,
        'GCS_total': float(gcs_total),
        # skala_nyeri dihapus — FEAT-007
        'sistole': sistole,
        'diastole': diastole,
        'denyut_jantung': hr,
        'laju_pernafasan': rr,
        'suhu_tubuh': temp,
        'SpO2': spo2,
        'MAP': MAP,
        'flag_takikardia': flag_takikardia,
        'flag_bradikardia': flag_bradikardia,
        'flag_hipotensi': flag_hipotensi,
        'flag_hipertensi': flag_hipertensi,
        'flag_takipnea': flag_takipnea,
        'flag_hipoksia': flag_hipoksia,
        'flag_demam': flag_demam,
        'flag_hipotermi': flag_hipotermi,
        'n_vital_abnormal': n_vital_abnormal,
        'shock_index': shock_index,
        'cardiorespiratory_score': cardiorespiratory_score,
        'pulse_pressure': pulse_pressure,
        'mews_approx': mews_approx,
        'flag_hypoxic_shock': flag_hypoxic_shock,
        'cara_datang_KLL': cara_kll,
        'cara_datang_Puskesmas': cara_puskesmas,
        'cara_datang_Sendiri': cara_sendiri,
    }

    df = pd.DataFrame([feature_dict], columns=FEATURE_COLS)
    df = df.astype(float)
    return df


def get_active_flags(input_data: dict) -> dict:
    """
    Get clinical flags for display.
    Returns dict: {flag_name: (label, severity)} where severity = 'danger' | 'warning'
    """
    sistole = float(input_data.get('sistole', 120))
    hr = float(input_data.get('denyut_jantung', 80))
    rr = float(input_data.get('laju_pernafasan', 16))
    temp = float(input_data.get('suhu_tubuh', 36.5))
    spo2 = float(input_data.get('SpO2', 98))

    active_flags = {}

    if hr > 100:
        active_flags['flag_takikardia'] = ('Takikardia', 'warning')
    if hr < 60:
        active_flags['flag_bradikardia'] = ('Bradikardia', 'warning')
    if sistole < 90:
        active_flags['flag_hipotensi'] = ('Hipotensi', 'danger')
    if sistole > 160:
        active_flags['flag_hipertensi'] = ('Hipertensi', 'warning')
    if rr > 20:
        active_flags['flag_takipnea'] = ('Takipnea', 'warning')
    if spo2 < 95:
        active_flags['flag_hipoksia'] = ('Hipoksia', 'danger')
    if temp > 37.5:
        active_flags['flag_demam'] = ('Demam', 'warning')
    if temp < 36.0:
        active_flags['flag_hipotermi'] = ('Hipotermi', 'warning')
    if spo2 < 90 and sistole < 90:
        active_flags['flag_hypoxic_shock'] = ('Hypoxic Shock', 'danger')

    return active_flags
