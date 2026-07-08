"""
backend/tews_calculator.py
TEWS (Triage Early Warning Score) scoring logic — sesuai notebook V3
"""

def compute_tews_subscores(rr: float, spo2: float, sbp: float, hr: float, temp: float, gcs_total: int) -> dict:
    """
    Hitung sub-skor TEWS untuk 6 parameter.
    Returns dict dengan semua sub-skor dan TEWS total.
    """
    # --- Respiratory Rate ---
    if rr < 9:
        rr_score = 3
    elif rr <= 11:
        rr_score = 1
    elif rr <= 20:
        rr_score = 0
    elif rr <= 29:
        rr_score = 2
    else:
        rr_score = 3

    # --- SpO2 ---
    if spo2 < 90:
        spo2_score = 3
    elif spo2 <= 94:
        spo2_score = 2
    elif spo2 <= 96:
        spo2_score = 1
    else:
        spo2_score = 0

    # --- Systolic Blood Pressure ---
    if sbp < 70:
        bp_score = 3
    elif sbp <= 89:
        bp_score = 2
    elif sbp <= 109:
        bp_score = 1
    elif sbp <= 149:
        bp_score = 0
    elif sbp <= 179:
        bp_score = 1
    else:
        bp_score = 2

    # --- Heart Rate ---
    if hr < 40:
        hr_score = 3
    elif hr <= 50:
        hr_score = 1
    elif hr <= 100:
        hr_score = 0
    elif hr <= 110:
        hr_score = 1
    elif hr <= 129:
        hr_score = 2
    else:
        hr_score = 3

    # --- Temperature ---
    if temp < 35.0:
        temp_score = 2
    elif temp <= 38.4:
        temp_score = 0
    else:
        temp_score = 1

    # --- GCS ---
    if gcs_total <= 8:
        gcs_score = 3
    elif gcs_total <= 12:
        gcs_score = 2
    elif gcs_total <= 14:
        gcs_score = 1
    else:
        gcs_score = 0

    tews_total = rr_score + spo2_score + bp_score + hr_score + temp_score + gcs_score

    return {
        'TEWS_rr_score': rr_score,
        'TEWS_spo2_score': spo2_score,
        'TEWS_bp_score': bp_score,
        'TEWS_hr_score': hr_score,
        'TEWS_temp_score': temp_score,
        'TEWS_gcs_score': gcs_score,
        'TEWS_total': tews_total,
    }


def check_override(gcs_total: int, sbp: float, spo2: float, rr: float) -> list[str]:
    """
    Cek kondisi override klinis — langsung Merah tanpa melihat TEWS total.
    Returns list kondisi override yang terpenuhi (kosong = tidak ada override).
    """
    overrides = []
    if gcs_total <= 8:
        overrides.append(f"GCS ≤ 8 (nilai: {gcs_total}) — Penurunan kesadaran berat")
    if sbp < 70:
        overrides.append(f"Sistole < 70 mmHg (nilai: {sbp:.0f}) — Perfusi organ tidak adekuat")
    if spo2 < 90 and (rr < 9 or rr >= 30):
        overrides.append(f"SpO₂ < 90% ({spo2:.0f}%) + RR abnormal ({rr:.0f} x/mnt) — Hipoksia + gangguan napas")
    return overrides


def get_sats_zone_from_tews(tews_total: int, overrides: list) -> int:
    """
    Map TEWS total → SATS label (0-4).
    Jika ada override → return 0 (Merah).
    """
    if overrides:
        return 0  # Merah
    if tews_total >= 7:
        return 1  # Oranye
    elif tews_total >= 5:
        return 2  # Kuning
    elif tews_total >= 3:
        return 3  # Hijau
    else:
        return 4  # Biru


def get_active_range_label(param: str, value: float, score: int) -> str:
    """Human-readable label untuk rentang aktif sub-skor TEWS."""
    ranges = {
        'rr': {
            3: '< 9 atau ≥ 30 x/mnt',
            2: '21–29 x/mnt',
            1: '9–11 x/mnt',
            0: '12–20 x/mnt (Normal)',
        },
        'spo2': {
            3: '< 90%',
            2: '90–94%',
            1: '95–96%',
            0: '≥ 97% (Normal)',
        },
        'bp': {
            3: '< 70 mmHg',
            2: '70–89 mmHg',
            1: '90–109 atau 150–179 mmHg',
            0: '110–149 mmHg (Normal)',
        },
        'hr': {
            3: '< 40 atau ≥ 130 bpm',
            2: '111–129 bpm',
            1: '40–50 atau 101–110 bpm',
            0: '51–100 bpm (Normal)',
        },
        'temp': {
            2: '< 35°C (Hipotermi)',
            1: '≥ 38.5°C (Demam)',
            0: '35.0–38.4°C (Normal)',
        },
        'gcs': {
            3: '≤ 8 (Koma berat)',
            2: '9–12 (Sedang)',
            1: '13–14 (Ringan)',
            0: '15 (Normal)',
        },
    }
    return ranges.get(param, {}).get(score, '-')
