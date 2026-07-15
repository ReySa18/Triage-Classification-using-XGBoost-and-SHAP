"""
backend/predictor.py
Model loading & prediction pipeline.
Gunakan @st.cache_resource agar model hanya di-load sekali.
Model final — tanpa skala_nyeri (FEAT-007)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings('ignore')

# app/backend/ -> app/ -> repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / 'model' / 'artifacts'

# Model final — artifact tanpa versi suffix (FEAT-007: tanpa skala_nyeri)
MODEL_FILES = {
    'model':    'model_triage_xgb.pkl',
    'scaler':   'scaler_minmax.pkl',
    'explainer': 'shap_explainer.pkl',
    'features': 'feature_names.pkl',
    'imputer':  'imputer.pkl',
    'thresholds': 'best_thresholds.pkl',
}


@st.cache_resource(show_spinner="Memuat model ML...")
def load_artifacts():
    """
    Load all model artifacts once and cache them.
    Returns dict with model, scaler, explainer, feature_names, imputer, thresholds.
    Returns None if files not found.
    """
    artifacts = {}
    try:
        for key, filename in MODEL_FILES.items():
            path = OUTPUT_DIR / filename
            if not path.is_file() or path.stat().st_size <= 0:
                st.error(f"File artefak tidak ditemukan: {path}")
                return None
            artifacts[key] = joblib.load(path)
        artifacts['model_version'] = 'final'
        # FEAT-007 verification: pastikan skala_nyeri tidak ada di feature names
        feat_names = artifacts.get('features', [])
        if hasattr(feat_names, '__iter__') and 'skala_nyeri' in list(feat_names):
            st.warning("⚠️ FEAT-007: feature_names.pkl masih mengandung skala_nyeri. Pastikan model sudah di-retrain.")
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None


def predict_triage_v3(input_data: dict, artifacts: dict) -> dict:
    """
    Full prediction pipeline.

    Parameters:
        input_data: dict dengan keys:
            usia_tahun, jenis_kelamin, cara_datang,
            GCS_E, GCS_M, GCS_V,
            sistole, diastole, denyut_jantung, laju_pernafasan, suhu_tubuh, SpO2
        artifacts: dict dari load_artifacts()

    Returns:
        dict dengan:
            predicted_class (int 0-4)
            predicted_label (str)
            probabilities (dict {label: prob})
            shap_values (array, 8 top features)
            shap_top_features (list of (feature_name, shap_val, display_name))
            tews_subscores (dict)
            clinical_scores (dict: shock_index, mews, tews_total, cardiorespiratory)
            active_flags (dict)
    """
    from .feature_engineering import (
        build_feature_vector, get_active_flags,
        FEATURE_COLS, SATS_LABELS
    )
    from .tews_calculator import compute_tews_subscores, check_override

    # FEAT-007: abaikan skala_nyeri jika masih ada di input_data (backward compat)
    if 'skala_nyeri' in input_data:
        import warnings as _w
        _w.warn(
            "FEAT-007: 'skala_nyeri' diterima di input_data tapi diabaikan. "
            "Hapus key ini dari input untuk menghindari kebingungan.",
            DeprecationWarning, stacklevel=2
        )

    model = artifacts['model']
    scaler = artifacts['scaler']
    explainer = artifacts['explainer']
    thresholds = artifacts['thresholds']

    # 1. Build feature vector
    X_raw = build_feature_vector(input_data)

    # 2. Impute (if needed)
    imputer = artifacts['imputer']
    try:
        X_imputed = pd.DataFrame(
            imputer.transform(X_raw),
            columns=FEATURE_COLS
        )
    except Exception:
        X_imputed = X_raw.copy()

    # 3. Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=FEATURE_COLS
    )

    # 4. Get probabilities
    proba = model.predict_proba(X_scaled)[0]  # shape (5,)

    # 5. Apply custom thresholds
    adjusted = np.zeros(5)
    for cls_idx, thresh in thresholds.items():
        adjusted[int(cls_idx)] = 1.0 if proba[int(cls_idx)] >= thresh else 0.0

    # If no class passes threshold, use argmax
    if adjusted.sum() == 0:
        predicted_class = int(np.argmax(proba))
    else:
        # Pick the class with highest probability among those passing threshold
        masked_proba = proba * adjusted
        predicted_class = int(np.argmax(masked_proba))

    # 6. SHAP values for predicted class
    try:
        shap_vals = explainer.shap_values(X_scaled)
        # shap_vals shape: (n_samples, n_features, n_classes) or list of arrays
        if isinstance(shap_vals, list):
            shap_for_class = np.array(shap_vals[predicted_class])[0]
        else:
            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 3:
                shap_for_class = shap_arr[0, :, predicted_class]
            else:
                shap_for_class = shap_arr[0]
    except Exception:
        shap_for_class = np.zeros(len(FEATURE_COLS))

    # Top 8 features by |SHAP|
    top_indices = np.argsort(np.abs(shap_for_class))[::-1][:8]
    shap_top = []
    feature_display_names = {
        'TEWS_total': 'TEWS Total',
        'GCS_total': 'GCS Total',
        'SpO2': 'SpO₂',
        'sistole': 'Sistole',
        'denyut_jantung': 'Denyut Jantung',
        'laju_pernafasan': 'Laju Pernafasan',
        'suhu_tubuh': 'Suhu Tubuh',
        'mews_approx': 'MEWS',
        'shock_index': 'Shock Index',
        'cardiorespiratory_score': 'Cardioresp. Score',
        'MAP': 'MAP',
        'skala_nyeri': 'Skala Nyeri',
        'flag_hipoksia': 'Flag Hipoksia',
        'flag_hipotensi': 'Flag Hipotensi',
        'flag_takikardia': 'Flag Takikardia',
        'flag_takipnea': 'Flag Takipnea',
        'n_vital_abnormal': 'Vital Abnormal (n)',
        'diastole': 'Diastole',
        'pulse_pressure': 'Pulse Pressure',
        'usia_tahun': 'Usia',
        'TEWS_rr_score': 'TEWS RR Score',
        'TEWS_spo2_score': 'TEWS SpO₂ Score',
        'TEWS_bp_score': 'TEWS BP Score',
        'TEWS_hr_score': 'TEWS HR Score',
        'TEWS_temp_score': 'TEWS Temp Score',
        'TEWS_gcs_score': 'TEWS GCS Score',
        'flag_hypoxic_shock': 'Flag Hypoxic Shock',
        'flag_hipertensi': 'Flag Hipertensi',
        'flag_demam': 'Flag Demam',
        'flag_hipotermi': 'Flag Hipotermi',
        'flag_bradikardia': 'Flag Bradikardia',
        'jenis_kelamin_enc': 'Jenis Kelamin',
        'kelompok_usia': 'Kelompok Usia',
        'GCS_E': 'GCS Eye',
        'GCS_M': 'GCS Motor',
        'GCS_V': 'GCS Verbal',
        'cara_datang_KLL': 'Cara Datang (KLL)',
        'cara_datang_Puskesmas': 'Cara Datang (Puskesmas)',
        'cara_datang_Sendiri': 'Cara Datang (Sendiri)',
    }
    for idx in top_indices:
        feat_name = FEATURE_COLS[idx]
        display_name = feature_display_names.get(feat_name, feat_name)
        shap_val = float(shap_for_class[idx])
        shap_top.append((feat_name, shap_val, display_name))

    # 7. TEWS subscores
    rr = float(input_data.get('laju_pernafasan', 16))
    spo2 = float(input_data.get('SpO2', 98))
    sbp = float(input_data.get('sistole', 120))
    hr = float(input_data.get('denyut_jantung', 80))
    temp = float(input_data.get('suhu_tubuh', 36.5))
    gcs_e = float(input_data.get('GCS_E', 4))
    gcs_m = float(input_data.get('GCS_M', 6))
    gcs_v = float(input_data.get('GCS_V', 5))
    gcs_total = int(gcs_e + gcs_m + gcs_v)

    tews_subscores = compute_tews_subscores(rr, spo2, sbp, hr, temp, gcs_total)

    # 8. Clinical scores for display
    shock_index = hr / sbp if sbp > 0 else 0
    clinical_scores = {
        'shock_index': round(shock_index, 2),
        'mews_approx': int(X_raw['mews_approx'].iloc[0]),
        'tews_total': int(tews_subscores['TEWS_total']),
        'cardiorespiratory_score': int(X_raw['cardiorespiratory_score'].iloc[0]),
    }

    # 9. Active clinical flags
    active_flags = get_active_flags(input_data)

    # 10. Probabilities as dict
    prob_dict = {SATS_LABELS[i]: float(proba[i]) for i in range(5)}

    return {
        'predicted_class': predicted_class,
        'predicted_label': SATS_LABELS[predicted_class],
        'probabilities': prob_dict,
        'shap_top_features': shap_top,
        'tews_subscores': tews_subscores,
        'clinical_scores': clinical_scores,
        'active_flags': active_flags,
        'gcs_total': gcs_total,
    }
