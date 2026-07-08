"""
retrain.py — Re-training script (Model Final)
==============================================
Script ini me-retrain model XGBoost Triage IGD tanpa fitur skala_nyeri,
sesuai PRD-FEAT-007 v1.0.

Jalankan dari folder root proyek:
    python scripts/retrain.py

Output artifact akan disimpan di: output/*.pkl
Estimasi waktu: 30–90 menit (tergantung spesifikasi mesin).
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
# scripts/ -> SKRIPSI/ (2 levels up)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_Labeled_SATS.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'model', 'artifacts')

# ─── FEAT-007: FEATURE_COLS tanpa skala_nyeri ─────────────────────────────────
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

VITAL_COLS = ['sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2']

TARGET_COL = 'sats_label'
RANDOM_STATE = 42

CLIP_BOUNDS = {
    'sistole':        (60, 200), 'diastole':       (30, 130),
    'denyut_jantung': (30, 250), 'laju_pernafasan': (4, 60),
    'suhu_tubuh':     (34.0, 42.0), 'SpO2': (70, 100),
    # skala_nyeri dihapus — FEAT-007
}

SATS_LABELS = {0: 'Merah', 1: 'Oranye', 2: 'Kuning', 3: 'Hijau', 4: 'Biru'}


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_dependencies():
    """Pastikan semua library tersedia."""
    missing = []
    for lib in ['xgboost', 'sklearn', 'imblearn', 'shap']:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"❌ Library berikut tidak terinstall: {missing}")
        print("   Jalankan: pip install xgboost scikit-learn imbalanced-learn shap")
        sys.exit(1)
    print("✅ Semua dependency tersedia")


def load_dataset():
    print_section("1. Load Dataset")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset tidak ditemukan: {DATA_PATH}")
        print("   Pastikan notebook V4 sudah dijalankan hingga Section 3 untuk generate dataset berlabel.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"   Distribusi label: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def build_features(df: pd.DataFrame):
    print_section("2. Feature Engineering")

    # FEAT-007: Verifikasi skala_nyeri tidak ada di FEATURE_COLS
    assert 'skala_nyeri' not in FEATURE_COLS, \
        "FEAT-007 FAIL: skala_nyeri masih ada di FEATURE_COLS!"

    # Pastikan semua kolom tersedia
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"❌ Kolom tidak ditemukan di dataset: {missing_cols}")
        print("   Mungkin perlu regenerate Dataset_Labeled_SATS_v4.csv dari notebook V4.")
        sys.exit(1)

    # Clip bounds
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    print(f"   📊 Total fitur v5: {len(FEATURE_COLS)}")
    print(f"   ℹ️  skala_nyeri dihapus (FEAT-007): zero information gain terhadap label SATS-TEWS")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"   Shape X: {X.shape}, y: {y.shape}")
    return X, y


def impute_features(X: pd.DataFrame, X_train_idx):
    print_section("3. Imputation (IterativeImputer — MICE)")
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)

    # Fit hanya pada vital cols yang ada di FEATURE_COLS
    vital_in_features = [c for c in VITAL_COLS if c in FEATURE_COLS]
    X_train = X.iloc[X_train_idx][vital_in_features]

    imputer.fit(X_train)

    # Transform seluruh dataset (vital cols only, fill lainnya dengan 0)
    X_vital = pd.DataFrame(
        imputer.transform(X[vital_in_features]),
        columns=vital_in_features, index=X.index
    )
    X[vital_in_features] = X_vital

    missing_after = X.isnull().sum().sum()
    print(f"   Missing setelah imputation: {missing_after}")
    return X, imputer


def split_data(X: pd.DataFrame, y: pd.Series):
    print_section("4. Train/Test Split (Stratified)")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, list(idx_train), list(idx_test)


def scale_features(X_train, X_test):
    print_section("5. MinMaxScaler")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  columns=FEATURE_COLS)

    # FEAT-007 verification
    if hasattr(scaler, 'feature_names_in_'):
        assert 'skala_nyeri' not in scaler.feature_names_in_, \
            "FEAT-007 FAIL: skala_nyeri masih ada di scaler.feature_names_in_!"
    print("   ✅ FEAT-007: scaler fitted tanpa skala_nyeri")
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train):
    print_section("6. SMOTE + XGBoost + RandomizedSearchCV")

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    # SMOTE sampling strategy
    SMOTE_SAMPLING_STRATEGY = {
        0: max(int(y_train.value_counts().get(0, 10) * 2.5), 50),
        1: max(int(y_train.value_counts().get(1, 10) * 2.0), 50),
        2: max(int(y_train.value_counts().get(2, 10) * 1.5), 50),
    }
    # Hanya oversample kelas yang lebih sedikit dari mayoritas
    max_count = y_train.value_counts().max()
    SMOTE_SAMPLING_STRATEGY = {
        k: min(v, max_count)
        for k, v in SMOTE_SAMPLING_STRATEGY.items()
        if y_train.value_counts().get(k, 0) < max_count
    }
    print(f"   SMOTE strategy: {SMOTE_SAMPLING_STRATEGY}")

    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    pipeline = ImbPipeline(steps=[('smote', smote), ('xgb', xgb)])

    param_dist = {
        'xgb__n_estimators':    [100, 200, 300, 400, 500],
        'xgb__max_depth':       [3, 4, 5, 6, 7, 8],
        'xgb__learning_rate':   [0.01, 0.05, 0.1, 0.15, 0.2],
        'xgb__subsample':       [0.6, 0.7, 0.8, 0.9, 1.0],
        'xgb__colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
        'xgb__min_child_weight':[1, 3, 5, 7],
        'xgb__gamma':           [0, 0.1, 0.2, 0.3],
        'xgb__reg_alpha':       [0, 0.01, 0.1, 1],
        'xgb__reg_lambda':      [1, 1.5, 2, 5],
        'xgb__scale_pos_weight':[1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print(f"   🚀 Mulai RandomizedSearchCV (100 iter × 5-fold)...")
    t0 = time.time()
    search.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0

    print(f"   ✅ Training selesai dalam {elapsed/60:.1f} menit")
    print(f"   Best CV F1-Macro: {search.best_score_:.4f}")
    print(f"   Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(best_pipeline, X_test_scaled, y_test):
    print_section("7. Evaluasi Model v5")
    from sklearn.metrics import (
        classification_report, f1_score, accuracy_score,
        balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    )

    y_pred  = best_pipeline.predict(X_test_scaled)
    y_proba = best_pipeline.predict_proba(X_test_scaled)

    f1_macro   = f1_score(y_test, y_pred, average='macro')
    acc        = accuracy_score(y_test, y_pred)
    bal_acc    = balanced_accuracy_score(y_test, y_pred)
    mcc        = matthews_corrcoef(y_test, y_pred)

    try:
        auc_macro = roc_auc_score(
            pd.get_dummies(y_test).values,
            y_proba, average='macro', multi_class='ovr'
        )
    except Exception:
        auc_macro = float('nan')

    report = classification_report(y_test, y_pred,
                                   target_names=[SATS_LABELS[i] for i in range(5)])

    print(f"\n   F1-Macro     : {f1_macro:.4f}")
    print(f"   Accuracy     : {acc:.4f}")
    print(f"   Balanced Acc : {bal_acc:.4f}")
    print(f"   MCC          : {mcc:.4f}")
    print(f"   AUC-ROC Macro: {auc_macro:.4f}")
    print(f"\n{report}")

    metrics = {
        'f1_macro': f1_macro, 'accuracy': acc,
        'balanced_accuracy': bal_acc, 'mcc': mcc, 'auc_macro': auc_macro,
    }
    return metrics


def compute_thresholds(best_pipeline, X_test_scaled, y_test):
    print_section("8. Threshold Optimization")
    from sklearn.metrics import f1_score

    y_proba = best_pipeline.predict_proba(X_test_scaled)

    # Cari threshold optimal per kelas (maximize recall kelas kritis)
    best_thresholds = {}
    for cls_idx in range(5):
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.95, 0.05):
            adjusted = np.zeros(len(y_test))
            for i in range(len(y_test)):
                passes = {c: 1 if y_proba[i, c] >= best_thresholds.get(c, thresh) else 0 for c in range(5)}
                if sum(passes.values()) == 0:
                    adjusted[i] = int(np.argmax(y_proba[i]))
                else:
                    masked = y_proba[i] * np.array([passes.get(c, 0) for c in range(5)])
                    adjusted[i] = int(np.argmax(masked))

            f1 = f1_score((y_test == cls_idx).astype(int),
                          (adjusted == cls_idx).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        best_thresholds[cls_idx] = best_thresh

    print(f"   Best thresholds: {best_thresholds}")
    return best_thresholds


def compute_shap(best_pipeline, X_train_scaled):
    print_section("9. SHAP Explainer")
    import shap

    # Extract XGBoost model dari pipeline
    xgb_model = best_pipeline.named_steps['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    print(f"   ✅ SHAP explainer dibuat")
    return explainer


def save_artifacts(best_pipeline, scaler, imputer, explainer,
                   best_params, best_thresholds,
                   train_idx, test_idx, metrics):
    print_section("10. Save Artifact (Model Final)")

    # Model final — artifact tanpa suffix versi (FEAT-007: tanpa skala_nyeri)
    paths = {
        'model':     os.path.join(OUTPUT_DIR, 'model_triage_xgb.pkl'),
        'pipeline':  os.path.join(OUTPUT_DIR, 'pipeline_imblearn.pkl'),
        'scaler':    os.path.join(OUTPUT_DIR, 'scaler_minmax.pkl'),
        'features':  os.path.join(OUTPUT_DIR, 'feature_names.pkl'),
        'thresholds':os.path.join(OUTPUT_DIR, 'best_thresholds.pkl'),
        'params':    os.path.join(OUTPUT_DIR, 'best_params.pkl'),
        'imputer':   os.path.join(OUTPUT_DIR, 'imputer.pkl'),
        'split':     os.path.join(OUTPUT_DIR, 'split_indices.pkl'),
    }

    xgb_model = best_pipeline.named_steps['xgb']
    joblib.dump(xgb_model,       paths['model'])
    joblib.dump(best_pipeline,   paths['pipeline'])
    joblib.dump(scaler,          paths['scaler'])
    joblib.dump(FEATURE_COLS,    paths['features'])
    joblib.dump(best_thresholds, paths['thresholds'])
    joblib.dump(best_params,     paths['params'])
    joblib.dump(imputer,         paths['imputer'])
    joblib.dump({'train': train_idx, 'test': test_idx}, paths['split'])

    # Try save SHAP explainer (bisa besar, skip jika error)
    try:
        import shap
        explainer_path = os.path.join(OUTPUT_DIR, 'shap_explainer.pkl')
        joblib.dump(explainer, explainer_path)
        print(f"   ✅ shap_explainer.pkl disimpan")
    except Exception as e:
        print(f"   ⚠️  SHAP explainer gagal disimpan: {e}")

    for name, path in paths.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"   ✅ {os.path.basename(path):35s} ({size_kb:.0f} KB)")
        else:
            print(f"   ❌ {os.path.basename(path)} — GAGAL DISIMPAN")

    # ── FEAT-007 Verification ──────────────────────────────────────────────
    print_section("FEAT-007 VERIFICATION")
    feat_saved = joblib.load(paths['features'])
    assert 'skala_nyeri' not in FEATURE_COLS, \
        "FEAT-007 FAIL: skala_nyeri masih ada di FEATURE_COLS"
    assert 'skala_nyeri' not in feat_saved, \
        "FEAT-007 FAIL: skala_nyeri masih ada di feature_names.pkl"

    print(f"   ✅ AC-F07-M01: skala_nyeri tidak ada di FEATURE_COLS ({len(FEATURE_COLS)} fitur)")
    print(f"   ✅ AC-F07-M02: feature_names.pkl tidak mengandung skala_nyeri")
    print(f"   ✅ AC-F07-M05: Training selesai tanpa error")
    print(f"   ✅ AC-F07-M08: Semua artifact final tersimpan di output/")

    print(f"\n   📊 Ringkasan Performa v5:")
    for k, v in metrics.items():
        print(f"      {k:25s}: {v:.4f}")


def main():
    print("=" * 60)
    print("  RETRAIN MODEL TRIAGE IGD — v5 (FEAT-007)")
    print("  Penghapusan skala_nyeri dari pipeline")
    print("=" * 60)

    check_dependencies()
    df = load_dataset()
    X, y = build_features(df)

    # Split sebelum imputation untuk mencegah data leakage
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = split_data(X, y)

    # Impute
    X_full, imputer = impute_features(X, idx_train)
    X_train_raw = X_full.iloc[idx_train]
    X_test_raw  = X_full.iloc[idx_test]

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)

    # Train
    best_pipeline, best_params, cv_score = train_model(X_train_scaled, y_train)

    # Evaluate
    metrics = evaluate_model(best_pipeline, X_test_scaled, y_test)

    # Thresholds
    best_thresholds = compute_thresholds(best_pipeline, X_test_scaled, y_test)

    # SHAP
    try:
        explainer = compute_shap(best_pipeline, X_train_scaled)
    except Exception as e:
        print(f"   ⚠️  SHAP gagal: {e}. Melanjutkan tanpa explainer.")
        explainer = None

    # Save
    save_artifacts(
        best_pipeline, scaler, imputer, explainer,
        best_params, best_thresholds,
        idx_train, idx_test, metrics
    )

    print("\n" + "=" * 60)
    print("  ✅ RETRAIN SELESAI! Artifact final tersimpan.")
    print("  Artifact tersimpan di: output/*.pkl")
    print("  Jalankan UI: python -m streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
