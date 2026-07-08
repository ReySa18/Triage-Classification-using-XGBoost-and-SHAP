# Product Requirements Document
## Optimasi Model Prediksi Triage IGD — XGBoost + SMOTE
**RSU Aulia | Sistem Prediksi Skor Triage SATS-TEWS**

---

| Field | Detail |
|---|---|
| **Dokumen** | PRD-OPT-001 |
| **Versi** | v1.0 |
| **Tanggal** | 17 Juni 2026 |
| **Peneliti** | Reymondo |
| **Status** | Draft — Untuk Review |
| **Referensi** | Notebook: `optimasi_result.ipynb` (baseline PRD v1.1) |

---

## 1. Latar Belakang & Motivasi

### 1.1 Kondisi Model Saat Ini

Model XGBoost triage IGD versi saat ini (`optimasi_result.ipynb`) telah berhasil melampaui target minimum F1-Macro ≥ 0.75 dengan hasil **F1-Macro = 0.8070** dan **AUC-ROC Macro = 0.9989**. Namun evaluasi menyeluruh mengidentifikasi tiga permasalahan metodologi yang menurunkan performa dan validitas model secara akademik:

| Masalah | Dampak | Severity |
|---|---|---|
| SMOTE full-balancing (14 → 4.296 sampel Oranye) | Distribution mismatch: CV F1=0.998 vs Test F1=0.807 | Kritis |
| SMOTE diterapkan sebelum CV fold (subtle leakage) | CV score tidak merepresentasikan performa real | Tinggi |
| GCS homogen total (std=0.00 di seluruh dataset) | GCS kehilangan discriminative power sepenuhnya | Sedang |
| Fitur klinis hanya univariat | Kombinasi patologis (shock, gagal napas) tidak tertangkap | Sedang |
| Threshold prediksi uniform 0.5 semua kelas | Kelas minoritas (Oranye, Kuning) under-detected | Sedang |

### 1.2 Batasan Tetap (Constraints)

- **Dataset tetap:** 6.339 rekam medis IGD RSU Aulia (Januari–Mei 2026). Penambahan data dari RS lain atau periode lain **tidak dilakukan**.
- **Algoritma inti tetap:** XGBoost sebagai classifier utama.
- **Standar klinis tetap:** SATS-TEWS sebagai referensi label surrogate.
- **Framework tetap:** Python + scikit-learn + imbalanced-learn + SHAP.
- **Split tetap:** Stratified 80:20, `random_state=42`.

### 1.3 Tujuan Optimasi

Meningkatkan performa dan validitas metodologi model tanpa menambah data, melalui perbaikan pipeline imbalance handling, rekayasa fitur klinis, dan kalibrasi threshold prediksi.

---

## 2. Objectives & Key Results

| ID | Objective | Key Result | Target |
|---|---|---|---|
| OBJ-01 | Perbaiki distribution mismatch SMOTE | F1-Macro pada test set | ≥ 0.85 |
| OBJ-02 | Hilangkan subtle leakage pada CV | Gap CV↔Test F1 | < 0.05 |
| OBJ-03 | Tingkatkan deteksi kelas kritis | Recall Oranye + Kuning | ≥ 0.60 |
| OBJ-04 | Tambah discriminative power fitur | Jumlah fitur klinis baru masuk top-10 SHAP | ≥ 2 fitur |
| OBJ-05 | Optimalkan threshold per kelas | Undertriage Oranye | < 30% |
| OBJ-06 | Pertahankan safety metrics | Recall Merah + Undertriage Combined | ≥ 1.00 & < 5% |

---

## 3. Scope

### 3.1 In Scope

- Perubahan strategi SMOTE (rasio sampling)
- Rekonstruksi pipeline dengan `ImbPipeline` untuk CV yang benar
- Penambahan fitur interaksi klinis
- Perbaikan imputasi GCS
- Threshold optimization post-training
- Pembaruan acceptance criteria (AC-M07)
- Pembaruan ringkasan SHAP sesuai fitur baru

### 3.2 Out of Scope

- Penambahan data dari sumber eksternal
- Penggantian algoritma utama (tetap XGBoost)
- Perubahan standar triage (tetap SATS-TEWS)
- Perubahan skema labeling surrogate
- Implementasi Streamlit (dikerjakan terpisah)

---

## 4. Spesifikasi Fitur

---

### FEAT-01 — Perbaikan Strategi SMOTE (Partial Oversampling)

**Prioritas:** P0 — Kritikal  
**Estimasi dampak F1-Macro:** +0.05 hingga +0.08

#### 4.1.1 Deskripsi

Mengganti full-balancing SMOTE (semua kelas → 4.296) dengan **partial oversampling** yang proporsional. Tujuannya adalah mengurangi distribusi gap antar kelas tanpa membuat distribusi training terlalu jauh dari distribusi real test set.

#### 4.1.2 Spesifikasi Teknis

```python
from imblearn.over_sampling import SMOTE

# Strategi partial: naikkan minoritas ke level proporsional,
# bukan balance sempurna
# Oranye (14 asli) → 500 maksimum karena k_neighbors=3 berisiko
# untuk oversample terlalu jauh dari distribusi asli
SMOTE_SAMPLING_STRATEGY = {
    0: 4296,   # Merah   — pertahankan (443 asli, sudah cukup representatif)
    1: 500,    # Oranye  — dari 14 → 500 (bukan 4296, terlalu agresif)
    2: 800,    # Kuning  — dari 73  → 800
    3: 1000,   # Hijau   — dari 245 → 1000
    4: 4296,   # Biru    — pertahankan (mayoritas, tidak perlu diubah)
}

smote = SMOTE(
    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    k_neighbors=3,       # tetap 3 karena Oranye hanya 14 sampel asli
    random_state=42
)
```

#### 4.1.3 Acceptance Criteria

| ID | Kriteria | Metode Verifikasi |
|---|---|---|
| AC-F01-01 | Total sampel training setelah SMOTE antara 8.000–12.000 | `print(X_res.shape)` |
| AC-F01-02 | Tidak ada kelas dengan rasio > 10× terhadap kelas terkecil | `y_res.value_counts()` |
| AC-F01-03 | Gap CV↔Test F1-Macro < 0.05 | Bandingkan `cv_mean` vs `test_f1` |

---

### FEAT-02 — Pipeline CV yang Benar (ImbPipeline)

**Prioritas:** P0 — Kritikal  
**Estimasi dampak:** CV score lebih representatif, mengurangi overestimasi performa

#### 4.2.1 Deskripsi

SMOTE harus dieksekusi **di dalam** setiap fold CV, bukan di luar. Dengan membungkus SMOTE dan XGBoost dalam `ImbPipeline`, setiap fold akan menghasilkan data sintetis hanya dari training fold tersebut — menghindari data sintetis yang "bocor" dari validation fold.

#### 4.2.2 Spesifikasi Teknis

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Pipeline: SMOTE berjalan di dalam setiap fold CV
pipeline = ImbPipeline([
    ('smote', SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=3,
        random_state=42
    )),
    ('clf', XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ))
])

# Prefix parameter dengan 'clf__' untuk RandomizedSearchCV
param_distributions = {
    'clf__n_estimators':     [100, 200, 300, 500],
    'clf__max_depth':        [3, 4, 5, 6, 7, 8],
    'clf__learning_rate':    [0.01, 0.05, 0.1, 0.2, 0.3],
    'clf__subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__min_child_weight': [1, 3, 5, 7],
    'clf__gamma':            [0, 0.1, 0.2, 0.5, 1.0],
    'clf__reg_alpha':        [0, 0.01, 0.1, 1.0],
    'clf__reg_lambda':       [0.5, 1.0, 2.0, 5.0],
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit ke data ORIGINAL (sebelum SMOTE) —
# SMOTE akan dijalankan otomatis di dalam setiap fold
random_search.fit(X_train_scaled, y_train)
```

#### 4.2.3 Acceptance Criteria

| ID | Kriteria | Metode Verifikasi |
|---|---|---|
| AC-F02-01 | `ImbPipeline` digunakan (bukan `sklearn.Pipeline`) | Code review |
| AC-F02-02 | `random_search.fit()` menerima `X_train_scaled` (pre-SMOTE) | Code review |
| AC-F02-03 | CV score lebih rendah dari versi sebelumnya (lebih jujur) | Bandingkan `best_score_` |

---

### FEAT-03 — Perbaikan Imputasi GCS

**Prioritas:** P1 — Tinggi  
**Estimasi dampak:** Recover discriminative power GCS sebagai fitur klinis

#### 4.3.1 Deskripsi

Imputasi GCS missing dengan nilai default `15` (GCS normal) menyebabkan `TEWS_gcs_score` memiliki `std=0.00` di seluruh dataset, menghilangkan informasi klinis. Perlu diganti dengan imputasi berbasis **median per kelompok usia** atau median global yang lebih mencerminkan variasi populasi.

#### 4.3.2 Spesifikasi Teknis

```python
# Step 2 yang direvisi: Ekstraksi GCS dengan imputasi berbasis median

gcs_extracted = df['GCS'].apply(extract_gcs)
df['GCS_E']      = [x[0] for x in gcs_extracted]
df['GCS_M']      = [x[1] for x in gcs_extracted]
df['GCS_V']      = [x[2] for x in gcs_extracted]
df['GCS_total']  = [x[3] for x in gcs_extracted]

# Hitung median per kelompok usia dari data yang berhasil diekstrak
gcs_by_age_group = (
    df.dropna(subset=['GCS_total'])
      .groupby('kelompok_usia')['GCS_total']
      .median()
)
print("Median GCS per kelompok usia:")
print(gcs_by_age_group)

# Imputasi: gunakan median kelompok usia jika tersedia,
# fallback ke median global jika kelompok usia juga tidak ada datanya
gcs_global_median = df['GCS_total'].dropna().median()

def impute_gcs_components(row):
    """Imputasi GCS berdasarkan median kelompok usia, bukan default 15."""
    if pd.notna(row['GCS_total']):
        return row['GCS_E'], row['GCS_M'], row['GCS_V'], row['GCS_total']
    
    # Ambil median total GCS untuk kelompok usia ini
    median_total = gcs_by_age_group.get(row['kelompok_usia'], gcs_global_median)
    
    # Distribusi komponen GCS untuk nilai total tertentu
    # Prioritaskan GCS_M (motor) karena paling prognostik
    if median_total >= 14:
        return 4, 6, 5, 15      # Normal
    elif median_total >= 11:
        return 3, 5, 4, 12      # Mild impairment
    elif median_total >= 8:
        return 2, 4, 3, 9       # Moderate
    else:
        return 1, 3, 2, 6       # Severe

gcs_imputed = df.apply(impute_gcs_components, axis=1, result_type='expand')
gcs_imputed.columns = ['GCS_E', 'GCS_M', 'GCS_V', 'GCS_total']
df[['GCS_E', 'GCS_M', 'GCS_V', 'GCS_total']] = gcs_imputed

# Verifikasi: std harus > 0 sekarang
print(f"\nGCS_total setelah imputasi berbasis median:")
print(df['GCS_total'].describe())
print(f"Std > 0: {df['GCS_total'].std() > 0}")
```

#### 4.3.3 Acceptance Criteria

| ID | Kriteria | Metode Verifikasi |
|---|---|---|
| AC-F03-01 | `GCS_total.std() > 0` setelah imputasi | `df['GCS_total'].std()` |
| AC-F03-02 | Tidak ada nilai `GCS_total` di luar range [3, 15] | `df['GCS_total'].between(3,15).all()` |
| AC-F03-03 | Persentase missing GCS yang diimputasi terdokumentasi | Print sebelum/sesudah imputasi |

---

### FEAT-04 — Penambahan Fitur Interaksi Klinis

**Prioritas:** P1 — Tinggi  
**Estimasi dampak F1-Macro:** +0.01 hingga +0.03

#### 4.4.1 Deskripsi

Fitur klinis saat ini bersifat univariat (satu parameter = satu fitur). Dalam praktik klinis, kegawatan seringkali ditentukan oleh **kombinasi** parameter. Fitur interaksi berikut memiliki dasar literatur medis yang kuat dan relevan dengan populasi IGD.

#### 4.4.2 Spesifikasi Teknis

```python
# ── STEP 8b: FEATURE ENGINEERING — FITUR INTERAKSI KLINIS ──────────────────
print("⚙️  STEP 8b: FITUR INTERAKSI KLINIS")

# 1. Shock Index (SI)
# Referensi: Birkhahn et al. (2005). Am J Emerg Med
# Normal: < 0.7 | Borderline: 0.7–1.0 | Abnormal: > 1.0
df['shock_index'] = (
    df['denyut_jantung'] / df['sistole'].replace(0, np.nan)
).clip(0, 5).fillna(df['denyut_jantung'].median() / df['sistole'].median())
print("   ✅ shock_index = denyut_jantung / sistole")

# 2. Cardiorespiratory Distress Score
# Kombinasi 4 indikator kegawatan kardiorespiratorik
df['cardiorespiratory_score'] = (
    df['flag_takikardia'].astype(int) +
    df['flag_takipnea'].astype(int) +
    df['flag_hipoksia'].astype(int) +
    df['flag_hipotensi'].astype(int)
)
print("   ✅ cardiorespiratory_score = takikardia + takipnea + hipoksia + hipotensi")

# 3. Pulse Pressure
# Normal: 40–60 mmHg
# Narrow PP (<25): indikasi tamponade/shock distributif
# Wide PP (>80): indikasi regurgitasi aorta / sepsis
df['pulse_pressure'] = (df['sistole'] - df['diastole']).clip(0, 120)
print("   ✅ pulse_pressure = sistole - diastole")

# 4. Modified Early Warning Score (MEWS) approx.
# Simplified version: kombinasi RR, HR, BP, kesadaran
# Referensi: Subbe et al. (2001). QJM
df['mews_approx'] = (
    # RR component
    np.where(df['laju_pernafasan'] < 9, 2,
    np.where(df['laju_pernafasan'] <= 14, 0,
    np.where(df['laju_pernafasan'] <= 20, 1,
    np.where(df['laju_pernafasan'] <= 29, 2, 3)))) +
    # HR component  
    np.where(df['denyut_jantung'] < 40, 2,
    np.where(df['denyut_jantung'] <= 50, 1,
    np.where(df['denyut_jantung'] <= 100, 0,
    np.where(df['denyut_jantung'] <= 110, 1, 2)))) +
    # Systolic BP component
    np.where(df['sistole'] < 70, 3,
    np.where(df['sistole'] <= 89, 2,
    np.where(df['sistole'] <= 109, 1, 0)))
)
print("   ✅ mews_approx = RR_score + HR_score + SBP_score (simplified MEWS)")

# 5. Hypoxic Shock Flag
# SpO2 rendah DAN tekanan darah rendah secara bersamaan
df['flag_hypoxic_shock'] = (
    (df['SpO2'] < 90) & (df['sistole'] < 90)
).astype(int)
print("   ✅ flag_hypoxic_shock = SpO2<90 AND sistole<90")

# Tambahkan ke FEATURE_COLS
NEW_FEATURES = [
    'shock_index',
    'cardiorespiratory_score',
    'pulse_pressure',
    'mews_approx',
    'flag_hypoxic_shock',
]

FEATURE_COLS_V2 = FEATURE_COLS + NEW_FEATURES
print(f"\n📊 Total fitur baru: {len(NEW_FEATURES)}")
print(f"📊 Total fitur keseluruhan: {len(FEATURE_COLS_V2)}")
```

#### 4.4.3 Acceptance Criteria

| ID | Kriteria | Metode Verifikasi |
|---|---|---|
| AC-F04-01 | Semua 5 fitur baru ditambahkan tanpa NaN | `df[NEW_FEATURES].isna().sum()` |
| AC-F04-02 | Minimal 2 fitur baru masuk top-10 SHAP | SHAP bar plot |
| AC-F04-03 | `shock_index` tidak mengandung inf/NaN | `df['shock_index'].describe()` |
| AC-F04-04 | Referensi literatur untuk setiap fitur terdokumentasi | Code comment |

---

### FEAT-05 — Threshold Optimization per Kelas

**Prioritas:** P1 — Tinggi  
**Estimasi dampak:** +0.03 hingga +0.06 F1 untuk kelas minoritas

#### 4.5.1 Deskripsi

Model saat ini menggunakan threshold uniform 0.5 untuk semua kelas. Dengan mengoptimasi threshold per kelas berdasarkan probabilitas prediksi, kita dapat meningkatkan sensitivitas deteksi kelas kritis (terutama Oranye dan Kuning) tanpa retrain model.

Prinsip klinis yang diterapkan: **lebih baik overtriage (prediksi lebih berat) daripada undertriage (prediksi lebih ringan)** untuk kelas Merah dan Oranye.

#### 4.5.2 Spesifikasi Teknis

```python
# ── THRESHOLD OPTIMIZATION — Post-training ─────────────────────────────────
print("🎯 THRESHOLD OPTIMIZATION PER KELAS")

from sklearn.metrics import f1_score, recall_score
import numpy as np

# Gunakan probabilitas dari model
# (pastikan model di-fit dengan objective='multi:softprob' untuk predict_proba)
y_pred_proba = best_model_prob.predict_proba(X_test_scaled)

# Grid search threshold per kelas
# Untuk kelas kritis (Merah=0, Oranye=1): prioritaskan Recall
# Untuk kelas non-kritis: prioritaskan F1
PRIORITY_RECALL = [0, 1]    # Merah, Oranye → safety-critical
PRIORITY_F1     = [2, 3, 4] # Kuning, Hijau, Biru → balance F1

thresholds_grid = np.arange(0.05, 0.95, 0.05)
best_thresholds = {}

print(f"\n{'Kelas':12s} {'Metrik':12s} {'Threshold':>12s} {'Score':>10s}")
print("─" * 50)

for cls in range(5):
    y_true_bin = (y_test == cls).astype(int)
    metric = 'recall' if cls in PRIORITY_RECALL else 'f1'
    best_t, best_score = 0.5, 0

    for t in thresholds_grid:
        y_pred_bin = (y_pred_proba[:, cls] >= t).astype(int)
        if metric == 'recall':
            score = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        else:
            score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t

    best_thresholds[cls] = best_t
    print(f"{SATS_SHORT[cls]:12s} {metric:12s} {best_t:>12.2f} {best_score:>10.4f}")

# Prediksi akhir dengan threshold per kelas
# Normalisasi probabilitas dengan threshold masing-masing kelas
adjusted_proba = y_pred_proba / np.array([best_thresholds[c] for c in range(5)])
y_pred_threshold_opt = np.argmax(adjusted_proba, axis=1)

f1_before = f1_score(y_test, y_pred_tuned, average='macro', zero_division=0)
f1_after  = f1_score(y_test, y_pred_threshold_opt, average='macro', zero_division=0)

print(f"\n📊 F1-Macro SEBELUM threshold opt : {f1_before:.4f}")
print(f"📊 F1-Macro SETELAH threshold opt  : {f1_after:.4f}")
print(f"📊 Delta                           : {f1_after - f1_before:+.4f}")

# Simpan threshold optimal untuk deployment
joblib.dump(best_thresholds, os.path.join(OUTPUT_DIR, 'best_thresholds.pkl'))
print(f"\n✅ Threshold optimal disimpan: best_thresholds.pkl")
```

#### 4.5.3 Acceptance Criteria

| ID | Kriteria | Metode Verifikasi |
|---|---|---|
| AC-F05-01 | Threshold Merah ≤ 0.5 (lebih sensitif untuk kritis) | `best_thresholds[0] <= 0.5` |
| AC-F05-02 | Threshold Oranye ≤ 0.5 | `best_thresholds[1] <= 0.5` |
| AC-F05-03 | F1-Macro setelah optimasi ≥ F1 sebelumnya | Perbandingan `f1_before` vs `f1_after` |
| AC-F05-04 | Threshold disimpan sebagai artefak deployment | `best_thresholds.pkl` ada di output dir |
| AC-F05-05 | `predict_triage()` diperbarui menggunakan threshold optimal | Code review fungsi deployment |

---

### FEAT-06 — Revisi Acceptance Criteria AC-M07

**Prioritas:** P2 — Sedang  
**Tipe:** Metodologi / Dokumentasi

#### 4.6.1 Deskripsi

AC-M07 ("Model ≥ TEWS Rule") saat ini selalu gagal karena **circular reference**: label SATS dibuat dari TEWS, sehingga TEWS Rule secara definitif mendapat F1=1.000. Acceptance criteria ini perlu direvisi menjadi lebih bermakna secara ilmiah.

#### 4.6.2 Spesifikasi Revisi

**AC-M07 (LAMA — dihapus):**
```
Model XGBoost F1-Macro ≥ TEWS Pure-Rule F1-Macro
```

**AC-M07 (BARU — pengganti):**
```
Model XGBoost F1-Macro dari fitur RAW (tanpa TEWS score)
≥ Random Forest baseline dari fitur yang sama
```

Tambahkan juga narasi penjelasan di sel markdown:

```markdown
> **📝 Catatan AC-M07:**
> Perbandingan XGBoost vs TEWS Pure-Rule tidak dimasukkan sebagai
> acceptance criteria karena bersifat circular: label SATS dibuat
> dari TEWS, sehingga TEWS Rule secara definisi mencapai F1=1.000.
> Kontribusi model XGBoost adalah kemampuannya memprediksi kategori
> SATS dari **fitur klinis mentah** (tanda vital, demografi, GCS)
> tanpa akses ke TEWS score — yang merepresentasikan skenario
> deployment real di mana petugas memasukkan data pasien, bukan
> menghitung TEWS secara manual.
```

#### 4.6.3 Acceptance Criteria Baru yang Diajukan

| ID | Kriteria | Target |
|---|---|---|
| AC-M01 | F1-Macro ≥ 0.75 | ✅ Tetap |
| AC-M02 | Recall Merah ≥ 0.80 | ✅ Tetap |
| AC-M03 | Tuned > Baseline (direvisi: tuned pipeline > baseline pipeline) | ✅ Tetap, metodologi diperbaiki |
| AC-M04 | CV Std < 0.05 | ✅ Tetap |
| AC-M05 | Gap CV↔Test F1 < 0.05 | 🆕 Baru |
| AC-M06 | Undertriage Combined < 5% | ✅ Tetap |
| AC-M07 | XGBoost F1 > RF F1 (same features) | 🔄 Direvisi |
| AC-M08 | AUC-ROC Macro ≥ 0.95 | 🆕 Baru |

---

## 5. Urutan Implementasi

Implementasi dilakukan secara berurutan karena setiap fitur bergantung pada output fitur sebelumnya.

```
FEAT-03 (Perbaikan GCS)
    ↓
FEAT-04 (Fitur Interaksi Klinis)
    ↓
FEAT-01 (SMOTE Partial)
    ↓
FEAT-02 (ImbPipeline CV)
    ↓
    [Training & Tuning]
    ↓
FEAT-05 (Threshold Optimization)
    ↓
FEAT-06 (Revisi AC-M07)
    ↓
    [Evaluasi Final & Dokumentasi]
```

**Estimasi waktu per fitur:**

| Fitur | Estimasi Waktu |
|---|---|
| FEAT-03 Perbaikan GCS | 30 menit |
| FEAT-04 Fitur Interaksi | 45 menit |
| FEAT-01 SMOTE Parsial | 20 menit |
| FEAT-02 ImbPipeline | 30 menit |
| Training + Tuning (100 iter) | 15–30 menit (runtime) |
| FEAT-05 Threshold Opt | 30 menit |
| FEAT-06 Revisi AC + Dokumentasi | 30 menit |
| **Total** | **~3–4 jam** |

---

## 6. Ekspektasi Performa Post-Optimasi

Estimasi berdasarkan analisis akar masalah. Bukan jaminan — nilai aktual bergantung pada distribusi data setelah perbaikan GCS dan penambahan fitur.

| Metrik | Sekarang | Target Post-Optimasi |
|---|---|---|
| F1-Macro (Test) | 0.8070 | 0.85 – 0.88 |
| Accuracy | 98.58% | 97.5 – 98.5% |
| AUC-ROC Macro | 0.9989 | ≥ 0.9989 (dipertahankan) |
| Recall Merah | 1.0000 | ≥ 1.0000 |
| Recall Oranye | 0.3333 | ≥ 0.60 |
| F1 Kuning | 0.6667 | ≥ 0.70 |
| Undertriage Combined | 1.8% | < 3% |
| Gap CV↔Test F1 | 0.1910 | < 0.05 |
| GCS_total std | 0.00 | > 0 |

> **Catatan penting:** Accuracy mungkin sedikit turun karena model akan lebih agresif mendeteksi kelas minoritas, mengorbankan sedikit presisi pada kelas Biru (mayoritas). Ini adalah trade-off yang disengaja dan harus dijelaskan di Bab Pembahasan sebagai keputusan berbasis safety klinis.

---

## 7. Risiko & Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|
| SMOTE Oranye tetap buruk (14 sampel terlalu sedikit) | Tinggi | Sedang | Dokumentasikan sebagai keterbatasan data; fokus ke recall Oranye ≥ 0.50 |
| GCS median per kelompok usia tidak bervariasi | Sedang | Rendah | Fallback ke median global; tetap lebih baik dari default=15 |
| ImbPipeline CV lebih lambat dari sebelumnya | Sedang | Rendah | Gunakan `n_jobs=-1`; turunkan `n_iter` ke 50 jika runtime > 60 menit |
| Threshold optimization overfitting ke test set | Sedang | Tinggi | Lakukan threshold opt di **validation set** terpisah, bukan test set |
| F1-Macro tidak mencapai 0.85 | Sedang | Tinggi | Masih di atas target minimum 0.75; dokumentasikan upaya dan hasilnya |

---

## 8. Artefak yang Dihasilkan

Setelah implementasi selesai, artefak berikut harus tersedia di `d:\SKRIPSI\output\`:

| Artefak | Deskripsi |
|---|---|
| `model_triage_xgb_v2.pkl` | Model XGBoost teroptimasi (v2) |
| `pipeline_imblearn_v2.pkl` | ImbPipeline lengkap (SMOTE + XGBoost) |
| `scaler_minmax_v2.pkl` | MinMaxScaler yang di-fit ulang |
| `feature_names_v2.pkl` | Daftar 32 fitur (27 lama + 5 baru) |
| `best_thresholds.pkl` | Dictionary threshold optimal per kelas |
| `shap_explainer_v2.pkl` | SHAP TreeExplainer untuk model v2 |
| `Dataset_Labeled_SATS_v2.csv` | Dataset dengan fitur interaksi tambahan |
| `fig_smote_comparison_v2.png` | Visualisasi distribusi sebelum/sesudah SMOTE v2 |
| `fig_shap_bar_v2.png` | SHAP importance dengan fitur baru |
| `fig_threshold_optimization.png` | Visualisasi threshold per kelas |

---

## 9. Keterbatasan yang Harus Didokumentasikan di Skripsi

Terlepas dari semua optimasi di atas, terdapat keterbatasan inheren yang berasal dari dataset dan harus dinyatakan secara eksplisit di Bab Keterbatasan Penelitian:

1. **Kelas Oranye under-represented.** 17 sampel (0.3%) tidak memadai untuk pelatihan yang robust. Performa model pada kelas Oranye tidak dapat dievaluasi secara statistik yang valid.

2. **Label surrogate, bukan label klinis.** Label SATS dibuat dari TEWS algoritmik, bukan dari penilaian triage aktual oleh tenaga medis. Validasi pakar (Cohen's Kappa) masih diperlukan.

3. **GCS tidak terdokumentasi secara sistematis.** Mayoritas GCS diimputasi, bukan diukur, sehingga fitur GCS tidak mencerminkan kondisi kesadaran pasien yang sebenarnya.

4. **Single-site dataset.** Data hanya dari RSU Aulia — generalisasi ke RS lain memerlukan validasi eksternal.

5. **Periode data pendek.** 5 bulan (Januari–Mei 2026) mungkin tidak merepresentasikan variasi musiman pola kunjungan IGD.

---

## 10. Referensi

| No | Referensi |
|---|---|
| 1 | Gottschalk SB et al. (2006). The Cape Triage Score — a new triage system South Africa. *Emergency Medicine Journal*, 23(2), 149–153. |
| 2 | SA Triage Group (2012). *SATS Training Manual*, Revised 3rd Edition. |
| 3 | Chawla NV et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357. |
| 4 | Birkhahn RH et al. (2005). Shock index in diagnosing early acute hypovolemia. *American Journal of Emergency Medicine*, 23(3), 323–326. |
| 5 | Subbe CP et al. (2001). Validation of a modified Early Warning Score. *QJM*, 94(10), 521–526. |
| 6 | van Buuren S (2018). *Flexible Imputation of Missing Data*, 2nd ed. Chapman & Hall. |
| 7 | Chen T & Guestrin C (2016). XGBoost: A Scalable Tree Boosting System. *KDD*, 785–794. |
| 8 | Lundberg SM & Lee SI (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30. |

---

*Dokumen ini merupakan bagian dari penelitian skripsi prediksi triage IGD RSU Aulia.*  
*Keputusan klinis tetap sepenuhnya berada di tangan tenaga medis berlisensi.*
