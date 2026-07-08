# PRD-FEAT-007 — Penghapusan Fitur `skala_nyeri` dari Model dan UI

| Field | Value |
|---|---|
| **Dokumen** | PRD-FEAT-007 |
| **Versi** | v1.0 |
| **Notebook Target** | `Klasifikasi_Triage_IGD_XGBoost_SHAP_SATS_CRISPDM_V4.ipynb` → output `V5` |
| **Keputusan** | Opsi A — Hapus dari Model, FEATURE_COLS, `predict_triage_v4()`, dan UI |
| **Prioritas** | P1 (memerlukan re-train penuh) |
| **Estimasi Waktu** | 3–5 jam (termasuk re-train RandomizedSearchCV) |
| **Status** | APPROVED — siap implementasi |

---

## 1. Executive Summary

`skala_nyeri` saat ini ada di `FEATURE_COLS` dan diterima sebagai input di `predict_triage_v4()`, namun **tidak pernah digunakan dalam konstruksi label `sats_label`** dan **tidak ada dalam formula TEWS**. Akibatnya, XGBoost secara internal memberikan importance ~0 pada fitur ini karena tidak ada korelasi statistik antara `skala_nyeri` dan label target.

**Dampak langsung:** pengguna yang menginput nyeri 10/10 mendapatkan hasil klasifikasi identik dengan pengguna yang menginput nyeri 0/10 — dengan semua parameter lain sama. Ini menciptakan ekspektasi yang salah dan merusak kepercayaan pengguna terhadap sistem DSS.

**Keputusan:** Opsi A dipilih — hapus `skala_nyeri` dari seluruh pipeline: FEATURE_COLS, model (re-train), fungsi prediksi, test cases, dan UI. Artifact model lama (v4) tidak lagi valid setelah perubahan ini dan harus digantikan dengan artifact v5.

---

## 2. Root Cause Analysis

### 2.1 Mengapa `skala_nyeri` Zero-Impact

**Layer 1 — Label Construction** (`assign_sats_label`, Section 3.7)

```python
# Kondisi saat ini — skala_nyeri TIDAK ADA di sini
def assign_sats_label(row):
    gcs  = row['GCS_total']
    sbp  = row['sistole']
    spo2 = row['SpO2']
    rr   = row['laju_pernafasan']
    tews = row['TEWS_total']

    if gcs <= 8:                             return 0  # Merah
    if sbp < 70:                             return 0  # Merah
    if spo2 < 90 and (rr < 9 or rr >= 30): return 0  # Merah
    if tews >= 7:   return 1  # Oranye
    elif tews >= 5: return 2  # Kuning
    elif tews >= 3: return 3  # Hijau
    else:           return 4  # Biru
```

Dua pasien dengan vital sign identik tetapi `skala_nyeri` berbeda (0 vs 10) akan mendapatkan **label SATS yang persis sama**. XGBoost tidak dapat mempelajari hubungan yang tidak ada di label.

**Layer 2 — TEWS Formula** (`compute_tews`, Section 3.6)

TEWS dihitung dari: SpO2, laju pernapasan, tekanan darah, denyut jantung, suhu, GCS. `skala_nyeri` **bukan komponen TEWS** — sesuai standar klinis SATS-TEWS internasional.

**Layer 3 — XGBoost Feature Importance**

Karena label tidak berkorelasi dengan `skala_nyeri`, algoritma gradient boosting akan menghasilkan split gain = 0 untuk fitur ini. Feature importance dan SHAP value untuk `skala_nyeri` mendekati nol di seluruh kelas.

### 2.2 Konsekuensi Status Quo (jika dibiarkan)

| Konsekuensi | Dampak |
|---|---|
| User input nyeri 10, dapat hasil Biru | Ekspektasi salah, potensi miskomunikasi klinis |
| Penguji sidang bertanya "kenapa ada fitur yang tidak berpengaruh?" | Sulit dijustifikasi tanpa analisis importance tertulis |
| SHAP plot menampilkan fitur dengan importance ~0 | Noise di visualisasi, mengurangi clarity eksplanasi |
| Artifact model v4 menyimpan `feature_names_v4.pkl` dengan `skala_nyeri` | Inkonsistensi jika UI sudah diubah tapi model belum |

---

## 3. Scope Perubahan

### 3.1 Komponen yang Berubah

| Komponen | File/Section | Jenis Perubahan | Re-train? |
|---|---|---|---|
| `FEATURE_COLS` | Section 3.10 | Hapus satu entry | Ya (downstream) |
| `CLIP_BOUNDS` | Section 3.4 | Hapus entry `skala_nyeri` | Tidak |
| `CLIP_BOUNDS` (post-impute) | Section 3.5 | Hapus entry `skala_nyeri` | Tidak |
| `X`, `X_train`, `X_test` | Section 3.10 | Kolom berkurang 1 | Ya |
| `scaler` (MinMaxScaler) | Section 3.11 | Re-fit tanpa `skala_nyeri` | Ya |
| `SMOTE` + `ImbPipeline` | Section 4.1–4.2 | Re-train dari awal | **Ya** |
| `RandomizedSearchCV` | Section 4.2 | Re-run | **Ya** |
| `best_pipeline` | Section 4.2 | Artifact baru | **Ya** |
| `predict_triage_v4()` | Section 6.2 | Hapus parameter input | Tidak |
| `test_patients` | Section 6.2 | Hapus `skala_nyeri` dari dict | Tidak |
| SHAP explainer | Section 5.3 | Re-compute | **Ya** |
| Semua artifact `.pkl` | Section 6.1 | Regenerate semua | **Ya** |
| Judul notebook | Section 1 | Update versi ke v5 | Tidak |
| `feature_names_v4.pkl` | Section 6.1 | Ganti `feature_names_v5.pkl` | Tidak |

### 3.2 Komponen yang TIDAK Berubah

- `assign_sats_label()` — tidak ada hubungannya dengan `skala_nyeri`
- `compute_tews()` — TEWS formula tidak berubah
- `VITAL_COLS` — `skala_nyeri` bukan vital sign (tidak ada di imputasi MICE)
- `IterativeImputer` — scope imputer adalah `VITAL_COLS`, bukan `skala_nyeri`
- Distribusi label `sats_label` — tidak berubah
- Hyperparameter search space — tidak berubah
- `SMOTE_SAMPLING_STRATEGY` — tidak berubah
- Semua analisis di Section 2 (EDA) dan Section 3.1–3.9 (preprocessing)

---

## 4. Spesifikasi Teknis

### 4.1 FEAT-007-A — Hapus dari CLIP_BOUNDS (Section 3.4 & 3.5)

**Lokasi:** Section 3.4 (Zero → NaN & Clip) dan Section 3.5 (post-imputation clip)

**Sebelum (v4):**
```python
CLIP_BOUNDS = {
    'sistole': (60, 200), 'diastole': (30, 130), 'denyut_jantung': (30, 250),
    'laju_pernafasan': (4, 60), 'suhu_tubuh': (34.0, 42.0), 'SpO2': (70, 100),
    'skala_nyeri': (0, 10),   # ← HAPUS BARIS INI
}
```

**Sesudah (v5):**
```python
CLIP_BOUNDS = {
    'sistole': (60, 200), 'diastole': (30, 130), 'denyut_jantung': (30, 250),
    'laju_pernafasan': (4, 60), 'suhu_tubuh': (34.0, 42.0), 'SpO2': (70, 100),
    # skala_nyeri dihapus — FEAT-007: tidak digunakan dalam label SATS-TEWS
}
```

> **Catatan:** `skala_nyeri` tidak ada di `VITAL_COLS` sehingga tidak masuk ke `IterativeImputer`. Entry `CLIP_BOUNDS` untuk `skala_nyeri` hanya melakukan clip kolom raw — aman untuk dihapus karena kolom tersebut juga akan dihapus dari `FEATURE_COLS`.

---

### 4.2 FEAT-007-B — Hapus dari FEATURE_COLS (Section 3.10)

**Lokasi:** Section 3.10, definisi `FEATURE_COLS`

**Sebelum (v4):**
```python
FEATURE_COLS = [
    'usia_tahun', 'kelompok_usia', 'jenis_kelamin_enc',
    'GCS_E', 'GCS_M', 'GCS_V', 'GCS_total',
    'skala_nyeri',      # ← HAPUS BARIS INI
    'sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2',
    'MAP',
    'flag_takikardia', 'flag_bradikardia', 'flag_hipotensi', 'flag_hipertensi',
    'flag_takipnea', 'flag_hipoksia', 'flag_demam', 'flag_hipotermi',
    'n_vital_abnormal',
    'shock_index', 'cardiorespiratory_score', 'pulse_pressure', 'mews_approx', 'flag_hypoxic_shock',
]
```

**Sesudah (v5):**
```python
FEATURE_COLS = [
    'usia_tahun', 'kelompok_usia', 'jenis_kelamin_enc',
    'GCS_E', 'GCS_M', 'GCS_V', 'GCS_total',
    # skala_nyeri dihapus (FEAT-007) — zero information gain terhadap label SATS-TEWS
    'sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2',
    'MAP',
    'flag_takikardia', 'flag_bradikardia', 'flag_hipotensi', 'flag_hipertensi',
    'flag_takipnea', 'flag_hipoksia', 'flag_demam', 'flag_hipotermi',
    'n_vital_abnormal',
    'shock_index', 'cardiorespiratory_score', 'pulse_pressure', 'mews_approx', 'flag_hypoxic_shock',
]
```

**Dampak downstream yang otomatis mengikuti perubahan ini:**

- `X = df[FEATURE_COLS]` → shape berubah dari `(N, n+1)` ke `(N, n)` di mana n = jumlah fitur sebelumnya
- `X_train`, `X_test` → kolom berkurang 1
- `scaler.fit_transform(X_train)` → scaler fitted tanpa `skala_nyeri`
- Print statement `Total fitur v2:` → update label menjadi `Total fitur v5:` dan angkanya berkurang 1

**Wajib update print statement:**
```python
# Sebelum:
print(f"📊 Total fitur v2: {len(FEATURE_COLS)}")
print(f"   Base: {len(FEATURE_COLS) - len(cara_datang_cols) - 5} + Interaksi: 5 + Cara Datang: {len(cara_datang_cols)}")

# Sesudah:
print(f"📊 Total fitur v5: {len(FEATURE_COLS)}")
print(f"   Base: {len(FEATURE_COLS) - len(cara_datang_cols) - 5} + Interaksi: 5 + Cara Datang: {len(cara_datang_cols)}")
print(f"   ℹ️  skala_nyeri dihapus (FEAT-007): zero information gain terhadap label SATS-TEWS")
```

---

### 4.3 FEAT-007-C — Re-Train Model (Section 4.1–4.2)

Tidak ada perubahan pada kode training. Re-train terjadi secara otomatis karena `FEATURE_COLS` sudah berubah. Yang perlu diperhatikan:

**Ekspektasi perubahan performa:**

| Metrik | Ekspektasi | Justifikasi |
|---|---|---|
| F1-Macro | Tidak berubah signifikan (±0.005) | Fitur dengan importance ~0 tidak berkontribusi |
| AUC-ROC | Tidak berubah signifikan | Sama |
| Recall Merah/Oranye | Tidak berubah | Fitur tidak digunakan dalam split tree |
| Waktu training | Sedikit lebih cepat (~1-3%) | Dimensi input berkurang 1 |

> **Catatan risiko:** Jika performa berubah signifikan (>0.01 F1-Macro), ini mengindikasikan ada interaksi tidak terduga. Dokumentasikan dan bandingkan dengan angka v4 secara eksplisit di Section 4.4 (tabel perbandingan model). Jangan suppress perbedaan ini — justru ini menjadi temuan menarik untuk dibahas di skripsi.

**Tambahkan verifikasi FEAT-007 setelah training:**
```python
# ============================================================
# VERIFIKASI FEAT-007 — skala_nyeri tidak ada di feature names
# ============================================================
assert 'skala_nyeri' not in FEATURE_COLS, \
    "FEAT-007 FAIL: skala_nyeri masih ada di FEATURE_COLS"
assert 'skala_nyeri' not in best_pipeline.feature_names_in_ \
    if hasattr(best_pipeline, 'feature_names_in_') else True, \
    "FEAT-007 FAIL: skala_nyeri masih ada di pipeline feature names"
print(f"✅ FEAT-007: skala_nyeri tidak ada di FEATURE_COLS ({len(FEATURE_COLS)} fitur)")
print(f"✅ FEAT-007: Model di-train tanpa skala_nyeri")
```

---

### 4.4 FEAT-007-D — Update `predict_triage_v4()` (Section 6.2)

Fungsi harus diupdate di dua tempat: (1) hapus penerimaan `skala_nyeri` dari `input_data`, (2) hapus dari `feature_dict`.

**Sebelum (v4):**
```python
def predict_triage_v4(input_data, pipeline, scaler, feature_cols, thresholds=None, explainer=None):
    # ...
    feature_dict = {
        'usia_tahun': usia, 'kelompok_usia': kp_usia, 'jenis_kelamin_enc': jk_enc,
        'GCS_E': gcs_e, 'GCS_M': gcs_m, 'GCS_V': gcs_v, 'GCS_total': gcs_total,
        'skala_nyeri': input_data['skala_nyeri'],   # ← HAPUS BARIS INI
        'sistole': sbp, 'diastole': dbp, 'denyut_jantung': hr,
        # ...
    }
```

**Sesudah (v5):**
```python
def predict_triage_v5(input_data, pipeline, scaler, feature_cols, thresholds=None, explainer=None):
    """
    Fungsi klasifikasi triage v5 untuk Streamlit deployment.
    
    CHANGELOG v5 (FEAT-007):
    - skala_nyeri dihapus dari input dan feature_dict
    - Alasan: label surrogate SATS-TEWS tidak menggunakan parameter nyeri,
      sehingga fitur ini memiliki zero information gain terhadap label target
    - Referensi: SATS-TEWS Clinical Protocol; PRD-FEAT-007
    
    Parameters:
        input_data   : dict — TIDAK lagi menerima 'skala_nyeri'
        pipeline     : ImbPipeline — best_pipeline dari training v5
        scaler       : MinMaxScaler — fitted tanpa skala_nyeri
        feature_cols : list[str] — FEATURE_COLS v5 (tanpa skala_nyeri)
        thresholds   : dict | None
        explainer    : shap.TreeExplainer | None
    
    Returns: dict hasil klasifikasi
    """
    # Demografis
    usia = input_data['usia_tahun']
    kp_usia = (0 if usia < 1 else 1 if usia < 12 else 2 if usia < 18 else
               3 if usia < 45 else 4 if usia < 60 else 5 if usia < 75 else 6)
    jk_enc = 1 if input_data.get('jenis_kelamin', 'L') == 'L' else 0

    # Vital signs
    sbp  = input_data['sistole']
    dbp  = input_data['diastole']
    hr   = input_data['denyut_jantung']
    rr   = input_data['laju_pernafasan']
    temp = input_data['suhu_tubuh']
    spo2 = input_data['SpO2']
    gcs_e, gcs_m, gcs_v = input_data['GCS_E'], input_data['GCS_M'], input_data['GCS_V']
    gcs_total = gcs_e + gcs_m + gcs_v

    # MAP
    MAP = dbp + (sbp - dbp) / 3

    # Binary flags
    flags = {
        'flag_takikardia':  int(hr   > 100), 'flag_bradikardia': int(hr < 60),
        'flag_hipotensi':   int(sbp  < 90),  'flag_hipertensi':  int(sbp > 160),
        'flag_takipnea':    int(rr   > 20),  'flag_hipoksia':    int(spo2 < 95),
        'flag_demam':       int(temp > 37.5),'flag_hipotermi':   int(temp < 36.0),
    }
    n_vital_abnormal = sum(flags.values())

    # FEAT-04: Fitur Interaksi Klinis
    shock_index             = np.clip(hr / sbp if sbp > 0 else 1.0, 0, 5)
    cardiorespiratory_score = (flags['flag_takikardia'] + flags['flag_takipnea'] +
                               flags['flag_hipoksia']   + flags['flag_hipotensi'])
    pulse_pressure          = np.clip(sbp - dbp, 0, 120)
    rr_s  = (2 if rr < 9 else 0 if rr <= 14 else 1 if rr <= 20 else 2 if rr <= 29 else 3)
    hr_s  = (2 if hr < 40 else 1 if hr <= 50 else 0 if hr <= 100 else 1 if hr <= 110 else 2)
    sbp_s = (3 if sbp < 70 else 2 if sbp <= 89 else 1 if sbp <= 109 else 0)
    mews_approx             = rr_s + hr_s + sbp_s
    flag_hypoxic_shock      = int(spo2 < 90 and sbp < 90)

    # Cara datang
    cara = input_data.get('cara_datang', 'Sendiri')
    cara_dummies_dict = {
        'cara_datang_Dokter':    int(cara == 'Dokter'),
        'cara_datang_KLL':       int(cara == 'KLL'),
        'cara_datang_Puskesmas': int(cara == 'Puskesmas'),
        'cara_datang_Sendiri':   int(cara == 'Sendiri'),
    }

    # FEAT-007: skala_nyeri TIDAK ada di feature_dict
    feature_dict = {
        'usia_tahun': usia, 'kelompok_usia': kp_usia, 'jenis_kelamin_enc': jk_enc,
        'GCS_E': gcs_e, 'GCS_M': gcs_m, 'GCS_V': gcs_v, 'GCS_total': gcs_total,
        # skala_nyeri dihapus — FEAT-007
        'sistole': sbp, 'diastole': dbp, 'denyut_jantung': hr,
        'laju_pernafasan': rr, 'suhu_tubuh': temp, 'SpO2': spo2,
        'MAP': MAP, 'n_vital_abnormal': n_vital_abnormal,
        'shock_index': shock_index, 'cardiorespiratory_score': cardiorespiratory_score,
        'pulse_pressure': pulse_pressure, 'mews_approx': mews_approx,
        'flag_hypoxic_shock': flag_hypoxic_shock,
    }
    feature_dict.update(flags)
    feature_dict.update(cara_dummies_dict)

    X_input = pd.DataFrame([feature_dict])
    for col in feature_cols:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input  = X_input[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=feature_cols)

    proba = pipeline.predict_proba(X_scaled)[0]
    if thresholds is not None:
        adjusted   = proba / np.array([thresholds.get(c, 0.5) for c in range(5)])
        pred_class = int(np.argmax(adjusted))
    else:
        pred_class = int(np.argmax(proba))

    return {
        'predicted_class':          pred_class,
        'predicted_label':          SATS_SHORT[pred_class],
        'predicted_name':           SATS_NAMES[pred_class],
        'probabilities':            {SATS_SHORT[i]: float(p) for i, p in enumerate(proba)},
        'shock_index':              shock_index,
        'mews_approx':              mews_approx,
        'cardiorespiratory_score':  cardiorespiratory_score,
        'thresholds_used':          thresholds,
    }
```

---

### 4.5 FEAT-007-E — Update Test Cases (Section 6.2)

**Sebelum (v4):**
```python
test_patients = [
    {'nama': 'Pasien A (Kritis)', 'data': {
        'usia_tahun': 65, 'jenis_kelamin': 'L', 'cara_datang': 'Sendiri',
        'GCS_E': 2, 'GCS_M': 3, 'GCS_V': 2,
        'skala_nyeri': 8,       # ← HAPUS
        'sistole': 80, 'diastole': 50, 'denyut_jantung': 130,
        'laju_pernafasan': 32, 'suhu_tubuh': 38.8, 'SpO2': 88
    }},
    {'nama': 'Pasien B (Stabil)', 'data': {
        'usia_tahun': 30, 'jenis_kelamin': 'P', 'cara_datang': 'Sendiri',
        'GCS_E': 4, 'GCS_M': 6, 'GCS_V': 5,
        'skala_nyeri': 2,       # ← HAPUS
        'sistole': 120, 'diastole': 80, 'denyut_jantung': 75,
        'laju_pernafasan': 18, 'suhu_tubuh': 36.5, 'SpO2': 98
    }},
]
```

**Sesudah (v5):**
```python
test_patients = [
    {'nama': 'Pasien A (Kritis)', 'data': {
        'usia_tahun': 65, 'jenis_kelamin': 'L', 'cara_datang': 'Sendiri',
        'GCS_E': 2, 'GCS_M': 3, 'GCS_V': 2,
        # skala_nyeri dihapus — FEAT-007
        'sistole': 80, 'diastole': 50, 'denyut_jantung': 130,
        'laju_pernafasan': 32, 'suhu_tubuh': 38.8, 'SpO2': 88
    }},
    {'nama': 'Pasien B (Stabil)', 'data': {
        'usia_tahun': 30, 'jenis_kelamin': 'P', 'cara_datang': 'Sendiri',
        'GCS_E': 4, 'GCS_M': 6, 'GCS_V': 5,
        # skala_nyeri dihapus — FEAT-007
        'sistole': 120, 'diastole': 80, 'denyut_jantung': 75,
        'laju_pernafasan': 18, 'suhu_tubuh': 36.5, 'SpO2': 98
    }},
]

# Update pemanggilan fungsi
for patient in test_patients:
    result = predict_triage_v5(patient['data'], best_pipeline, scaler, FEATURE_COLS, best_thresholds)
```

---

### 4.6 FEAT-007-F — Update Artifact Names (Section 6.1)

Semua artifact harus di-regenerate dan disimpan dengan suffix `_v5` untuk menghindari ambiguitas dengan artifact v4 yang masih memuat `skala_nyeri`.

**Sebelum (v4):**
```python
model_path     = os.path.join(OUTPUT_DIR, 'model_triage_xgb_v4.pkl')
pipeline_path  = os.path.join(OUTPUT_DIR, 'pipeline_imblearn_v4.pkl')
scaler_path    = os.path.join(OUTPUT_DIR, 'scaler_minmax_v4.pkl')
features_path  = os.path.join(OUTPUT_DIR, 'feature_names_v4.pkl')
thresh_path    = os.path.join(OUTPUT_DIR, 'best_thresholds_v4.pkl')
explainer_path = os.path.join(OUTPUT_DIR, 'shap_explainer_v4.pkl')
params_path    = os.path.join(OUTPUT_DIR, 'best_params_v4.pkl')
imputer_path   = os.path.join(OUTPUT_DIR, 'imputer_v4.pkl')
```

**Sesudah (v5):**
```python
# FEAT-007: Artifact v5 — model di-train tanpa skala_nyeri
model_path     = os.path.join(OUTPUT_DIR, 'model_triage_xgb_v5.pkl')
pipeline_path  = os.path.join(OUTPUT_DIR, 'pipeline_imblearn_v5.pkl')
scaler_path    = os.path.join(OUTPUT_DIR, 'scaler_minmax_v5.pkl')
features_path  = os.path.join(OUTPUT_DIR, 'feature_names_v5.pkl')
thresh_path    = os.path.join(OUTPUT_DIR, 'best_thresholds_v5.pkl')
explainer_path = os.path.join(OUTPUT_DIR, 'shap_explainer_v5.pkl')
params_path    = os.path.join(OUTPUT_DIR, 'best_params_v5.pkl')
imputer_path   = os.path.join(OUTPUT_DIR, 'imputer_v5.pkl')
split_path     = os.path.join(OUTPUT_DIR, 'split_indices_v5.pkl')

# Tambahkan verifikasi feature list setelah save
assert 'skala_nyeri' not in joblib.load(features_path), \
    "FEAT-007 FAIL: skala_nyeri masih ada di feature_names_v5.pkl"
print("✅ FEAT-007: feature_names_v5.pkl tidak mengandung skala_nyeri")
```

---

### 4.7 FEAT-007-G — Update UI Input (Streamlit / Tab Prediksi)

Jika sudah ada komponen Streamlit atau rencana UI, hapus field input `skala_nyeri` sepenuhnya. Tidak ada penggantian dengan field lain dan tidak ada nilai default yang dikirim ke backend.

**Input fields yang DIHAPUS dari UI:**

```python
# HAPUS dari Tab 1 (Input Pasien):
skala_nyeri = st.slider(
    "Skala Nyeri (NRS)",
    min_value=0, max_value=10, value=0, step=1,
    help="..."
)

# HAPUS dari payload yang dikirim ke predict_triage_v5():
input_data = {
    # ...
    # 'skala_nyeri': skala_nyeri,   ← HAPUS
    # ...
}
```

**Penanganan backward compatibility:**

Jika `input_data` yang diterima fungsi masih mengandung key `skala_nyeri` (misalnya dari form lama atau API call lama), fungsi `predict_triage_v5()` cukup mengabaikannya karena key tersebut tidak digunakan dalam `feature_dict`. Tidak perlu raise error — cukup log warning:

```python
# Optional: tambahkan di awal predict_triage_v5()
if 'skala_nyeri' in input_data:
    import warnings
    warnings.warn(
        "FEAT-007: 'skala_nyeri' diterima di input_data tapi diabaikan. "
        "Hapus key ini dari input untuk menghindari kebingungan.",
        DeprecationWarning, stacklevel=2
    )
```

---

### 4.8 FEAT-007-H — Update Dokumentasi Keterbatasan (Section 5.10)

Tambahkan atau update poin L-07 di Section 5.10:

```markdown
### L-07: Skala Nyeri Tidak Dimodelkan (FEAT-007)

Parameter skala nyeri (NRS 0–10) tidak dimasukkan dalam model klasifikasi v5.

**Alasan teknis:**
Label surrogate `sats_label` dikonstruksi dari aturan SATS-TEWS yang tidak
memasukkan parameter nyeri. Akibatnya, `skala_nyeri` memiliki zero information
gain terhadap label target — XGBoost tidak dapat mempelajari hubungan yang
tidak ada dalam data label. Feature importance dan SHAP value untuk `skala_nyeri`
mendekati nol di semua kelas.

**Implikasi klinis:**
Sistem ini tidak dapat mendeteksi kegawatan yang dimediasi nyeri pada pasien
dengan tanda vital yang masih dalam batas kompensasi normal. Contoh: nyeri dada
10/10 dengan tekanan darah dan denyut jantung masih normal akan diklasifikasikan
sebagai Biru atau Hijau, yang mungkin tidak sesuai dengan penilaian klinis dokter.

**Mitigasi yang disarankan:**
Tenaga medis tetap perlu melakukan asesmen nyeri independen sebagai bagian dari
protokol triage. Sistem ini berfungsi sebagai DSS berbasis tanda vital objektif,
bukan pengganti asesmen nyeri subjektif.

**Referensi:** PRD-FEAT-007 v1.0
```

---

## 5. Acceptance Criteria

### 5.1 Acceptance Criteria Model

| ID | Kriteria | Cara Verifikasi |
|---|---|---|
| AC-F07-M01 | `skala_nyeri` tidak ada di `FEATURE_COLS` | `assert 'skala_nyeri' not in FEATURE_COLS` |
| AC-F07-M02 | `skala_nyeri` tidak ada di `feature_names_v5.pkl` | `assert 'skala_nyeri' not in joblib.load(features_path)` |
| AC-F07-M03 | Jumlah fitur di `FEATURE_COLS` = jumlah v4 dikurangi 1 | `len(FEATURE_COLS) == len(FEATURE_COLS_v4) - 1` |
| AC-F07-M04 | `scaler` fitted tanpa kolom `skala_nyeri` | `'skala_nyeri' not in scaler.feature_names_in_` (jika sklearn ≥ 1.0) |
| AC-F07-M05 | Re-training berhasil tanpa error | Notebook eksekusi penuh tanpa exception |
| AC-F07-M06 | Delta F1-Macro antara v4 dan v5 ≤ 0.01 | Dokumentasikan angka v4 dan v5 berdampingan |
| AC-F07-M07 | `best_pipeline` bisa predict tanpa `skala_nyeri` di input | `predict_triage_v5(test_case_tanpa_nyeri, ...)` tidak error |
| AC-F07-M08 | Semua artifact tersimpan dengan suffix `_v5` | `os.path.exists(features_path)` untuk setiap `_v5.pkl` |

### 5.2 Acceptance Criteria Fungsi Prediksi

| ID | Kriteria | Cara Verifikasi |
|---|---|---|
| AC-F07-P01 | `predict_triage_v5()` tidak menerima `skala_nyeri` sebagai required input | Eksekusi dengan dict tanpa key `skala_nyeri` → tidak KeyError |
| AC-F07-P02 | `feature_dict` di dalam fungsi tidak mengandung key `skala_nyeri` | Code review |
| AC-F07-P03 | Fungsi dipanggil dengan nama `predict_triage_v5` (bukan v4) | Code review — grep untuk `predict_triage_v4` tidak ada di test section |
| AC-F07-P04 | Docstring fungsi mencantumkan FEAT-007 dan alasan penghapusan | Code review |
| AC-F07-P05 | Jika input dict berisi `skala_nyeri`, fungsi tidak crash (diabaikan atau warning) | Uji dengan `{'skala_nyeri': 10, ...}` → tidak raise exception |

### 5.3 Acceptance Criteria UI

| ID | Kriteria | Cara Verifikasi |
|---|---|---|
| AC-F07-U01 | Tidak ada field input `skala_nyeri` di UI / form prediksi | Visual review UI |
| AC-F07-U02 | Payload yang dikirim ke `predict_triage_v5()` tidak mengandung key `skala_nyeri` | Code review payload dict |
| AC-F07-U03 | Tidak ada label, slider, input, atau referensi nyeri di Tab 1 | Visual + code review |

### 5.4 Acceptance Criteria Dokumentasi

| ID | Kriteria | Cara Verifikasi |
|---|---|---|
| AC-F07-D01 | Section 5.10 mengandung poin L-07 tentang penghapusan `skala_nyeri` | Code review markdown cell |
| AC-F07-D02 | Tabel perbandingan performa menampilkan F1-Macro v4 vs v5 berdampingan | Lihat output Section 4.4 |
| AC-F07-D03 | Header/judul notebook sudah update ke `V5` | Visual check |
| AC-F07-D04 | Komentar `FEAT-007` ada di setiap titik perubahan kode | Code review |

---

## 6. Risiko dan Mitigasi

### 6.1 Risiko Teknis

| Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|
| F1-Macro berubah >0.01 setelah re-train | Rendah (fitur importance ~0) | Sedang (perlu penjelasan) | Dokumentasikan sebelum dan sesudah; jika berubah, jadikan temuan menarik di bab analisis |
| RandomizedSearchCV menemukan hyperparameter berbeda | Sedang (search adalah stochastic) | Rendah (performa setara) | RANDOM_STATE=42 dipertahankan; catat `best_params_v5` untuk dibandingkan dengan `best_params_v4` |
| SHAP values bergeser untuk fitur lain | Rendah | Rendah | Re-generate SHAP plot — nilai redistribusi minor adalah normal |
| Artifact v4 tidak sengaja di-load untuk v5 model | Sedang (human error) | Tinggi (silent wrong prediction) | Suffix `_v5` wajib; tambahkan assert di load: `assert 'skala_nyeri' not in feature_names` |

### 6.2 Risiko Akademik

| Risiko | Mitigasi |
|---|---|
| Penguji bertanya "mengapa fitur dihapus di v5?" | Jawab: feature importance ~0, label tidak menggunakan nyeri, penghapusan meningkatkan clarity model dan konsistensi UI |
| Penguji meminta bukti importance ~0 | Siapkan SHAP bar chart v4 yang menunjukkan `skala_nyeri` di posisi terbawah sebelum dihapus |
| Penguji menilai penghapusan sebagai "data dredging" | Jelaskan bahwa ini adalah keputusan yang didasarkan pada analisis domain klinis (SATS-TEWS tidak memasukkan nyeri), bukan dari melihat angka performa |

---

## 7. Rencana Implementasi

### 7.1 Urutan Pengerjaan

```
Fase 1 — Persiapan (15 menit)
├── 1.1  Backup notebook V4 → simpan salinan sebagai referensi
├── 1.2  Catat semua angka performa V4 (F1-Macro, Recall, AUC) untuk perbandingan
└── 1.3  Buka notebook V4 untuk diedit menjadi V5

Fase 2 — Perubahan Preprocessing (20 menit)
├── 2.1  Hapus 'skala_nyeri' dari CLIP_BOUNDS (Section 3.4)  [FEAT-007-A]
└── 2.2  Hapus 'skala_nyeri' dari CLIP_BOUNDS post-impute (Section 3.5)  [FEAT-007-A]

Fase 3 — Perubahan Feature Engineering (10 menit)
├── 3.1  Hapus 'skala_nyeri' dari FEATURE_COLS (Section 3.10)  [FEAT-007-B]
└── 3.2  Update print statement total fitur  [FEAT-007-B]

Fase 4 — Re-Train (30–90 menit, tergantung mesin)
└── 4.1  Run Section 3.10 → 4.2 (split, impute, scale, SMOTE, train, tune)  [FEAT-007-C]

Fase 5 — Perubahan Deployment (20 menit)
├── 5.1  Update predict_triage_v4() → predict_triage_v5()  [FEAT-007-D]
├── 5.2  Update test_patients dict (hapus skala_nyeri)  [FEAT-007-E]
└── 5.3  Update artifact names ke _v5  [FEAT-007-F]

Fase 6 — Update UI (15 menit)
└── 6.1  Hapus field skala_nyeri dari form prediksi  [FEAT-007-G]

Fase 7 — Dokumentasi (20 menit)
├── 7.1  Tambahkan L-07 di Section 5.10  [FEAT-007-H]
├── 7.2  Tambahkan tabel perbandingan v4 vs v5
└── 7.3  Update judul notebook → V5

Fase 8 — Verifikasi AC (15 menit)
└── 8.1  Jalankan semua assertion AC-F07-M01 s/d AC-F07-D04
```

### 7.2 Total Estimasi Waktu

| Fase | Estimasi |
|---|---|
| Persiapan + Preprocessing + Feature | 45 menit |
| Re-train (RandomizedSearchCV 100 iter × 5 fold) | 30–90 menit |
| Deployment + UI | 35 menit |
| Dokumentasi + Verifikasi | 35 menit |
| **Total** | **~2.5–4.5 jam** |

---

## 8. Tabel Perbandingan Performa yang Harus Diisi

Isi tabel ini setelah re-training v5 selesai:

| Metrik | v4 (dengan skala_nyeri) | v5 (tanpa skala_nyeri) | Delta | Status |
|---|---|---|---|---|
| F1-Macro | _isi setelah run v4_ | _isi setelah run v5_ | — | — |
| Accuracy | — | — | — | — |
| Balanced Accuracy | — | — | — | — |
| MCC | — | — | — | — |
| Recall Merah | — | — | — | — |
| Recall Oranye | — | — | — | — |
| AUC-ROC Macro | — | — | — | — |
| Undertriage Combined | — | — | — | — |
| CV Mean F1 | — | — | — | — |
| CV Std | — | — | — | — |
| Jumlah Fitur | n (dengan nyeri) | n-1 (tanpa nyeri) | -1 | Expected |

> Tabel ini wajib ada di Section 4.4 notebook v5 sebagai bukti bahwa penghapusan `skala_nyeri` tidak mendegradasi performa.

---

## 9. Changelog

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 | 2026-06-23 | Initial PRD — Opsi A approved, spesifikasi lengkap |

---

*PRD-FEAT-007 v1.0 — Klasifikasi Triage IGD RSU Aulia — XGBoost + SHAP + SATS-TEWS v5*
