# 🔬 Analisis Hasil Model ML Prediksi Triage IGD
## XGBoost + SHAP + SATS-TEWS | RSU Aulia

> [!CAUTION]
> **MASALAH KRITIS TERDETEKSI** — Hasil model `F1-Macro = 1.0000` dan `Accuracy = 1.0000` pada **semua kelas** adalah sinyal **data leakage** yang serius. Ini bukan performa sempurna yang sesungguhnya dan **WAJIB diselesaikan sebelum skripsi diajukan**.

---

## 📊 Rekap Hasil Model

| Metrik | Baseline | Tuned (RandomizedSearchCV) | CV Mean |
|--------|----------|---------------------------|---------|
| Accuracy | 1.0000 | 1.0000 | — |
| F1-Macro | 1.0000 | 1.0000 | 0.9941 |
| Precision (W) | 1.0000 | 1.0000 | — |
| Recall (W) | 1.0000 | 1.0000 | — |
| CV Std | 0.0241 | **0.0119** | — |

**Distribusi kelas (imbalanced):**
| Kelas | Jumlah | % |
|-------|--------|---|
| 🔴 Merah (Resusitasi) | 554 | 8.7% |
| 🟠 Oranye (Emergent) | 17 | 0.3% |
| 🟡 Kuning (Urgent) | 92 | 1.5% |
| 🟢 Hijau (Less Urgent) | 306 | 4.8% |
| 🔵 Biru (Not Urgent) | 5,370 | **84.7%** |

---

## 🚨 MASALAH KRITIS — Harus Diperbaiki

### 1. DATA LEAKAGE — Root Cause Utama Perfection Score

**Diagnosis: Model mendapat skor 1.0 karena label target (`sats_label`) dibangun DARI fitur-fitur yang juga masuk ke model.**

Ini adalah **circular logic / tautologi**:

```
TEWS_total    (fitur) ──┐
TEWS_rr_score (fitur) ──┤
TEWS_bp_score (fitur) ──┼──→ sats_label (target) ──→ model belajar rule deterministik
GCS_total     (fitur) ──┤
sistole       (fitur) ──┘
```

- Label `sats_label` dibuat via fungsi `assign_sats_label()` yang menggunakan `TEWS_total`, `GCS_total`, `sistole`, `SpO2`, `laju_pernafasan`
- Semua variabel tersebut **juga masuk sebagai fitur** ke XGBoost
- Model XGBoost cukup belajar rule deterministic `TEWS_total >= 7 → Oranye`, dll → F1 = 1.0
- Ini bukan prediksi yang generalisable ke data baru di dunia nyata

**Bukti dari output SHAP:**
```
1. TEWS_total    : 1.8735  ← sangat dominan (50x lipat fitur kedua)
2. sistole       : 0.6942  ← komponen utama TEWS_bp_score
3. diastole      : 0.0462
```
TEWS_total mendominasi dengan gap ekstrem → model hanya mengenal rule labeling, bukan pola klinis independen.

**Bukti dari GCS:**
```
TEWS_gcs_score: mean=0.00, std=0.00, max=0.00
GCS_total     : semua 15.00 (6339 records)
```
Seluruh pasien memiliki GCS = 15 (normal), sehingga tidak ada variasi GCS yang bisa dipelajari model.

---

### 2. IMBALANCE KELAS EKSTREM — Tidak Ditangani

| Kelas | Support Test | % |
|-------|-------------|---|
| Oranye | **3** | 0.2% |
| Kuning | 19 | 1.5% |
| Merah | 111 | 8.8% |
| Biru | **1,074** | 84.7% |

- Kelas Oranye hanya memiliki **3 sampel** pada test set → tidak dapat dievaluasi secara statistik
- Tidak ada penanganan imbalance (SMOTE, class_weight, threshold adjustment)
- Pada kondisi normal (bukan data leakage), F1 kelas minority akan sangat rendah

> [!IMPORTANT]
> Skripsi harus mendiskusikan dan mencoba minimal satu teknik: `class_weight='balanced'`, SMOTE, atau `scale_pos_weight` di XGBoost.

---

### 3. TEST SET LEAKAGE — MinMaxScaler

```
Range X_test_scaled:
   Min: 0.0000
   Max: 1.0460  ← MELEBIHI 1.0 → ada nilai test > nilai training max
```

Nilai `1.0460 > 1.0` menunjukkan bahwa scaler yang di-fit pada training set tidak cukup mewakili rentang data test. Ini normal dalam pengertian teknis (scaler hanya fit di train), namun menunjukkan bahwa beberapa outlier di test set melebihi max training — perlu didokumentasikan.

---

### 4. GCS TIDAK BERVARIASI — Fitur Tidak Informatif

```
GCS_total distribusi:
15.00    6339   ← semua pasien GCS = 15 (NORMAL semua)
TEWS_gcs_score: mean=0.00, std=0.00
```

- **Seluruh 6,339 pasien memiliki GCS = 15** — ini tidak realistis secara klinis
- Kemungkinan besar GCS tidak dicatat untuk pasien ringan (default 15)
- GCS_total, GCS_E, GCS_M, GCS_V tidak memberikan informasi diskriminatif sama sekali
- Ini harus didiskusikan sebagai **limitasi** dalam skripsi

---

### 5. MISSING DATA SANGAT TINGGI — ≈33% Tanda Vital

```
sistole      : 2105 (33.2%) missing → diimputasi
diastole     : 2115 (33.4%) missing → diimputasi
denyut_jantung: 2069 (32.6%) missing → diimputasi
laju_pernafasan: 2073 (32.7%) missing → diimputasi
suhu_tubuh   : 2078 (32.8%) missing → diimputasi
SpO2         : 2073 (32.7%) missing → diimputasi
```

- Lebih dari **1/3 data tanda vital tidak tercatat** — bias besar
- Kemungkinan pasien-pasien yang tidak diukur tanda vitalnya adalah pasien ringan (Biru)
- Imputed values via IterativeImputer mungkin menyebabkan TEWS tidak akurat untuk pasien ini
- **Analisis MCAR/MAR/MNAR** tidak dilakukan

---

## ⚠️ MASALAH PENTING — Perlu Diperjelas di Skripsi

### 6. Perbandingan XGBoost vs TEWS Pure-Rule Tidak Bermakna

```
TEWS Rule   : Accuracy=1.0000, F1-Macro=1.0000
XGBoost     : Accuracy=1.0000, F1-Macro=1.0000
Delta       : +0.0000
```

Karena label dibuat dari TEWS, **tentu saja TEWS pure-rule juga sempurna**. Perbandingan ini tidak membuktikan XGBoost lebih baik dari rule-based — justru membuktikan adanya data leakage.

### 7. Pasien A (Kritis) Diprediksi Oranye, bukan Merah

```
Pasien A: TEWS = 15, GCS <= 8 → harusnya MERAH (Resusitasi)
Prediksi model: 🟠 Oranye (Emergent) — SALAH secara klinis
```
- Model keliru mengklasifikasikan pasien kritis (TEWS=15) ke Oranye, bukan Merah
- Probabilitas Merah = 0.110, Oranye = 0.417
- Ini adalah **clinical plausibility failure** meski AC-M02 (Recall Merah) = 1.0 pada test set

### 8. SHAP Analysis Kurang Mendalam untuk Skripsi

**Yang ada:**
- ✅ Summary plot (bee swarm)
- ✅ Bar plot fitur importance
- ✅ Dependence plot
- ✅ Waterfall plot
- ✅ Force plot

**Yang kurang/perlu ditambah:**
- ❌ **Interpretasi naratif per kelas SATS** (apa yang membuat model mendeteksi Merah vs Biru)
- ❌ **SHAP interaction values** antara fitur kritis (TEWS × GCS, sistole × SpO2)
- ❌ **Contoh prediksi per kelas** dengan waterfall/force plot yang dibahas secara klinis
- ❌ **Perbandingan SHAP importance vs feature importance XGBoost internal** (gain, cover)

---

## 📋 Yang Sudah Baik (Kelebihan)

| Aspek | Status |
|-------|--------|
| CRISP-DM pipeline lengkap (6 fase) | ✅ |
| Zero-encoded → NaN handling | ✅ |
| IterativeImputer (MICE) | ✅ |
| Clinical bounds clipping | ✅ |
| Stratified 80:20 split (anti data leakage scaler) | ✅ |
| TEWS kalkulasi sesuai SA Triage Group 2012 | ✅ |
| Feature engineering (MAP, flags, n_vital_abnormal) | ✅ |
| RandomizedSearchCV 100 iter × 5-Fold CV | ✅ |
| Undertriage analysis (patient safety) | ✅ |
| TreeSHAP explainability (5 tipe plot) | ✅ |
| Model deployment + artefak (.pkl) | ✅ |
| SHAP values disimpan: (1268, 34, 5) | ✅ |

---

## 🔧 Rekomendasi Perbaikan untuk Skripsi

### Prioritas KRITIS (Harus Dilakukan)

1. **Pisahkan fitur dari label-building variables**
   
   **Opsi A — Hapus TEWS sub-scores dari fitur:** Biarkan model belajar dari raw vitals saja, TEWS digunakan hanya untuk labeling
   ```python
   # Hapus dari FEATURE_COLS:
   TEWS_COLS = ['TEWS_rr_score', 'TEWS_spo2_score', 'TEWS_bp_score',
                'TEWS_hr_score', 'TEWS_temp_score', 'TEWS_gcs_score', 'TEWS_total']
   ```
   
   **Opsi B — Gunakan label triage dari rekam medis asli** (jika tersedia) sebagai ground truth

2. **Tangani class imbalance:**
   ```python
   # Option 1: class_weight di XGBoost
   xgb = XGBClassifier(class_weight='balanced', ...)
   
   # Option 2: SMOTE sebelum training (hanya pada train set)
   from imblearn.over_sampling import SMOTE
   X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
   ```

### Prioritas TINGGI (Sangat Disarankan untuk Skripsi S1)

3. **Tambah analisis MCAR/MAR/MNAR untuk missing data** — apakah pasien yang tidak diukur tanda vitalnya random atau sistematis

4. **Diskusikan limitasi GCS** — 100% pasien GCS=15, tidak ada variasi → fitur ini tidak berguna dan harus didiskusikan

5. **Tambah ROC-AUC per kelas (One-vs-Rest)** sebagai metrik tambahan yang lebih robust untuk imbalanced data

6. **Tambah Learning Curve** untuk membuktikan model tidak overfitting

### Prioritas MENENGAH (Memperkuat Skripsi)

7. **Tambah confusion matrix normalisasi (%)** agar lebih mudah dibaca

8. **Interpretasi klinis SHAP per kelas** — "Fitur apa yang membuat model memprediksi pasien sebagai Merah?"

9. **Analisis kesalahan (Error Analysis)** — pasien mana yang salah diklasifikasikan dan mengapa

10. **Tambah tabel hyperparameter search space** vs best parameter yang dipilih

---

## 📐 Checklist Kelengkapan untuk Skripsi

### Bab 4 — Hasil & Pembahasan

- [ ] Jelaskan mengapa F1=1.0 terjadi (data leakage dari TEWS → label → fitur)
- [ ] Diskusikan distribusi kelas yang sangat imbalanced (Oranye=0.3%)
- [ ] Tampilkan tabel missing value rate ≈33% dan justifikasi IterativeImputer
- [ ] Diskusikan GCS tidak bervariasi sebagai limitasi dataset
- [ ] Sajikan CV scores per fold (bukan hanya mean) → variabilitas antar fold
- [ ] Interpretasi SHAP: mengapa TEWS_total dominan (SHAP=1.87 vs sistole=0.69)
- [ ] Analisis clinical plausibility: apakah top SHAP features masuk akal klinis?
- [ ] Bandingkan dengan penelitian sejenis (referensi jurnal triage ML)

### Bab 5 — Kesimpulan & Saran

- [ ] Sebutkan limitasi model (data leakage, GCS uniform, high missing rate)
- [ ] Saran: validasi eksternal dengan data baru
- [ ] Saran: ground truth dari label triage asli (bukan konstruksi TEWS)
- [ ] Saran: integrasi ke sistem informasi RS (deployment readiness)

---

## 📚 Referensi Tambahan yang Direkomendasikan

Tambahkan ke daftar pustaka untuk memperkuat argumentasi:

1. **Data Leakage in ML**: Kaufman et al. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." *ACM TKDD*
2. **SHAP Clinical**: Lundberg SM et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*
3. **Triage ML**: Levin S et al. (2018). "Machine-Learning-Based Electronic Triage More Accurately Differentiates Patients with Respect to Clinical Outcomes." *Annals of Emergency Medicine*
4. **Class Imbalance**: Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*
5. **MICE Imputation**: van Buuren S (2018). *Flexible Imputation of Missing Data*. Chapman & Hall
