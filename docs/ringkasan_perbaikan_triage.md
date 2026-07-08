# 📋 Ringkasan Perbaikan — Skripsi Prediksi Triage IGD
### XGBoost + SHAP + SATS-TEWS | RSU Aulia

> [!CAUTION]
> **MASALAH KRITIS:** Hasil `F1-Macro = 1.0000` pada semua kelas merupakan sinyal **data leakage** — bukan performa sesungguhnya. Wajib diselesaikan sebelum sidang.

---

## 🔴 PRIORITAS 1 — Kritis (Wajib Selesai Sebelum Sidang)

### 1. Perbaiki Data Leakage (Root Cause F1 = 1.0)

**Masalah:** Label `sats_label` dibangun dari `TEWS_total`, `GCS_total`, `sistole`, `SpO2`, dan `laju_pernafasan` — namun **semua variabel itu juga masuk sebagai fitur ke model**. Model hanya menghafal rule deterministik, bukan belajar pola klinis.

**Bukti:** SHAP `TEWS_total = 1.8735` (50× lebih dominan dari fitur kedua `sistole = 0.694`). Perbandingan XGBoost vs TEWS pure-rule pun sama-sama 1.0 — tidak bermakna.

**Solusi — Pilih salah satu:**

- **Opsi A (Direkomendasikan):** Hapus seluruh TEWS scores dari `FEATURE_COLS`, biarkan model belajar dari raw vitals saja:
  ```python
  TEWS_COLS = ['TEWS_rr_score', 'TEWS_spo2_score', 'TEWS_bp_score',
               'TEWS_hr_score', 'TEWS_temp_score', 'TEWS_gcs_score', 'TEWS_total']
  FEATURE_COLS = [c for c in FEATURE_COLS if c not in TEWS_COLS]
  ```
- **Opsi B:** Gunakan label triage asli dari rekam medis (jika tersedia) sebagai *ground truth*

---

### 2. Tangani Class Imbalance Ekstrem

**Masalah:** Biru mendominasi **84.7%** data. Kelas Oranye hanya **3 sampel** di test set — tidak bisa dievaluasi statistik. Tidak ada penanganan imbalance sama sekali.

**Solusi — Terapkan minimal satu:**
```python
# Opsi 1: SMOTE (hanya pada training set)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

# Opsi 2: sample_weight di XGBoost
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y_train)
xgb.fit(X_train_scaled, y_train, sample_weight=weights)
```
Tampilkan distribusi sebelum dan sesudah resampling.

---

### 3. Jalankan Ulang Notebook & Simpan Semua Output

**Masalah (sebelum hasil ada):** Penguji perlu melihat angka aktual, bukan hanya kode. Pastikan semua sel sudah dieksekusi dan tersimpan outputnya.

**Output wajib tersimpan di notebook:**
- Nilai F1-Macro, Accuracy, Precision, Recall (baseline vs tuned)
- Classification report per kelas
- Confusion matrix (visual + normalisasi %)
- Distribusi label SATS
- Best hyperparameters dari RandomizedSearchCV
- Top 10 SHAP features dengan nilai SHAP

---

## 🟠 PRIORITAS 2 — Tinggi (Sangat Disarankan untuk Skripsi S1)

### 4. Tambah AUC-ROC per Kelas

F1-Macro saja tidak cukup untuk data imbalanced di konteks medis. Tambahkan:
```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_test, classes=[0,1,2,3,4])
auc = roc_auc_score(y_bin, y_pred_proba, average='macro', multi_class='ovr')
print(f"AUC-ROC Macro (OvR): {auc:.4f}")
```
Tampilkan AUC kelas Merah (Resusitasi) secara terpisah — ini sangat relevan klinis.

---

### 5. Tambah Model Pembanding (Komparasi Algoritma)

Tunjukkan XGBoost dipilih karena alasan objektif. Tambahkan minimal:
- **Random Forest** — paling relevan untuk dibandingkan
- **Logistic Regression** — sebagai simple baseline

Sajikan tabel perbandingan F1-Macro, Accuracy, dan training time.

---

### 6. Validasi Pakar (Cohen's Kappa)

Karena label SATS dibuat secara algoritmik (surrogate via TEWS), bukan dari label triage asli dokter, **validasi pakar adalah syarat wajib** untuk klaim validitas klinis.

- Validasi dengan dokter/perawat IGD RSU Aulia (sampel 50–100 kasus)
- Hitung **Cohen's Kappa** antara label TEWS-algoritmik vs penilaian pakar
- Target: Kappa > 0.60 (acceptable agreement)

Tanpa ini, penguji dapat mempertanyakan validitas seluruh dataset berlabel.

---

### 7. Diskusikan Limitasi GCS (Semua = 15)

**Masalah:** Seluruh 6.339 pasien memiliki GCS = 15 — tidak ada variasi. GCS_E, GCS_M, GCS_V, GCS_total dan TEWS_gcs_score semuanya `std = 0`, tidak informatif.

Kemungkinan penyebab: GCS hanya dicatat untuk pasien kritis, sisanya di-default 15. Diskusikan ini sebagai **limitasi dataset** dan justifikasi keputusan impute default GCS=15 dengan referensi atau sensitivity analysis.

---

### 8. Tambah Learning Curve

Buktikan model tidak overfitting:
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_scaled, y_train, cv=5, scoring='f1_macro'
)
```

---

### 9. Analisis Missing Data (MCAR/MAR/MNAR)

**Masalah:** ±33% tanda vital tidak tercatat di semua 6 variabel vital. Ini bias besar — kemungkinan pasien ringan (Biru) yang tidak diukur.

Tambahkan analisis: apakah pola missing bersifat **random (MCAR)** atau **sistematis (MAR/MNAR)**? Ini mempengaruhi validitas IterativeImputer.

---

## 🟡 PRIORITAS 3 — Menengah (Memperkuat Skripsi)

### 10. Interpretasi Klinis SHAP per Kelas
Saat ini SHAP plot ada tapi belum diinterpretasi secara naratif. Tambahkan penjelasan: *"Fitur apa yang membuat model memprediksi seorang pasien sebagai Merah? Apa yang membedakan Kuning dari Hijau?"* Sertakan contoh waterfall/force plot yang dibahas secara klinis per kelas.

### 11. Error Analysis (Analisis Kesalahan Prediksi)
Identifikasi pasien mana yang salah diklasifikasikan, dan analisis pola kesalahannya. Perhatikan juga: **Pasien A (TEWS=15) diprediksi Oranye, bukan Merah** — ini adalah clinical plausibility failure yang perlu dijelaskan.

### 12. Justifikasi MinMaxScaler
XGBoost berbasis pohon, **secara teori tidak memerlukan normalisasi**. Jika dipertahankan, tambahkan penjelasan bahwa scaler diterapkan untuk konsistensi pipeline deployment. Catatan: nilai test set `max = 1.046 > 1.0` perlu didokumentasikan.

### 13. Buat Streamlit Prototype
Fungsi `predict_triage()` sudah siap. Buat minimal 1 halaman Streamlit dan sertakan screenshot/demo video di skripsi — sering menjadi nilai plus signifikan saat sidang.

### 14. Tambah Tabel Search Space Hyperparameter
Tampilkan search space vs best parameter yang dipilih RandomizedSearchCV agar proses tuning lebih transparan dan reproducible.

---

## ✅ Yang Sudah Baik — Pertahankan

| Aspek | Keterangan |
|-------|-----------|
| CRISP-DM 6 fase | Pipeline terstruktur dan lengkap |
| Zero-encoded → NaN handling | Penanganan missing yang benar |
| IterativeImputer (MICE) | Imputation multivariate yang valid |
| Clinical bounds clipping | Berbasis batas fisiologis klinis |
| Stratified 80:20 split | Anti-leakage pada scaler |
| TEWS scoring (SA Triage 2012) | Referensi standar valid |
| Feature engineering (MAP, flags) | Menunjukkan domain knowledge klinis |
| RandomizedSearchCV 100 iter × 5-Fold | Tuning sistematis |
| Undertriage analysis (<5%) | Metrik patient safety yang cerdas |
| TreeSHAP (5 tipe plot) | Explainability lengkap |
| Model deployment + artefak .pkl | Siap deployment |
| Disclaimer medis konsisten | Etika penelitian yang baik |

---

## 📐 Checklist Skripsi

### Bab 4 — Hasil & Pembahasan
- [ ] Jelaskan data leakage & mengapa F1=1.0 terjadi (TEWS → label → fitur)
- [ ] Diskusikan distribusi kelas imbalanced (Oranye = 0.3%, Biru = 84.7%)
- [ ] Tampilkan tabel missing value ≈33% + justifikasi IterativeImputer + analisis MCAR/MAR
- [ ] Diskusikan GCS seragam = 15 sebagai limitasi dataset
- [ ] Sajikan CV scores per fold + mean + std
- [ ] Tabel perbandingan algoritma (XGBoost vs RF vs LR)
- [ ] AUC-ROC per kelas (khususnya kelas Merah)
- [ ] Confusion matrix normalisasi (%)
- [ ] Interpretasi naratif SHAP per kelas SATS
- [ ] Bandingkan dengan penelitian sejenis (referensi jurnal)

### Bab 5 — Kesimpulan & Saran
- [ ] Nyatakan limitasi: data leakage, GCS uniform, missing rate tinggi, label surrogate
- [ ] Saran: validasi pakar (Cohen's Kappa)
- [ ] Saran: ground truth dari label triage asli dokter
- [ ] Saran: validasi eksternal dengan data baru / RS lain
- [ ] Saran: integrasi ke sistem informasi RS

---

## 📚 Referensi Tambahan

| Topik | Referensi |
|-------|-----------|
| Data Leakage | Kaufman et al. (2012). *Leakage in Data Mining.* ACM TKDD |
| SHAP Klinis | Lundberg SM et al. (2020). *From local explanations to global understanding.* Nature MI |
| Triage ML | Levin S et al. (2018). *ML-Based Electronic Triage.* Annals of Emergency Medicine |
| Class Imbalance | Chawla et al. (2002). *SMOTE.* JAIR |
| MICE Imputation | van Buuren S (2018). *Flexible Imputation of Missing Data.* Chapman & Hall |
