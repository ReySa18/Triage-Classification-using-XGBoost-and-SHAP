# PRD: Prediksi Skor Triage IGD — XGBoost + SHAP | RSU Aulia | v1.1 | Juni 2026

---

**PRODUCT REQUIREMENTS DOCUMENT**

PRD v1.1

## Sistem Prediksi Skor Triage IGD
### Menggunakan XGBoost + Explainable AI (SHAP)

*RSU Aulia — Sistem Pendukung Keputusan Klinis IGD*

| **Atribut** | **Detail** |
|---|---|
| Proyek | Prediksi Skor Triage IGD — XGBoost + SHAP |
| Institusi | RSU Aulia |
| Peneliti | Reymondo |
| Metodologi | CRISP-DM (Cross-Industry Standard Process for Data Mining) |
| Versi PRD | v1.1 — SATS Reference Update |
| Tanggal Dokumen | Juni 2026 |
| Status | DRAFT — For Review |
| Tipe Sistem | ML Decision Support System (Prototype) |
| Standar Triage | **SATS (South African Triage Scale) — 5 Level** |

> **CONFIDENTIAL — INTERNAL RESEARCH USE ONLY**

---

# 1. EXECUTIVE SUMMARY

## 1.1 Ringkasan Proyek

Proyek ini membangun sistem prediksi skor triage berbasis machine learning untuk Instalasi Gawat Darurat (IGD) RSU Aulia. Sistem menggunakan algoritma XGBoost dengan pendekatan klasifikasi multi-kelas dan diintegrasikan dengan Explainable AI melalui SHAP (SHapley Additive exPlanations) untuk memberikan transparansi prediksi yang dapat dipahami tenaga medis.

Dataset aktual dari RSU Aulia terdiri dari **6.339 rekam medis kunjungan IGD** periode Januari–Mei 2026. Dataset mencakup 18 variabel raw yang setelah preprocessing akan menghasilkan fitur-fitur klinis relevan sebagai input model.

Sistem ini mengadopsi standar **SATS (South African Triage Scale)** sebagai framework klasifikasi triage, yang menggunakan **TEWS (Triage Early Warning Score)** berbasis tanda vital dan GCS sebagai dasar pembentukan label target.

## 1.2 Problem Statement

**Masalah inti:** Proses triage IGD di RSU Aulia saat ini bersifat manual dan subjektif, bergantung sepenuhnya pada penilaian individual tenaga medis. Hal ini berpotensi menghasilkan inkonsistensi antar-petugas, terutama pada kondisi beban kerja tinggi.

| **Dimensi Problem** | **Kondisi Saat Ini** | **Risiko** |
|---|---|---|
| Konsistensi | Tergantung pengalaman individu | Variasi skor antar-shift/petugas |
| Kecepatan | Manual assessment tiap pasien | Bottleneck di waktu sibuk |
| Dokumentasi | Tidak terstandarisasi | Audit trail terbatas |
| Akuntabilitas | Subyektif | Sulit dipertanggungjawabkan |
| Undertriage Risk | Tidak terukur sistemik | Pasien kritis tidak tertangani tepat waktu |

## 1.3 Proposed Solution

- Model XGBoost multi-kelas teroptimasi sebagai mesin prediksi skor triage
- SHAP (TreeSHAP) untuk explainability per prediksi — global dan lokal
- Prototipe Streamlit sebagai antarmuka decision support untuk tenaga medis IGD
- Pipeline CRISP-DM lengkap dari data preparation hingga deployment
- **Konstruksi label target menggunakan SATS-TEWS** (rule-based scoring berbasis tanda vital + GCS)

## 1.4 Critical Dataset Finding

> **⚠️ PERINGATAN KRITIS:** Dataset RSU Aulia yang ada (`Dataset_Klinis_Edit.csv`) **TIDAK memiliki kolom target triage eksplisit**. Variabel target (kategori triage) harus diturunkan/dikonstruksi melalui pendekatan berikut:
> - **(Rekomendasi Utama)** Rule-based labeling menggunakan **SATS-TEWS** (kalkulasi skor dari tanda vital dan GCS)
> - Labeling oleh dokter/pakar klinis IGD RSU Aulia sebagai validasi ground truth
> - Proxy labeling dari kode ICD-10 (alternatif terakhir)
>
> Keputusan ini merupakan **gating factor** untuk seluruh tahap modeling.

---

# 2. PROJECT OVERVIEW & OBJECTIVES

## 2.1 Latar Belakang

Triage adalah proses kritis di IGD yang menentukan urutan prioritas penanganan pasien berdasarkan derajat kegawatan kondisi medisnya. Keterlambatan atau kesalahan dalam menentukan kategori triage dapat berujung pada risiko klinis serius, termasuk undertriage (pasien kritis tidak mendapatkan penanganan cepat) atau overtriage (sumber daya habis untuk kasus non-kritis).

RSU Aulia memerlukan sistem pendukung keputusan berbasis data untuk meningkatkan objektivitas dan konsistensi proses triage IGD, sambil tetap mempertahankan kontrol keputusan di tangan tenaga medis terlatih.

**Standar triage yang diadopsi dalam proyek ini adalah SATS (South African Triage Scale)**, yang telah divalidasi secara klinis dan direkomendasikan oleh WHO untuk setting IGD dengan sumber daya terbatas di negara berkembang. SATS menggunakan TEWS (Triage Early Warning Score) sebagai mekanisme objektif skoring berbasis tanda vital, GCS, dan faktor trauma untuk menghasilkan 5 kategori triage.

## 2.2 Tujuan Bisnis

| **ID** | **Tujuan Bisnis** | **Metrik Keberhasilan** | **Prioritas** |
|---|---|---|---|
| BO-01 | Meningkatkan konsistensi triage lintas petugas | Variance skor triage antar-petugas menurun | HIGH |
| BO-02 | Menyediakan decision support berbasis data | Tool digunakan dalam >50% kasus triage | HIGH |
| BO-03 | Transparansi dan akuntabilitas keputusan triage | Setiap prediksi memiliki SHAP explanation | HIGH |
| BO-04 | Fondasi standardisasi triage RSU Aulia menggunakan SATS | SOPs baru berbasis model output selaras SATS | MEDIUM |
| BO-05 | Reduksi risiko undertriage/overtriage | Clinical error rate terukur dan menurun | HIGH |

## 2.3 Tujuan Teknis

| **ID** | **Tujuan Teknis** | **Target Kualitas** |
|---|---|---|
| TO-01 | Bangun model klasifikasi multi-kelas XGBoost | F1-Macro >= 0.75 pada test set |
| TO-02 | Optimasi hyperparameter via RandomizedSearchCV | 100 iterasi, 5-Fold CV, F1-macro objective |
| TO-03 | Integrasi SHAP (TreeSHAP) explainability | Summary, Bar, Dependence, Waterfall, Force Plot |
| TO-04 | Pipeline preprocessing anti-data-leakage | Stratified 80:20 split + 5-Fold CV validation |
| TO-05 | Konstruksi variabel target triage via **SATS-TEWS** | Label surrogate SATS (Merah/Oranye/Kuning/Hijau/Biru) |
| TO-06 | Deployment prototipe Streamlit | Functional prototype: input → TEWS score → prediksi → SHAP |

## 2.4 Ruang Lingkup (Scope)

### 2.4.1 In-Scope

- Dataset dari RSU Aulia: 6.339 rekam medis IGD (Januari–Mei 2026)
- Variabel klinis: usia, jenis kelamin, cara datang, GCS, skala nyeri, tanda vital (sistole, diastole, denyut jantung, laju pernapasan, suhu, SpO2)
- **Konstruksi variabel target triage menggunakan SATS-TEWS** (5 kategori: Merah/Oranye/Kuning/Hijau/Biru)
- **Kalkulasi TEWS score** sebagai intermediate feature (TEWS = sum of weighted vital sign sub-scores)
- Feature engineering: MAP, age grouping, abnormal vital flags
- Model XGBoost + hyperparameter tuning + SHAP analysis
- Prototipe Streamlit (bukan sistem produksi penuh)
- Dokumentasi CRISP-DM lengkap untuk skripsi

### 2.4.2 Out-of-Scope

- Integrasi dengan sistem HIS/SIMRS RSU Aulia yang ada
- Real-time data streaming dari perangkat medis
- Mobile application deployment
- Model untuk pasien pediatrik <1 tahun (edge case klinis — SATS memiliki SATS-Paeds secara terpisah)
- Prediksi prognosis atau outcome pasien jangka panjang
- Sistem produksi/live deployment (hanya prototipe penelitian)

---

# 3. ANALISIS DATASET — DATA UNDERSTANDING

## 3.1 Profil Dataset

| **Parameter** | **Nilai** | **Catatan** |
|---|---|---|
| Nama file | Dataset_Klinis_Edit.csv | Dataset aktual RSU Aulia |
| Total rekam medis | 6.339 baris | Periode Jan–Mei 2026 |
| Total variabel raw | 18 kolom | Sebelum feature engineering |
| Periode data | 1 Januari – 9 Mei 2026 | 129 hari unik |
| Variabel target | **TIDAK ADA — harus dikonstruksi via SATS-TEWS** | **CRITICAL ISSUE** |
| Format | CSV (comma-separated) | Encoding: UTF-8 |

## 3.2 Inventaris Variabel Dataset

| **Kolom** | **Tipe Raw** | **Deskripsi** | **Peran dalam Model** | **Isu Kualitas** |
|---|---|---|---|---|
| noreg | String | Nomor registrasi kunjungan | ID — exclude dari model | — |
| tgl_periksa | DateTime | Tanggal pemeriksaan | Fitur waktu (opsional) | — |
| rm | Integer | Nomor rekam medis pasien | ID — exclude dari model | — |
| pasien | String | Nama pasien | PII — exclude dari model | Privacy concern |
| usia | String | Usia pasien (format: Th/Bl/Hr) | Prediktor klinis | Perlu parsing numeric |
| jenis_kelamin | String | Laki-laki / Perempuan | Prediktor klinis | Encoding diperlukan |
| cara_datang | String | Metode kedatangan pasien | Prediktor klinis | Trailing whitespace |
| keluhan_utama | String | Keluhan utama teks bebas | NLP/drop/encode | 30,9% missing (1.958) |
| GCS | String | Skor Glasgow Coma Scale (teks) | **Prediktor klinis KRITIS + komponen TEWS** | Perlu ekstraksi E/M/V numeric |
| skala_nyeri | Integer | Skala nyeri 0-10 | Prediktor klinis | 7 outlier >10 (max=80) |
| sistole | Integer | Tekanan darah sistolik (mmHg) | **Prediktor klinis KRITIS + komponen TEWS** | 33,2% nilai 0 |
| diastole | Integer | Tekanan darah diastolik (mmHg) | **Prediktor klinis KRITIS + komponen TEWS** | 33,4% nilai 0 + outlier 853 |
| denyut_jantung | Integer | Denyut jantung (bpm) | **Prediktor klinis KRITIS + komponen TEWS** | 32,6% nilai 0 + outlier 870 |
| laju_pernafasan | Integer | Laju pernapasan (bpm) | **Prediktor klinis KRITIS + komponen TEWS** | 32,7% nilai 0 |
| suhu_tubuh | Float | Suhu tubuh (Celsius) | Prediktor klinis | 32,8% nilai 0 + outlier 380 |
| SpO2 | Float | Saturasi oksigen (%) | **Prediktor klinis KRITIS + komponen TEWS** | 32,7% nilai 0 + outlier 990 |
| kode_ICD | String | Kode diagnosa ICD-10 | Label proxy / exclude | 509 kode unik |
| Diagnosa | String | Nama diagnosa lengkap | Label proxy / exclude | 509 diagnosa unik |

## 3.3 Analisis Kualitas Data — Temuan Kritis

### 3.3.1 Missing Values & Zero-Encoded Data

Sekitar 32–34% dari seluruh rekam medis memiliki nilai 0 pada variabel tanda vital. Ini **BUKAN data valid** — nilai 0 untuk sistole, SpO2, atau denyut jantung secara klinis tidak mungkin pada pasien hidup. Ini mengindikasikan data tidak dicatat (missing) dan di-encode sebagai 0.

| **Variabel** | **Jumlah Nol** | **% Total** | **Interpretasi** | **Penanganan** |
|---|---|---|---|---|
| keluhan_utama | 1.958 | 30,9% | Tidak diisi petugas | Impute 'Tidak diketahui' atau drop |
| GCS | 75 | 1,2% | Tidak diisi | Impute dengan GCS normal (E4M6V5=15) → TEWS sub-score = 0 |
| sistole | 2.105 | 33,2% | Missing encoded as 0 | Treat as NaN → impute median kelas |
| diastole | 2.115 | 33,4% | Missing encoded as 0 | Treat as NaN → impute median kelas |
| denyut_jantung | 2.069 | 32,6% | Missing encoded as 0 | Treat as NaN → impute median kelas |
| laju_pernafasan | 2.073 | 32,7% | Missing encoded as 0 | Treat as NaN → impute median kelas |
| suhu_tubuh | 2.078 | 32,8% | Missing encoded as 0 | Treat as NaN → impute median kelas |
| SpO2 | 2.073 | 32,7% | Missing encoded as 0 | Treat as NaN → impute median kelas |

> **Catatan Penting untuk SATS-TEWS:** Kalkulasi TEWS harus dilakukan **setelah** imputation, bukan pada data raw. Record dengan >3 komponen TEWS missing (setelah imputation gagal) dieksklusi dari pembentukan label.

### 3.3.2 Outlier Ekstrem — Tanda Vital

| **Variabel** | **Range Valid Klinis** | **Nilai Max Dataset** | **Jumlah Outlier** | **Penanganan** |
|---|---|---|---|---|
| skala_nyeri | 0–10 | 80 (max raw) | 7 kasus >10 | Clip ke 10 |
| diastole | 40–130 mmHg | 853 mmHg | 4 kasus >200 | Clip batas klinis |
| denyut_jantung | 30–250 bpm | 870 bpm | 11 kasus >300 | Clip batas klinis |
| suhu_tubuh | 34–42°C | 380°C | 11 kasus >45 | Clip batas klinis |
| SpO2 | 70–100% | 990% | 25 kasus >100 | Clip ke 100 |

### 3.3.3 GCS — Parsing Kompleks

Kolom GCS berisi teks klinis lengkap. Contoh: `'Keadaan Umum GCS E : 4  M : 6 V : 5  Kepala: normocephali...'`. Perlu regex extraction untuk nilai E, M, V secara terpisah.

- **GCS-E** (Eye opening): nilai 1–4
- **GCS-M** (Motor response): nilai 1–6
- **GCS-V** (Verbal response): nilai 1–5
- **GCS Total** = E + M + V (range 3–15)

> Dalam konteks SATS-TEWS, **GCS Total** merupakan salah satu dari 6 parameter TEWS yang mendapatkan sub-score berbobot. GCS Total ≤ 13 mulai mendapatkan skor TEWS positif, dengan skor tertinggi untuk GCS ≤ 8.

### 3.3.4 Variabel Target — Masalah Kritis

**Dataset TIDAK memiliki kolom kategori triage.** Pendekatan yang direkomendasikan adalah konstruksi label surrogate menggunakan **SATS-TEWS**:

| **Opsi** | **Metode** | **Pros** | **Cons** | **Rekomendasi** |
|---|---|---|---|---|
| **A** | **Rule-based SATS-TEWS scoring** dari tanda vital + GCS | Terstandar klinis, reproducible, berbasis evidence | Tidak 100% mencerminkan keputusan dokter RSU Aulia | **REKOMENDASI UTAMA** |
| B | Manual labeling oleh dokter/pakar IGD RSU Aulia | Paling akurat, ground truth sebenarnya | Time-consuming, perlu resources tambahan | Ideal untuk validasi label SATS |
| C | Proxy dari kode ICD-10 (mapping ke triage level) | Tidak perlu labeling manual | Tidak langsung, akurasi dipertanyakan | Alternatif terakhir |

---

# 4. FUNCTIONAL REQUIREMENTS

## 4.1 Target Variable — Sistem Kategori Triage SATS

Sistem akan memprediksi **5 kategori triage** berdasarkan standar **SATS (South African Triage Scale)**. SATS menggunakan **TEWS (Triage Early Warning Score)** sebagai mekanisme objektif yang mengagregasikan sub-score dari 6 parameter klinis utama.

### 4.1.1 Kategori Triage SATS

| **Kategori** | **Kode SATS** | **Warna** | **Deskripsi Klinis** | **Waktu Penanganan Target** |
|---|---|---|---|---|
| **Resusitasi** | Red | 🔴 Merah | Mengancam jiwa segera. Henti jantung-napas, airway tidak aman, syok berat, GCS ≤ 8, SpO2 < 90% | **Segera (0 menit)** |
| **Emergent** | Orange | 🟠 Oranye | Kondisi tidak stabil, deteriorasi cepat. TEWS tinggi, distress pernapasan, nyeri dada | **≤ 10 menit** |
| **Urgent** | Yellow | 🟡 Kuning | Kondisi stabil namun memerlukan evaluasi segera. TEWS sedang, nyeri sedang-berat | **≤ 60 menit** |
| **Less Urgent** | Green | 🟢 Hijau | Kondisi tidak akut. TEWS rendah, nyeri ringan-sedang, tanda vital mendekati normal | **≤ 240 menit** |
| **Not Urgent** | Blue | 🔵 Biru | Kondisi minor/kronis. TEWS 0, semua tanda vital normal, keluhan ringan | **≤ 480 menit** |

### 4.1.2 TEWS — Mekanisme Skoring SATS

TEWS merupakan sistem skoring agregat dari 6 parameter klinis. Setiap parameter mendapat sub-score berdasarkan deviasi dari nilai normal. Total TEWS = jumlah semua sub-score.

**Tabel Sub-Score TEWS:**

| **Parameter** | **Nilai** | **Sub-Score** |
|---|---|---|
| **Laju Pernapasan (rpm)** | < 9 | 3 |
| | 9–11 | 1 |
| | 12–20 | 0 |
| | 21–29 | 2 |
| | ≥ 30 | 3 |
| **SpO2 (%)** | < 90 | 3 |
| | 90–94 | 2 |
| | 95–96 | 1 |
| | ≥ 97 | 0 |
| **Tekanan Darah Sistolik (mmHg)** | < 70 | 3 |
| | 70–89 | 2 |
| | 90–109 | 1 |
| | 110–149 | 0 |
| | 150–179 | 1 |
| | ≥ 180 | 2 |
| **Denyut Jantung (bpm)** | < 40 | 3 |
| | 40–50 | 1 |
| | 51–100 | 0 |
| | 101–110 | 1 |
| | 111–129 | 2 |
| | ≥ 130 | 3 |
| **Suhu Tubuh (°C)** | < 35 | 2 |
| | 35–38,4 | 0 |
| | ≥ 38,5 | 1 |
| **GCS Total** | ≤ 8 | 3 |
| | 9–12 | 2 |
| | 13–14 | 1 |
| | 15 | 0 |

**Mapping TEWS Total ke Kategori SATS:**

| **Total TEWS** | **Kategori SATS** | **Label Model** |
|---|---|---|
| Override klinis (henti napas/jantung, tidak sadar) | Resusitasi | 0 — Merah |
| ≥ 7 | Emergent | 1 — Oranye |
| 5–6 | Urgent | 2 — Kuning |
| 3–4 | Less Urgent | 3 — Hijau |
| 0–2 | Not Urgent | 4 — Biru |

> **Catatan Implementasi:** Override klinis (kategori Resusitasi) berlaku untuk kondisi: (a) GCS total ≤ 8 secara standalone, (b) sistole < 70 mmHg, (c) SpO2 < 90% dengan laju pernapasan < 9 atau > 30 rpm. Kondisi override selalu menghasilkan label Merah tanpa memperhatikan total TEWS.

## 4.2 Input Features — Model Prediktor

| **Feature Group** | **Variabel** | **Derivasi** | **Tipe Model** |
|---|---|---|---|
| Demografis | usia_tahun | Parse dari kolom 'usia' (regex Th/Bl/Hr) | Numeric (continuous) |
| Demografis | kelompok_usia | Binning: Bayi/Anak/Remaja/Dewasa/Lansia/Lansia Tua | Ordinal encoded |
| Demografis | jenis_kelamin_enc | Binary encoding: L=1, P=0 | Binary |
| Akses | cara_datang_enc | One-Hot: Sendiri/Puskesmas/Dokter/KLL | One-Hot |
| Kesadaran | GCS_E | Regex extract dari teks GCS | Numeric (1–4) |
| Kesadaran | GCS_M | Regex extract dari teks GCS | Numeric (1–6) |
| Kesadaran | GCS_V | Regex extract dari teks GCS | Numeric (1–5) |
| Kesadaran | GCS_total | GCS_E + GCS_M + GCS_V | Numeric (3–15) |
| Nyeri | skala_nyeri_clip | Clip ke [0, 10] | Numeric (0–10) |
| Vital Signs | sistole | Outlier clip: [60, 200] mmHg, 0→NaN→impute | Numeric |
| Vital Signs | diastole | Outlier clip: [30, 130] mmHg, 0→NaN→impute | Numeric |
| Vital Signs | denyut_jantung | Outlier clip: [30, 250] bpm, 0→NaN→impute | Numeric |
| Vital Signs | laju_pernafasan | Outlier clip: [4, 60] rpm, 0→NaN→impute | Numeric |
| Vital Signs | suhu_tubuh | Outlier clip: [34, 42] °C, 0→NaN→impute | Numeric |
| Vital Signs | SpO2 | Outlier clip: [70, 100] %, 0→NaN→impute | Numeric |
| **SATS-TEWS** | **TEWS_total** | **Kalkulasi sub-score SATS-TEWS (0–17)** | **Numeric (engineered)** |
| **SATS-TEWS** | **TEWS_rr_score** | Sub-score laju pernapasan (0–3) | Numeric |
| **SATS-TEWS** | **TEWS_spo2_score** | Sub-score SpO2 (0–3) | Numeric |
| **SATS-TEWS** | **TEWS_bp_score** | Sub-score tekanan darah sistolik (0–3) | Numeric |
| **SATS-TEWS** | **TEWS_hr_score** | Sub-score denyut jantung (0–3) | Numeric |
| **SATS-TEWS** | **TEWS_temp_score** | Sub-score suhu tubuh (0–2) | Numeric |
| **SATS-TEWS** | **TEWS_gcs_score** | Sub-score GCS total (0–3) | Numeric |
| Engineered | MAP | MAP = diastole + (sistole - diastole)/3 | Numeric |
| Engineered | flag_takikardia | denyut_jantung > 100 bpm → 1 | Binary |
| Engineered | flag_bradikardia | denyut_jantung < 60 bpm → 1 | Binary |
| Engineered | flag_hipotensi | sistole < 90 mmHg → 1 | Binary |
| Engineered | flag_hipertensi | sistole > 160 mmHg → 1 | Binary |
| Engineered | flag_takipnea | laju_pernafasan > 20 rpm → 1 | Binary |
| Engineered | flag_hipoksia | SpO2 < 95% → 1 | Binary |
| Engineered | flag_demam | suhu_tubuh > 37,5°C → 1 | Binary |
| Engineered | flag_hipotermi | suhu_tubuh < 36°C → 1 | Binary |
| Engineered | n_vital_abnormal | Sum semua flag vital abnormal (0–8) | Count (0–8) |

## 4.3 Model Requirements

| **Requirement ID** | **Deskripsi** | **Spesifikasi** | **Prioritas** |
|---|---|---|---|
| MR-01 | Algoritma ML | XGBoost (xgboost>=1.7), objective=multi:softmax | MUST HAVE |
| MR-02 | Jumlah kelas output | 5 kelas: Merah/Oranye/Kuning/Hijau/Biru (SATS) | MUST HAVE |
| MR-03 | Metrik optimasi tuning | F1-Score Macro Average | MUST HAVE |
| MR-04 | Hyperparameter tuning | RandomizedSearchCV, n_iter=100, cv=5-Fold Stratified | MUST HAVE |
| MR-05 | Data split | Stratified Hold-Out 80:20 (train/test) | MUST HAVE |
| MR-06 | Cross-validation | Stratified 5-Fold CV pada training set | MUST HAVE |
| MR-07 | Data leakage prevention | Scikit-learn Pipeline (fit hanya pada train) | MUST HAVE |
| MR-08 | Normalisasi | Min-Max Normalization untuk fitur numerik | MUST HAVE |
| MR-09 | Target encoding | Target Encoding untuk cara_datang (high-cardinality) | SHOULD HAVE |
| MR-10 | Model persistence | Simpan dalam format Joblib (.pkl) | MUST HAVE |
| MR-11 | Explainability — Global | SHAP Summary Plot, Bar Plot, Dependence Plot | MUST HAVE |
| MR-12 | Explainability — Lokal | SHAP Waterfall Plot dan Force Plot per prediksi | MUST HAVE |
| MR-13 | Baseline comparison | Bandingkan model default vs teroptimasi | MUST HAVE |
| MR-14 | Class imbalance handling | Analisis distribusi; pertimbangkan class_weight atau SMOTE | SHOULD HAVE |
| **MR-15** | **TEWS kalkulasi engine** | **Fungsi Python kalkulasi TEWS dari raw vitals (reproducible, unit-tested)** | **MUST HAVE** |
| **MR-16** | **TEWS sebagai fitur model** | **TEWS_total dan 6 sub-score TEWS dimasukkan sebagai input fitur XGBoost** | **MUST HAVE** |

## 4.4 Performance Requirements

| **Metrik** | **Target Minimum** | **Target Ideal** | **Justifikasi Klinis** |
|---|---|---|---|
| F1-Score Macro | >= 0.75 | >= 0.85 | Keseimbangan performa semua kelas triage SATS |
| F1-Score Merah (Resusitasi) | >= 0.80 | >= 0.90 | Undertriage kelas kritis = risiko kematian |
| F1-Score Oranye (Emergent) | >= 0.75 | >= 0.85 | High-risk emergent class |
| Accuracy (overall) | >= 0.78 | >= 0.88 | Benchmark performa umum |
| Precision Weighted | >= 0.78 | >= 0.87 | Minimasi false positives |
| Recall Weighted | >= 0.78 | >= 0.87 | Minimasi false negatives |
| Undertriage Rate (Merah/Oranye) | < 5% | < 2% | Standard safety klinis: SATS benchmark ≤ 5% undertriage |
| Inference Time (Streamlit) | < 500ms | < 200ms | Usability di IGD — real-time requirement |

---

# 5. CRISP-DM — SPESIFIKASI TEKNIS PIPELINE

## 5.1 Phase 1: Business Understanding

**Deliverable:** Problem definition, tujuan bisnis, tujuan data mining, rencana proyek. Dokumen ini (PRD) merupakan output utama fase ini.

| **Task** | **Output** | **Status** |
|---|---|---|
| Identifikasi kebutuhan klinis RSU Aulia | Problem statement terdokumentasi | DONE |
| Penetapan tujuan bisnis | Business objectives (PRD §2.2) | DONE |
| Penetapan tujuan data mining | Technical objectives (PRD §2.3) | DONE |
| Penilaian situasi (data, SDM, risiko) | Risk register dan constraint list | DONE |
| Perencanaan proyek CRISP-DM | Project plan & timeline | DONE |
| **Pemilihan standar triage** | **SATS dipilih sebagai referensi label surrogate** | **DONE** |

## 5.2 Phase 2: Data Understanding

| **Task** | **Metode/Tool** | **Output** | **Temuan Kritis** |
|---|---|---|---|
| Load & inspect dataset | pandas read_csv, df.info() | Data inventory | 18 col, 6.339 baris |
| Statistik deskriptif | df.describe(), df.dtypes | Summary statistics | Zero-values 33% |
| Missing value analysis | df.isnull().sum() + zero-masking | Missing value report | GCS: 75 null, Vital: ~33% |
| Distribusi target (awal) | value_counts() pada kode ICD | Class distribution proxy | 509 kode ICD unik |
| Outlier detection | Boxplot, IQR, clinical bounds | Outlier inventory | SpO2 max=990, DJ max=870 |
| GCS parsing analysis | regex pattern matching | GCS extraction prototype | E4M6V5 pattern dominan |
| Temporal analysis | groupby tgl_periksa | Visit trend by date | Jan-Mei 2026, 129 hari |
| **TEWS feasibility check** | **Cek completeness vital signs post-imputation** | **% record dengan TEWS calculable** | **Target >80% valid TEWS** |

## 5.3 Phase 3: Data Preparation

### 5.3.1 Data Cleaning Pipeline

| **Step** | **Operasi** | **Fungsi Python** | **Catatan** |
|---|---|---|---|
| 1 | Remove non-feature columns | df.drop(['noreg','rm','pasien'], axis=1) | PII + ID columns |
| 2 | Parse usia ke numeric (tahun) | regex r'(\d+) Th' + convert | Ekstrak tahun saja |
| 3 | Convert 0→NaN vital signs | df[vitals] = df[vitals].replace(0, np.nan) | Zero = missing |
| 4 | Clip outlier tanda vital | np.clip() dengan batas klinis | Lihat PRD §3.3.2 |
| 5 | Clip skala nyeri | np.clip(df.skala_nyeri, 0, 10) | Max valid = 10 |
| 6 | Parse GCS E/M/V dari teks | regex r'GCS E : (\d) M : (\d) V : (\d)' | E+M+V → GCS_total |
| 7 | Impute missing (numeric) | IterativeImputer / median per kelas | Post-split: fit train only |
| 8 | Clean cara_datang | str.strip() + mapping ke kategori bersih | Trailing whitespace |
| 9 | Encode jenis_kelamin | Binary: L=1, P=0 | |
| **10** | **Kalkulasi TEWS sub-scores** | **fungsi `compute_tews(df)` — 6 sub-score + TEWS_total** | **Dilakukan post-imputation** |
| **11** | **Konstruksi label SATS** | **fungsi `assign_sats_label(tews, overrides)` → Merah/Oranye/Kuning/Hijau/Biru** | **Gating step sebelum modeling** |
| 12 | Feature engineering | MAP, flags, n_vital_abnormal (PRD §4.2) | ~32 final features (termasuk TEWS sub-scores) |

### 5.3.2 Encoding Strategy

| **Variabel** | **Encoding Method** | **Justifikasi** |
|---|---|---|
| jenis_kelamin | Binary Encoding (L=1, P=0) | 2 kategori, tidak ada ordinality |
| cara_datang | One-Hot Encoding (4 kategori) | Nominal, tidak ada ordinality antar kategori |
| kelompok_usia | Ordinal Encoding | Ada urutan natural: Bayi < Anak < Dewasa < Lansia |
| TEWS_total + sub-scores | Numerik langsung | Sudah berupa integer terstruktur |
| flag_* (binary) | Sudah binary (0/1) | Tidak perlu encoding tambahan |
| Semua numerik | Min-Max Normalization [0,1] | XGBoost tidak sensitif skala, tapi membantu SHAP comparability |

### 5.3.3 Target Variable Construction — SATS-TEWS

Label triage dikonstruksi menggunakan **SATS-TEWS rule-based scoring**. Implementasi dilakukan dalam dua tahap:

**Tahap 1 — Kalkulasi TEWS per rekam medis:**

```python
def compute_tews(row):
    """
    Kalkulasi TEWS berdasarkan SATS (South African Triage Scale).
    Input: row pandas dengan kolom vital signs post-imputation & post-clip.
    Output: dict dengan TEWS_total dan 6 sub-scores.
    """
    score = {}
    
    # Laju Pernapasan (rpm)
    rr = row['laju_pernafasan']
    if rr < 9: score['rr'] = 3
    elif rr <= 11: score['rr'] = 1
    elif rr <= 20: score['rr'] = 0
    elif rr <= 29: score['rr'] = 2
    else: score['rr'] = 3

    # SpO2 (%)
    spo2 = row['SpO2']
    if spo2 < 90: score['spo2'] = 3
    elif spo2 <= 94: score['spo2'] = 2
    elif spo2 <= 96: score['spo2'] = 1
    else: score['spo2'] = 0

    # Tekanan Darah Sistolik (mmHg)
    sbp = row['sistole']
    if sbp < 70: score['bp'] = 3
    elif sbp <= 89: score['bp'] = 2
    elif sbp <= 109: score['bp'] = 1
    elif sbp <= 149: score['bp'] = 0
    elif sbp <= 179: score['bp'] = 1
    else: score['bp'] = 2

    # Denyut Jantung (bpm)
    hr = row['denyut_jantung']
    if hr < 40: score['hr'] = 3
    elif hr <= 50: score['hr'] = 1
    elif hr <= 100: score['hr'] = 0
    elif hr <= 110: score['hr'] = 1
    elif hr <= 129: score['hr'] = 2
    else: score['hr'] = 3

    # Suhu Tubuh (°C)
    temp = row['suhu_tubuh']
    if temp < 35: score['temp'] = 2
    elif temp <= 38.4: score['temp'] = 0
    else: score['temp'] = 1

    # GCS Total
    gcs = row['GCS_total']
    if gcs <= 8: score['gcs'] = 3
    elif gcs <= 12: score['gcs'] = 2
    elif gcs <= 14: score['gcs'] = 1
    else: score['gcs'] = 0

    score['TEWS_total'] = sum(score.values())
    return score
```

**Tahap 2 — Mapping TEWS ke Label SATS:**

| **Kondisi** | **Label SATS** | **Kode** |
|---|---|---|
| Override: GCS_total ≤ 8 ATAU sistole < 70 ATAU (SpO2 < 90 DAN RR < 9 atau RR ≥ 30) | Resusitasi — Merah | 0 |
| TEWS_total ≥ 7 (tanpa override) | Emergent — Oranye | 1 |
| TEWS_total 5–6 | Urgent — Kuning | 2 |
| TEWS_total 3–4 | Less Urgent — Hijau | 3 |
| TEWS_total 0–2 | Not Urgent — Biru | 4 |

> **Validasi Wajib:** Rules SATS-TEWS ini harus divalidasi dengan minimal 1 dokter IGD RSU Aulia sebelum finalisasi label. Dokumentasikan semua penyesuaian lokal yang disepakati.

## 5.4 Phase 4: Modeling

### 5.4.1 Model Configuration

| **Parameter XGBoost** | **Nilai / Range Search Space** | **Justifikasi** |
|---|---|---|
| objective | multi:softmax | Klasifikasi multi-kelas langsung (argmax) |
| num_class | 5 | Jumlah kategori triage SATS: Merah/Oranye/Kuning/Hijau/Biru |
| eval_metric | mlogloss | Logarithmic loss untuk multi-class |
| n_estimators | [100, 200, 300, 500] | Jumlah pohon — lebih banyak = lebih akurat tapi lebih lambat |
| max_depth | [3, 4, 5, 6, 7, 8] | Kedalaman pohon — kontrol kompleksitas |
| learning_rate | [0.01, 0.05, 0.1, 0.2, 0.3] | Shrinkage — tradeoff kecepatan vs akurasi |
| subsample | [0.6, 0.7, 0.8, 0.9, 1.0] | Row sampling — regularisasi overfitting |
| colsample_bytree | [0.6, 0.7, 0.8, 0.9, 1.0] | Feature sampling per tree |
| min_child_weight | [1, 3, 5, 7] | Min sum of instance weight in leaf |
| gamma | [0, 0.1, 0.2, 0.5, 1.0] | Min loss reduction for split |
| reg_alpha | [0, 0.01, 0.1, 1.0] | L1 regularization |
| reg_lambda | [0.5, 1.0, 2.0, 5.0] | L2 regularization |
| random_state | 42 | Reproducibility |

### 5.4.2 Training Pipeline (Scikit-learn)

Pipeline scikit-learn memastikan tidak terjadi data leakage — semua transformation di-fit hanya pada training data:

1. Stratified train/test split (80:20, random_state=42)
2. Fit IterativeImputer pada X_train, transform X_train dan X_test
3. **Kalkulasi TEWS_total + 6 sub-scores pada X_train_imputed dan X_test_imputed**
4. **Konstruksi label SATS dari TEWS (hanya dilakukan pada training set; test set menggunakan label yang sudah dikonstruksi di awal)**
5. Fit MinMaxScaler pada X_train_final, transform keduanya
6. RandomizedSearchCV(XGBClassifier, param_distributions, n_iter=100, cv=StratifiedKFold(5), scoring='f1_macro')
7. Fit best model pada X_train_final, predict pada X_test
8. Evaluasi metrik (classification_report, confusion_matrix)
9. SHAP analysis pada test set
10. joblib.dump(best_model, 'model_triage_xgb_sats.pkl')

## 5.5 Phase 5: Evaluation

| **Evaluasi** | **Metode** | **Output** | **Benchmark** |
|---|---|---|---|
| Performa overall | accuracy_score, classification_report | Tabel precision/recall/F1 per kelas SATS | F1-macro >= 0.75 |
| Analisis per kelas | Confusion matrix heatmap (5x5) | Identifikasi kelas terlemah | Merah/Oranye recall >= 0.80 |
| Bias undertriage | Analisis FN Merah+Oranye vs total Merah+Oranye | Undertriage rate | < 5% (SATS benchmark) |
| Baseline vs Tuned | Perbandingan tabel metrik | Improvement delta | Tuned > Baseline |
| CV stability | Cross-val scores per fold | Variance antar fold | Std < 0.05 |
| **TEWS vs Model** | **Akurasi prediksi model vs pure TEWS rule** | **Delta improvement model di atas TEWS baseline** | **Model >= TEWS rule accuracy** |
| Validasi klinis SHAP | Review feature importance dengan dokter IGD | Clinical plausibility check | TEWS_total, GCS, SpO2 di top-5 SHAP |

## 5.6 Phase 6: Deployment — Streamlit Prototype

| **Komponen** | **Deskripsi** | **Implementasi** |
|---|---|---|
| Input Form | Form entri data pasien IGD: usia, jenis kelamin, cara datang, GCS, skala nyeri, tanda vital | st.form() dengan st.number_input(), st.selectbox() |
| **TEWS Calculator** | **Kalkulasi TEWS real-time dari input; tampilkan TEWS_total + sub-scores sebelum prediksi** | **compute_tews(input_dict) → st.metric() per sub-score** |
| Preprocessing Engine | Validasi input, feature engineering otomatis (MAP, TEWS, flags), normalisasi Min-Max | Python function: preprocess_input(raw_input) |
| Prediction Engine | Load model Joblib, predict kategori SATS dan probabilitas per kelas | joblib.load() + xgb_model.predict_proba() |
| Output Panel | Tampilkan kategori SATS dengan color coding (Merah/Oranye/Kuning/Hijau/Biru), probabilitas tiap kelas | st.metric(), color-coded badge |
| SHAP Explainer | SHAP Waterfall Plot + Force Plot per prediksi — kontribusi tiap fitur (termasuk TEWS sub-scores) | shap.TreeExplainer + st.pyplot() |
| Input Validation | Cek range klinis tiap input, alert jika outlier, required field check | try/except + st.warning() |
| Model Info Panel | Tampilkan versi model, standar triage (SATS), tanggal training, metrik performa | Sidebar st.sidebar.info() |

---

# 6. NON-FUNCTIONAL REQUIREMENTS

| **ID** | **Kategori** | **Requirement** | **Target** | **Prioritas** |
|---|---|---|---|---|
| NFR-01 | Performance | Waktu inferensi model (single prediction) | < 500ms | HIGH |
| NFR-02 | Performance | SHAP computation per prediksi | < 2 detik | MEDIUM |
| NFR-03 | Reliability | Model menghasilkan output valid untuk semua input legal | 100% non-null output | HIGH |
| NFR-04 | Reliability | Graceful error handling pada input invalid/missing | Error message informatif | HIGH |
| NFR-05 | Maintainability | Kode Python: PEP8 compliant, terdokumentasi | Docstrings semua fungsi utama | MEDIUM |
| NFR-06 | Reproducibility | Semua random seed di-set eksplisit | random_state=42 konsisten | HIGH |
| NFR-07 | Security | Tidak ada logging data pasien identifiable (PII) | No PHI in logs/output | HIGH |
| NFR-08 | Usability | UI Streamlit dapat digunakan tanpa training teknis | UX testing dengan 2+ tenaga medis | MEDIUM |
| NFR-09 | Portability | Berjalan di Python >= 3.9, requirements.txt tersedia | requirements.txt terdokumentasi | MEDIUM |
| NFR-10 | Scalability | Prototipe: tidak dirancang untuk concurrent load tinggi | Single-user prototype scope | LOW |
| NFR-11 | Explainability | Setiap prediksi harus memiliki SHAP explanation | 100% predictions explained | HIGH |
| NFR-12 | Clinical Safety | Model tidak override keputusan dokter — pure advisory | Disclaimer jelas di UI | CRITICAL |
| **NFR-13** | **Standards Compliance** | **TEWS kalkulasi mengikuti SATS versi resmi (Gottschalk et al., 2006; direvisi 2012)** | **Terdokumentasi referensi versi SATS yang digunakan** | **HIGH** |

---

# 7. RISK REGISTER

| **ID** | **Risk** | **Kategori** | **Likelihood** | **Impact** | **Severity** | **Mitigasi** |
|---|---|---|---|---|---|---|
| R-01 | Tidak ada kolom target triage di dataset | Data | **HIGH** | **CRITICAL** | **CRITICAL** | Rule-based labeling SATS-TEWS + validasi pakar |
| R-02 | 33% missing data pada tanda vital (zero-encoded) | Data | **HIGH** | **HIGH** | **HIGH** | Zero→NaN, IterativeImputer per kelas; TEWS dihitung post-imputation |
| R-03 | GCS parsing gagal pada variasi format teks | Data/Eng | **MEDIUM** | **HIGH** | **HIGH** | Robust regex + fallback ke GCS default (15 → TEWS_gcs_score = 0) |
| R-04 | Class imbalance ekstrem (Merah/Resusitasi sangat sedikit) | Model | **HIGH** | **HIGH** | **HIGH** | SMOTE / class_weight / F1-macro objective |
| R-05 | Overfitting pada kelas mayoritas (Biru/Hijau) | Model | **MEDIUM** | **HIGH** | **MEDIUM** | Regularization, cross-validation monitoring |
| R-06 | Label noise dari SATS-TEWS rule-based labeling | Data | **HIGH** | **MEDIUM** | **HIGH** | Validasi klinis rules dengan dokter IGD RSU Aulia |
| R-07 | TEWS sub-score terlalu dominan dalam SHAP (model "hanya belajar TEWS") | Model | **MEDIUM** | **MEDIUM** | **MEDIUM** | Analisis ablation: bandingkan model dengan/tanpa TEWS features |
| R-08 | SHAP tidak intuitif bagi tenaga medis non-teknis | UX | **MEDIUM** | **MEDIUM** | **MEDIUM** | UI labeling yang jelas, demo langsung |
| R-09 | Generalisasi model buruk ke periode/RS lain | Model | **HIGH** | **MEDIUM** | **MEDIUM** | Scope penelitian: RSU Aulia saja, disclaimer jelas |
| R-10 | Undertriage pasien kritis (False Negative Merah/Oranye) | Clinical | **MEDIUM** | **CRITICAL** | **CRITICAL** | Prioritaskan recall Merah/Oranye, threshold adjustment |
| R-11 | Data pasien terekspos (PII leak) | Security | **LOW** | **CRITICAL** | **HIGH** | Anonymisasi data training, no PHI in artifacts |

---

# 8. TECH STACK & DEPENDENCIES

| **Komponen** | **Library/Tool** | **Versi Min** | **Fungsi** |
|---|---|---|---|
| Core ML | xgboost | >=1.7.0 | Algoritma klasifikasi utama (XGBoost Softmax) |
| ML Framework | scikit-learn | >=1.2.0 | Pipeline, RandomizedSearchCV, StratifiedKFold, metrics |
| XAI/Explainability | shap | >=0.42.0 | TreeSHAP: Summary, Bar, Dependence, Waterfall, Force Plot |
| Data Manipulation | pandas | >=1.5.0 | Data loading, manipulation, feature engineering |
| Numerical Computing | numpy | >=1.23.0 | Array ops, clipping, TEWS kalkulasi |
| Visualization (model) | matplotlib | >=3.6.0 | Confusion matrix, learning curves, SHAP plots |
| Visualization (model) | seaborn | >=0.12.0 | Heatmaps, distribution plots |
| Web UI | streamlit | >=1.25.0 | Prototipe antarmuka pengguna deployment |
| Model Persistence | joblib | >=1.2.0 | Serialisasi model ke .pkl format |
| Imputation | sklearn.impute | >=1.2.0 | IterativeImputer untuk missing value handling |
| Runtime | Python | >=3.9 | Environment utama |
| Package Management | pip / conda | Latest | Dependency management |

## 8.1 Requirements.txt (Estimasi)

```
xgboost>=1.7.0
scikit-learn>=1.2.0
shap>=0.42.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.25.0
joblib>=1.2.0
imbalanced-learn>=0.10.0  # opsional: SMOTE
```

---

# 9. PROJECT TIMELINE & MILESTONES

| **Fase CRISP-DM** | **Task Utama** | **Estimasi Durasi** | **Deliverable** | **Dependencies** |
|---|---|---|---|---|
| Phase 1: Business Understanding | Problem def, tujuan, PRD finalisasi, penetapan standar SATS | 1 minggu | PRD Final (dokumen ini) | — |
| Phase 2: Data Understanding | EDA lengkap, profiling kualitas, visualisasi distribusi, analisis GCS | 1 minggu | Data Understanding Report + EDA Notebook | Dataset tersedia |
| Phase 3a: Target Labeling (SATS-TEWS) | Implementasi compute_tews(), assign_sats_label(), validasi distribusi kelas | 1 minggu | Labeled dataset dengan kolom SATS_label + TEWS_total | Phase 2 complete |
| Phase 3b: Validasi Klinis Label | Review rules SATS-TEWS dengan dokter IGD RSU Aulia | 0.5 minggu | Validated labeling documentation | Dokter IGD tersedia |
| Phase 3c: Data Preparation | Cleaning, parsing usia/GCS, encoding, feature engineering, split | 1 minggu | Clean dataset + Pipeline code | Phase 3b complete |
| Phase 4: Modeling | Baseline XGBoost, RandomizedSearchCV tuning, SHAP analysis, ablation TEWS | 2 minggu | Best model .pkl, SHAP plots, tuning report | Phase 3c complete |
| Phase 5: Evaluation | Metrik lengkap, confusion matrix, undertriage analysis, TEWS baseline comparison | 1 minggu | Evaluation report, BAB 4 skripsi draft | Phase 4 complete |
| Phase 6: Deployment | Streamlit prototype build, TEWS calculator widget, testing, UI refinement | 1.5 minggu | Functional Streamlit app | Phase 5 complete |
| Documentation | Finalisasi BAB 2, 3, 4, 5 skripsi + revisi | Paralel all phases | Draft skripsi lengkap | All phases |

## 9.1 Gating Milestones

| **Milestone** | **Kondisi Gate** | **Consequence jika Gagal** |
|---|---|---|
| GATE-1: SATS-TEWS Label Constructed | TEWS kalkulasi berhasil pada >80% records; distribusi 5 kelas SATS reasonable | Seluruh modeling tidak dapat dilanjutkan — BLOCKER |
| GATE-2: Clinical Label Validation | Minimal 1 dokter IGD menyetujui rule-based SATS mapping | Label dianggap tidak valid secara klinis — perlu revisi rules |
| GATE-3: Data Quality Cleared | Zero-encoding ter-handle, GCS berhasil diparsing, outlier di-clip | Model akan bias/tidak valid |
| GATE-4: Baseline Model Passes | F1-macro baseline >= 0.60 (sanity check) | Perlu review ulang feature engineering dan labeling |
| GATE-5: Tuned Model Passes | F1-macro >= 0.75, Recall Merah >= 0.80 | Perlu iterasi tambahan atau re-labeling |
| GATE-6: SHAP Clinical Plausibility | TEWS_total, GCS_total, SpO2 ada di top-5 SHAP features | Review model validity dengan pembimbing klinis |

---

# 10. ACCEPTANCE CRITERIA

## 10.1 Model Acceptance

| **ID** | **Kriteria** | **Metode Verifikasi** | **Status** |
|---|---|---|---|
| AC-M01 | F1-Score Macro >= 0.75 pada hold-out test set | classification_report sklearn | PENDING |
| AC-M02 | Recall kelas Merah (Resusitasi) >= 0.80 | Per-class recall dari confusion matrix | PENDING |
| AC-M03 | Model teroptimasi > model baseline untuk semua metrik utama | Comparison table baseline vs tuned | PENDING |
| AC-M04 | 5-Fold CV F1-macro: std < 0.05 (stability check) | cross_val_score() report | PENDING |
| AC-M05 | Semua 5 kelas SATS terprediksi dalam test set | Confusion matrix coverage check | PENDING |
| AC-M06 | Undertriage rate (FN Merah+Oranye / Total Merah+Oranye) < 5% | Custom metric dari confusion matrix | PENDING |
| **AC-M07** | **Model accuracy >= TEWS pure-rule accuracy pada test set** | **Perbandingan tabel vs TEWS rule baseline** | **PENDING** |

## 10.2 SHAP Explainability Acceptance

| **ID** | **Kriteria** | **Metode Verifikasi** | **Status** |
|---|---|---|---|
| AC-S01 | SHAP Summary Plot menampilkan semua fitur (termasuk TEWS sub-scores) | Visual check plot | PENDING |
| AC-S02 | TEWS_total, GCS_total, SpO2 ada di top-5 SHAP features | SHAP bar plot ranking check | PENDING |
| AC-S03 | Waterfall Plot tersedia untuk minimal 3 sample kasus per kelas SATS | SHAP local explanation demo | PENDING |
| AC-S04 | Force Plot berfungsi di Streamlit tanpa error rendering | Functional test di Streamlit | PENDING |
| AC-S05 | Penjelasan SHAP dapat dipahami oleh non-technical clinician | User acceptance test dengan 1 dokter IGD | PENDING |

## 10.3 System/Prototype Acceptance

| **ID** | **Kriteria** | **Metode Verifikasi** | **Status** |
|---|---|---|---|
| AC-P01 | Input semua 12 variabel klinis berfungsi tanpa error | Functional test all inputs | PENDING |
| AC-P02 | Output menampilkan kategori SATS dengan color indicator (5 warna) | UI visual test | PENDING |
| AC-P03 | TEWS_total dan probabilitas tiap kelas SATS ditampilkan | Output panel test | PENDING |
| AC-P04 | Inference time < 500ms dari submit ke output | Timer test | PENDING |
| AC-P05 | Input invalid menghasilkan error message yang jelas | Negative test cases | PENDING |
| AC-P06 | Disclaimer 'Decision Support Only — Final decision by licensed clinician' tampil jelas di UI | Visual check | PENDING |
| **AC-P07** | **TEWS Calculator Widget menampilkan sub-score tiap parameter secara real-time** | **Functional test widget** | **PENDING** |

---

# 11. APPENDIX

## A. Distribusi ICD-10 Teratas (Dataset RSU Aulia)

| **Rank** | **Kode ICD-10** | **Frekuensi** | **Deskripsi (Referensi)** |
|---|---|---|---|
| 1 | J06.9 | 774 | Acute upper respiratory infection, unspecified |
| 2 | R10.4 | 485 | Other and unspecified abdominal pain |
| 3 | A09.0 | 259 | Other and unspecified gastroenteritis and colitis of infectious origin |
| 4 | R11 | 218 | Nausea and vomiting |
| 5 | A68.9 | 208 | Relapsing fever, unspecified |
| 6 | R10.1 | 183 | Pain localized to upper abdomen |
| 7 | J00 | 180 | Acute nasopharyngitis (common cold) |
| 8 | Z37.0 | 161 | Single liveborn infant (obstetric context) |
| 9 | A49.9 | 159 | Bacterial infection, unspecified |
| 10 | R50.9 | 136 | Fever, unspecified |

## B. Distribusi Demografis Dataset

| **Variabel** | **Kategori** | **Jumlah** | **Persentase** |
|---|---|---|---|
| Jenis Kelamin | Perempuan | 4.056 | 64,0% |
| Jenis Kelamin | Laki-laki | 2.283 | 36,0% |
| Cara Datang | Datang Sendiri | 6.189 | 97,6% |
| Cara Datang | Puskesmas | 139 | 2,2% |
| Cara Datang | Kecelakaan Lalu Lintas | 9 | 0,1% |
| Cara Datang | Dokter / Dokter Gigi | 2 | 0,03% |

## C. Referensi Klinis

- **Gottschalk SB, Wood D, DeVries S, Wallis LA, Bruijns S. (2006). The Cape Triage Score: A Triage System for South Africa. Emergency Medicine Journal. (Dasar SATS)**
- **South African Triage Group. (2012). The South African Triage Scale — Training Manual (Revised 3rd ed.). Cape Town. (Versi SATS yang digunakan dalam proyek ini)**
- **Wallis LA, Twomey M. (2007). Workload and case mix in Cape Town emergency departments. South African Medical Journal.**
- WHO. (2008). Emergency Triage Assessment and Treatment (ETAT). Geneva.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Lundberg, S.M. & Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
- CRISP-DM 1.0 Step-by-step data mining guide — SPSS.
- Permenkes No. 47 Tahun 2018 tentang Pelayanan Kegawatdaruratan.

## D. Perbandingan Standar Triage (Rationale Pemilihan SATS)

| **Dimensi** | **ESI (Emergency Severity Index)** | **MTS (Manchester Triage System)** | **SATS (South African Triage Scale)** |
|---|---|---|---|
| Asal | USA | UK | Afrika Selatan / WHO |
| Jumlah Level | 5 | 5 | 5 |
| Mekanisme Skoring | Resource-based (resources yang dibutuhkan) | Discriminator flowchart klinis | **TEWS (objective vital sign scoring)** |
| Berbasis Tanda Vital Numerik | Parsial | Parsial | **Ya — terstruktur (TEWS)** |
| Cocok untuk rule-based labeling | Sedang | Rendah | **Tinggi** |
| Cocok untuk negara berkembang | Rendah | Rendah | **Tinggi (WHO-recommended)** |
| Dapat dikonstruksi dari dataset RSU Aulia | Parsial | Sulit | **Ya — semua komponen tersedia** |
| **Rekomendasi untuk proyek ini** | Tidak digunakan | Tidak digunakan | **✅ DIPILIH** |

> **Justifikasi Pemilihan SATS:** SATS dipilih karena: (1) mekanisme TEWS sepenuhnya berbasis tanda vital numerik yang tersedia di dataset RSU Aulia, (2) direkomendasikan WHO untuk setting IGD sumber daya terbatas, (3) memungkinkan konstruksi label surrogate yang objektif dan reproducible tanpa membutuhkan flowchart klinis subjektif, dan (4) relevan untuk konteks Indonesia sebagai negara berkembang.

## E. Glossary

| **Term** | **Definisi** |
|---|---|
| Triage | Proses pengelompokan pasien berdasarkan tingkat kegawatan untuk prioritas penanganan |
| SATS | South African Triage Scale — standar triage 5-level berbasis TEWS, direkomendasikan WHO |
| TEWS | Triage Early Warning Score — skor agregat 6 parameter klinis dalam sistem SATS (range 0–17) |
| XGBoost | eXtreme Gradient Boosting — algoritma ensemble tree-based learning yang efisien |
| SHAP | SHapley Additive exPlanations — metode game-theory untuk interpretabilitas model ML |
| TreeSHAP | Implementasi SHAP yang dioptimasi untuk model berbasis pohon keputusan |
| CRISP-DM | Cross-Industry Standard Process for Data Mining — metodologi standar data mining |
| F1-Macro | Rata-rata F1-Score semua kelas tanpa mempertimbangkan jumlah sampel per kelas |
| MAP | Mean Arterial Pressure = Diastole + (Sistole-Diastole)/3, indikator perfusi organ |
| Undertriage | Pasien gawat darurat dikategorikan lebih rendah dari seharusnya — risiko fatal |
| Overtriage | Pasien non-kritis dikategorikan lebih tinggi — membebani sumber daya IGD |
| GCS | Glasgow Coma Scale — skala penilaian tingkat kesadaran (Eye+Motor+Verbal, range 3–15) |
| PHI/PII | Protected Health Information / Personally Identifiable Information — data privasi pasien |
| Decision Support | Sistem bantu pengambilan keputusan — tidak menggantikan keputusan klinis dokter |
| Surrogate Label | Label target yang dikonstruksi dari rules (SATS-TEWS) karena ground truth tidak tersedia |
| Label Noise | Ketidakakuratan dalam label surrogate akibat gap antara rules dan keputusan klinis nyata |

---

*Dokumen ini adalah PRD internal penelitian skripsi undergraduate. Model yang dikembangkan bersifat decision support prototype, bukan sistem produksi klinis. Keputusan triage final tetap berada di tangan tenaga medis berlisensi.*

*Standar triage yang digunakan: **South African Triage Scale (SATS) — Triage Early Warning Score (TEWS)**, South African Triage Group, Revised 3rd Edition, 2012.*

---

**Reymondo | RSU Aulia | XGBoost + SHAP Triage Prediction | PRD v1.1 — SATS Reference | Juni 2026**

*CONFIDENTIAL — RESEARCH USE ONLY*
