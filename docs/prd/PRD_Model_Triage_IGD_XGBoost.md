# Product Requirements Document (PRD)
## Model Prediksi Skor Triage IGD — RSU Aulia
**Versi:** 1.1 | **Tanggal:** 2026 | **Peneliti:** Reymondo
**Metode:** CRISP-DM | **Algoritma:** XGBoost + SHAP

---

## 1. Ringkasan Eksekutif

Dokumen ini mendefinisikan seluruh persyaratan teknis dan fungsional untuk membangun sistem prediksi kategori triage IGD berbasis machine learning. Model dibangun menggunakan dataset klinis rekam medis RSU Aulia yang berisi **6.339 rekam kunjungan** dengan **18 kolom** data klinis pasien, periode Januari 2026.

> **Pembaruan v1.1:** Berdasarkan konfirmasi langsung dari staff RSU Aulia (Bapak Shendy Arief Prastyanto), sistem SIMRS RSU Aulia **tidak mencatat kategori triage secara terpisah**. Seluruh pasien IGD diproses dalam alur yang sama tanpa pembedaan prioritas triage di dalam sistem. Oleh karena itu, label target harus dikonstruksi secara mandiri menggunakan pendekatan rule-based berbasis skor klinis baku (MEWS/NEWS). Strategi konstruksi label ini didokumentasikan pada Fase 1.

---

## 2. Identifikasi Dataset

### 2.1 Profil Dataset

| Atribut | Detail |
|---------|--------|
| Nama file | `Dataset_Klinis_Edit.csv` |
| Sumber | RSU Aulia (via Bapak Shendy Arief Prastyanto) |
| Jumlah rekam | **6.339 baris** |
| Jumlah kolom | **18 kolom** |
| Periode data | Januari 2026 (estimasi 2023–2024 perlu dikonfirmasi) |
| Format | CSV |

### 2.2 Deskripsi Kolom

| No | Nama Kolom | Tipe Data | Deskripsi | Catatan |
|----|-----------|-----------|-----------|---------|
| 1 | `noreg` | String | Nomor registrasi kunjungan | ID unik per kunjungan |
| 2 | `tgl_periksa` | String (datetime) | Tanggal dan waktu pemeriksaan | Perlu parsing ke datetime |
| 3 | `rm` | Integer | Nomor rekam medis pasien | ID unik per pasien |
| 4 | `pasien` | String | Nama lengkap pasien | PII — perlu anonimisasi |
| 5 | `usia` | String | Usia dalam format teks (contoh: "13 Th 9 Bl 3 Hr") | Perlu ekstraksi ke angka (tahun) |
| 6 | `jenis_kelamin` | String | Jenis kelamin pasien | Perempuan / Laki-laki |
| 7 | `cara_datang` | String | Cara kedatangan ke IGD | 4 kategori unik |
| 8 | `keluhan_utama` | String | Keluhan utama pasien | **1.958 missing values (30.9%)** |
| 9 | `GCS` | String | Kolom berisi teks pemeriksaan fisik lengkap | **Mismatch nama** — bukan nilai GCS numerik |
| 10 | `skala_nyeri` | Integer | Skala nyeri pasien (0–10) | Ada outlier nilai 80 |
| 11 | `sistole` | Integer | Tekanan darah sistolik (mmHg) | **33.2% nilai nol** |
| 12 | `diastole` | Integer | Tekanan darah diastolik (mmHg) | **33.4% nilai nol** |
| 13 | `denyut_jantung` | Integer | Denyut jantung (bpm) | **32.6% nilai nol** |
| 14 | `laju_pernafasan` | Integer | Laju pernapasan (napas/menit) | **32.7% nilai nol** |
| 15 | `suhu_tubuh` | Float | Suhu tubuh (°C) | **32.8% nilai nol**, ada outlier (max 380) |
| 16 | `SpO2` | Float | Saturasi oksigen (%) | **32.7% nilai nol**, ada outlier (max 990) |
| 17 | `kode_ICD` | String | Kode diagnosis ICD-10 | Dapat digunakan sebagai proxy label triage |
| 18 | `Diagnosa` | String | Deskripsi diagnosis lengkap | Teks panjang |

### 2.3 Temuan Kritis Dataset

#### ⚠️ Masalah Utama yang Teridentifikasi

**1. Tidak Ada Kolom Label Triage — Dikonfirmasi oleh Staff RSU Aulia**
Dataset tidak memiliki kolom target triage secara eksplisit. Hal ini **bukan kelalaian input data**, melainkan karena SIMRS RSU Aulia memang tidak mencatat kategori triage secara terpisah — seluruh pasien IGD diproses dalam alur yang sama di dalam sistem. Konfirmasi ini diperoleh langsung dari Bapak Shendy Arief Prastyanto (staff RSU Aulia). Label target harus dikonstruksi secara mandiri menggunakan pendekatan rule-based berbasis skor kegawatan klinis baku (lihat Fase 1).

**2. Kolom `GCS` Berisi Data Salah Kolom**
Kolom bernama `GCS` faktanya berisi teks pemeriksaan fisik lengkap (anamnesis), bukan nilai GCS numerik (E/M/V). Nilai GCS aktual perlu diekstraksi melalui text parsing (regex pola `GCS E : X M : X V : X`).

**3. Nilai Nol Masif pada Tanda Vital (~33%)**
Sekitar sepertiga data memiliki nilai nol pada semua kolom tanda vital (sistole, diastole, denyut jantung, laju pernapasan, suhu tubuh, SpO2) secara bersamaan — mengindikasikan data tidak terisi (missing), bukan nilai nol fisiologis.

**4. Kolom `usia` Berupa Teks**
Format usia adalah string seperti `"13 Th 9 Bl 3 Hr"` — perlu parsing ke nilai numerik (tahun) sebelum digunakan sebagai fitur.

**5. Outlier Ekstrem**
- `skala_nyeri`: max = 80 (seharusnya 0–10)
- `suhu_tubuh`: max = 380 (seharusnya 35–42°C)
- `SpO2`: max = 990 (seharusnya 70–100%)

**6. Missing Values `keluhan_utama`**
30.9% (1.958 baris) tidak memiliki keluhan utama tercatat.

**7. Kolom Identitas PII**
Kolom `pasien` (nama lengkap) perlu dianonimisasi sebelum pemodelan.

---

## 3. Tujuan Produk

### 3.1 Problem Statement
RSU Aulia membutuhkan sistem yang dapat membantu tenaga medis IGD dalam menentukan kategori triage pasien secara konsisten, cepat, dan transparan — mengurangi ketergantungan pada penilaian subyektif yang bisa bervariasi.

### 3.2 Tujuan Utama
1. Membangun model klasifikasi multi-kelas XGBoost yang memprediksi kategori triage IGD
2. Mengintegrasikan SHAP untuk menjelaskan setiap prediksi secara klinis
3. Mengembangkan prototipe antarmuka Streamlit yang dapat digunakan tenaga medis

### 3.3 Kriteria Keberhasilan
| Metrik | Target Minimum |
|--------|---------------|
| F1-Score Macro | ≥ 0.75 |
| Akurasi | ≥ 0.80 |
| Recall kelas kritis (P1/P2) | ≥ 0.85 |
| Waktu inferensi per pasien | < 2 detik |

---

## 4. Tahapan Pembangunan Model

---

### FASE 1 — Konstruksi Label Target

> **Prasyarat sebelum seluruh tahapan lain.** Berdasarkan konfirmasi staff RSU Aulia, SIMRS tidak mencatat kategori triage — sehingga label harus dibangun secara mandiri menggunakan skor kegawatan klinis yang sudah tervalidasi secara medis.

#### 4.1.1 Latar Belakang Keputusan

Terdapat tiga opsi yang dievaluasi untuk konstruksi label:

| Opsi | Pendekatan | Kelebihan | Kekurangan | Keputusan |
|------|-----------|-----------|-----------|-----------|
| A | Mapping kode ICD → triage | Mudah diimplementasi | Subjektif, tidak berbasis parameter klinis | ✗ Tidak dipilih |
| **B** | **Rule-based MEWS/NEWS dari tanda vital** | **Berbasis standar medis tervalidasi, objektif, reproducible** | **Bergantung kelengkapan tanda vital** | **✓ Dipilih** |
| C | Ubah target ke prediksi kode ICD | Menghindari masalah label | Mengubah framing penelitian secara mendasar | ✗ Cadangan |

**Opsi B (MEWS/NEWS)** dipilih karena paling dapat dipertanggungjawabkan secara klinis — label dihasilkan dari parameter fisiologis terukur menggunakan sistem skoring yang sudah divalidasi di literatur medis internasional, bukan dari interpretasi subjektif kode diagnosis.

#### 4.1.2 Metode Konstruksi Label — Modified Early Warning Score (MEWS)

MEWS adalah sistem skoring kegawatan pasien berbasis 5 parameter tanda vital yang sudah digunakan secara luas di IGD. Skor dihitung dari penjumlahan skor per parameter.

**Tabel Skoring MEWS:**

| Parameter | Skor 3 | Skor 2 | Skor 1 | Skor 0 | Skor 1 | Skor 2 | Skor 3 |
|-----------|--------|--------|--------|--------|--------|--------|--------|
| Sistole (mmHg) | < 70 | 70–80 | 81–100 | 101–199 | — | ≥ 200 | — |
| Denyut jantung (bpm) | — | < 40 | 41–50 | 51–100 | 101–110 | 111–129 | ≥ 130 |
| Laju pernapasan (/mnt) | — | < 9 | — | 9–14 | 15–20 | 21–29 | ≥ 30 |
| Suhu tubuh (°C) | — | < 35 | — | 35–38.4 | ≥ 38.5 | — | — |
| Tingkat kesadaran | — | — | — | Alert | Reaksi suara | Reaksi nyeri | Tidak respons |

**Mapping Skor MEWS → Kategori Triage:**

| Total Skor MEWS | Kategori | Label | Warna | Tindakan |
|----------------|----------|-------|-------|---------|
| ≥ 5 | Gawat Darurat | **P1** | Merah | Penanganan segera |
| 3 – 4 | Darurat | **P2** | Kuning | Pantau ketat, tangani cepat |
| 0 – 2 | Tidak Darurat | **P3** | Hijau | Dapat menunggu |

#### 4.1.3 Penanganan Kasus Data Tidak Lengkap

Karena ~33% baris memiliki tanda vital nol (tidak terisi), perhitungan MEWS tidak dapat dilakukan langsung. Strategi berikut diterapkan:

1. **Baris dengan ≥ 3 tanda vital tersedia** → Hitung MEWS dari parameter yang ada, gunakan nilai rata-rata untuk parameter yang hilang
2. **Baris dengan < 3 tanda vital tersedia** → Gunakan fallback: mapping kode ICD ke kategori triage sebagai proxy (Opsi A sebagai fallback)
3. **Baris yang tidak dapat diklasifikasikan** → Ekslusi dari dataset pemodelan, dokumentasikan sebagai kriteria eksklusi

#### 4.1.4 Fallback — Mapping Kode ICD (untuk baris tanda vital tidak lengkap)

| Kategori | Label | Contoh kode ICD |
|----------|-------|-----------------|
| Gawat Darurat | P1 | I21 (AMI), I60 (SAH), J18 (Pneumonia berat), R09 (gagal napas) |
| Darurat | P2 | A09.0 (gastroenteritis infeksius), R10.4 (nyeri abdomen), J06.9 (ISPA) |
| Tidak Darurat | P3 | B86 (scabies), Z37.0 (persalinan normal), H81.4 (vertigo) |

#### 4.1.5 Output Fase 1

- Kolom baru: `mews_score` (integer, skor total MEWS)
- Kolom baru: `kategori_triage` (P1 / P2 / P3) — label target model
- Kolom baru: `metode_label` ("mews" / "icd_fallback") — transparansi sumber label
- Laporan distribusi kelas: jumlah dan proporsi P1/P2/P3
- Dokumentasi mapping dan keputusan ekslusi

---

### FASE 2 — Data Understanding

#### 4.2.1 Eksplorasi Statistik Deskriptif

**Variabel numerik yang dianalisis:**
- `skala_nyeri`, `sistole`, `diastole`, `denyut_jantung`, `laju_pernafasan`, `suhu_tubuh`, `SpO2`

**Analisis yang dilakukan:**
- Mean, median, std, min, max, kuartil (Q1, Q3, IQR)
- Distribusi per kolom (histogram)
- Korelasi antar variabel (heatmap Pearson/Spearman)

#### 4.2.2 Analisis Distribusi Kelas Target
- Distribusi frekuensi per kategori triage
- Identifikasi class imbalance
- Visualisasi pie chart dan bar chart distribusi

#### 4.2.3 Analisis Kualitas Data

| Pemeriksaan | Detail |
|-------------|--------|
| Missing values | Per kolom — lihat Tabel 2.2 |
| Nilai nol fisiologis | Tanda vital: 32–33% nol (bukan nilai valid) |
| Duplikasi | Cek duplikasi berdasarkan `noreg` dan `rm + tgl_periksa` |
| Outlier | Identifikasi menggunakan IQR dan rentang fisiologis normal |
| Konsistensi format | `usia` (string), `tgl_periksa` (string datetime), `GCS` (mismatch) |

**Rentang fisiologis normal sebagai acuan:**

| Variabel | Min Normal | Max Normal |
|----------|-----------|-----------|
| Sistole | 70 | 220 |
| Diastole | 40 | 130 |
| Denyut jantung | 30 | 200 |
| Laju pernapasan | 8 | 40 |
| Suhu tubuh | 34.0 | 42.0 |
| SpO2 | 70 | 100 |
| Skala nyeri | 0 | 10 |

---

### FASE 3 — Data Preparation

#### 4.3.1 Seleksi dan Anonimisasi Data

**Kolom yang DIHAPUS dari dataset pemodelan:**
- `noreg` — ID administratif, tidak informatif untuk model
- `rm` — ID pasien, berpotensi data leakage
- `pasien` — PII (nama lengkap), wajib dihapus
- `tgl_periksa` — kecuali diekstraksi fitur temporal (bulan, jam, hari)
- `Diagnosa` — teks diagnosis panjang, redundan dengan `kode_ICD`
- `kode_ICD` — digunakan untuk membentuk label, tidak boleh menjadi fitur (data leakage)
- `GCS` (kolom asli) — berisi teks pemeriksaan fisik, bukan GCS numerik

**Kolom yang DIPERTAHANKAN sebagai fitur:**
`usia_tahun` (hasil parsing), `jenis_kelamin`, `cara_datang`, `keluhan_utama`, `gcs_total` (hasil ekstraksi), `skala_nyeri`, `sistole`, `diastole`, `denyut_jantung`, `laju_pernafasan`, `suhu_tubuh`, `SpO2`

#### 4.3.2 Penanganan Masalah Spesifik Dataset

**A. Parsing kolom `usia`**
```
Input : "13 Th 9 Bl 3 Hr"
Output: 13  (diambil nilai tahun saja)
Metode: regex ekstraksi angka sebelum "Th"
```

**B. Ekstraksi nilai GCS dari kolom `GCS`**
```
Input : "Keadaan Umum GCS E : 4  M : 6 V : 5 ..."
Output: gcs_e=4, gcs_m=6, gcs_v=5, gcs_total=15
Metode: regex pola "GCS E : (\d) M : (\d) V : (\d)"
Fallback: nilai NaN jika pola tidak ditemukan
```

**C. Penanganan nilai nol tanda vital (33% data)**
Nilai nol pada kolom tanda vital diperlakukan sebagai missing values, bukan nol fisiologis.
```
Metode: Ganti 0 → NaN, lalu imputasi median per kolom
Catatan: Pertimbangkan imputasi berbasis kelompok usia/diagnosa untuk akurasi lebih tinggi
```

**D. Penanganan outlier tanda vital**
```
Pendekatan: Winsorizing berbasis rentang fisiologis klinis
- Nilai di luar rentang normal → clip ke batas terdekat
- Contoh: suhu_tubuh > 42 → 42.0; SpO2 > 100 → 100.0
```

**E. Penanganan `keluhan_utama` (30.9% missing)**
```
Opsi 1: Imputasi dengan token "TIDAK DIKETAHUI"
Opsi 2: Ekstraksi fitur TF-IDF dari teks yang ada (top-N kata kunci)
Opsi 3: Flag binary "keluhan_tersedia" (1/0)
Rekomendasi: Kombinasi Opsi 1 + Opsi 3
```

#### 4.3.3 Encoding Variabel Kategorikal

| Variabel | Metode Encoding | Alasan |
|----------|----------------|--------|
| `jenis_kelamin` | Binary Encoding (0/1) | 2 kategori |
| `cara_datang` | One-Hot Encoding | 4 kategori, nominal |
| `keluhan_utama` | TF-IDF (top-50 fitur) | Teks bebas |
| `kategori_triage` | Label Encoding (0,1,2,3) | Target multi-kelas |

#### 4.3.4 Rekayasa Fitur (Feature Engineering)

| Fitur Baru | Formula | Justifikasi Klinis |
|-----------|---------|-------------------|
| `usia_tahun` | Parsing "X Th" | Fitur numerik dari string |
| `kelompok_usia` | Bayi/Anak/Remaja/Dewasa/Lansia | Risiko berbeda per kelompok |
| `gcs_total` | E + M + V | GCS total lebih informatif |
| `map` | Diastole + (Sistole − Diastole)/3 | Mean Arterial Pressure |
| `vital_abnormal_count` | Jumlah tanda vital di luar normal | Skor keparahan agregat |
| `flag_sistole_abnormal` | 1 jika sistole < 90 atau > 180 | Indikator hipotensi/hipertensi |
| `flag_spo2_rendah` | 1 jika SpO2 < 95 | Indikator hipoksia |
| `flag_takikardi` | 1 jika denyut > 100 | Indikator stres kardiovaskular |
| `flag_demam` | 1 jika suhu > 37.5 | Indikator infeksi |
| `keluhan_tersedia` | 1 jika keluhan tidak null | Kelengkapan dokumentasi |

#### 4.3.5 Normalisasi
- **Metode:** Min-Max Normalization ke rentang [0, 1]
- **Variabel:** Semua variabel numerik kontinu
- **Implementasi:** `sklearn.preprocessing.MinMaxScaler` di dalam Pipeline

#### 4.3.6 Penanganan Class Imbalance
- **Metode utama:** `class_weight='balanced'` pada parameter XGBoost (`scale_pos_weight`)
- **Metode alternatif:** SMOTE pada data training saja (cegah leakage)
- **Evaluasi:** Gunakan F1-score macro (sensitif terhadap imbalance)

#### 4.3.7 Pembagian Dataset

```
Total dataset → Stratified Hold-Out 80:20
├── Training set (80%) → Stratified 5-Fold Cross-Validation
│   ├── Fold 1-4: Training
│   └── Fold 5: Validation
└── Test set (20%) → Evaluasi final (tidak disentuh sampai akhir)

Implementasi: sklearn Pipeline untuk mencegah data leakage
Scaler dan encoder HANYA difit pada data training
```

---

### FASE 4 — Modeling

#### 4.4.1 Pemilihan Algoritma
XGBoost dipilih berdasarkan evaluasi komparatif:

| Algoritma | Kelebihan | Kekurangan | Keputusan |
|-----------|-----------|-----------|-----------|
| Logistic Regression | Interpretabel, cepat | Asumsi linearitas kuat | ✗ Baseline saja |
| Decision Tree | Mudah diinterpretasi | Prone overfitting | ✗ Baseline saja |
| Random Forest | Robust, akurat | Lambat, kurang interpretabel | ✗ Pembanding |
| **XGBoost** | **Akurat, cepat, kompatibel SHAP** | **Perlu tuning** | **✓ Dipilih** |
| SVM | Baik untuk dimensi tinggi | Lambat, tidak kompatibel SHAP | ✗ Tidak dipilih |
| Deep Learning | Sangat akurat | Data kecil, black-box | ✗ Tidak dipilih |

#### 4.4.2 Konfigurasi Awal Model

| Parameter | Nilai Awal | Keterangan |
|-----------|-----------|-----------|
| `objective` | `multi:softmax` | Klasifikasi multi-kelas |
| `num_class` | 3 atau 4 | Sesuai jumlah kategori triage |
| `eval_metric` | `mlogloss` | Multi-class log loss |
| `n_estimators` | 300 | Jumlah pohon awal |
| `max_depth` | 6 | Kedalaman maksimum pohon |
| `learning_rate` | 0.1 | Learning rate awal |
| `subsample` | 0.8 | Subsampling baris |
| `colsample_bytree` | 0.8 | Subsampling kolom |
| `min_child_weight` | 1 | Bobot minimum child node |
| `gamma` | 0 | Regularisasi minimum gain |
| `reg_alpha` | 0 | L1 regularization |
| `reg_lambda` | 1 | L2 regularization |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Semua core CPU |

#### 4.4.3 Hyperparameter Tuning

```
Metode   : RandomizedSearchCV
Iterasi  : 100
CV       : Stratified 5-Fold
Metrik   : F1-score macro
```

**Ruang pencarian parameter:**

| Parameter | Rentang Pencarian |
|-----------|------------------|
| `n_estimators` | [100, 200, 300, 500] |
| `max_depth` | [3, 4, 5, 6, 7, 8] |
| `learning_rate` | [0.01, 0.05, 0.1, 0.2] |
| `subsample` | [0.6, 0.7, 0.8, 0.9, 1.0] |
| `colsample_bytree` | [0.6, 0.7, 0.8, 0.9, 1.0] |
| `min_child_weight` | [1, 3, 5, 7] |
| `gamma` | [0, 0.1, 0.2, 0.5] |
| `reg_alpha` | [0, 0.01, 0.1, 1] |
| `reg_lambda` | [0.5, 1, 2, 5] |

#### 4.4.4 Analisis SHAP

**Analisis Global (keseluruhan model):**
- Summary Plot — distribusi SHAP values semua fitur
- Bar Plot — rata-rata |SHAP| per fitur (feature importance)
- Dependence Plot — efek fitur tertentu terhadap prediksi

**Analisis Lokal (per pasien):**
- Waterfall Plot — kontribusi setiap fitur pada satu prediksi
- Force Plot — visualisasi kompak force diagram

---

### FASE 5 — Evaluation

#### 4.5.1 Metrik Evaluasi

| Metrik | Formula | Justifikasi Klinis |
|--------|---------|-------------------|
| **Akurasi** | TP+TN / Total | Proporsi kebenaran keseluruhan |
| **Precision** | TP / (TP+FP) | Hindari over-triage (P1 yang bukan P1) |
| **Recall** | TP / (TP+FN) | Kritis — hindari under-triage (P1 terlewat) |
| **F1-Score Macro** | 2×P×R / (P+R) | Keseimbangan per kelas, sensitif imbalance |
| **F1-Score Weighted** | Weighted by support | Gambaran performa keseluruhan |

> **Catatan klinis:** Recall kelas P1 (gawat darurat) diprioritaskan di atas precision — lebih baik false alarm daripada melewatkan kasus kritis.

#### 4.5.2 Analisis Confusion Matrix
- Confusion matrix per kelas triage
- Identifikasi pola misklasifikasi (P1 salah prediksi sebagai P2, dll.)
- Analisis per kelompok (usia, jenis kelamin, cara datang)

#### 4.5.3 Perbandingan Model

| Model | Akurasi | F1 Macro | F1 Weighted |
|-------|---------|----------|-------------|
| Baseline (tanpa tuning) | — | — | — |
| Setelah tuning | — | — | — |
| Logistic Regression (pembanding) | — | — | — |
| Random Forest (pembanding) | — | — | — |

*(Diisi setelah eksperimen)*

---

### FASE 6 — Deployment

#### 4.6.1 Arsitektur Sistem

```
[Antarmuka Streamlit]
        │
        ▼
[Input Data Pasien]  →  [Feature Engineering Real-time]
        │
        ▼
[Prediction Engine: XGBoost model (.joblib)]
        │
        ▼
[SHAP Explainer: TreeSHAP]
        │
        ▼
[Panel Hasil: Kategori + Probabilitas + SHAP Visualization]
```

#### 4.6.2 Komponen Sistem

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| Antarmuka pengguna | Streamlit | Input data klinis, tampilkan hasil |
| Prediction Engine | XGBoost + joblib | Load model, inferensi |
| SHAP Explainer | shap.TreeExplainer | Hitung SHAP values |
| Preprocessing | sklearn Pipeline | Transformasi input real-time |
| Visualisasi | matplotlib, shap | Waterfall & Force plot |

#### 4.6.3 Spesifikasi Antarmuka

**Panel Input:** Form dengan field untuk setiap variabel klinis:
- Usia (angka tahun)
- Jenis kelamin (dropdown)
- Cara datang (dropdown)
- Keluhan utama (text input)
- Tanda vital: sistole, diastole, denyut jantung, laju pernapasan, suhu, SpO2 (number input)
- Skala nyeri (slider 0–10)
- Skor GCS E/M/V (dropdown/number)

**Panel Hasil:**
- Kategori triage (P1/P2/P3) dengan warna: Merah/Kuning/Hijau
- Probabilitas per kelas (bar chart)
- SHAP Waterfall Plot — 5 fitur teratas
- Tombol reset

#### 4.6.4 Error Handling
- Validasi input: nilai tanda vital di luar rentang fisiologis → peringatan
- Nilai kosong → imputasi otomatis dengan nilai median training
- Kegagalan load model → pesan error informatif

---

## 5. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|---------|
| Label triage tidak tersedia di SIMRS (dikonfirmasi) | Tinggi | Konstruksi label menggunakan MEWS — standar klinis tervalidasi |
| ~33% data tanda vital adalah nol → MEWS tidak dapat dihitung | Tinggi | Imputasi + fallback ICD mapping untuk baris tidak lengkap |
| Label MEWS bukan ground truth aktual petugas triage | Sedang | Dokumentasikan sebagai keterbatasan, validasi dengan literatur |
| Mismatch kolom GCS | Sedang | Ekstraksi teks via regex |
| Class imbalance (kemungkinan P3 dominan) | Sedang | class_weight balanced + F1 macro |
| Data hanya dari 1 periode | Sedang | Dokumentasi batasan generalisasi |
| Outlier ekstrem pada tanda vital | Sedang | Winsorizing berbasis rentang klinis |

---

## 6. Batasan Sistem

1. Model bersifat **decision support**, bukan pengganti keputusan klinis tenaga medis
2. **Label triage dikonstruksi menggunakan MEWS**, bukan ground truth aktual dari petugas triage RSU Aulia — terdapat kemungkinan perbedaan antara prediksi MEWS dengan penilaian klinis petugas
3. Performa model bergantung pada kualitas dan kelengkapan input data
4. Prototipe belum tervalidasi secara prospektif di lapangan
5. Dataset dari satu rumah sakit — generalisasi ke RS lain belum terjamin
6. Sistem tidak terintegrasi dengan EMR/HIS RSU Aulia (berdiri sendiri)

---

## 7. Teknologi Stack

| Layer | Teknologi | Versi |
|-------|-----------|-------|
| Bahasa | Python | ≥ 3.9 |
| Algoritma ML | XGBoost | ≥ 1.7 |
| Preprocessing | Scikit-learn | ≥ 1.2 |
| Interpretabilitas | SHAP | ≥ 0.42 |
| Antarmuka | Streamlit | ≥ 1.28 |
| Manipulasi data | Pandas, NumPy | Latest stable |
| Visualisasi | Matplotlib, Seaborn | Latest stable |
| Penyimpanan model | Joblib | Latest stable |

---

## 8. Milestone dan Timeline

| Fase | Aktivitas Utama | Output |
|------|----------------|--------|
| **M1** | Konstruksi label triage | Dataset berlabel |
| **M2** | Data understanding & EDA | Laporan eksplorasi + visualisasi |
| **M3** | Data preparation & feature engineering | Dataset siap model |
| **M4** | Baseline modeling | Model baseline + metrik awal |
| **M5** | Hyperparameter tuning | Model teroptimasi |
| **M6** | Analisis SHAP | Global & lokal interpretation report |
| **M7** | Evaluasi & komparasi | Classification report, confusion matrix |
| **M8** | Deployment prototipe | Aplikasi Streamlit fungsional |
| **M9** | Dokumentasi & penulisan | BAB III–V skripsi selesai |

---

*Dokumen ini adalah PRD teknis versi 1.1 untuk skripsi undergraduate Reymondo — Model Prediksi Skor Triage IGD menggunakan XGBoost + SHAP di RSU Aulia. Diperbarui berdasarkan konfirmasi staff RSU Aulia bahwa SIMRS tidak mencatat kategori triage secara terpisah.*
