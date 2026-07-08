# Ringkasan Bisnis: Pembangunan Model Prediksi Skor Triage IGD
**RSU Aulia | Penelitian Skripsi — Reymondo**
**Metode: CRISP-DM | Algoritma: XGBoost + SHAP**

---

## Latar Belakang

Proses triage di Instalasi Gawat Darurat (IGD) merupakan langkah kritis yang menentukan prioritas penanganan pasien berdasarkan tingkat kegawatan kondisi medisnya. Di RSU Aulia, proses ini masih sangat bergantung pada penilaian subyektif tenaga medis yang dapat bervariasi antar-petugas, terutama di saat kondisi pelayanan sedang sibuk. Ketidakkonsistenan ini berpotensi memengaruhi kualitas dan keselamatan pelayanan pasien.

Untuk menjawab tantangan tersebut, penelitian ini membangun model machine learning berbasis **XGBoost** yang diintegrasikan dengan metode **Explainable AI (SHAP)** sebagai alat bantu keputusan klinis triage IGD.

---

## Tujuan

### Tujuan Bisnis
Meningkatkan kualitas, konsistensi, dan transparansi proses triage IGD di RSU Aulia melalui sistem pendukung keputusan berbasis data yang dapat digunakan langsung oleh tenaga medis.

### Tujuan Teknis
1. Membangun model klasifikasi multi-kelas menggunakan XGBoost yang mampu memprediksi kategori triage berdasarkan 12 variabel klinis pasien.
2. Mengoptimalkan performa model melalui hyperparameter tuning dengan RandomizedSearchCV (100 iterasi, 5-Fold Cross-Validation).
3. Mengintegrasikan SHAP (TreeSHAP) agar setiap prediksi dapat dijelaskan secara klinis — baik secara global maupun per kasus pasien.
4. Mengembangkan prototipe sistem berbasis Streamlit yang dapat dioperasikan tenaga medis di IGD secara langsung.

---

## Tahapan Pembangunan Model (CRISP-DM)

### 1. Business Understanding
Identifikasi kebutuhan institusional RSU Aulia, penetapan tujuan bisnis dan tujuan data mining, penilaian situasi (ketersediaan data, sumber daya, risiko), serta perencanaan proyek keseluruhan.

### 2. Data Understanding
Pengumpulan 1.000–3.000 rekam medis kunjungan IGD periode 2023–2024 dari RSU Aulia. Eksplorasi mencakup statistik deskriptif, distribusi kelas target, distribusi variabel prediktor, dan verifikasi kualitas data (missing values, outlier, duplikasi) terhadap 13 variabel dataset.

**Variabel dataset (13 variabel):**
- Usia, jenis kelamin, cara kedatangan, keluhan utama
- Skor GCS, skala nyeri
- Tekanan darah sistolik & diastolik, detak jantung, laju pernapasan, suhu tubuh, SpO₂
- Kategori keputusan triage aktual *(variabel target)*

### 3. Data Preparation
- **Pembersihan data** — imputasi median untuk variabel numerik, imputasi modus untuk variabel kategorikal, penanganan outlier berbasis konteks klinis
- **Transformasi** — Binary Encoding, One-Hot Encoding, Target Encoding, normalisasi Min-Max
- **Rekayasa fitur** — pengelompokan usia, perhitungan Mean Arterial Pressure (MAP), pembentukan fitur indikator tanda vital abnormal dan fitur agregat jumlah tanda vital abnormal
- **Pembagian dataset** — Stratified Hold-Out 80:20, Stratified 5-Fold Cross-Validation, pencegahan data leakage menggunakan Pipeline

### 4. Modeling
- **Algoritma** — XGBoost dengan fungsi objektif softmax (klasifikasi multi-kelas)
- **Optimasi** — RandomizedSearchCV, 100 iterasi, 5-Fold CV, optimasi berbasis F1-score makro
- **Interpretabilitas** — analisis SHAP global (Summary Plot, Bar Plot, Dependence Plot) dan lokal (Waterfall Plot, Force Plot)
- **Penyimpanan model** — format binary Joblib

### 5. Evaluation
- **Metrik evaluasi** — Akurasi, Precision, Recall, F1-Score (macro average & weighted average)
- **Analisis** — confusion matrix per kelas triage, perbandingan model baseline vs model teroptimasi
- **Validasi klinis** — verifikasi konsistensi fitur dominan SHAP terhadap relevansi klinis yang diakui dalam standar triage medis

### 6. Deployment
- **Arsitektur** — aplikasi berbasis Streamlit (Python), terdiri dari tiga komponen: antarmuka pengguna, Prediction Engine, dan SHAP Explainer
- **Fitur sistem** — input data pasien, prediksi kategori triage real-time, panel hasil dengan indikator warna triage, visualisasi SHAP per prediksi, dan rekayasa fitur otomatis
- **Error handling** — validasi input dan penanganan kesalahan pada setiap tahap pemrosesan

---

## Hasil yang Diharapkan

| # | Hasil | Deskripsi |
|---|-------|-----------|
| 1 | Model prediksi akurat | XGBoost teroptimasi yang mampu mengklasifikasikan kategori triage IGD dengan F1-score macro yang memadai secara klinis |
| 2 | Prediksi yang dapat dijelaskan | Analisis SHAP mengidentifikasi fitur paling berpengaruh per prediksi, memvalidasi kesesuaian model dengan pengetahuan klinis |
| 3 | Prototipe sistem fungsional | Aplikasi Streamlit siap digunakan tenaga medis IGD dengan antarmuka input, output prediksi, dan visualisasi SHAP real-time |
| 4 | Dasar keputusan berbasis data | RSU Aulia memiliki fondasi awal untuk standardisasi triage yang lebih objektif dan terdokumentasi |
| 5 | Kontribusi akademik | Dokumentasi lengkap proses CRISP-DM beserta justifikasi teknis dan klinis setiap keputusan penelitian |

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Algoritma utama | XGBoost (softmax multi-class) |
| Interpretabilitas | SHAP / TreeSHAP |
| Optimasi hyperparameter | RandomizedSearchCV (Scikit-learn) |
| Preprocessing pipeline | Scikit-learn Pipeline |
| Normalisasi | Min-Max Normalization |
| Framework antarmuka | Streamlit (Python) |
| Penyimpanan model | Joblib binary format |
| Evaluasi | Stratified 5-Fold Cross-Validation |

---

## Catatan Penting

> **Model ini bersifat decision support, bukan pengganti keputusan klinis.** Rekomendasi yang dihasilkan sistem dimaksudkan sebagai alat bantu bagi dokter dan perawat — keputusan akhir tetap berada di tangan tenaga medis yang bertugas. Sistem yang dikembangkan merupakan prototipe penelitian, bukan sistem produksi siap deploy penuh.

---

*Dokumen ini merupakan ringkasan bisnis dari skripsi undergraduate Reymondo mengenai Prediksi Skor Triage IGD menggunakan XGBoost dan Explainable AI (SHAP) di RSU Aulia.*
