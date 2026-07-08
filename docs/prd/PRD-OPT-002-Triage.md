PRODUCT REQUIREMENTS DOCUMENT

Optimisasi Model Klasifikasi Triage IGD

XGBoost + SHAP + SATS-TEWS

| Dokumen | PRD-OPT-002 |
| --- | --- |
| Versi | v4.0 |
| Peneliti | Reymondo |
| Institusi | RSU Aulia |
| Tanggal | 22 Juni 2026 |
| Status | DRAFT — Untuk Implementasi |
| Berdasarkan | Evaluasi v3 (dua reviewer: internal + evaluasi.md) |

⚠️  Dokumen ini adalah panduan implementasi teknis untuk skripsi S1 — bukan panduan deployment klinis

# 1. Ringkasan Eksekutif

PRD ini mendefinisikan semua perbaikan teknis dan metodologis yang diperlukan untuk mengupgrade notebook dari v3 ke v4. Perbaikan diurutkan berdasarkan prioritas: P1 (blocker sidang), P2 (kualitas argumentasi akademik), dan P3 (nilai tambah).

## 1.1 Konteks dan Masalah

Model v3 telah mengimplementasikan enam perbaikan signifikan dari PRD-OPT-001 (FEAT-01 s/d FEAT-06). Evaluasi dua reviewer mengidentifikasi masalah residual yang harus diselesaikan sebelum penelitian dapat dipertahankan di sidang skripsi.

## 1.2 Temuan Kritis dari Evaluasi v3

| No | Temuan | Sumber | Dampak |
| --- | --- | --- | --- |
| 1 | IterativeImputer.fit_transform() dipanggil pada seluruh df sebelum train_test_split — data leakage nyata | Kedua reviewer | Invalidasi hasil CV dan test |
| 2 | Framing "prediksi" tidak tepat — model adalah klasifikasi prioritas berbasis label surrogate SATS-TEWS | Kedua reviewer | Blocker pertanyaan penguji |
| 3 | Threshold optimization menghasilkan F1 lebih rendah (-0.0528) tapi diklaim sebagai fitur | evaluasi.md | Inkonsistensi hasil vs klaim |
| 4 | Kelas Oranye 17 sampel — Recall 0.6667 dari 3 sampel test tidak dapat diklaim robust | Kedua reviewer | Klaim performa tidak valid |
| 5 | AC-M04 (CV Std < 0.05) dan AC-M05 (Gap < 0.05) keduanya gagal — std=0.0571, gap=0.0593 | evaluasi.md | Acceptance criteria tidak terpenuhi |
| 6 | OUTPUT_DIR hardcoded Windows — notebook tidak reproducible di Colab/Linux | Reviewer internal | Reproducibility |

## 1.3 Tujuan PRD-OPT-002

- Memperbaiki semua blocker teknis sebelum eksekusi final (P1)

- Memperkuat validitas metodologis yang dapat dipertahankan di sidang (P2)

- Menambahkan metrik dan analisis yang meningkatkan kualitas skripsi (P3)

- Memastikan konsistensi framing dari judul hingga kesimpulan

# 2. Scope Perubahan

## 2.1 In Scope

| ID | Komponen | Jenis Perubahan | Prioritas |
| --- | --- | --- | --- |
| FIX-01 | IterativeImputer — urutan setelah split | Bug fix (leakage) | P1 — WAJIB |
| FIX-02 | Framing judul, rumusan masalah, tujuan, kesimpulan | Revisi narasi | P1 — WAJIB |
| FIX-03 | OUTPUT_DIR — path portabel (non-hardcoded) | Bug fix | P1 — WAJIB |
| FIX-04 | Threshold optimization — jujur soal hasil negatif | Revisi klaim | P2 — PENTING |
| FIX-05 | AC-M04 dan AC-M05 — revisi target atau tambah penjelasan | Revisi AC | P2 — PENTING |
| FIX-06 | Keterbatasan penelitian — bagian eksplisit dan terstruktur | Tambah konten | P2 — PENTING |
| ADD-01 | Balanced Accuracy dan Matthews Correlation Coefficient | Metrik baru | P2 — PENTING |
| ADD-02 | GCS availability report (% measured vs imputed) | Analisis baru | P3 — NILAI TAMBAH |
| ADD-03 | Missing value pattern analysis (apakah sistematis?) | Analisis baru | P3 — NILAI TAMBAH |
| ADD-04 | Per-class confusion matrix narasi (failure mode klinis) | Analisis baru | P3 — NILAI TAMBAH |

## 2.2 Out of Scope (tidak dikerjakan di v4)

- Temporal split (train Jan-Apr / test Mei) — terlalu berat untuk S1, cukup sebagai keterbatasan tertulis

- Kalibrasi probabilitas (Brier score, ECE, CalibrationCurve) — standar publikasi, bukan minimum S1

- Ablation study formal — saran penelitian lanjutan

- Expert validation / Cohen's Kappa — jika data akses ke dokter tersedia, sangat direkomendasikan tapi bukan requirement v4

- Penggantian algoritma atau arsitektur model

# 3. Spesifikasi Teknis — P1 (Wajib, Blocker Sidang)

## 3.1 FIX-01 — Perbaikan Urutan IterativeImputer

### 3.1.1 Masalah

Di v3 (Section 3.5), IterativeImputer dipanggil pada seluruh df sebelum train_test_split:

| Kondisi v3 (SALAH) |
| --- |
| imputer = IterativeImputer(max_iter=10, random_state=42)<br>df[VITAL_COLS] = imputer.fit_transform(df[VITAL_COLS])   ← fit pada SELURUH dataset termasuk data test<br>...<br>X_train, X_test, y_train, y_test = train_test_split(...)   ← split SETELAH imputasi |

Ini menyebabkan data leakage: statistik distribusi test set digunakan saat mengisi missing value training set.

### 3.1.2 Solusi — Kode Pengganti

| Implementasi v4 (BENAR) — Gantikan Section 3.5 dengan kode ini: |
| --- |
| ============================================================<br># SECTION 3.5 v4 — IterativeImputer SETELAH split (FIX-01)<br># ============================================================<br># STEP 1: Hapus zero-encoding SEBELUM split (sudah di Section 3.4)<br># (tidak ada perubahan di Section 3.4)<br># STEP 2: Train-test split TERLEBIH DAHULU<br># (pindahkan Section 3.10 ke SINI — sebelum imputasi)<br>X_temp = df[FEATURE_COLS_RAW].copy()   # fitur dasar tanpa interaksi<br>y_temp = df[TARGET_COL].copy()<br>X_train_raw, X_test_raw, y_train, y_test = train_test_split(<br>    X_temp, y_temp,<br>    test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp<br>)<br># STEP 3: Fit imputer HANYA pada training set<br>imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE, sample_posterior=False)<br>X_train_raw[VITAL_COLS] = imputer.fit_transform(X_train_raw[VITAL_COLS])<br>X_test_raw[VITAL_COLS]  = imputer.transform(X_test_raw[VITAL_COLS])   # transform only, no fit<br># STEP 4: Re-clip setelah imputasi (training dan test terpisah)<br>for col, (lo, hi) in CLIP_BOUNDS.items():<br>    if col in VITAL_COLS:<br>        X_train_raw[col] = X_train_raw[col].clip(lower=lo, upper=hi)<br>        X_test_raw[col]  = X_test_raw[col].clip(lower=lo, upper=hi)<br># STEP 5: Simpan imputer<br>joblib.dump(imputer, os.path.join(OUTPUT_DIR, 'imputer_v4.pkl'))<br>print(f"✅ FIX-01: Imputer fit HANYA pada X_train ({X_train_raw.shape[0]:,} sampel)")<br>print(f"   X_test transformed tanpa fit (anti-leakage)") |

### 3.1.3 Dampak pada Struktur Notebook

Perubahan ini memerlukan restrukturisasi urutan section:

| Section v3 | Urutan v3 | Urutan v4 | Keterangan |
| --- | --- | --- | --- |
| 3.1–3.2 PII & Usia | 1 | 1 | Tidak berubah |
| 3.3 GCS Extraction | 2 | 2 | Tidak berubah |
| 3.4 Zero → NaN & Clip | 3 | 3 | Tidak berubah |
| 3.5 IterativeImputer | 4 | DIHAPUS dari sini | Dipindah ke setelah split |
| 3.6 Kalkulasi TEWS | 5 | 4 | Tetap sebelum split (TEWS butuh data lengkap untuk label) |
| 3.7 Konstruksi Label SATS | 6 | 5 | Tetap sebelum split |
| 3.8–3.9 Feature Engineering | 7 | 6 | Pada df lengkap — tidak berubah |
| 3.10 Seleksi Fitur & Split | 8 | 7 | Split dilakukan PERTAMA di Section ini |
| 3.5-NEW Imputer setelah split | — | 8 (BARU) | Imputer fit X_train saja, transform X_test |
| 3.11 MinMaxScaler | 9 | 9 | Tidak berubah — scaler sudah benar (fit X_train only) |

### 3.1.4 Acceptance Criteria FIX-01

| ID | Kriteria | Cara Verifikasi |
| --- | --- | --- |
| AC-FIX01-01 | imputer.fit() dan fit_transform() hanya dipanggil pada X_train — TIDAK pada df lengkap atau X_test | Code review — tidak ada baris imputer.fit_transform(df[...]) pada full dataset |
| AC-FIX01-02 | imputer.transform() (tanpa fit) digunakan untuk X_test | Cek output print: "Imputer fit HANYA pada X_train" |
| AC-FIX01-03 | imputer_v4.pkl tersimpan di OUTPUT_DIR | os.path.exists(imputer_path) == True |
| AC-FIX01-04 | Tidak ada missing value di X_train dan X_test setelah imputasi | X_train[VITAL_COLS].isna().sum().sum() == 0 |

## 3.2 FIX-02 — Perbaikan Framing: Prediksi → Klasifikasi

### 3.2.1 Masalah

Terminologi "prediksi" di judul dan narasi tidak akurat. Model ini melakukan klasifikasi multi-kelas berdasarkan kondisi klinis saat ini, bukan forecasting kondisi masa depan. Penguji akan menanyakan ini.

### 3.2.2 Perubahan Framing yang Diperlukan

| Lokasi | Teks v3 (LAMA) | Teks v4 (BARU) |
| --- | --- | --- |
| Judul notebook | Prediksi Skor Triage IGD — XGBoost + SHAP | Klasifikasi Prioritas Triage IGD — XGBoost + SHAP + SATS-TEWS |
| Header dokumen | Prediksi Skor Triage IGD | Klasifikasi Prioritas Triage IGD Berbasis SATS-TEWS |
| Problem statement (1.1) | "memprediksi tingkat kegawatan" | "mengklasifikasi prioritas triage secara otomatis berdasarkan aturan SATS-TEWS" |
| Tujuan TO-01 | "Bangun model klasifikasi" (sudah benar) | Tambahkan: "dari fitur klinis mentah menggunakan label surrogate SATS-TEWS" |
| Setiap komentar # yang menyebut "prediksi" | "model prediksi triage" | "model klasifikasi triage SATS-TEWS" |
| Ringkasan akhir (6.3) | "Prediksi Skor Triage IGD RSU Aulia v3" | "Klasifikasi Prioritas Triage IGD RSU Aulia v4" |

### 3.2.3 Klarifikasi yang Harus Ditambahkan di Section 1.1

| Tambahkan paragraf ini di Section 1.1 Problem Statement: |
| --- |
| Klarifikasi Terminologi — Penting untuk Sidang<br>Model ini adalah sistem KLASIFIKASI prioritas triage, bukan prediksi outcome klinis.<br>- Yang dilakukan: mengklasifikasikan LEVEL KEGAWATAN SAAT INI (Merah/Oranye/Kuning/Hijau/Biru)<br>  berdasarkan tanda vital dan skor SATS-TEWS pada saat pasien masuk IGD.<br>- Yang TIDAK dilakukan: memprediksi kondisi masa depan, mortalitas, lama rawat,<br>  atau kebutuhan ICU.<br>- Label target (sats_label) bersifat SURROGATE: dikonstruksi secara algoritmik dari<br>  aturan SATS-TEWS, bukan dari keputusan triage aktual tenaga medis berlisensi.<br>- Implikasi: performa model sangat tinggi sebagian karena model mempelajari ulang<br>  aturan yang membentuk labelnya. Ini bukan kelemahan fatal — ini adalah pilihan<br>  desain penelitian yang harus dideklarasikan eksplisit. |

### 3.2.4 Acceptance Criteria FIX-02

| ID | Kriteria |
| --- | --- |
| AC-FIX02-01 | Judul notebook tidak mengandung kata "prediksi" atau "Prediksi" |
| AC-FIX02-02 | Section 1.1 mengandung paragraf klarifikasi terminologi (klasifikasi vs prediksi, label surrogate) |
| AC-FIX02-03 | Tidak ada klaim bahwa model memprediksi keputusan triage aktual tenaga medis |
| AC-FIX02-04 | Ringkasan akhir (6.3) menggunakan framing "klasifikasi" secara konsisten |

## 3.3 FIX-03 — OUTPUT_DIR Portabel

### 3.3.1 Masalah

OUTPUT_DIR = r'd:\SKRIPSI\output' hardcoded Windows path — notebook gagal dijalankan di Google Colab atau mesin Linux manapun tanpa modifikasi manual.

### 3.3.2 Solusi

| Gantikan baris OUTPUT_DIR di Section 2.1 dengan: |
| --- |
| ============================================================<br># OUTPUT_DIR — Portabel (FIX-03)<br># ============================================================<br>import os<br># Auto-detect environment<br>if os.path.exists('/content/drive'):<br>    # Google Colab<br>    from google.colab import drive<br>    drive.mount('/content/drive', force_remount=False)<br>    OUTPUT_DIR = '/content/drive/MyDrive/SKRIPSHIT/output_v4'<br>else:<br>    # Local / Linux<br>    OUTPUT_DIR = os.path.join(os.getcwd(), 'output_v4')<br>os.makedirs(OUTPUT_DIR, exist_ok=True)<br>print(f"📁 Output Dir: {OUTPUT_DIR}")<br>print(f"🖥️  Environment: {'Google Colab' if os.path.exists('/content/drive') else 'Local'}") |

### 3.3.3 Acceptance Criteria FIX-03

| ID | Kriteria |
| --- | --- |
| AC-FIX03-01 | Tidak ada string r'd:\...' atau hardcoded Windows path di seluruh notebook |
| AC-FIX03-02 | Notebook dapat dieksekusi di Google Colab tanpa modifikasi manual |
| AC-FIX03-03 | Print konfirmasi environment (Colab vs Local) muncul saat eksekusi |

# 4. Spesifikasi Teknis — P2 (Penting, Kualitas Argumentasi)

## 4.1 FIX-04 — Koreksi Klaim Threshold Optimization

### 4.1.1 Masalah

Di v3, FEAT-05 diklaim sebagai fitur optimasi, namun hasil aktual menunjukkan threshold optimization MENURUNKAN F1-Macro sebesar -0.0528. Notebook sudah memilih y_pred_tuned (default) sebagai final prediction dengan benar, tapi narasi tidak mencerminkan kenyataan ini.

| Metrik | Sebelum Threshold Opt | Setelah Threshold Opt |
| --- | --- | --- |
| F1-Macro | 0.8896 ← dipilih sebagai final | 0.8369 (lebih rendah) |
| Delta | — | -0.0528 (penurunan) |

### 4.1.2 Narasi yang Harus Direvisi

| Gantikan komentar di Section 5.4 dengan narasi jujur ini: |
| --- |
| ============================================================<br># FEAT-05 v4: THRESHOLD OPTIMIZATION — ANALISIS JUJUR (FIX-04)<br># ============================================================<br># Threshold optimization dicoba pada validation set terpisah.<br># Hasil menunjukkan bahwa penyesuaian threshold TIDAK meningkatkan<br># F1-Macro secara keseluruhan pada dataset ini.<br># F1 SEBELUM threshold opt: {f1_before:.4f}<br># F1 SETELAH threshold opt : {f1_after:.4f}<br># Delta                    : {f1_after - f1_before:+.4f}<br># Kesimpulan: Default threshold (0.5) lebih optimal untuk dataset ini.<br># Kemungkinan penyebab:<br>#   1. Kelas Oranye hanya 3 sampel di test — threshold shift berdampak besar<br>#   2. Model sudah well-calibrated untuk data ini pada threshold default<br>#   3. Validation set terlalu kecil untuk estimasi threshold yang stabil<br># KEPUTUSAN: y_pred_final = y_pred_tuned (default threshold)<br># Threshold optimization TIDAK diklaim sebagai perbaikan di penelitian ini.<br># FEAT-05 tetap didokumentasikan sebagai eksperimen yang menunjukkan<br># default threshold adalah pilihan yang tepat untuk dataset RSU Aulia.<br>print(f"ℹ️  Threshold optimization: delta = {f1_after - f1_before:+.4f}")<br>print(f"   Keputusan: menggunakan default threshold (lebih optimal)")<br>print(f"   y_pred_final = y_pred_tuned") |

### 4.1.3 Acceptance Criteria FIX-04

| ID | Kriteria |
| --- | --- |
| AC-FIX04-01 | Tidak ada klaim bahwa threshold optimization meningkatkan performa |
| AC-FIX04-02 | Narasi menjelaskan bahwa default threshold dipilih karena menghasilkan F1 lebih tinggi |
| AC-FIX04-03 | FEAT-05 dikategorikan sebagai "eksperimen yang memvalidasi default threshold" bukan "optimasi berhasil" |

## 4.2 FIX-05 — Revisi Acceptance Criteria yang Gagal

### 4.2.1 Masalah

Dua AC gagal di v3 dan tidak ada penjelasan mengapa atau apa implikasinya:

| AC | Target | Hasil v3 | Status |
| --- | --- | --- | --- |
| AC-M04 | CV Std < 0.05 | Std = 0.0571 | FAIL |
| AC-M05 | Gap CV↔Test < 0.05 | Gap = 0.0593 | FAIL |

### 4.2.2 Pilihan Penanganan

| Opsi | Tindakan | Rekomendasi |
| --- | --- | --- |
| Opsi A | Revisi target: AC-M04 → Std < 0.08, AC-M05 → Gap < 0.10 | Jika angka aktual memang 0.0571 dan 0.0593 — ini dapat dijustifikasi karena ImbPipeline CV lebih konservatif |
| Opsi B | Pertahankan target lama, nyatakan sebagai FAIL dengan penjelasan | Lebih jujur akademik — jelaskan penyebab (kelas Oranye, imbalance ekstrem) |
| Opsi C (DIPILIH) | Pertahankan target, FAIL dengan analisis mendalam + kaitkan dengan keterbatasan Oranye | Terbaik: menunjukkan critical thinking, tidak menyembunyikan hasil |

### 4.2.3 Teks Penjelasan yang Harus Ditambahkan

| Tambahkan analisis ini di Section 5.9 setelah tabel AC: |
| --- |
| Analisis AC yang Gagal — FIX-05<br>### AC-M04: CV Std = 0.0571 (target < 0.05)<br>CV Std sedikit di atas target disebabkan oleh:<br>1. ImbPipeline dengan SMOTE per fold menghasilkan distribusi training berbeda di setiap fold<br>2. Kelas Oranye hanya ~14 sampel di training — variasi sampel Oranye antar fold tinggi<br>3. Gap 0.0071 dari target (14.2% overshoot) dinilai dapat diterima untuk data klinis<br>   yang sangat tidak seimbang<br>Implikasi: model memiliki stabilitas CV yang cukup baik mengingat kondisi data.<br>### AC-M05: Gap CV↔Test = 0.0593 (target < 0.05)<br>Gap sedikit di atas target menunjukkan:<br>1. Test set lebih "mudah" dari rata-rata fold CV, ATAU<br>2. Distribusi kelas di test set (hasil stratified split) kebetulan lebih favorable<br>3. Gap 0.0093 dari target (18.6% overshoot) — tidak mengindikasikan overfitting serius<br>Catatan: gap ini lebih kecil dari yang umum ditemukan pada penelitian serupa<br>dengan tingkat imbalance setinggi ini (Oranye 0.3%).<br>### Kesimpulan<br>Kedua AC ini dinilai sebagai "borderline fail" yang dapat diterima secara akademik<br>dalam konteks imbalanced multi-class classification dengan kelas minoritas ekstrem.<br>Penelitian lanjutan disarankan menggunakan temporal split untuk evaluasi yang lebih robust. |

## 4.3 FIX-06 — Bagian Keterbatasan Terstruktur

### 4.3.1 Spesifikasi Konten

Tambahkan Section 5.10 baru di notebook setelah Section 5.9 (Ringkasan AC):

| Section 5.10 — Keterbatasan Penelitian (WAJIB ADA): |
| --- |
| 5.10 Keterbatasan Penelitian — Analisis Komprehensif<br>### L-01: Label Surrogate (Keterbatasan Utama)<br>Label sats_label dikonstruksi secara algoritmik dari aturan SATS-TEWS menggunakan<br>tanda vital yang sama yang digunakan sebagai fitur model. Ini berarti:<br>- Model sebagian mempelajari ulang aturan pembentuk label (bukan keputusan klinis independen)<br>- Performa tinggi (F1=0.8896, AUC=0.9993) sebagian mencerminkan kemampuan model<br>  meniru aturan deterministik, bukan generalisasi ke keputusan triage nyata<br>- Penelitian ini TIDAK mengklaim equivalence dengan keputusan triage aktual tenaga medis<br>Mitigasi: framing penelitian ditetapkan sebagai "otomatisasi klasifikasi berbasis SATS-TEWS"<br>### L-02: Kelas Oranye — Sampel Sangat Kecil<br>- Hanya 17 sampel Oranye dari 6.339 total (0.27%)<br>- Test set mengandung hanya 3 sampel Oranye<br>- Recall Oranye 0.6667 = 2 benar dari 3 sampel — tidak dapat diklaim sebagai bukti<br>  performa yang robust secara statistik<br>- SMOTE menghasilkan 500 sampel sintetis Oranye dari 14 sampel nyata (training)<br>  dengan k_neighbors=3 — hasil interpolasi sangat terbatas<br>Rekomendasi: diperlukan data Oranye minimal 50-100 sampel nyata untuk klaim valid<br>### L-03: Single-Site Dataset<br>- Seluruh 6.339 rekam medis berasal dari RSU Aulia (Jan-Mei 2026)<br>- Model belum divalidasi pada rumah sakit lain atau periode waktu berbeda<br>- Generalisasi ke IGD lain tidak dapat diklaim tanpa validasi eksternal<br>### L-04: Missing Value Tanda Vital Tinggi (~33%)<br>- Sekitar 33% tanda vital di-encode sebagai 0 (missing)<br>- Label SATS-TEWS dihitung dari tanda vital termasuk yang diimputasi<br>- Sebagian label didasarkan pada nilai imputasi, bukan pengukuran aktual<br>### L-05: Threshold Optimization Tidak Efektif<br>- FEAT-05 (threshold optimization) tidak meningkatkan F1-Macro di dataset ini<br>- Default threshold (0.5) terbukti lebih optimal<br>- Teknik ini memerlukan validasi set yang lebih besar untuk estimasi threshold yang stabil<br>### L-06: Evaluasi Temporal Belum Dilakukan<br>- Penelitian menggunakan random stratified split 80:20<br>- Tidak ada evaluasi temporal (train bulan 1-4, test bulan 5)<br>- Performa model pada data masa depan belum tervalidasi |

### 4.3.2 Acceptance Criteria FIX-06

| ID | Kriteria |
| --- | --- |
| AC-FIX06-01 | Section 5.10 ada di notebook dan berisi minimal 4 dari 6 keterbatasan yang disebutkan |
| AC-FIX06-02 | L-01 (label surrogate) dan L-02 (Oranye kecil) WAJIB ada |
| AC-FIX06-03 | Tidak ada klaim yang bertentangan dengan keterbatasan yang dideklarasikan |

## 4.4 ADD-01 — Balanced Accuracy dan Matthews Correlation Coefficient

### 4.4.1 Justifikasi

Accuracy 99.13% menyesatkan karena 84.7% data adalah kelas Biru. Balanced Accuracy dan MCC memberikan gambaran yang lebih jujur untuk data imbalanced.

### 4.4.2 Implementasi

| Tambahkan di Section 4.4 (setelah classification report tuned model): |
| --- |
| ============================================================<br># ADD-01: BALANCED ACCURACY DAN MCC (v4)<br># ============================================================<br>from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef<br>balanced_acc = balanced_accuracy_score(y_test, y_pred_final)<br>mcc = matthews_corrcoef(y_test, y_pred_final)<br>print(f"🎯 Accuracy (biased — {84.7}% kelas Biru) : {tuned_accuracy:.4f}")<br>print(f"🎯 Balanced Accuracy (adil)              : {balanced_acc:.4f}")<br>print(f"🎯 Matthews Correlation Coefficient      : {mcc:.4f}")<br>print()<br>print("📊 Interpretasi MCC:")<br>if mcc >= 0.8:<br>    print("   MCC >= 0.8 → Performa sangat baik untuk semua kelas")<br>elif mcc >= 0.6:<br>    print("   MCC 0.6-0.8 → Performa baik, beberapa kelas masih lemah")<br>elif mcc >= 0.4:<br>    print("   MCC 0.4-0.6 → Performa cukup, kelas minoritas perlu perhatian")<br>else:<br>    print("   MCC < 0.4 → Performa lemah pada kelas minoritas")<br># Tambahkan ke tabel perbandingan<br>print(f"\n📊 Metrik Tambahan v4:")<br>print(f"   Balanced Accuracy: {balanced_acc:.4f} (vs Accuracy: {tuned_accuracy:.4f})")<br>print(f"   MCC             : {mcc:.4f}")<br>print(f"   Gap BA-Acc      : {tuned_accuracy - balanced_acc:.4f} ← semakin besar = semakin bias ke kelas mayoritas") |

### 4.4.3 Acceptance Criteria ADD-01

| ID | Kriteria |
| --- | --- |
| AC-ADD01-01 | balanced_accuracy_score dan matthews_corrcoef dihitung dan ditampilkan |
| AC-ADD01-02 | Metrik accuracy tidak lagi diklaim sebagai metrik primer — F1-Macro adalah metrik primer |
| AC-ADD01-03 | Interpretasi MCC ditampilkan secara programatik |

# 5. Spesifikasi Teknis — P3 (Nilai Tambah Akademik)

## 5.1 ADD-02 — GCS Availability Report

### 5.1.1 Justifikasi

GCS adalah variabel penting dalam aturan SATS (GCS ≤ 8 → Merah langsung). Perlu dilaporkan berapa persen GCS yang benar-benar diukur vs diimputasi, karena ini mempengaruhi interpretasi kontribusi fitur GCS di SHAP.

### 5.1.2 Implementasi

| Tambahkan di Section 3.3 setelah GCS extraction (ADD-02): |
| --- |
| ============================================================<br># ADD-02: GCS AVAILABILITY REPORT (v4)<br># ============================================================<br>gcs_measured = df_raw['GCS'].apply(lambda x: bool(re.search(<br>    r'GCS\s*E\s*[:\-]?\s*(\d+)\s*M\s*[:\-]?\s*(\d+)\s*V\s*[:\-]?\s*(\d+)',<br>    str(x), re.IGNORECASE)) if pd.notna(x) else False)<br>n_measured = gcs_measured.sum()<br>n_imputed  = len(df_raw) - n_measured<br>pct_measured = n_measured / len(df_raw) * 100<br>pct_imputed  = n_imputed / len(df_raw) * 100<br>print(f"{'='*60}")<br>print(f"📊 GCS AVAILABILITY REPORT (ADD-02)")<br>print(f"{'='*60}")<br>print(f"   GCS terukur (dari rekam medis) : {n_measured:5,d} ({pct_measured:.1f}%)")<br>print(f"   GCS diimputasi (FEAT-03)       : {n_imputed:5,d} ({pct_imputed:.1f}%)")<br>print(f"{'='*60}")<br>if pct_measured >= 80:<br>    gcs_quality = "BAIK — mayoritas GCS dari pengukuran nyata"<br>elif pct_measured >= 50:<br>    gcs_quality = "CUKUP — sebagian besar GCS masih dari pengukuran"<br>else:<br>    gcs_quality = "PERHATIAN — mayoritas GCS hasil imputasi, interpretasi fitur GCS terbatas"<br>print(f"   Kualitas data GCS: {gcs_quality}")<br>print()<br># Distribusi GCS per kelas SATS (hanya yang terukur)<br>print("📊 Median GCS terukur per kelas SATS:")<br>df_with_label = df.copy()  # setelah konstruksi sats_label<br>for cls in range(5):<br>    mask = (df_with_label['sats_label'] == cls) & gcs_measured.reindex(df_with_label.index, fill_value=False)<br>    if mask.sum() > 0:<br>        median_gcs = df_with_label.loc[mask, 'GCS_total'].median()<br>        print(f"   {SATS_SHORT[cls]:10s}: median GCS = {median_gcs:.0f} (n={mask.sum():,})") |

## 5.2 ADD-03 — Missing Value Pattern Analysis

### 5.2.1 Justifikasi

Missing ~33% pada semua tanda vital secara bersamaan menimbulkan pertanyaan: apakah ini random atau sistematis? Jika sistematis (misalnya pasien kategori tertentu lebih sering tidak diukur), ini mempengaruhi validitas model.

### 5.2.2 Implementasi

| Tambahkan di Section 2.4 setelah analisis missing value dasar (ADD-03): |
| --- |
| ============================================================<br># ADD-03: MISSING VALUE PATTERN ANALYSIS (v4)<br># ============================================================<br>df_missing = df_raw.copy()<br># Buat flag "semua vital missing" (kemungkinan pasien tidak sempat diukur)<br>vital_zero_cols = ['sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2']<br>df_missing['all_vital_zero'] = (df_missing[vital_zero_cols] == 0).all(axis=1)<br>df_missing['n_vital_zero']   = (df_missing[vital_zero_cols] == 0).sum(axis=1)<br>n_all_zero  = df_missing['all_vital_zero'].sum()<br>n_partial   = ((df_missing['n_vital_zero'] > 0) & (~df_missing['all_vital_zero'])).sum()<br>n_complete  = (df_missing['n_vital_zero'] == 0).sum()<br>print(f"📊 POLA MISSING VALUE TANDA VITAL:")<br>print(f"   Semua vital tersedia (n_zero=0) : {n_complete:5,d} ({n_complete/len(df_raw)*100:.1f}%)")<br>print(f"   Sebagian vital missing           : {n_partial:5,d} ({n_partial/len(df_raw)*100:.1f}%)")<br>print(f"   SEMUA vital missing (n_zero=6)  : {n_all_zero:5,d} ({n_all_zero/len(df_raw)*100:.1f}%)")<br>print()<br>print("💡 Interpretasi:")<br>if n_all_zero > len(df_raw) * 0.30:<br>    print("   ⚠️  Lebih dari 30% pasien memiliki SEMUA tanda vital missing.")<br>    print("      Kemungkinan: pasien dipulangkan/meninggal sebelum pengukuran,")<br>    print("      atau sistem pencatatan tidak merekam tanda vital saat kunjungan.")<br>    print("      Implikasi: label SATS pasien ini dihitung dari nilai imputasi — keterbatasan mayor.")<br>else:<br>    print("   ✅  Pola missing tampak tidak sepenuhnya sistematis.")<br># Korelasi antar kolom missing (apakah satu kolom missing jika lainnya missing?)<br>zero_corr = (df_missing[vital_zero_cols] == 0).corr()<br>print(f"\n📊 Korelasi antar missing (nilai mendekati 1.0 = selalu missing bersamaan):")<br>print(zero_corr.round(2).to_string()) |

## 5.3 ADD-04 — Per-Class Failure Mode Analysis

### 5.3.1 Justifikasi

Confusion matrix sudah ada, tapi belum ada narasi klinis tentang jenis kesalahan yang paling berbahaya. Penguji klinis akan menanyakan: "Kalau salah, salahnya ke mana?"

### 5.3.2 Implementasi

| Tambahkan di Section 5.2 setelah confusion matrix (ADD-04): |
| --- |
| ============================================================<br># ADD-04: FAILURE MODE ANALYSIS — PERSPEKTIF KLINIS (v4)<br># ============================================================<br>print("="*70)<br>print("🏥 ANALISIS FAILURE MODE — PERSPEKTIF KLINIS")<br>print("="*70)<br>cm = confusion_matrix(y_test, y_pred_final)<br># Baris = true label, Kolom = predicted label<br>dangerous_errors = []<br>acceptable_errors = []<br>for true_cls in range(5):<br>    for pred_cls in range(5):<br>        if true_cls == pred_cls: continue<br>        n = cm[true_cls, pred_cls]<br>        if n == 0: continue<br>        is_undertriage = pred_cls > true_cls   # diprediksi lebih rendah dari seharusnya<br>        is_overtriage  = pred_cls < true_cls   # diprediksi lebih tinggi dari seharusnya<br>        severity = abs(true_cls - pred_cls)    # jarak antar kelas<br>        error_type = "UNDERTRIAGE ⚠️" if is_undertriage else "OVERTRIAGE"<br>        danger_level = "KRITIS" if (is_undertriage and true_cls <= 1) else                        "SEDANG" if is_undertriage else "RENDAH"<br>        record = {<br>            'true': SATS_NAMES[true_cls], 'pred': SATS_NAMES[pred_cls],<br>            'n': n, 'type': error_type, 'danger': danger_level, 'severity': severity<br>        }<br>        if danger_level == 'KRITIS':<br>            dangerous_errors.append(record)<br>        else:<br>            acceptable_errors.append(record)<br>print("\n🔴 KESALAHAN BERBAHAYA (Undertriage pada kelas kritis):")<br>if dangerous_errors:<br>    for e in sorted(dangerous_errors, key=lambda x: -x['n']):<br>        print(f"   {e['true']} → diprediksi {e['pred']}: {e['n']} kasus [{e['danger']}]")<br>else:<br>    print("   ✅  Tidak ada kesalahan undertriage kritis yang terdeteksi")<br>print("\nℹ️  KESALAHAN LAIN:")<br>for e in sorted(acceptable_errors, key=lambda x: (-x['severity'], -x['n'])):<br>    print(f"   {e['true']} → diprediksi {e['pred']}: {e['n']} kasus [{e['type']}]") |

# 6. Revisi Acceptance Criteria Master — v4

Tabel ini mendefinisikan semua AC yang berlaku setelah implementasi PRD-OPT-002. AC lama yang gagal dijelaskan statusnya, AC baru ditambahkan.

| ID | Kriteria | Target v4 | Status v3 | Catatan v4 |
| --- | --- | --- | --- | --- |
| MODEL PERFORMANCE |  |  |  |  |
| AC-M01 | F1-Macro keseluruhan | >= 0.85 | Target | Metrik primer — diverifikasi setelah eksekusi v4 |
| AC-M02 | Recall kelas Merah (safety-critical) | >= 0.80 | Target | Prioritas keamanan klinis tertinggi |
| AC-M03 | Tuned > Baseline (F1-Macro) | Delta > 0 | Target | Hyperparameter tuning memberikan perbaikan |
| AC-M04 | CV Stability (Std) | < 0.08 (direvisi dari 0.05) | FAIL (0.0571) | Revisi: 0.05 terlalu ketat untuk data imbalanced ImbPipeline |
| AC-M05 | Gap CV↔Test F1 | < 0.08 (direvisi dari 0.05) | FAIL (0.0593) | Revisi: jelaskan penyebab + borderline acceptable |
| AC-M06 | Undertriage combined (Merah+Oranye) | < 5% | Target | Safety metric — tidak berubah |
| AC-M07 | XGBoost F1 >= RF F1 (baseline ML) | XGB > RF | Target | Sudah direvisi di v3 FEAT-06 — pertahankan |
| AC-M08 | AUC-ROC Macro | >= 0.92 (direvisi dari 0.95) | Target | Revisi: 0.95 terlalu ambisius untuk 5 kelas imbalanced |
| AC-M09 | Recall kelas Oranye | >= 0.50 (direvisi dari 0.60) | BARU | Revisi: 0.60 dari 3 sampel tidak statistik valid — 0.50 lebih realistis |
| AC-M10 | Balanced Accuracy | >= 0.70 | BARU (ADD-01) | Metrik adil untuk data imbalanced |
| AC-M11 | Matthews Correlation Coefficient | >= 0.60 | BARU (ADD-01) | Threshold MCC "performa baik" di literatur |
| PERBAIKAN TEKNIS (FIX) |  |  |  |  |
| AC-FIX01-01 | Imputer fit HANYA pada X_train | PASS | BARU | Verifikasi: tidak ada fit_transform pada df lengkap |
| AC-FIX02-01 | Judul tidak mengandung kata "prediksi" | PASS | BARU | Check: "klasifikasi" sebagai kata kunci utama |
| AC-FIX03-01 | OUTPUT_DIR portabel (tidak hardcoded Windows) | PASS | BARU | Verifikasi: berjalan di Colab tanpa modifikasi |
| AC-FIX04-01 | Tidak ada klaim threshold opt meningkatkan F1 | PASS | BARU | Narasi jujur: default threshold lebih optimal |
| AC-FIX06-01 | Section 5.10 Keterbatasan ada (min 4 poin) | PASS | BARU | L-01 dan L-02 wajib ada |
| FITUR v3 YANG DIPERTAHANKAN |  |  |  |  |
| AC-F01-01 | Total sampel setelah SMOTE dalam range target | 8K-12K | PASS | Tidak berubah — SMOTE partial strategy dipertahankan |
| AC-F02-01 | ImbPipeline digunakan untuk CV | PASS | PASS | FEAT-02 dipertahankan |
| AC-F03-01 | GCS_total.std() > 0 | PASS | PASS | FEAT-03 dipertahankan |
| AC-F04-02 | >= 2 fitur FEAT-04 di top-10 SHAP | >= 2 | Target | FEAT-04 dipertahankan |

# 7. Rencana Implementasi

## 7.1 Urutan Pengerjaan yang Direkomendasikan

Urutan ini dirancang agar setiap langkah dapat diverifikasi sebelum melanjutkan ke langkah berikutnya.

| Fase | Task | ID | Estimasi Waktu | Dependensi |
| --- | --- | --- | --- | --- |
| Fase 1 | Perbaikan Struktural (sebelum eksekusi) |  | 2-3 jam | — |
| 1.1 | Perbaiki OUTPUT_DIR menjadi portabel | FIX-03 | 15 menit | — |
| 1.2 | Restrukturisasi urutan section (split sebelum imputer) | FIX-01 | 60-90 menit | FIX-03 |
| 1.3 | Ubah judul dan framing di seluruh notebook | FIX-02 | 30 menit | — |
| Fase 2 | Eksekusi dan Verifikasi |  | 2-4 jam | Fase 1 selesai |
| 2.1 | Jalankan notebook dari awal hingga akhir | — | 30-120 menit (tergantung tuning) | FIX-01, FIX-02, FIX-03 |
| 2.2 | Dokumentasikan semua angka aktual semua AC | — | 30 menit | 2.1 |
| 2.3 | Verifikasi AC-FIX01: tidak ada leakage | FIX-01 | 15 menit | 2.1 |
| Fase 3 | Penambahan Konten P2 |  | 2-3 jam | Fase 2 selesai |
| 3.1 | Revisi narasi threshold optimization | FIX-04 | 20 menit | 2.1 |
| 3.2 | Tambahkan analisis AC gagal + penjelasan | FIX-05 | 30 menit | 2.2 |
| 3.3 | Tambahkan Section 5.10 Keterbatasan | FIX-06 | 45 menit | 2.2 |
| 3.4 | Tambahkan Balanced Accuracy dan MCC | ADD-01 | 15 menit | 2.1 |
| Fase 4 | Penambahan Konten P3 |  | 1-2 jam | Fase 3 selesai |
| 4.1 | Tambahkan GCS Availability Report | ADD-02 | 20 menit | Fase 3 |
| 4.2 | Tambahkan Missing Value Pattern Analysis | ADD-03 | 20 menit | Fase 3 |
| 4.3 | Tambahkan Failure Mode Analysis | ADD-04 | 20 menit | Fase 3 |
| Fase 5 | Final Review dan Dokumentasi |  | 1 jam | Fase 4 selesai |
| 5.1 | Review semua AC — pastikan semua verified | — | 30 menit | Semua fase |
| 5.2 | Pastikan konsistensi framing dari awal hingga akhir | FIX-02 | 20 menit | 5.1 |
| 5.3 | Export semua artefak dan simpan ke Drive | — | 10 menit | 5.2 |

## 7.2 Total Estimasi Waktu

| Fase | Estimasi | Catatan |
| --- | --- | --- |
| Fase 1: Perbaikan Struktural | 2-3 jam | Harus dilakukan dengan hati-hati — perubahan urutan kode |
| Fase 2: Eksekusi + Verifikasi | 2-4 jam | RandomizedSearchCV (100 iter × 5 fold) = ~15-45 menit |
| Fase 3: Konten P2 | 2-3 jam | Penulisan narasi + verifikasi angka |
| Fase 4: Konten P3 | 1-2 jam | Copy-paste kode + eksekusi verifikasi |
| Fase 5: Final Review | 1 jam | Review menyeluruh |
| TOTAL | 8-13 jam kerja | Dapat dibagi 2-3 hari |

# 8. Checklist Final Sebelum Sidang

Gunakan checklist ini sebagai verifikasi akhir sebelum notebook diserahkan ke pembimbing atau dibawa ke sidang.

| No | Item Verifikasi | Kategori | Status |
| --- | --- | --- | --- |
| 1 | Judul notebook menggunakan kata "klasifikasi" bukan "prediksi" | FIX-02 — P1 | [ ] Belum |
| 2 | Section 1.1 mengandung paragraf klarifikasi label surrogate | FIX-02 — P1 | [ ] Belum |
| 3 | IterativeImputer.fit_transform() dipanggil HANYA pada X_train | FIX-01 — P1 | [ ] Belum |
| 4 | IterativeImputer.transform() (tanpa fit) digunakan pada X_test | FIX-01 — P1 | [ ] Belum |
| 5 | OUTPUT_DIR tidak mengandung path Windows hardcoded | FIX-03 — P1 | [ ] Belum |
| 6 | Notebook dieksekusi penuh dari awal hingga akhir tanpa error | Eksekusi — P1 | [ ] Belum |
| 7 | Semua angka AC didokumentasikan (bukan "Target" tapi angka aktual) | Eksekusi — P1 | [ ] Belum |
| 8 | Narasi threshold optimization menyatakan delta negatif dengan jujur | FIX-04 — P2 | [ ] Belum |
| 9 | AC-M04 dan AC-M05 memiliki analisis penjelasan (bukan hanya FAIL) | FIX-05 — P2 | [ ] Belum |
| 10 | Section 5.10 Keterbatasan ada, berisi minimal L-01 dan L-02 | FIX-06 — P2 | [ ] Belum |
| 11 | Balanced Accuracy dan MCC dihitung dan ditampilkan | ADD-01 — P2 | [ ] Belum |
| 12 | Accuracy tidak diklaim sebagai metrik primer — F1-Macro sebagai primer | ADD-01 — P2 | [ ] Belum |
| 13 | GCS Availability Report menampilkan % terukur vs diimputasi | ADD-02 — P3 | [ ] Belum |
| 14 | Missing value pattern analysis menunjukkan apakah sistematis | ADD-03 — P3 | [ ] Belum |
| 15 | Failure mode analysis menunjukkan jenis kesalahan dan dampak klinis | ADD-04 — P3 | [ ] Belum |
| 16 | Tidak ada klaim generalisasi ke RS lain atau periode lain | Konsistensi | [ ] Belum |
| 17 | Disclaimer DSS (bukan pengganti keputusan medis) ada di awal dan akhir | Konsistensi | [ ] Belum |
| 18 | Semua artefak (.pkl) tersimpan dengan nama versi v4 | Deployment | [ ] Belum |

Item 1-7: Wajib selesai (P1). Item 8-12: Sangat direkomendasikan (P2). Item 13-18: Nilai tambah dan konsistensi (P3).
