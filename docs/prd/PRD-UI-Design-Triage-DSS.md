# PRD — UI Design: Sistem Prediksi Triage IGD
**Dokumen:** PRD-UI-001  
**Versi:** 1.0  
**Proyek:** Prediksi Triage IGD RSU Aulia — XGBoost + SHAP + SATS  
**Peneliti:** Reymondo  
**Tanggal:** 2026-06-18  
**Status:** Draft — Wireframe Review

---

## 1. Overview

Dokumen ini mendefinisikan spesifikasi desain UI untuk aplikasi **Decision Support System (DSS) Prediksi Triage IGD** berbasis model XGBoost + SHAP dengan standar SATS-TEWS. Aplikasi terdiri dari **3 tab utama** yang dirancang untuk digunakan oleh petugas IGD (perawat/dokter jaga) pada perangkat desktop atau tablet.

### 1.1 Tujuan Dokumen

- Mendefinisikan komponen UI per tab secara lengkap
- Menetapkan spesifikasi input/output setiap komponen
- Menjadi acuan implementasi Streamlit atau framework frontend lain
- Mendukung kelengkapan dokumentasi skripsi (bab Implementasi/Perancangan Sistem)

### 1.2 Pengguna Target

| Pengguna | Konteks | Prioritas |
|----------|---------|-----------|
| Perawat IGD | Input data pasien saat triase awal | Primer |
| Dokter jaga IGD | Verifikasi hasil prediksi | Primer |
| Mahasiswa/residen | Edukasi & simulasi | Sekunder |

### 1.3 Prinsip Desain

- **Kecepatan input:** Target waktu pengisian form < 60 detik per pasien
- **Safety-first:** Tampilan kritis (Merah/Oranye) harus dominan secara visual
- **Transparansi:** Setiap prediksi disertai penjelasan SHAP yang dapat dibaca petugas
- **Disclaimer wajib:** Semua output dilengkapi disclaimer peran DSS — bukan pengganti keputusan medis
- **Responsive:** Mendukung desktop (1280px+) dan tablet (768px+)

---

## 2. Arsitektur Navigasi

```
Aplikasi DSS Triage IGD
├── Tab 1: Input Pasien & Hasil Prediksi     [TAB-01]
├── Tab 2: Kalkulator TEWS                   [TAB-02]
└── Tab 3: Panduan SATS                      [TAB-03]
```

### Header Global (Semua Tab)

| Elemen | Spesifikasi |
|--------|-------------|
| Judul aplikasi | "Sistem Prediksi Triage IGD" |
| Sub-judul | "RSU Aulia · SATS-TEWS · XGBoost + SHAP" |
| Badge versi model | "Model v3.0" — dinamis dari metadata file `.pkl` |
| Badge status model | "Aktif" (hijau) / "Tidak Tersedia" (merah) |
| Navigasi tab | Tab bar horizontal: Input Pasien / Kalkulator TEWS / Panduan SATS |

---

## 3. TAB-01 — Input Pasien & Hasil Prediksi

### 3.1 Deskripsi

Tab utama. Petugas mengisi data pasien, menekan tombol prediksi, dan melihat hasil klasifikasi SATS beserta penjelasan SHAP. Layout dua kolom pada desktop (kiri: input, kanan: output), satu kolom pada tablet/mobile.

### 3.2 Panel Kiri — Input Pasien

#### 3.2.1 Seksi A — Identitas & Demografis

| ID Komponen | Tipe | Label | Validasi | Keterangan |
|-------------|------|-------|----------|------------|
| INP-A01 | Text input | Nama pasien | Opsional, max 100 char | Tidak disimpan ke model; hanya untuk referensi petugas |
| INP-A02 | Number input | Usia (tahun) | Integer, 0–120 | Wajib. Bayi < 1 tahun dimasukkan sebagai 0 |
| INP-A03 | Toggle button (2 opsi) | Jenis kelamin | Wajib | Pilihan: "Laki-laki" / "Perempuan" |
| INP-A04 | Dropdown | Cara datang | Wajib | Pilihan: Sendiri / KLL (Kecelakaan) / Rujukan Puskesmas / Rujukan Dokter |

#### 3.2.2 Seksi B — Tanda Vital

Setiap field tanda vital memiliki:
- **Hint teks** menampilkan range normal klinis
- **Real-time flag** — field berubah warna jika nilai di luar batas klinis (warning: kuning, danger: merah)
- **Label satuan** tertampil di sisi kanan field

| ID Komponen | Tipe | Label | Satuan | Range Normal | Batas Klinis Warning | Batas Klinis Danger |
|-------------|------|-------|--------|--------------|---------------------|---------------------|
| INP-B01 | Number input | Sistole | mmHg | 110–149 | 90–109 atau 150–179 | < 90 atau ≥ 180 |
| INP-B02 | Number input | Diastole | mmHg | 60–90 | 50–59 atau 91–100 | < 50 |
| INP-B03 | Number input | Denyut jantung | bpm | 51–100 | 40–50 atau 101–110 | < 40 atau ≥ 130 |
| INP-B04 | Number input | Laju pernafasan | x/mnt | 12–20 | 9–11 atau 21–29 | < 9 atau ≥ 30 |
| INP-B05 | Number input (1 desimal) | Suhu tubuh | °C | 36.0–37.5 | 35.0–35.9 atau 37.6–38.4 | < 35.0 atau ≥ 38.5 |
| INP-B06 | Number input | SpO₂ | % | 97–100 | 95–96 | < 90 |

> **Catatan implementasi:** Jika nilai 0 dimasukkan, sistem memperlakukan sebagai nilai kosong (missing). Imputation dilakukan oleh `IterativeImputer` yang sudah di-fit pada training data.

#### 3.2.3 Seksi C — Neurologis & Nyeri

**GCS Sub-input (3 komponen terpisah):**

| ID Komponen | Tipe | Label | Pilihan |
|-------------|------|-------|---------|
| INP-C01 | Dropdown | Eye (E) | 4 — Spontan / 3 — Respons suara / 2 — Respons nyeri / 1 — Tidak ada |
| INP-C02 | Dropdown | Motor (M) | 6 — Ikuti perintah / 5 — Lokasi nyeri / 4 — Fleksi normal / 3 — Fleksi abnormal / 2 — Ekstensi / 1 — Tidak ada |
| INP-C03 | Dropdown | Verbal (V) | 5 — Orientasi baik / 4 — Bingung / 3 — Hanya kata-kata / 2 — Hanya suara / 1 — Tidak ada |

**GCS Total Display (read-only, auto-kalkulasi):**

| ID Komponen | Tipe | Deskripsi |
|-------------|------|-----------|
| INP-C04 | Display card | GCS Total = E + M + V, range 3–15 |
| INP-C05 | Display text | Interpretasi: ≤8 Koma berat / 9–12 Sedang / 13–14 Ringan / 15 Normal |

**Skala Nyeri:**

| ID Komponen | Tipe | Label | Spesifikasi |
|-------------|------|-------|-------------|
| INP-C06 | Range slider | Skala nyeri | Range 0–10, step 1. Label: 0 = Tidak nyeri, 5 = Nyeri sedang, 10 = Tidak tertahankan |

#### 3.2.4 Seksi D — Tombol Aksi

| ID Komponen | Tipe | Label | Behaviour |
|-------------|------|-------|-----------|
| ACT-01 | Button (primer, full-width) | "Prediksi Triage" | Validasi form → jalankan `predict_triage_v3()` → tampilkan output panel. Disabled jika ada field wajib kosong |
| ACT-02 | Button (sekunder) | "Reset Form" | Clear semua field ke nilai default. Confirmation dialog sebelum eksekusi |

**Validasi form sebelum prediksi:**
- Semua field bertanda "Wajib" harus terisi
- Nilai harus dalam batas klinis yang dapat diterima (batas absolut, bukan hanya warning)
- Jika tidak valid: tampilkan inline error message per field

---

### 3.3 Panel Kanan — Hasil Prediksi

Panel ini hanya muncul setelah tombol "Prediksi Triage" ditekan. Sebelum prediksi, tampilkan placeholder: *"Isi data pasien dan tekan Prediksi untuk melihat hasil."*

#### 3.3.1 Komponen A — SATS Badge Utama

| ID Komponen | Spesifikasi |
|-------------|-------------|
| OUT-A01 | Card besar dengan warna background dan border sesuai kelas SATS yang diprediksi |
| OUT-A02 | Dot warna bulat (36×36px) + Nama kelas besar (18px bold): "Merah — Resusitasi" dst. |
| OUT-A03 | Sub-teks waktu target: "Penanganan segera — 0 menit" / "< 10 menit" / "< 60 menit" / "< 4 jam" / "< 6 jam" |

**Mapping warna SATS:**

| Kelas | Background | Border | Teks judul | Dot warna |
|-------|-----------|--------|------------|-----------|
| Merah (0) | #FCEBEB | #F09595 | #791F1F | #E24B4A |
| Oranye (1) | #FAEEDA | #FAC775 | #633806 | #BA7517 |
| Kuning (2) | #FAEEDA | #EF9F27 | #412402 | #EF9F27 |
| Hijau (3) | #EAF3DE | #C0DD97 | #173404 | #639922 |
| Biru (4) | #E6F1FB | #85B7EB | #042C53 | #378ADD |

#### 3.3.2 Komponen B — Distribusi Probabilitas

| ID Komponen | Tipe | Spesifikasi |
|-------------|------|-------------|
| OUT-B01 | Progress bar set (5 baris) | Satu baris per kelas SATS. Label kiri: nama kelas. Bar fill: warna kelas. Label kanan: persentase (1 desimal) |
| OUT-B02 | Sorting | Kelas dengan probabilitas tertinggi selalu di urutan pertama |

#### 3.3.3 Komponen C — Clinical Flags Aktif

| ID Komponen | Spesifikasi |
|-------------|-------------|
| OUT-C01 | Chip/badge row | Tampilkan semua flag klinis yang aktif (bernilai 1) dari: flag_takikardia, flag_bradikardia, flag_hipotensi, flag_hipertensi, flag_takipnea, flag_hipoksia, flag_demam, flag_hipotermi, flag_hypoxic_shock |
| OUT-C02 | Warna chip | Danger (merah): hipotensi, hipoksia, hypoxic_shock. Warning (kuning): sisanya |
| OUT-C03 | Empty state | Jika tidak ada flag aktif: tampilkan "Tidak ada flag klinis abnormal" |

#### 3.3.4 Komponen D — Skor Klinis Turunan

Grid 2×2 metric card:

| ID | Label | Sumber | Interpretasi |
|----|-------|--------|--------------|
| OUT-D01 | Shock Index | `denyut_jantung / sistole` | Normal < 0.9 / Perlu perhatian 0.9–1.0 / Kritis > 1.0 |
| OUT-D02 | MEWS Approx | `mews_approx` dari `compute_tews()` | Rendah 0–2 / Sedang 3–4 / Tinggi ≥ 5 |
| OUT-D03 | TEWS Total | `TEWS_total` | Ditampilkan + zone SATS yang sesuai |
| OUT-D04 | Cardiorespiratory | `cardiorespiratory_score` (0–4) | 0 = Normal / 1–2 = Perhatian / 3–4 = Distres berat |

#### 3.3.5 Komponen E — SHAP Explainability Panel

| ID Komponen | Spesifikasi |
|-------------|-------------|
| OUT-E01 | Judul section | "Alasan prediksi — top fitur (SHAP)" |
| OUT-E02 | Legend | Merah = mendorong ke kelas lebih kritis / Biru = menekan ke kelas lebih aman |
| OUT-E03 | Horizontal bar chart | 8 fitur teratas (berdasarkan |SHAP value| untuk kelas yang diprediksi). Bar merah ke kanan (positif), bar biru ke kiri (negatif) dari garis tengah |
| OUT-E04 | Label per bar | Nama fitur (kiri, 12px) + nilai SHAP (kanan, 11px, warna sesuai arah) |
| OUT-E05 | Default state | Panel expand/collapse. Default: expanded |

#### 3.3.6 Komponen F — Riwayat Sesi

| ID Komponen | Spesifikasi |
|-------------|-------------|
| OUT-F01 | Tabel list | Daftar pasien yang sudah diprediksi dalam sesi aktif. Kolom: No urut / Waktu / Usia / Prediksi (dot warna + label) / Confidence (%) |
| OUT-F02 | Maksimum entri | 20 entri terakhir. Otomatis scroll |
| OUT-F03 | Tombol export | "Export CSV" — download riwayat sesi sebagai file `.csv` |
| OUT-F04 | Persistence | Data sesi tidak disimpan permanen (session state saja). Reset saat aplikasi ditutup |

#### 3.3.7 Disclaimer Wajib

| ID Komponen | Spesifikasi |
|-------------|-------------|
| OUT-G01 | Alert bar | Selalu tampil di bagian bawah panel output, tidak dapat di-hide |
| OUT-G02 | Teks | "Hasil ini adalah rekomendasi sistem pendukung keputusan (DSS). Keputusan triage final adalah wewenang eksklusif tenaga medis berlisensi." |
| OUT-G03 | Style | Background secondary, border-left warning, font 11px |

---

## 4. TAB-02 — Kalkulator TEWS

### 4.1 Deskripsi

Kalkulator TEWS (Triage Early Warning Score) mandiri. Memungkinkan petugas menghitung skor TEWS secara manual untuk edukasi, cross-check, atau penggunaan tanpa model ML. Hasil kalkulasi bersifat real-time — skor berubah seiring perubahan input.

### 4.2 Layout

Dua kolom: kiri (6 input parameter), kanan (output TEWS + breakdown skor).

### 4.3 Komponen Input

#### 4.3.1 Parameter Input (6 komponen)

| ID | Label | Tipe | Satuan | Spesifikasi |
|----|-------|------|--------|-------------|
| TEWS-I01 | Laju pernafasan | Number input | x/mnt | Range 0–80. Placeholder: 16 |
| TEWS-I02 | SpO₂ | Number input | % | Range 0–100. Placeholder: 98 |
| TEWS-I03 | Tekanan darah sistole | Number input | mmHg | Range 0–300. Placeholder: 120 |
| TEWS-I04 | Denyut jantung | Number input | bpm | Range 0–300. Placeholder: 80 |
| TEWS-I05 | Suhu tubuh | Number input (1 desimal) | °C | Range 30.0–45.0. Placeholder: 36.5 |
| TEWS-I06 | GCS total | Number input ATAU dropdown | — | Range 3–15 ATAU pilihan: 15/14/13/.../3. Placeholder: 15 |

> **Catatan UX:** Setiap field input langsung memicu recalculation skor TEWS (onChange event). Tidak ada tombol "Hitung" yang perlu ditekan.

### 4.4 Komponen Output Real-time

#### 4.4.1 TEWS Total Score Display

| ID Komponen | Spesifikasi |
|-------------|-------------|
| TEWS-O01 | Angka besar (32px bold) menampilkan total TEWS (0–18) |
| TEWS-O02 | Warna angka berubah sesuai zona: Biru (0–2) / Hijau (3–4) / Kuning (5–6) / Oranye (7+) |
| TEWS-O03 | Badge zona SATS yang sesuai skor |

#### 4.4.2 Sub-score Breakdown (6 baris)

Untuk setiap parameter, tampilkan:

| ID | Spesifikasi |
|----|-------------|
| TEWS-O04 | Label parameter + nilai yang dimasukkan |
| TEWS-O05 | Skor sub-parameter (badge 0/1/2/3 berwarna) |
| TEWS-O06 | Rentang nilai yang dipilih (highlight aktif) |

Contoh tampilan satu baris:
```
Laju Pernafasan: 32 x/mnt    [Skor: 3]    Rentang aktif: ≥ 30
```

#### 4.4.3 Tabel Referensi Scoring (Accordion, Collapsed Default)

| ID | Spesifikasi |
|----|-------------|
| TEWS-O07 | Toggle "Lihat tabel referensi scoring lengkap" |
| TEWS-O08 | Tabel 6 parameter × semua range nilai × skor, dengan highlight baris yang aktif |

#### 4.4.4 Override Warning

| ID | Spesifikasi |
|----|-------------|
| TEWS-O09 | Alert merah jika kondisi override Merah terpenuhi, meskipun TEWS total < 7 |
| TEWS-O10 | Teks: "Perhatian: kondisi override Merah terdeteksi — [nama kondisi]" |

**Kondisi trigger override:**
- GCS total ≤ 8
- Sistole < 70 mmHg
- SpO₂ < 90% DAN (RR < 9 ATAU RR ≥ 30)

#### 4.4.5 Rekomendasi Tindakan

| Zona SATS | Teks Rekomendasi |
|-----------|-----------------|
| Merah | "Resusitasi segera. Panggil tim resusitasi. Monitor kontinu." |
| Oranye | "Tangani dalam 10 menit. Pasang IV line, monitor vital signs." |
| Kuning | "Evaluasi dalam 60 menit. Pasang monitoring, pantau perubahan." |
| Hijau | "Tangani dalam 4 jam. Triage ulang jika kondisi berubah." |
| Biru | "Tangani dalam 6 jam. Pertimbangkan rujuk ke fasilitas primer." |

#### 4.4.6 Tombol Utilitas

| ID | Label | Behaviour |
|----|-------|-----------|
| TEWS-ACT01 | "Reset" | Clear semua field ke placeholder default |
| TEWS-ACT02 | "Salin ke Tab Input" | Auto-fill field tanda vital di Tab-01 dengan nilai yang sudah diisi di kalkulator ini |

---

## 5. TAB-03 — Panduan SATS

### 5.1 Deskripsi

Tab referensi klinis statis. Berisi panduan penggunaan standar SATS, tabel scoring TEWS lengkap, kondisi override, dan alur keputusan triage. Dirancang sebagai referensi cepat yang dapat dibaca petugas kapan saja.

### 5.2 Komponen

#### 5.2.1 Seksi A — 5 Level Triage SATS (Card Grid)

Grid 5 kartu (auto-fit, responsive). Setiap kartu berisi:

| Field | Spesifikasi |
|-------|-------------|
| Dot warna + nama level | Merah/Oranye/Kuning/Hijau/Biru dalam bahasa Indonesia |
| Nama dalam bahasa Inggris | Resuscitation / Emergent / Urgent / Less Urgent / Not Urgent |
| Badge waktu target | "Segera — 0 mnt" / "< 10 mnt" / "< 60 mnt" / "< 4 jam" / "< 6 jam" |
| Definisi klinis | 1–2 kalimat deskripsi kondisi pasien di level ini |
| Contoh kondisi | 3–4 contoh klinis (italic, font 10px) |

**Warna kartu mengikuti mapping SATS** (lihat Bagian 3.3.1).

#### 5.2.2 Seksi B — Tabel Scoring TEWS Lengkap

Tabel referensi 6 parameter:

| Kolom | Spesifikasi |
|-------|-------------|
| Parameter | Nama parameter + satuan |
| Rentang nilai | Semua range untuk parameter tersebut |
| Skor | Badge berwarna 0 (hijau) / 1 (kuning) / 2 (oranye) / 3 (merah) |
| Keterangan klinis | Interpretasi singkat tiap range |

Baris bergantian background untuk readability. Baris header dengan background secondary.

**Footer tabel:**
- Legenda skor: 0 Normal / 1 Perhatian / 2 Waspada / 3 Kritis
- Catatan: TEWS total = jumlah semua sub-skor (range 0–18)

#### 5.2.3 Seksi C — Override Criteria (Langsung Merah)

Grid 3 kartu dengan background merah muda:

| Kartu | Kondisi | Keterangan Klinis |
|-------|---------|-------------------|
| 1 | GCS total ≤ 8 | Airway tidak terlindungi, risiko aspirasi |
| 2 | Sistole < 70 mmHg | Perfusi organ tidak adekuat, syok berat |
| 3 | SpO₂ < 90% DAN (RR < 9 ATAU RR ≥ 30) | Kombinasi hipoksia + gangguan frekuensi napas |

Setiap kartu dilengkapi ikon klinis dan catatan implikasi tindakan.

#### 5.2.4 Seksi D — Alur Keputusan Triage (Flowchart Visual)

Flowchart 4-langkah dalam format horizontal/vertikal responsif:

```
Step 1: Pasien tiba → Ukur tanda vital & GCS
         ↓
Step 2: Cek override criteria
         ├── Ya (salah satu terpenuhi) → MERAH (langsung resusitasi)
         └── Tidak ada → Hitung TEWS total
              ↓
Step 3: Tentukan zona SATS berdasarkan TEWS
         ├── TEWS ≥ 7   → ORANYE
         ├── TEWS 5–6   → KUNING
         ├── TEWS 3–4   → HIJAU
         └── TEWS 0–2   → BIRU
              ↓
Step 4: Verifikasi dokter/perawat → Tindakan dimulai
```

Setiap step berupa box dengan warna netral (abu-abu), kecuali outcome SATS yang berwarna sesuai level.

#### 5.2.5 Seksi E — Referensi Ilmiah

Row horizontal berisi sumber ilmiah yang digunakan:

| Referensi | Kutipan |
|-----------|---------|
| SATS | South African Triage Scale (SATS), 2nd Edition, 2012 |
| TEWS | Twomey M. et al. — Emerg Med J, 2007 |
| MEWS | Subbe CP. et al. — QJM, 2001 |
| Shock Index | Birkhahn RH. et al. — Am J Emerg Med, 2005 |

---

## 6. Spesifikasi Teknis Implementasi

### 6.1 Stack Teknologi (Rekomendasi)

| Layer | Teknologi | Keterangan |
|-------|-----------|------------|
| Framework UI | Streamlit ≥ 1.30 | Kompatibel dengan model `.pkl` yang sudah ada |
| Model serving | `joblib.load()` | Load `pipeline_imblearn_v3.pkl`, `scaler_minmax_v3.pkl`, `best_thresholds.pkl` |
| SHAP | `shap.TreeExplainer` | Instance di-cache dengan `@st.cache_resource` |
| Styling | Custom CSS via `st.markdown()` | Warna SATS, tabel, badge, flag chips |
| State management | `st.session_state` | Riwayat sesi, form state |

### 6.2 File Artefak yang Dibutuhkan

| File | Fungsi | Tab yang Menggunakan |
|------|--------|---------------------|
| `pipeline_imblearn_v3.pkl` | ImbPipeline (SMOTE + XGBoost) | Tab-01 |
| `scaler_minmax_v3.pkl` | MinMaxScaler untuk preprocessing | Tab-01 |
| `best_thresholds.pkl` | Threshold optimal per kelas | Tab-01 |
| `shap_explainer_v3.pkl` | TreeExplainer untuk SHAP values | Tab-01 |
| `feature_names_v3.pkl` | Daftar 32 fitur (urutan penting) | Tab-01 |
| `imputer.pkl` | IterativeImputer untuk missing values | Tab-01 |

### 6.3 Fungsi Backend yang Dipanggil UI

| Fungsi | Tab | Input | Output |
|--------|-----|-------|--------|
| `predict_triage_v3()` | Tab-01 | Dict input pasien | Dict: kelas, probabilitas, skor klinis |
| `compute_tews()` | Tab-01, Tab-02 | Row pandas (tanda vital + GCS) | Dict sub-skor + TEWS total |
| SHAP `explainer(X_input)` | Tab-01 | DataFrame 1 baris (scaled) | Array SHAP values per fitur per kelas |

### 6.4 Validasi Input (Frontend)

| Parameter | Batas Minimum | Batas Maksimum | Tipe |
|-----------|--------------|----------------|------|
| Usia | 0 | 120 | Integer |
| Sistole | 40 | 300 | Integer |
| Diastole | 20 | 200 | Integer |
| Denyut jantung | 10 | 300 | Integer |
| Laju pernafasan | 0 | 80 | Integer |
| Suhu tubuh | 30.0 | 45.0 | Float (1 desimal) |
| SpO₂ | 0 | 100 | Integer |
| GCS total | 3 | 15 | Integer |
| Skala nyeri | 0 | 10 | Integer |

### 6.5 Responsivitas

| Breakpoint | Layout Tab-01 | Layout Tab-02 | Layout Tab-03 |
|-----------|---------------|---------------|---------------|
| ≥ 1280px (desktop) | 2 kolom (input | output) | 2 kolom | 1 kolom lebar |
| 768–1279px (tablet) | 1 kolom (input atas, output bawah) | 1 kolom | 1 kolom |
| < 768px (mobile) | 1 kolom | 1 kolom | 1 kolom |

---

## 7. Acceptance Criteria UI

| ID | Kriteria | Cara Verifikasi |
|----|----------|----------------|
| UI-AC01 | Form Tab-01 dapat diisi dan mengirim prediksi dalam < 60 detik | User testing dengan perawat IGD |
| UI-AC02 | Badge SATS Merah tampil dengan warna merah dominan, tidak bisa terlewat secara visual | Visual inspection |
| UI-AC03 | SHAP panel menampilkan minimal 5 fitur teratas dengan nilai dan arah yang benar | Cross-check dengan output `shap_values` di notebook |
| UI-AC04 | Kalkulator TEWS Tab-02 update real-time tanpa perlu tombol submit | Manual testing semua kombinasi nilai |
| UI-AC05 | Override warning muncul ketika GCS ≤ 8 atau Sistole < 70 atau kondisi hipoksia+RR abnormal | Testing per kondisi |
| UI-AC06 | Disclaimer wajib tampil di semua state output, tidak dapat di-hide | Visual inspection |
| UI-AC07 | Tab-03 dapat diakses tanpa loading > 500ms (konten statis) | Performance measurement |
| UI-AC08 | Tombol "Salin ke Tab Input" di Tab-02 mengisi field Tab-01 dengan benar | Functional testing |
| UI-AC09 | Export CSV riwayat sesi menghasilkan file dengan kolom yang benar | Download dan verifikasi isi file |
| UI-AC10 | UI responsive dan tidak ada horizontal scroll pada tablet 768px | Testing pada device atau browser resize |

---

## 8. Keterbatasan & Catatan Pengembangan

### 8.1 Keterbatasan Lingkup (v1.0)

- Tidak ada autentikasi pengguna — aplikasi bersifat open access dalam jaringan lokal RS
- Tidak ada penyimpanan permanen data pasien — sesuai regulasi privasi dan ruang lingkup prototype penelitian
- SHAP force plot per pasien tidak diimplementasi di v1.0 (hanya bar chart sederhana)
- Tidak ada integrasi dengan sistem HIS (Hospital Information System) RSU Aulia

### 8.2 Fitur yang Direncanakan untuk Versi Selanjutnya

| Fitur | Prioritas | Estimasi |
|-------|-----------|----------|
| Autentikasi petugas (login sederhana) | Medium | v1.1 |
| Penyimpanan log prediksi ke database lokal | Medium | v1.1 |
| SHAP force plot interaktif per pasien | Low | v1.2 |
| Multi-bahasa (Indonesia + Inggris) | Low | v1.2 |
| Integrasi API HIS | High (jika deploy produksi) | v2.0 |

---

## 9. Riwayat Versi Dokumen

| Versi | Tanggal | Perubahan | Author |
|-------|---------|-----------|--------|
| 1.0 | 2026-06-18 | Dokumen awal — 3 tab lengkap | Reymondo |

---

*Dokumen ini adalah bagian dari dokumentasi skripsi: "Prediksi Skor Triage IGD Menggunakan XGBoost dan SHAP Berbasis Standar SATS-TEWS di RSU Aulia"*

> **Disclaimer:** Sistem yang didokumentasikan ini adalah prototype penelitian (Decision Support System). Keputusan triage final adalah wewenang eksklusif tenaga medis berlisensi.
