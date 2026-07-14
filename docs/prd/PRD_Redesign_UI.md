# PRD — Redesign UI/UX Sistem Prediksi Triage IGD (RSU Aulia)

**Dokumen untuk:** Agentic AI / AI coding agent yang akan mengeksekusi redesign
**Jenis proyek:** Redesign UI/UX (visual & interaksi), **bukan** perubahan logika model/skoring
**Sistem saat ini:** Sistem Prediksi Triage IGD — Standar SATS-TEWS, Model XGBoost + SHAP
**Versi dokumen:** 1.0
**Tanggal:** 8 Juli 2026

---

## 1. Ringkasan Eksekutif

Sistem Prediksi Triage IGD adalah *decision support tool* yang membantu perawat/dokter IGD RSU Aulia menentukan level triase pasien (Merah/Oranye/Kuning/Hijau/Biru) berdasarkan skor SATS-TEWS dan prediksi model XGBoost. Sistem terdiri dari 3 tab: **Input Pasien & Hasil Prediksi**, **Kalkulator TEWS**, dan **Panduan SATS**.

Fungsionalitas dan logika klinis sistem sudah baik dan **tidak diubah** dalam proyek ini. Yang perlu diredesign adalah **lapisan visual dan pengalaman pengguna**: sistem warna triase yang sulit dibedakan, kepadatan informasi yang tinggi, hierarki visual yang lemah, dan keterbatasan responsivitas di layar kecil.

Karena sistem ini digunakan dalam konteks **keputusan klinis berisiko tinggi** (level triase menentukan kecepatan penanganan pasien gawat darurat), redesign ini harus diperlakukan sebagai **isu keselamatan pasien**, bukan sekadar isu estetika. Kesalahan membaca warna Oranye vs Kuning di ruang IGD yang bising, terburu-buru, dengan pencahayaan buruk, dan kemungkinan pengguna buta warna, dapat berakibat pada keterlambatan penanganan pasien gawat darurat.

---

## 2. Latar Belakang & Masalah Saat Ini

Berdasarkan review tangkapan layar sistem saat ini (3 tab, tema gelap dengan toggle "Light"):

### 2.1 Masalah Kritis (Prioritas Tertinggi)
1. **Warna Oranye dan Kuning nyaris identik.** Pada badge level triase, tabel skor, dan grafik probabilitas, kedua warna ini memiliki hue, saturasi, dan lightness yang berdekatan sehingga sulit dibedakan sekilas — terutama pada tema gelap, oleh pengguna dengan gangguan penglihatan warna (deuteranopia/protanopia), atau pada layar dengan kalibrasi buruk (umum terjadi pada monitor triage IGD).
2. **Identitas level triase hanya bergantung pada warna.** Tidak ada redundansi visual yang kuat (ikon, nomor level, pola) sehingga jika warna gagal terbaca (color-blind, silau, layar buram), pengguna kehilangan informasi kritis.

### 2.2 Masalah Utama (Prioritas Tinggi)
3. **Kepadatan informasi terlalu tinggi tanpa hierarki yang jelas** — terutama di tab "Input Pasien & Hasil Prediksi" (Gambar 4), di mana hasil prediksi, distribusi probabilitas, clinical flags, skor turunan, SHAP explanation, dan riwayat sesi semuanya bersaing dalam satu layar tanpa penekanan visual pada informasi paling kritis (level triase pasien saat ini).
4. **Kontras teks-latar tidak konsisten** — teks italic kecil abu-abu pada kartu penjelasan (Gambar 1) berisiko gagal memenuhi standar kontras WCAG AA, padahal berisi contoh klinis penting.
5. **Alur keputusan triase (flowchart) di Panduan SATS (Gambar 2) berbentuk kotak-kotak generik**, kurang menяmpaikan alur logika secara visual (tidak ada connector/branching yang jelas untuk kondisi override vs jalur normal).
6. **Tidak ada bukti desain responsif.** Layout dua-kolom lebar (form kiri, hasil kanan di Gambar 4; tabel lebar di Gambar 1-3) kemungkinan besar rusak atau sulit digunakan di tablet/mobile — padahal staf IGD sering menggunakan tablet di sisi tempat tidur pasien.

### 2.3 Masalah Sekunder
7. Target tap/klik tombol +/- (stepper) pada input tanda vital (Gambar 3, 4) tampak kecil untuk digunakan cepat dalam kondisi darurat atau dengan sarung tangan medis.
8. Urutan kategori pada grafik "Distribusi Probabilitas" (Gambar 4) tidak konsisten dengan urutan keparahan klinis (Merah→Oranye→Kuning→Hijau→Biru), berpotensi membingungkan pembacaan cepat.
9. Header memuat banyak badge (Model final, Aktif, Light) tanpa hierarki yang jelas antara identitas sistem, status model, dan kontrol pengguna.
10. Belum ada state kosong/loading/error yang terlihat terdefinisi dengan baik untuk form prediksi dan kalkulator.

---

## 3. Tujuan Redesign

| # | Tujuan | Metrik Keberhasilan |
|---|--------|---------------------|
| G1 | Level triase dapat dibedakan sekilas (<1 detik) tanpa bergantung pada warna saja | Setiap badge level memiliki kombinasi warna + ikon + nomor + label teks yang unik |
| G2 | Palet warna 5-level lulus uji aksesibilitas warna | Kontras teks/latar ≥ WCAG AA (4.5:1 teks normal, 3:1 komponen besar); simulasi buta warna (protanopia/deuteranopia/tritanopia) tetap dapat dibedakan |
| G3 | Informasi paling kritis (hasil triase pasien) langsung terlihat tanpa scroll | Hasil triase menjadi elemen hero di viewport pertama pada semua ukuran layar |
| G4 | Sistem sepenuhnya responsif | Berfungsi baik pada 3 breakpoint: mobile (<640px), tablet (640–1024px), desktop (>1024px), tanpa horizontal scroll atau elemen terpotong |
| G5 | Kepadatan informasi diturunkan lewat hierarki & progressive disclosure | Informasi sekunder (SHAP detail, riwayat sesi, tabel referensi lengkap) collapsible/tersembunyi secara default namun mudah diakses |
| G6 | Konsistensi desain lintas 3 tab | Satu design system (token warna, tipografi, spacing, komponen) dipakai di seluruh tab |

---

## 4. Target Pengguna

- **Perawat triase IGD** — pengguna utama, sering dalam kondisi terburu-buru, kadang menggunakan tablet di samping pasien, prioritas: kecepatan baca & input.
- **Dokter jaga IGD** — memverifikasi hasil prediksi & rekomendasi tindakan, butuh detail klinis (SHAP, skor turunan) saat diperlukan tapi tidak selalu.
- **Admin/staf rekam medis** — melihat riwayat sesi, ekspor data.
- **Semua pengguna** berpotensi mengakses dari desktop stasiun triase, tablet mobile cart, atau (kasus darurat) ponsel pribadi.

Asumsikan sebagian pengguna adalah **buta warna parsial** (prevalensi ~8% pria) — ini bukan edge case, harus menjadi desain default.

---

## 5. Prinsip Desain

1. **Keselamatan pasien di atas estetika.** Setiap keputusan desain warna/kontras/ukuran diuji dengan pertanyaan: "Apakah ini bisa menyebabkan salah baca level triase?"
2. **Redundansi kode visual.** Warna TIDAK PERNAH menjadi satu-satunya penanda arti. Selalu gabungkan warna + ikon + nomor level (1–5) + label teks.
3. **Clarity over decoration.** Tidak ada elemen dekoratif yang mengalihkan perhatian dari data klinis. Ruang kosong (whitespace) digunakan untuk memberi napas, bukan untuk gaya.
4. **Progressive disclosure.** Tampilkan ringkasan dahulu (hasil, level, rekomendasi tindakan); detail teknis (breakdown skor, SHAP, tabel referensi) dapat diakses lewat expand/collapse, tidak membanjiri layar utama.
5. **Konsisten & dapat diprediksi.** Komponen yang sama (badge triase, tabel skor, kartu parameter) terlihat & berperilaku sama di semua tab.
6. **Scannable dalam <5 detik.** Perawat harus bisa menangkap level triase dan tindakan yang direkomendasikan dalam sekali pandang.
7. **Mobile-first, bukan mobile-adapted.** Desain dimulai dari layar sempit lalu diperluas — bukan mengecilkan layout desktop.

---

## 6. Design System — Spesifikasi Wajib

### 6.1 Palet Warna Triase (PALING KRITIS)

Gunakan konvensi warna standar SATS internasional (Merah–Oranye–Kuning–Hijau–Biru) namun dengan **jarak hue, saturasi, dan lightness yang cukup lebar** antar-warna agar tidak ambigu — khususnya Oranye vs Kuning. Nilai berikut adalah **rekomendasi awal**, agen desain wajib memvalidasi ulang dengan color-contrast checker & color-blindness simulator sebelum finalisasi:

| Level | Nama | Hue (approx) | Warna Solid (light bg) | Warna Latar Muda | Ikon wajib | Nomor |
|---|---|---|---|---|---|---|
| 1 | **Merah** — Resuscitation | ~0° (merah murni) | `#DC2626` | `#FEE2E2` | ⚡ petir/kilat (kegawatan ekstrem) | `1` |
| 2 | **Oranye** — Emergent | ~25° (merah-jingga) | `#EA580C` | `#FFEDD5` | 🔺 segitiga peringatan | `2` |
| 3 | **Kuning** — Urgent | ~45–48° (kuning-emas, digeser menjauhi oranye) | `#CA8A04` (teks/ikon), `#EAB308` (fill besar) | `#FEF9C3` | ⏱ jam/waktu tunggu | `3` |
| 4 | **Hijau** — Less Urgent | ~142° | `#16A34A` | `#DCFCE7` | ✔ centang | `4` |
| 5 | **Biru** — Not Urgent | ~217° | `#2563EB` | `#DBEAFE` | ℹ info | `5` |

**Aturan wajib:**
- Jarak hue minimum antar-warna bertetangga (terutama Oranye↔Kuning) ≥ 20°, dan pastikan **lightness/value** kuning dinaikkan cukup jauh dari oranye (kuning harus terasa "lebih cerah/emas", oranye terasa "lebih merah/gelap") agar tetap terpisah pada simulasi protanopia/deuteranopia.
- Setiap badge/kartu level SELALU menampilkan: warna + ikon + angka level (1–5) + label teks ("Merah", "Resuscitation", dst) — minimal 3 dari 4 kode ini harus terlihat bersamaan, tidak boleh warna saja.
- Untuk mode gelap (dark theme, sesuai toggle "Light/Dark" yang sudah ada), buat varian warna yang sama-sama teruji kontrasnya — jangan hanya menurunkan brightness warna light-mode secara mentah.
- Semua kombinasi teks-di-atas-warna wajib lulus WCAG AA (rasio kontras ≥ 4.5:1 untuk teks normal, ≥ 3:1 untuk teks besar/ikon).
- Urutan warna pada grafik/legend/list SELALU mengikuti urutan keparahan klinis: Merah → Oranye → Kuning → Hijau → Biru (jangan diurutkan berdasarkan probabilitas tertinggi seperti pada tampilan saat ini di Gambar 4).
- Uji palet akhir dengan simulator color-blindness (mis. Coblis atau setara) untuk protanopia, deuteranopia, tritanopia — dokumentasikan hasil sebelum sign-off.

### 6.2 Tipografi
- Gunakan 2 peran font: **display/heading** (tegas, untuk angka skor & level triase besar) dan **body/data** (sangat legible untuk angka & tabel — prioritaskan font dengan angka tabular/monospace-numeral agar kolom angka rapi, mis. Inter, IBM Plex Sans, atau setara yang tersedia).
- Skala tipe jelas dan terbatas (mis. 12/14/16/20/24/32/48px) — jangan gunakan ukuran ad-hoc.
- Angka skor besar (TEWS Total, level triase) harus menjadi elemen tipografi paling dominan di layar hasil.

### 6.3 Layout & Spacing
- Gunakan sistem spacing konsisten berbasis grid 4px atau 8px.
- Card/panel memiliki padding internal konsisten, radius sudut konsisten, dan elevasi (shadow/border) yang konsisten antar tab.
- Maksimal 1 kolom di mobile; 2 kolom di tablet/desktop untuk layout form+hasil (lihat §7.2).

### 6.4 Ikonografi
- Satu ikon set konsisten (outline atau filled — pilih satu gaya, jangan campur) untuk seluruh sistem: parameter vital (RR, SpO2, TD, HR, Suhu, GCS), level triase, status (aktif/nonaktif), aksi (reset, export, salin).
- Ikon untuk level triase harus mudah dikenali tanpa warna (bentuk berbeda, bukan hanya warna berbeda).

### 6.5 Mode Gelap/Terang
- Pertahankan toggle Light/Dark yang sudah ada, tapi definisikan token warna sebagai variabel semantik (mis. `--color-bg-surface`, `--color-triage-merah-bg`, `--color-text-primary`) yang punya nilai berbeda di tiap tema — bukan hardcode hex di komponen.
- Uji ulang kontras §6.1 secara terpisah untuk tema gelap dan terang.

---

## 7. Kebutuhan Fungsional per Halaman

### 7.1 Global (berlaku di semua tab)
- **Header** disederhanakan: judul sistem & sub-info (RSU Aulia · SATS-TEWS · XGBoost+SHAP) sebagai identitas utama; status badge ("Model final", "Aktif") dikelompokkan terpisah dari kontrol pengguna (tema Light/Dark) secara visual agar tidak terbaca sebagai satu deret badge yang sama pentingnya.
- **Navigasi tab** (Input & Hasil / Kalkulator TEWS / Panduan SATS) tetap dipertahankan, dibuat jelas tab aktifnya, dan pada mobile berubah menjadi bottom-tab atau dropdown yang tetap mudah dijangkau ibu jari.
- **Badge level triase** menjadi komponen reusable tunggal (lihat §6.1) dipakai identik di ketiga tab — jangan reimplementasi ulang per halaman.
- Sediakan **state kosong, loading, dan error** yang terdefinisi untuk form (mis. saat prediksi sedang diproses, saat input tidak valid, saat model gagal merespons).

### 7.2 Tab "Input Pasien & Hasil Prediksi" (Gambar 4)
**Masalah saat ini:** terlalu banyak informasi bersaing tanpa hierarki; hasil triase tidak cukup menonjol relatif terhadap grafik SHAP/riwayat.

**Requirement:**
- Hasil triase (badge besar level + waktu target penanganan, mis. "<4 JAM") menjadi **hero element** — elemen visual terbesar & paling atas di kolom hasil, terlihat tanpa scroll di semua breakpoint.
- Kelompokkan informasi hasil ke dalam **hierarki 3 lapis**:
  1. **Utama (selalu terlihat):** Level triase + waktu target + rekomendasi tindakan singkat.
  2. **Sekunder (terlihat, ringkas):** Distribusi probabilitas 5 kategori (urutan sesuai §6.1), clinical flags aktif.
  3. **Detail (collapsed by default, expandable):** Skor klinis turunan (Shock Index, MEWS, TEWS, Cardiorespi), penjelasan SHAP/top fitur, riwayat sesi + export CSV.
- Form input (Identitas, Tanda Vital, Neurologis/GCS) di kolom terpisah dengan pengelompokan visual jelas per section, label satuan (mmHg, x/mnt, °C) selalu terlihat menempel pada field, bukan hanya sebagai placeholder kecil.
- Perbesar target tap pada tombol +/- stepper (minimal 44×44px sesuai pedoman aksesibilitas tap-target) untuk mobile/tablet dan penggunaan darurat.
- Disclaimer DSS ("Keputusan final tetap wewenang tenaga medis...") tetap ditampilkan tapi dengan styling yang jelas sebagai catatan penting, tidak sebagai info box generik yang mudah diabaikan.
- Pada **mobile**: form dan hasil disusun stack vertikal — form input dahulu (atau collapsible), hasil prediksi setelah submit muncul sebagai panel hero di atas.

### 7.3 Tab "Kalkulator TEWS" (Gambar 3)
**Masalah saat ini:** breakdown sub-skor & tabel referensi lengkap membuat layar padat; skor total kurang menonjol dibanding form input.

**Requirement:**
- Skor **TEWS Total** + badge level hasil (mis. "Merah — Resusitasi — 0 menit") menjadi elemen paling menonjol di kolom kanan, sticky/tetap terlihat saat user scroll mengisi form (khususnya di desktop/tablet).
- Alert **"Override Merah Terdeteksi"** menggunakan pola visual peringatan yang konsisten dengan level Merah di §6.1 (warna + ikon petir + label), ditempatkan tepat di bawah skor total agar langsung terlihat.
- **Breakdown Sub-skor**: tabel tetap ada namun beri indikator warna skor (0/1/2/3) per baris menggunakan palet §6.1 yang sudah diperbaiki (bukan warna hijau/kuning/oranye/merah versi lama yang ambigu).
- **Tabel Referensi Scoring Lengkap** tetap collapsible (sudah ada di desain saat ini — pertahankan pola ini, sudah baik untuk progressive disclosure), tapi pastikan mudah discan dengan zebra-striping/border ringan antar baris parameter.
- Tombol "Reset" dan "Salin ke Tab Input" dibedakan secara visual (primer vs sekunder) agar user tidak salah klik saat terburu-buru.

### 7.4 Tab "Panduan SATS" (Gambar 1–2)
**Masalah saat ini:** 5 kartu level triase punya warna oranye/kuning mirip; flowchart alur keputusan kurang menyampaikan logika percabangan (override vs skor normal) secara visual.

**Requirement:**
- 5 kartu level triase (Merah/Oranye/Kuning/Hijau/Biru) redesign mengikuti §6.1 — tambahkan ikon + nomor level di setiap kartu, bukan hanya warna & nama.
- **Tabel Scoring TEWS Lengkap**: pastikan kolom "Skor" menggunakan badge angka dengan warna dari §6.1 (bukan warna lama), dan teks "Keterangan Klinis" (saat ini italic kecil abu-abu) dinaikkan kontrasnya agar mudah dibaca (uji WCAG AA).
- **Kondisi Override — Langsung MERAH**: 3 kartu kondisi override dipertahankan sebagai pola "kartu peringatan merah", tapi tambahkan ikon yang membedakan tiap kondisi (GCS, tekanan darah, SpO2) secara jelas.
- **Alur Keputusan Triase**: redesign dari kotak-kotak linear generik menjadi **diagram alur (flowchart) visual sesungguhnya** — gunakan connector/panah yang menunjukkan percabangan jelas antara "jalur override → langsung Merah" vs "jalur normal → hitung TEWS → tentukan level via skor". Setiap langkah diberi nomor & ikon.
- **Referensi Ilmiah**: kartu referensi (SATS, TEWS, MEWS, Shock Index) dipertahankan sebagai pola kartu ringkas — cukup selaraskan tipografi & spacing dengan design system baru.

---

## 8. Responsivitas

| Breakpoint | Lebar | Perilaku Layout |
|---|---|---|
| Mobile | <640px | 1 kolom, form & hasil di-stack, nav tab jadi bottom-bar/dropdown, tabel lebar diubah jadi card list atau horizontal-scroll dengan indikator jelas |
| Tablet | 640–1024px | 1–2 kolom fleksibel, prioritaskan hasil triase tetap terlihat tanpa scroll berlebihan |
| Desktop | >1024px | 2 kolom (form kiri, hasil kanan) seperti saat ini, dengan hero result yang lebih menonjol |

- Tidak boleh ada horizontal scroll yang tidak disengaja pada breakpoint manapun.
- Tap target minimum 44×44px di seluruh breakpoint mobile/tablet.
- Uji pada minimal: layar ponsel kecil (~375px), tablet (~768px), desktop (~1440px).

---

## 9. Aksesibilitas (Wajib, bukan Nice-to-have)

- WCAG 2.1 level **AA** minimum untuk seluruh teks, ikon fungsional, dan komponen interaktif.
- Warna tidak pernah menjadi satu-satunya pembawa informasi (lihat §6.1).
- Semua elemen interaktif dapat diakses via keyboard (focus state terlihat jelas) — penting untuk workstation IGD yang mungkin tidak selalu pakai mouse/touch.
- Kontras fokus (focus ring) terlihat jelas di kedua tema (light/dark).
- Hormati preferensi `prefers-reduced-motion` jika ada animasi/transisi.
- Label form & satuan ukuran selalu terbaca oleh screen reader (gunakan `<label>` yang terasosiasi dengan benar, bukan hanya placeholder).

---

## 10. Batasan Teknis & Asumsi

- **Tidak mengubah** logika perhitungan skor TEWS/SATS, kriteria override, model XGBoost, maupun output SHAP — proyek ini murni lapisan presentasi (UI/UX).
- Pertahankan struktur informasi & data yang sudah ada (semua field, tabel referensi, dan hasil yang ditampilkan saat ini harus tetap ada di desain baru — hanya disusun ulang, tidak dihilangkan, kecuali dipindah ke bagian collapsible/detail sesuai §7).
- Pertahankan fitur toggle tema Light/Dark yang sudah ada.
- Agen desain bebas memilih stack implementasi (HTML/CSS/JS, React, dsb.) sesuai dengan stack proyek yang sudah berjalan — sesuaikan dengan kode sumber yang ada, jangan membangun ulang dari nol tanpa alasan kuat.
- Jika stack asli tidak diketahui/tidak tersedia, agen dapat mengklarifikasi ke pengguna sebelum membangun implementasi penuh, namun tetap dapat menghasilkan spesifikasi desain (design tokens, wireframe, komponen) tanpa perlu klarifikasi tersebut.

---

## 11. Deliverables yang Diharapkan dari Agentic AI

1. **Design tokens** (warna, tipografi, spacing, radius, shadow) dalam format terstruktur (mis. CSS variables/JSON) sesuai §6.
2. **Komponen reusable**: Badge Level Triase, Kartu Parameter Vital, Tabel Skor, Kartu Kondisi Override, Diagram Alur Keputusan.
3. Implementasi ulang UI ketiga tab sesuai §7, responsif sesuai §8.
4. Dokumentasi singkat hasil uji aksesibilitas (kontras warna & simulasi buta warna) untuk palet triase final.
5. Sebelum implementasi penuh, agen sebaiknya menghasilkan **preview/mockup** (mis. artifact HTML atau screenshot) dari komponen Badge Level Triase (5 varian) dan satu contoh layar hasil prediksi untuk direview pengguna terlebih dahulu — mengingat ini adalah aplikasi safety-critical, validasi visual sebelum full-build sangat dianjurkan.

---

## 12. Kriteria Penerimaan (Definition of Done)

- [ ] Warna Oranye dan Kuning dapat dibedakan dengan jelas oleh mata normal maupun dalam simulasi color-blindness.
- [ ] Setiap badge/level triase menampilkan warna + ikon + nomor + label teks secara bersamaan.
- [ ] Semua kombinasi teks-di-atas-warna lulus rasio kontras WCAG AA.
- [ ] Hasil triase (level + waktu target) terlihat tanpa scroll di semua breakpoint setelah prediksi dijalankan.
- [ ] Tidak ada horizontal scroll tak disengaja di mobile (375px), tablet (768px), desktop (1440px).
- [ ] Semua field form dan tombol punya tap target ≥44×44px di mobile/tablet.
- [ ] Ketiga tab menggunakan komponen & token desain yang konsisten (bukan style terpisah per tab).
- [ ] Flowchart "Alur Keputusan Triase" menampilkan percabangan override vs jalur normal secara visual, bukan kotak linear.
- [ ] Informasi detail (SHAP, riwayat sesi, tabel referensi lengkap) tetap ada namun dikemas sebagai collapsible/expandable, tidak membebani tampilan utama.
- [ ] Tidak ada perubahan pada logika skor, kriteria override, atau output model.
- [ ] Toggle tema Light/Dark tetap berfungsi dan kontras tervalidasi di kedua tema.

---

## 13. Di Luar Cakupan (Out of Scope)

- Perubahan algoritma/model prediksi (XGBoost), bobot fitur, atau perhitungan SHAP.
- Perubahan kriteria klinis SATS-TEWS (ambang skor, kriteria override).
- Integrasi sistem baru (rekam medis elektronik, autentikasi, dsb.) di luar yang sudah ada.
- Penerjemahan sistem ke bahasa lain (sistem tetap berbahasa Indonesia).

---

## 14. Lampiran — Referensi Tangkapan Layar Sistem Saat Ini

| Gambar | Konten |
|---|---|
| Gambar 1 | Tab "Panduan SATS" — 5 kartu level triase, tabel scoring TEWS (RR, SpO2, TD Sistol) |
| Gambar 2 | Tab "Panduan SATS" (lanjutan) — kartu kondisi override Merah, alur keputusan triase, referensi ilmiah |
| Gambar 3 | Tab "Kalkulator TEWS" — input parameter vital, skor total, breakdown sub-skor, rekomendasi tindakan |
| Gambar 4 | Tab "Input Pasien & Hasil Prediksi" — form identitas/vital/GCS, hasil triase, distribusi probabilitas, SHAP, riwayat sesi |