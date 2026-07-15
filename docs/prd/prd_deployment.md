# PRD Perbaikan Deployment Streamlit dengan Auto-Retraining Model

## 1. Informasi Dokumen

| Komponen        | Keterangan                                                                   |
| --------------- | ---------------------------------------------------------------------------- |
| Nama fitur      | Auto-Retraining Model saat Deployment                                        |
| Sistem          | Sistem Prediksi Triage IGD                                                   |
| Platform        | Streamlit Community Cloud                                                    |
| Prioritas       | Tinggi                                                                       |
| Ruang lingkup   | Perbaikan proses startup aplikasi agar model dibuat sebelum dimuat           |
| Di luar cakupan | Perubahan UI, algoritma model, evaluasi model, dan penambahan fitur prediksi |

---

## 2. Latar Belakang

Aplikasi Sistem Prediksi Triage IGD berhasil berjalan di GitHub Codespaces, tetapi mengalami kegagalan saat dijalankan pada Streamlit Community Cloud.

Pesan kesalahan menunjukkan bahwa artefak model tidak ditemukan:

```text
File artefak tidak ditemukan:
model/artifacts/shap_explainer.pkl
```

Codespaces dapat menjalankan aplikasi karena artefak model telah dihasilkan secara lokal melalui proses training. Namun, artefak tersebut belum tersedia pada filesystem Streamlit saat aplikasi pertama kali dijalankan.

Akibatnya, aplikasi mencoba memuat model sebelum file model dan SHAP explainer berhasil dibuat.

Perbaikan difokuskan pada penambahan mekanisme startup yang:

1. Memeriksa ketersediaan seluruh artefak model.
2. Menjalankan `scripts/retrain.py` apabila artefak belum tersedia.
3. Memastikan proses training selesai dengan sukses.
4. Memuat model setelah seluruh artefak berhasil dibuat.
5. Menghentikan aplikasi dan menampilkan pesan kesalahan apabila retraining gagal.

---

## 3. Permasalahan

### 3.1 Kondisi saat ini

Alur aplikasi saat ini:

```text
Streamlit menjalankan app/app.py
        ↓
Aplikasi langsung memuat model
        ↓
Artefak model belum tersedia
        ↓
FileNotFoundError
        ↓
Aplikasi tidak dapat digunakan
```

### 3.2 Kondisi yang diharapkan

Alur aplikasi setelah perbaikan:

```text
Streamlit menjalankan app/app.py
        ↓
Periksa seluruh artefak model
        ↓
Apakah artefak tersedia?
        ├── Ya → Muat model
        └── Tidak → Jalankan scripts/retrain.py
                          ↓
                   Validasi artefak
                          ↓
                   Muat model
                          ↓
                   Jalankan aplikasi
```

---

## 4. Tujuan

Perbaikan ini bertujuan agar aplikasi Streamlit dapat menjalankan proses retraining secara otomatis ketika artefak model belum tersedia.

Tujuan khususnya adalah:

* Mencegah kegagalan startup akibat file model yang tidak ditemukan.
* Memastikan model dilatih sebelum digunakan oleh aplikasi.
* Memastikan seluruh path file bekerja konsisten di Codespaces dan Streamlit Cloud.
* Mencegah proses training berjalan ulang setiap kali pengguna berinteraksi dengan aplikasi.
* Menampilkan status proses training yang mudah dipahami.
* Menampilkan pesan kesalahan teknis ketika retraining gagal.

---

## 5. Ruang Lingkup

### 5.1 Termasuk dalam perbaikan

Perbaikan hanya mencakup:

* Pemeriksaan keberadaan artefak model.
* Validasi ukuran file artefak.
* Pemanggilan `scripts/retrain.py` dari `app/app.py`.
* Pembuatan direktori `model/artifacts`.
* Penyesuaian path agar menggunakan root repository.
* Validasi hasil retraining.
* Penanganan kegagalan retraining.
* Caching proses pemuatan model.
* Tampilan status retraining pada halaman Streamlit.
* Penyesuaian dependensi yang diperlukan untuk proses training.

### 5.2 Tidak termasuk dalam perbaikan

Perbaikan ini tidak mencakup:

* Perubahan algoritma XGBoost.
* Perubahan parameter model.
* Perubahan metode SMOTE.
* Perubahan perhitungan SATS-TEWS.
* Perubahan SHAP.
* Perubahan dataset.
* Perubahan input prediksi.
* Perubahan hasil klasifikasi.
* Perubahan desain antarmuka.
* Penambahan autentikasi.
* Penambahan database.
* Penambahan pipeline CI/CD.
* Penyimpanan model pada cloud storage eksternal.

---

## 6. Pengguna dan Aktor Sistem

| Aktor                     | Peran                                       |
| ------------------------- | ------------------------------------------- |
| Streamlit Community Cloud | Menjalankan aplikasi dan proses startup     |
| Aplikasi Streamlit        | Memeriksa, membuat, dan memuat artefak      |
| Script retraining         | Melatih model dan menyimpan artefak         |
| Pengguna                  | Mengakses aplikasi setelah model siap       |
| Developer                 | Melihat log apabila proses retraining gagal |

---

## 7. Kebutuhan Fungsional

### FR-01 — Menentukan root repository

Sistem harus menentukan root repository berdasarkan lokasi file, bukan berdasarkan current working directory.

Contoh:

```python
ROOT_DIR = Path(__file__).resolve().parent.parent
```

Tujuannya agar path tetap konsisten pada:

* Codespaces.
* Local development.
* Streamlit Community Cloud.

---

### FR-02 — Mendefinisikan artefak wajib

Sistem harus memiliki daftar seluruh file artefak yang wajib tersedia sebelum aplikasi memuat model.

Contoh:

```python
REQUIRED_ARTIFACTS = [
    ROOT_DIR / "model" / "artifacts" / "model.pkl",
    ROOT_DIR / "model" / "artifacts" / "shap_explainer.pkl",
]
```

Daftar tersebut harus disesuaikan dengan seluruh file yang benar-benar digunakan aplikasi, misalnya:

```text
model.pkl
shap_explainer.pkl
preprocessor.pkl
label_encoder.pkl
feature_columns.pkl
metadata.json
```

---

### FR-03 — Memeriksa ketersediaan artefak

Sistem harus memeriksa bahwa:

* File tersedia.
* File bukan direktori.
* Ukuran file lebih dari 0 byte.

Contoh logika:

```python
def artifacts_available():
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in REQUIRED_ARTIFACTS
    )
```

Artefak kosong tidak boleh dianggap valid.

---

### FR-04 — Menjalankan retraining saat artefak tidak tersedia

Apabila satu atau lebih artefak tidak tersedia, aplikasi harus menjalankan:

```text
scripts/retrain.py
```

Proses harus dijalankan sebelum fungsi pemuatan model dipanggil.

Pemanggilan retraining dapat menggunakan:

```python
subprocess.run(
    [sys.executable, str(RETRAIN_SCRIPT)],
    cwd=str(ROOT_DIR),
)
```

Penggunaan `sys.executable` diwajibkan agar retraining menggunakan environment Python yang sama dengan aplikasi Streamlit.

---

### FR-05 — Membuat direktori artefak

Sebelum menyimpan model, `retrain.py` harus memastikan direktori artefak tersedia.

```python
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
```

Sistem tidak boleh mengasumsikan bahwa folder tersebut sudah tersedia.

---

### FR-06 — Menggunakan absolute path berbasis repository

Seluruh pembacaan dan penyimpanan file dalam `retrain.py` harus menggunakan path yang diturunkan dari root repository.

Tidak diperbolehkan menggunakan path yang bergantung pada lokasi terminal seperti:

```python
"../model/artifacts/model.pkl"
```

Gunakan:

```python
ROOT_DIR / "model" / "artifacts" / "model.pkl"
```

Ketentuan ini berlaku untuk:

* Dataset.
* Model.
* SHAP explainer.
* Preprocessor.
* Encoder.
* Metadata.
* File konfigurasi.

---

### FR-07 — Menampilkan status retraining

Saat retraining berjalan, halaman Streamlit harus menampilkan status yang informatif.

Contoh status:

```text
Model belum tersedia. Sedang melakukan retraining...
```

Setelah berhasil:

```text
Retraining selesai. Model siap digunakan.
```

Tampilan status dapat menggunakan:

```python
st.status()
```

atau:

```python
st.spinner()
```

---

### FR-08 — Memvalidasi hasil retraining

Setelah `retrain.py` selesai, aplikasi harus memeriksa kembali seluruh artefak.

Apabila proses retraining selesai tetapi artefak masih tidak lengkap, aplikasi harus menganggap proses gagal.

Contoh pesan:

```text
Retraining selesai, tetapi satu atau lebih artefak model tidak terbentuk.
```

---

### FR-09 — Menghentikan aplikasi ketika retraining gagal

Apabila retraining gagal, aplikasi harus:

1. Menampilkan pesan kesalahan.
2. Menampilkan informasi log yang relevan.
3. Tidak melanjutkan proses pemuatan model.
4. Memanggil `st.stop()`.

Aplikasi tidak boleh menampilkan form prediksi ketika model belum siap.

---

### FR-10 — Menangkap output retraining

Sistem harus menangkap:

* Standard output.
* Standard error.
* Return code.

Retraining dianggap berhasil hanya apabila:

```python
result.returncode == 0
```

Apabila gagal, pesan error harus menyertakan bagian penting dari output retraining.

Informasi sensitif seperti secrets dan credential tidak boleh ditampilkan.

---

### FR-11 — Memberikan batas waktu retraining

Proses retraining harus memiliki timeout untuk mencegah startup berjalan tanpa batas.

Contoh:

```python
timeout=900
```

Batas awal yang direkomendasikan adalah 15 menit dan dapat disesuaikan berdasarkan durasi training aktual.

---

### FR-12 — Memuat model setelah artefak tersedia

Fungsi pemuatan model hanya boleh dijalankan setelah:

```python
artifacts_available() is True
```

Urutan wajib:

```text
Periksa artefak
→ Retrain jika diperlukan
→ Validasi artefak
→ Load model
→ Render aplikasi
```

---

### FR-13 — Menggunakan cache untuk pemuatan model

Model dan SHAP explainer harus dimuat menggunakan:

```python
@st.cache_resource
```

Tujuannya agar artefak tidak dibaca ulang pada setiap rerun Streamlit.

Contoh:

```python
@st.cache_resource
def load_model_artifacts():
    ...
```

Cache digunakan untuk proses pemuatan model, bukan sebagai satu-satunya penanda bahwa retraining telah dilakukan.

---

### FR-14 — Mencegah retraining pada setiap rerun

Retraining tidak boleh dijalankan pada setiap perubahan widget.

Kondisi retraining hanya boleh bergantung pada validitas artefak:

```python
if not artifacts_available():
    run_retraining()
```

Apabila seluruh artefak sudah tersedia dan valid, aplikasi harus langsung memuat model.

---

### FR-15 — Mendukung proses import yang aman

File `scripts/retrain.py` harus memiliki entry point yang jelas:

```python
def train_and_save_artifacts():
    ...

if __name__ == "__main__":
    train_and_save_artifacts()
```

Hal ini mencegah proses training berjalan secara tidak sengaja ketika modul hanya di-import.

---

## 8. Kebutuhan Nonfungsional

### NFR-01 — Reliabilitas

Aplikasi tidak boleh melanjutkan prediksi apabila artefak model belum lengkap.

### NFR-02 — Portabilitas

Implementasi harus berjalan pada:

* Windows local development.
* Linux Codespaces.
* Linux Streamlit Community Cloud.

### NFR-03 — Observability

Log retraining harus dapat dilihat pada log deployment Streamlit.

Minimal log mencakup:

```text
Memulai retraining
Dataset berhasil dimuat
Training selesai
Artefak berhasil disimpan
Retraining selesai
```

### NFR-04 — Keamanan data

Dataset yang digunakan pada Streamlit Cloud tidak boleh mengandung:

* Nama pasien.
* Nomor rekam medis.
* Nomor registrasi.
* Informasi identitas pribadi lainnya.

### NFR-05 — Efisiensi

Retraining hanya dijalankan ketika artefak tidak tersedia atau tidak valid.

### NFR-06 — Fail-safe

Apabila model gagal dibuat, sistem harus berhenti dalam kondisi aman dan tidak menghasilkan prediksi.

---

## 9. Alur Proses

### 9.1 Startup normal

```text
Aplikasi dimulai
    ↓
Periksa artefak
    ↓
Semua artefak tersedia
    ↓
Load model dengan cache
    ↓
Tampilkan aplikasi
```

### 9.2 Startup tanpa artefak

```text
Aplikasi dimulai
    ↓
Periksa artefak
    ↓
Artefak tidak lengkap
    ↓
Tampilkan status retraining
    ↓
Jalankan scripts/retrain.py
    ↓
Periksa return code
    ↓
Validasi ulang artefak
    ↓
Load model
    ↓
Tampilkan aplikasi
```

### 9.3 Retraining gagal

```text
Aplikasi dimulai
    ↓
Artefak tidak lengkap
    ↓
Jalankan retraining
    ↓
Retraining gagal atau timeout
    ↓
Tampilkan pesan kesalahan
    ↓
Hentikan aplikasi
```

---

## 10. Rancangan Implementasi

### 10.1 Perubahan `app/app.py`

Tambahkan modul bootstrap sebelum pemuatan model.

Struktur yang diharapkan:

```python
# Import dasar
# Penentuan ROOT_DIR
# Daftar REQUIRED_ARTIFACTS
# Fungsi artifacts_available()
# Fungsi run_retraining()
# Validasi dan retraining
# Fungsi load_model_artifacts()
# Render aplikasi
```

Contoh implementasi inti:

```python
from pathlib import Path
import subprocess
import sys
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
RETRAIN_SCRIPT = ROOT_DIR / "scripts" / "retrain.py"

REQUIRED_ARTIFACTS = [
    ROOT_DIR / "model" / "artifacts" / "model.pkl",
    ROOT_DIR / "model" / "artifacts" / "shap_explainer.pkl",
]


def artifacts_available() -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in REQUIRED_ARTIFACTS
    )


def run_retraining() -> None:
    result = subprocess.run(
        [sys.executable, str(RETRAIN_SCRIPT)],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


if not artifacts_available():
    with st.status(
        "Model belum tersedia. Sedang melakukan retraining...",
        expanded=True,
    ) as status:
        try:
            run_retraining()

            if not artifacts_available():
                raise RuntimeError(
                    "Artefak model belum lengkap setelah retraining."
                )

            status.update(
                label="Retraining selesai. Model siap digunakan.",
                state="complete",
                expanded=False,
            )

        except Exception as error:
            status.update(
                label="Retraining model gagal.",
                state="error",
            )
            st.error(str(error))
            st.stop()
```

---

### 10.2 Perubahan `scripts/retrain.py`

Script harus:

1. Menentukan root repository.
2. Membuat direktori artefak.
3. Membaca dataset menggunakan absolute path.
4. Melatih model.
5. Membuat SHAP explainer.
6. Menyimpan seluruh artefak.
7. Mengembalikan exit code gagal apabila terjadi exception.

Struktur:

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "model" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def train_and_save_artifacts():
    # Load dataset
    # Preprocessing
    # Training
    # Membuat SHAP explainer
    # Menyimpan artefak
    pass


if __name__ == "__main__":
    train_and_save_artifacts()
```

---

### 10.3 Perubahan `requirements.txt`

Pastikan seluruh library untuk inference dan training tersedia.

Minimal:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
joblib
matplotlib
openpyxl
```

Versi library sebaiknya disamakan dengan environment yang telah berhasil menjalankan training di Codespaces.

---

## 11. Penanganan Error

| Kondisi                                 | Respons sistem                                               |
| --------------------------------------- | ------------------------------------------------------------ |
| `retrain.py` tidak ditemukan            | Tampilkan path script yang dicari dan hentikan aplikasi      |
| Dataset tidak ditemukan                 | Tampilkan nama file dataset yang tidak tersedia              |
| Dependency tidak tersedia               | Tampilkan modul yang gagal di-import                         |
| Training menghasilkan exception         | Tampilkan ringkasan error dan hentikan aplikasi              |
| Proses melebihi timeout                 | Hentikan subprocess dan tampilkan pesan timeout              |
| Model berhasil dibuat tetapi SHAP gagal | Anggap retraining gagal                                      |
| Artefak berukuran 0 byte                | Anggap artefak tidak valid                                   |
| Model gagal dibuka                      | Hapus cache, tampilkan error, dan hentikan aplikasi          |
| Memori deployment tidak cukup           | Tampilkan pesan kegagalan proses dan arahkan pemeriksaan log |

---

## 12. Acceptance Criteria

Perbaikan dinyatakan selesai apabila seluruh kriteria berikut terpenuhi.

### AC-01

Ketika folder `model/artifacts` kosong, aplikasi otomatis menjalankan `scripts/retrain.py`.

### AC-02

Aplikasi tidak memanggil fungsi pemuatan model sebelum seluruh artefak tersedia.

### AC-03

Setelah retraining berhasil, file berikut terbentuk sesuai kebutuhan aplikasi:

```text
model/artifacts/model.pkl
model/artifacts/shap_explainer.pkl
```

Beserta artefak tambahan lain yang digunakan aplikasi.

### AC-04

Setelah artefak terbentuk, aplikasi menampilkan status model aktif dan form prediksi dapat digunakan.

### AC-05

Ketika aplikasi mengalami rerun akibat perubahan input, retraining tidak berjalan ulang.

### AC-06

Ketika aplikasi direstart dan artefak masih tersedia, aplikasi langsung memuat model.

### AC-07

Ketika `retrain.py` gagal, aplikasi menampilkan pesan kesalahan dan tidak menampilkan hasil prediksi.

### AC-08

Path yang digunakan tidak bergantung pada current working directory.

### AC-09

Implementasi berhasil dijalankan di Codespaces dan Streamlit Community Cloud.

### AC-10

Tidak terdapat perubahan pada hasil klasifikasi, desain UI, fitur input, dan logika bisnis sistem.

---

## 13. Skenario Pengujian

### Test Case 1 — Artefak lengkap

**Persiapan:**

Seluruh file artefak tersedia.

**Hasil yang diharapkan:**

* Retraining tidak dijalankan.
* Model langsung dimuat.
* Aplikasi dapat digunakan.

---

### Test Case 2 — Semua artefak tidak tersedia

**Persiapan:**

Hapus folder `model/artifacts`.

**Hasil yang diharapkan:**

* Folder dibuat otomatis.
* Retraining dijalankan.
* Artefak tersimpan.
* Model dimuat.
* Aplikasi tampil normal.

---

### Test Case 3 — Salah satu artefak tidak tersedia

**Persiapan:**

Hapus hanya:

```text
shap_explainer.pkl
```

**Hasil yang diharapkan:**

* Artefak dianggap tidak lengkap.
* Retraining dijalankan.
* Seluruh artefak dibuat kembali.

---

### Test Case 4 — Artefak kosong

**Persiapan:**

Buat file `model.pkl` berukuran 0 byte.

**Hasil yang diharapkan:**

* File dianggap tidak valid.
* Retraining dijalankan.

---

### Test Case 5 — Dataset tidak tersedia

**Persiapan:**

Ubah atau hapus file dataset yang dibaca `retrain.py`.

**Hasil yang diharapkan:**

* Retraining gagal.
* Pesan dataset tidak ditemukan ditampilkan.
* Aplikasi dihentikan.

---

### Test Case 6 — Retraining error

**Persiapan:**

Tambahkan exception sementara pada `retrain.py`.

**Hasil yang diharapkan:**

* Return code subprocess tidak nol.
* Error ditampilkan.
* Model tidak dimuat.
* Form prediksi tidak ditampilkan.

---

### Test Case 7 — Streamlit rerun

**Persiapan:**

Setelah aplikasi aktif, ubah beberapa nilai input.

**Hasil yang diharapkan:**

* Script Streamlit mengalami rerun.
* Retraining tidak berjalan ulang.
* Model diambil dari cache.

---

## 14. Definition of Done

Perbaikan dianggap selesai ketika:

* Mekanisme pemeriksaan artefak telah ditambahkan.
* Retraining otomatis berjalan ketika artefak tidak tersedia.
* Seluruh path telah menggunakan `pathlib`.
* `retrain.py` menghasilkan seluruh artefak yang diperlukan.
* Pemuatan model menggunakan `st.cache_resource`.
* Kegagalan training ditangani dengan `st.stop()`.
* Seluruh test case utama berhasil.
* Deployment Streamlit dapat dibuka tanpa pesan `File artefak tidak ditemukan`.
* Tidak ada perubahan di luar ruang lingkup perbaikan ini.
