# SUMMARY — Klasifikasi Prioritas Triage IGD RSU Aulia

> Dokumen ini merangkum seluruh pekerjaan pengembangan model ML Triage IGD RSU Aulia,
> mencakup metodologi, hasil model, perbaikan iteratif, dan keputusan teknis.

---

## 1. Gambaran Umum Proyek

**Judul:** Klasifikasi Prioritas Triage IGD menggunakan XGBoost, SHAP, dan Standar SATS-TEWS  
**Institusi:** RSU Aulia  
**Periode Data:** Januari – Mei 2026  
**Dataset:** 6.339 rekam medis pasien IGD  
**Metodologi:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

### Tujuan Penelitian

| ID | Tujuan | Target |
|---|---|---|
| TO-01 | Bangun model klasifikasi triage dari fitur klinis menggunakan label surrogate SATS-TEWS | F1-Macro >= 0.85 |
| TO-02 | Prioritaskan safety: Recall Merah (kritis) | Recall Merah >= 0.80 |
| TO-03 | Jelaskan keputusan model menggunakan SHAP | >= 2 fitur interaksi di top-10 SHAP |
| TO-04 | Minimalisasi undertriage pada kelas kritis | Combined undertriage < 5% |

---

## 2. Standar Triage: SATS-TEWS

Model mengklasifikasikan pasien ke dalam 5 level kegawatan berdasarkan aturan SATS-TEWS:

| Kode | Warna | Makna | Aksi Klinis |
|---|---|---|---|
| 0 | Merah | Immediate — sangat kritis | Penanganan segera |
| 1 | Oranye | Very Urgent — kritis | Penanganan dalam 10 menit |
| 2 | Kuning | Urgent — mendesak | Penanganan dalam 60 menit |
| 3 | Hijau | Less Urgent — tidak mendesak | Penanganan dalam 4 jam |
| 4 | Biru | Not Urgent — sangat ringan | Dapat menunggu atau dirujuk |

### Formula Label Surrogate (assign_sats_label)

```python
if GCS <= 8:                          -> Merah
if SBP < 70:                          -> Merah
if SpO2 < 90 and (RR < 9 or RR >= 30) -> Merah
if TEWS >= 7:                          -> Oranye
elif TEWS >= 5:                        -> Kuning
elif TEWS >= 3:                        -> Hijau
else:                                  -> Biru
```

---

## 3. Fitur Model Final (31 Fitur)

### Demografis (3)
- `usia_tahun`, `kelompok_usia`, `jenis_kelamin_enc`

### GCS (4)
- `GCS_E`, `GCS_M`, `GCS_V`, `GCS_total`

### Tanda Vital (6)
- `sistole`, `diastole`, `denyut_jantung`, `laju_pernafasan`, `suhu_tubuh`, `SpO2`

### Turunan Vital (1)
- `MAP` (Mean Arterial Pressure)

### Binary Flags (8)
- `flag_takikardia`, `flag_bradikardia`, `flag_hipotensi`, `flag_hipertensi`
- `flag_takipnea`, `flag_hipoksia`, `flag_demam`, `flag_hipotermi`

### Composite Score (1)
- `n_vital_abnormal`

### Fitur Interaksi Klinis / FEAT-04 (5)
- `shock_index` — Birkhahn et al. (2005)
- `cardiorespiratory_score` — composite 4 indikator kegawatan
- `pulse_pressure` — selisih sistole-diastole
- `mews_approx` — Modified Early Warning Score (Subbe 2001)
- `flag_hypoxic_shock` — SpO2 < 90 AND sistole < 90

### Cara Datang / One-Hot (3)
- `cara_datang_Dokter`, `cara_datang_KLL`, `cara_datang_Puskesmas`, `cara_datang_Sendiri`

> **FEAT-007:** `skala_nyeri` DIHAPUS dari FEATURE_COLS karena zero information gain
> terhadap label SATS-TEWS. Label surrogate tidak menggunakan parameter nyeri,
> sehingga XGBoost tidak dapat mempelajari hubungan yang tidak ada di data.

---

## 4. Arsitektur Pipeline

```
Raw Data
  -> Zero-to-NaN conversion + Clip Outlier (CLIP_BOUNDS)
  -> IterativeImputer MICE (fit pada X_train only — anti-leakage FIX-01)
  -> Feature Engineering (MAP, flags, interaction features)
  -> Label Construction (assign_sats_label via TEWS)
  -> Stratified Train-Test Split (80:20, random_state=42)
  -> MinMaxScaler (fit pada X_train only)
  -> ImbPipeline:
       SMOTE Partial (dalam CV fold — FEAT-02)
       XGBoost (multi:softmax, num_class=5)
  -> RandomizedSearchCV (100 iter x 5-fold StratifiedKFold)
  -> Threshold Optimization per Kelas (FEAT-05)
  -> SHAP TreeExplainer
```

---

## 5. Perbaikan Iteratif

### v4 — PRD-OPT-002

| ID | Perbaikan | Dampak |
|---|---|---|
| FIX-01 | IterativeImputer setelah split (anti-leakage) | Evaluasi lebih valid |
| FIX-02 | Framing: prediksi -> klasifikasi + label surrogate | Kejujuran akademik |
| FIX-03 | OUTPUT_DIR portabel (Colab/Local) | Reproducibility |
| FIX-04 | Narasi jujur threshold optimization | Transparansi |
| FIX-05 | Analisis AC yang gagal (M04, M05) | Dokumentasi lengkap |
| FIX-06 | Section Keterbatasan Penelitian terstruktur (7 poin) | Kelengkapan akademik |
| ADD-01 | Balanced Accuracy + MCC | Metrik adil untuk imbalanced |
| ADD-02 | GCS Availability Report | Transparansi data |
| ADD-03 | Missing Value Pattern Analysis | Analisis kualitas data |
| ADD-04 | Failure Mode Analysis (perspektif klinis) | Analisis kesalahan model |

### v5 / Final — FEAT-007

| ID | Perbaikan | Dampak |
|---|---|---|
| FEAT-007 | Hapus `skala_nyeri` dari FEATURE_COLS dan model | Model lebih clean, konsisten dengan SATS-TEWS |

---

## 6. Keterbatasan Penelitian

| ID | Keterbatasan | Mitigasi |
|---|---|---|
| L-01 | Label surrogate — bukan keputusan triage aktual tenaga medis | Deklarasi eksplisit di framing penelitian |
| L-02 | Kelas Oranye hanya ~17 sampel (0.27%) | SMOTE synthetic, Recall dilaporkan dengan catatan |
| L-03 | Single-site: RSU Aulia (generalisasi belum tervalidasi) | Perlu validasi eksternal di RS lain |
| L-04 | Missing value tanda vital ~33% | IterativeImputer MICE |
| L-05 | Threshold optimization tidak selalu efektif | Default threshold (0.5) dipertahankan jika delta negatif |
| L-06 | Evaluasi temporal belum dilakukan | Disarankan untuk penelitian lanjutan |
| L-07 | Skala nyeri tidak dimodelkan (FEAT-007) | Asesmen nyeri independen oleh tenaga medis |

---

## 7. Struktur File Kunci

| File | Lokasi | Fungsi |
|---|---|---|
| `klasifikasi_triage.ipynb` | `notebook/` | Notebook CRISP-DM lengkap (final) |
| `model_triage_xgb.pkl` | `model/artifacts/` | Model XGBoost final |
| `pipeline_imblearn.pkl` | `model/artifacts/` | ImbPipeline (SMOTE + XGB) |
| `scaler_minmax.pkl` | `model/artifacts/` | MinMaxScaler fitted |
| `feature_names.pkl` | `model/artifacts/` | Daftar 31 fitur model |
| `best_thresholds.pkl` | `model/artifacts/` | Threshold optimal per kelas |
| `shap_explainer.pkl` | `model/artifacts/` | SHAP TreeExplainer |
| `imputer.pkl` | `model/artifacts/` | IterativeImputer MICE |
| `retrain.py` | `scripts/` | Script re-training model |
| `predictor.py` | `app/backend/` | Inference pipeline untuk Streamlit |
| `app.py` | `app/` | Streamlit DSS entry point |

---

## 8. Changelog Repositori

| Tanggal | Perubahan |
|---|---|
| 2026-06-23 | FEAT-007: Hapus `skala_nyeri`, PRD-FEAT-007 v1.0 approved |
| 2026-07-08 | Pembersihan notebook V5 (FEAT-007 cleanup) |
| 2026-07-08 | Rename artifact model (hapus suffix `_v5`) |
| 2026-07-08 | Reorganisasi folder Git-ready: notebook/, data/, model/, app/, scripts/, docs/ |
