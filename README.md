# 🏥 Klasifikasi Prioritas Triage IGD — RSU Aulia

> **Sistem Decision Support (DSS) berbasis Machine Learning untuk klasifikasi prioritas triage IGD**  
> Menggunakan XGBoost + SHAP + Standar SATS-TEWS (South African Triage Scale)

---

## 📋 Ringkasan Proyek

Penelitian ini membangun model klasifikasi otomatis untuk menentukan **level kegawatan triage** pasien IGD RSU Aulia berdasarkan tanda vital dan skor SATS-TEWS. Model ini berfungsi sebagai **Decision Support System (DSS)** — bukan pengganti keputusan tenaga medis berlisensi.

| Item | Detail |
|---|---|
| **Peneliti** | Reymondo |
| **Metodologi** | CRISP-DM |
| **Algoritma** | XGBoost (multi:softmax) dalam ImbPipeline |
| **Standar Triage** | SATS-TEWS — 5 Level (Merah / Oranye / Kuning / Hijau / Biru) |
| **Dataset** | 6.339 rekam medis IGD RSU Aulia (Januari–Mei 2026) |
| **Fitur** | 31 fitur (tanpa `skala_nyeri` — FEAT-007) |

---

## 🗂️ Struktur Folder

```
SKRIPSI/
├── notebook/
│   └── klasifikasi_triage.ipynb     # Notebook CRISP-DM final
│
├── data/
│   ├── raw/
│   │   └── Dataset_Klinis_Edit.csv  # Dataset mentah (.gitignore)
│   └── processed/
│       └── Dataset_Labeled_SATS.csv # Dataset berlabel SATS-TEWS
│
├── model/
│   ├── artifacts/                   # Model artifacts (.pkl)
│   │   ├── model_triage_xgb.pkl
│   │   ├── pipeline_imblearn.pkl
│   │   ├── scaler_minmax.pkl
│   │   ├── feature_names.pkl
│   │   ├── best_thresholds.pkl
│   │   ├── best_params.pkl
│   │   ├── imputer.pkl
│   │   └── split_indices.pkl
│   └── figures/                     # Visualisasi hasil model
│       └── fig_*.png
│
├── app/                             # Streamlit DSS Application
│   ├── app.py                       # Entry point
│   ├── assets/style.css             # Custom CSS
│   ├── backend/                     # Model loading & prediction logic
│   │   ├── predictor.py
│   │   ├── feature_engineering.py
│   │   └── tews_calculator.py
│   └── ui/                          # UI tab components
│       ├── tab_prediction.py
│       ├── tab_tews.py
│       └── tab_guidelines.py
│
├── scripts/
│   ├── M1_konstruksi_label.py       # Konstruksi label SATS dari data mentah
│   └── retrain.py                   # Re-training model
│
├── docs/
│   ├── prd/                         # Product Requirement Documents
│   ├── wireframe/                   # UI wireframes & guidelines HTML
│   └── SUMMARY.md                   # Ringkasan hasil penelitian
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi DSS

```bash
streamlit run app/app.py
```

### 3. Re-train Model (opsional)

```bash
python scripts/retrain.py
```

> Estimasi waktu: 30-90 menit (tergantung spesifikasi mesin)

---

## 🔧 Changelog Versi

| Versi | Perubahan Utama |
|---|---|
| v1-v3 | Prototype awal, eksplorasi model |
| v4 (PRD-OPT-002) | Anti-leakage imputer, ImbPipeline CV, fitur interaksi klinis, threshold optimization |
| v5 / final (FEAT-007) | Hapus `skala_nyeri` (zero information gain), model final |

---

## Disclaimer

Model ini adalah prototype Decision Support System (DSS).
Keputusan triage final tetap berada di tangan tenaga medis berlisensi.
Label target `sats_label` bersifat surrogate — dikonstruksi algoritmik dari aturan SATS-TEWS.
