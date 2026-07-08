import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
=============================================================================
TAHAP M1 - KONSTRUKSI LABEL TRIAGE IGD
=============================================================================
PRD: Model Prediksi Skor Triage IGD — RSU Aulia
Peneliti: Reymondo
Metode: MEWS (Modified Early Warning Score) + ICD Fallback
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi tampilan ──────────────────────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# =============================================================================
# LANGKAH 1: LOAD DATASET
# =============================================================================
print("=" * 70)
print("LANGKAH 1: LOAD DATASET")
print("=" * 70)

df = pd.read_csv(r"d:\SKRIPSI\Dataset_Klinis Edit.csv")

print(f"Shape dataset: {df.shape}")
print(f"\nKolom: {df.columns.tolist()}")
print(f"\n5 baris pertama:")
print(df.head())

# =============================================================================
# LANGKAH 2: PARSING KOLOM USIA
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 2: PARSING KOLOM USIA")
print("=" * 70)

def parse_usia(usia_str):
    """
    Ekstraksi nilai tahun dari string seperti '13 Th 9 Bl 3 Hr'
    Output: integer tahun, atau NaN jika gagal
    """
    if pd.isna(usia_str):
        return np.nan
    match = re.search(r'(\d+)\s*[Tt]h', str(usia_str))
    if match:
        return int(match.group(1))
    # Jika format hanya angka bulan/hari (bayi), return 0
    match_bl = re.search(r'(\d+)\s*[Bb]l', str(usia_str))
    if match_bl:
        return 0
    return np.nan

df['usia_tahun'] = df['usia'].apply(parse_usia)

print(f"Sample parsing usia:")
sample_usia = df[['usia', 'usia_tahun']].dropna(subset=['usia']).head(10)
print(sample_usia.to_string())
print(f"\nMissing usia_tahun: {df['usia_tahun'].isna().sum()} baris")
print(f"Statistik usia_tahun:\n{df['usia_tahun'].describe()}")

# =============================================================================
# LANGKAH 3: EKSTRAKSI GCS DARI KOLOM 'GCS'
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 3: EKSTRAKSI GCS DARI KOLOM 'GCS'")
print("=" * 70)

def extract_gcs(text):
    """
    Ekstraksi nilai GCS E, M, V dari teks pemeriksaan fisik.
    Pola: 'GCS E : X M : X V : X' atau variasi spasi
    """
    if pd.isna(text):
        return np.nan, np.nan, np.nan, np.nan

    text = str(text)
    # Berbagai variasi pola
    pattern = r'GCS\s*E\s*[:\-]?\s*(\d+)\s*M\s*[:\-]?\s*(\d+)\s*V\s*[:\-]?\s*(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        e = int(match.group(1))
        m = int(match.group(2))
        v = int(match.group(3))
        total = e + m + v
        return e, m, v, total
    return np.nan, np.nan, np.nan, np.nan

gcs_extracted = df['GCS'].apply(extract_gcs)
df['gcs_e']     = [x[0] for x in gcs_extracted]
df['gcs_m']     = [x[1] for x in gcs_extracted]
df['gcs_v']     = [x[2] for x in gcs_extracted]
df['gcs_total'] = [x[3] for x in gcs_extracted]

gcs_found = df['gcs_total'].notna().sum()
print(f"GCS berhasil diekstraksi: {gcs_found} dari {len(df)} baris ({gcs_found/len(df)*100:.1f}%)")
print(f"GCS tidak ditemukan (NaN): {df['gcs_total'].isna().sum()} baris")
print(f"\nStatistik GCS total:\n{df['gcs_total'].describe()}")

# =============================================================================
# LANGKAH 4: PENANGANAN NILAI NOL TANDA VITAL (GANTI → NaN)
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 4: PENANGANAN NILAI NOL TANDA VITAL")
print("=" * 70)

VITAL_COLS = ['sistole', 'diastole', 'denyut_jantung', 'laju_pernafasan', 'suhu_tubuh', 'SpO2']

print("Jumlah nilai nol sebelum pembersihan:")
for col in VITAL_COLS:
    n_zero = (df[col] == 0).sum()
    pct = n_zero / len(df) * 100
    print(f"  {col:20s}: {n_zero} ({pct:.1f}%)")

# Ganti 0 → NaN pada tanda vital (nilai 0 tidak fisiologis)
for col in VITAL_COLS:
    df[col] = df[col].replace(0, np.nan)

print("\nJumlah NaN setelah konversi 0 -> NaN:")
for col in VITAL_COLS:
    n_nan = df[col].isna().sum()
    pct = n_nan / len(df) * 100
    print(f"  {col:20s}: {n_nan} ({pct:.1f}%)")

# =============================================================================
# LANGKAH 5: WINSORIZING OUTLIER BERDASARKAN RENTANG FISIOLOGIS KLINIS
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 5: WINSORIZING OUTLIER TANDA VITAL")
print("=" * 70)

PHYSIOLOGICAL_BOUNDS = {
    'sistole':          (70,  220),
    'diastole':         (40,  130),
    'denyut_jantung':   (30,  200),
    'laju_pernafasan':  (8,   40),
    'suhu_tubuh':       (34.0, 42.0),
    'SpO2':             (70,  100),
    'skala_nyeri':      (0,   10),
}

for col, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
    if col in df.columns:
        n_outlier = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"  {col:20s}: {n_outlier} outlier di-clip ke [{lo}, {hi}]")

# =============================================================================
# LANGKAH 6: HITUNG MEWS SCORE
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 6: HITUNG MEWS SCORE")
print("=" * 70)

def score_sistole(val):
    if pd.isna(val): return np.nan
    if val < 70:    return 3
    if val <= 80:   return 2
    if val <= 100:  return 1
    if val <= 199:  return 0
    return 2  # >= 200

def score_denyut(val):
    if pd.isna(val): return np.nan
    if val < 40:    return 2
    if val <= 50:   return 1
    if val <= 100:  return 0
    if val <= 110:  return 1
    if val <= 129:  return 2
    return 3  # >= 130

def score_laju_pernafasan(val):
    if pd.isna(val): return np.nan
    if val < 9:     return 2
    if val <= 14:   return 0
    if val <= 20:   return 1
    if val <= 29:   return 2
    return 3  # >= 30

def score_suhu(val):
    if pd.isna(val): return np.nan
    if val < 35:    return 2
    if val <= 38.4: return 0
    return 1  # >= 38.5

def score_gcs(gcs_total):
    """
    Mapping GCS total ke skor MEWS kesadaran:
    15 (Alert)=0, 13-14 (Suara)=1, 9-12 (Nyeri)=2, <9 (Tidak Respons)=3
    """
    if pd.isna(gcs_total): return np.nan
    if gcs_total == 15:   return 0
    if gcs_total >= 13:   return 1
    if gcs_total >= 9:    return 2
    return 3

# Hitung skor komponen
df['mews_sistole']   = df['sistole'].apply(score_sistole)
df['mews_denyut']    = df['denyut_jantung'].apply(score_denyut)
df['mews_pernafasan']= df['laju_pernafasan'].apply(score_laju_pernafasan)
df['mews_suhu']      = df['suhu_tubuh'].apply(score_suhu)
df['mews_gcs']       = df['gcs_total'].apply(score_gcs)

MEWS_COMPONENTS = ['mews_sistole', 'mews_denyut', 'mews_pernafasan', 'mews_suhu', 'mews_gcs']

# Hitung jumlah komponen valid per baris
df['mews_n_valid'] = df[MEWS_COMPONENTS].notna().sum(axis=1)

print(f"Distribusi jumlah komponen MEWS yang valid:")
print(df['mews_n_valid'].value_counts().sort_index())

# Hitung skor MEWS — hanya untuk baris dengan >= 3 komponen valid
def hitung_mews(row):
    """
    Strategi PRD §4.1.3:
    - >= 3 komponen valid: hitung MEWS, imputasi rata2 untuk yang missing
    - < 3 komponen valid: return NaN (gunakan fallback ICD)
    """
    komponen = row[MEWS_COMPONENTS]
    n_valid = komponen.notna().sum()
    if n_valid < 3:
        return np.nan
    if n_valid == 5:
        return komponen.sum()
    # Imputasi rata-rata komponen yang valid untuk yang missing
    mean_val = komponen.mean()
    return komponen.fillna(mean_val).sum()

df['mews_score'] = df.apply(hitung_mews, axis=1)

n_mews_valid = df['mews_score'].notna().sum()
n_mews_null  = df['mews_score'].isna().sum()
print(f"\nMEWS berhasil dihitung: {n_mews_valid} baris ({n_mews_valid/len(df)*100:.1f}%)")
print(f"MEWS tidak dapat dihitung (< 3 vital): {n_mews_null} baris ({n_mews_null/len(df)*100:.1f}%)")
print(f"\nStatistik MEWS score:\n{df['mews_score'].describe()}")

# =============================================================================
# LANGKAH 7: MAPPING MEWS → KATEGORI TRIAGE
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 7: MAPPING MEWS → KATEGORI TRIAGE")
print("=" * 70)

def mews_to_triage(score):
    if pd.isna(score): return np.nan
    if score >= 5:  return 'P1'
    if score >= 3:  return 'P2'
    return 'P3'

df['kategori_triage'] = df['mews_score'].apply(mews_to_triage)
df['metode_label']    = df['mews_score'].apply(lambda x: 'mews' if pd.notna(x) else None)

print("Distribusi triage dari MEWS:")
print(df['kategori_triage'].value_counts())

# =============================================================================
# LANGKAH 8: FALLBACK ICD MAPPING (untuk baris tanpa MEWS)
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 8: ICD FALLBACK")
print("=" * 70)

# Mapping prefix kode ICD → kategori triage
ICD_P1 = [
    'I21', 'I22', 'I60', 'I61', 'I62', 'I63', 'I64',   # AMI, stroke hemoragik
    'J18', 'J96', 'J80',                                  # Pneumonia berat, gagal napas
    'R09', 'R57', 'R00',                                   # Gagal napas, syok
    'S06', 'S09', 'T07', 'T71',                           # Cedera kepala, trauma multipel
    'A41', 'A40',                                          # Sepsis
    'K92', 'K25', 'K26',                                   # Perdarahan GI
    'O14', 'O15',                                          # Preeklamsia/eklamsia
    'E11', 'E10',                                          # DM dengan komplikasi (kode kombinasi)
]

ICD_P2 = [
    'A09', 'A00',          # Gastroenteritis
    'R10',                 # Nyeri abdomen
    'J06', 'J20', 'J22',  # ISPA, bronkitis
    'K35', 'K36', 'K37',  # Apendisitis
    'M54', 'M79',          # Nyeri punggung/muskuloskeletal
    'I10', 'I11',          # Hipertensi
    'N23', 'N20',          # Kolik renal
    'G43', 'G44',          # Migren, nyeri kepala
    'R50', 'R51',          # Demam, nyeri kepala
    'J45',                 # Asma
    'F10', 'F19',          # Intoksikasi
]

def icd_fallback(kode_icd):
    """Mapping kode ICD ke triage berdasarkan prefix"""
    if pd.isna(kode_icd):
        return np.nan
    kode = str(kode_icd).strip().upper()
    # Cek P1 (3 karakter prefix)
    for prefix in ICD_P1:
        if kode.startswith(prefix):
            return 'P1'
    # Cek P2
    for prefix in ICD_P2:
        if kode.startswith(prefix):
            return 'P2'
    # Default: P3 (tidak darurat)
    return 'P3'

# Terapkan fallback hanya pada baris yang tidak punya MEWS
mask_no_mews = df['mews_score'].isna()
df.loc[mask_no_mews, 'kategori_triage'] = df.loc[mask_no_mews, 'kode_ICD'].apply(icd_fallback)
df.loc[mask_no_mews & df['kategori_triage'].notna(), 'metode_label'] = 'icd_fallback'

# Identifikasi baris yang tidak bisa diklasifikasikan sama sekali
mask_no_label = df['kategori_triage'].isna()
df.loc[mask_no_label, 'metode_label'] = 'ekslusi'

n_fallback = (df['metode_label'] == 'icd_fallback').sum()
n_ekslusi  = (df['metode_label'] == 'ekslusi').sum()
print(f"Fallback ICD berhasil: {n_fallback} baris")
print(f"Tidak dapat diklasifikasikan (ekslusi): {n_ekslusi} baris")

# =============================================================================
# LANGKAH 9: RINGKASAN DISTRIBUSI LABEL
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 9: RINGKASAN DISTRIBUSI LABEL AKHIR")
print("=" * 70)

print("\n── Metode pelabelan ──")
print(df['metode_label'].value_counts())

print("\n── Distribusi kategori triage (termasuk semua metode) ──")
dist = df[df['metode_label'] != 'ekslusi']['kategori_triage'].value_counts()
total_labeled = dist.sum()
for cat, count in dist.items():
    print(f"  {cat}: {count:5d} ({count/total_labeled*100:.1f}%)")

print(f"\n  Total berlabel: {total_labeled}")
print(f"  Total dieksklusi: {n_ekslusi}")

print("\n── Distribusi per metode ──")
cross = pd.crosstab(df['metode_label'], df['kategori_triage'])
print(cross)

# =============================================================================
# LANGKAH 10: SIMPAN DATASET BERLABEL
# =============================================================================
print("\n" + "=" * 70)
print("LANGKAH 10: SIMPAN OUTPUT")
print("=" * 70)

# Dataset lengkap (termasuk baris ekslusi, untuk transparansi)
OUTPUT_FULL = r"d:\SKRIPSI\output_M1\Dataset_M1_full.csv"
# Dataset siap modeling (tanpa baris ekslusi)
OUTPUT_MODEL = r"d:\SKRIPSI\output_M1\Dataset_M1_berlabel.csv"

import os
os.makedirs(r"d:\SKRIPSI\output_M1", exist_ok=True)

df.to_csv(OUTPUT_FULL, index=False)

df_model = df[df['metode_label'] != 'ekslusi'].copy()
df_model.to_csv(OUTPUT_MODEL, index=False)

print(f"Dataset lengkap disimpan: {OUTPUT_FULL}")
print(f"  Shape: {df.shape}")
print(f"\nDataset berlabel (siap M2) disimpan: {OUTPUT_MODEL}")
print(f"  Shape: {df_model.shape}")

# Ringkasan kolom baru yang ditambahkan
print("\n── Kolom baru yang ditambahkan ──")
new_cols = ['usia_tahun', 'gcs_e', 'gcs_m', 'gcs_v', 'gcs_total',
            'mews_sistole', 'mews_denyut', 'mews_pernafasan', 'mews_suhu', 'mews_gcs',
            'mews_n_valid', 'mews_score', 'kategori_triage', 'metode_label']
for c in new_cols:
    print(f"  ✓ {c}")

print("\n" + "=" * 70)
print("M1 SELESAI — Dataset berlabel siap untuk Fase M2 (EDA)")
print("=" * 70)
