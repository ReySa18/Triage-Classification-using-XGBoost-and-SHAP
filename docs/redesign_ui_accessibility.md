# Validasi Aksesibilitas Redesign UI Triage

Tanggal: 8 Juli 2026

## Palet SATS Final

| Level | Light BG | Light Text | Rasio | Dark BG | Dark Text | Rasio |
|---|---:|---:|---:|---:|---:|---:|
| Merah | `#FEE2E2` | `#7F1D1D` | 8.20:1 | `#3F1518` | `#FECACA` | 10.90:1 |
| Oranye | `#FFEDD5` | `#7C2D12` | 8.18:1 | `#431F0A` | `#FED7AA` | 10.78:1 |
| Kuning | `#FEF9C3` | `#713F12` | 8.07:1 | `#3A2A05` | `#FEF08A` | 11.92:1 |
| Hijau | `#DCFCE7` | `#14532D` | 8.30:1 | `#102F1C` | `#BBF7D0` | 11.99:1 |
| Biru | `#DBEAFE` | `#1E3A8A` | 8.49:1 | `#102A55` | `#BFDBFE` | 9.96:1 |

Semua kombinasi teks pada latar badge lulus WCAG AA untuk teks normal (>= 4.5:1).

## Redundansi Visual

Setiap level triase sekarang ditampilkan dengan kombinasi warna, ikon bentuk berbeda, nomor level `L1-L5`, label Indonesia, label klinis Inggris, dan target waktu. Informasi level tidak lagi bergantung pada warna saja.

## Oranye vs Kuning

Oranye menggunakan hue merah-jingga yang lebih gelap (`#EA580C` / `#FB923C`), sedangkan Kuning menggunakan emas cerah (`#CA8A04` / `#FACC15`). Keduanya juga dibedakan oleh ikon berbeda: segitiga peringatan untuk Oranye dan jam untuk Kuning.

Catatan: simulasi color-blindness eksternal tidak dijalankan di lingkungan ini karena browser visual tidak tersedia, tetapi desain final tidak mengandalkan warna sebagai satu-satunya sinyal.
