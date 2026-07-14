"""
Reusable UI components for SATS/TEWS presentation.

These helpers intentionally keep clinical/model logic outside the UI layer.
Redesigned — solid color hero banners matching triage-redesign.html.
"""

from backend.feature_engineering import SATS_FULL_LABELS, SATS_LABELS, SATS_WAKTU


TRIAGE_META = {
    0: {
        "slug": "red",
        "name": "Merah",
        "english": "Resuscitation",
        "icon": "&#9889;",
        "number": "1",
        "action": "Resusitasi segera. Aktifkan tim resusitasi dan monitor kontinu.",
    },
    1: {
        "slug": "orange",
        "name": "Oranye",
        "english": "Emergent",
        "icon": "&#9650;",
        "number": "2",
        "action": "Tangani dalam 10 menit. Siapkan monitoring, akses IV, dan evaluasi dokter.",
    },
    2: {
        "slug": "yellow",
        "name": "Kuning",
        "english": "Urgent",
        "icon": "&#9201;",
        "number": "3",
        "action": "Evaluasi dalam 60 menit. Pantau perubahan tanda vital.",
    },
    3: {
        "slug": "green",
        "name": "Hijau",
        "english": "Less Urgent",
        "icon": "&#10003;",
        "number": "4",
        "action": "Tangani dalam 4 jam. Triage ulang bila kondisi berubah.",
    },
    4: {
        "slug": "blue",
        "name": "Biru",
        "english": "Not Urgent",
        "icon": "i",
        "number": "5",
        "action": "Tangani dalam 6 jam. Pertimbangkan rujukan/fasilitas primer bila sesuai.",
    },
}

TRIAGE_ORDER = [0, 1, 2, 3, 4]
LABEL_TO_CLASS = {label: cls for cls, label in SATS_LABELS.items()}


def triage_badge_html(level: int, variant: str = "compact", extra_class: str = "") -> str:
    """Return a color-accessible triage badge/card with icon, number, and label."""
    meta = TRIAGE_META[level]
    full_label = SATS_FULL_LABELS[level]
    time_label = SATS_WAKTU[level]
    classes = f"triage-badge triage-{meta['slug']} triage-badge-{variant} {extra_class}".strip()

    return (
        f'<div class="{classes}">'
        f'<div class="triage-code" aria-hidden="true">'
        f'<span class="triage-icon">{meta["icon"]}</span>'
        f'<span class="triage-number">L{meta["number"]}</span>'
        f'</div>'
        f'<div class="triage-copy">'
        f'<div class="triage-name">{full_label}</div>'
        f'<div class="triage-subtitle">{meta["english"]}</div>'
        f'</div>'
        f'<div class="triage-time">{time_label}</div>'
        f'</div>'
    )


def triage_result_hero_html(level: int) -> str:
    """Result hero banner — solid color background, white text.
    Matches the result-banner design from triage-redesign.html."""
    meta = TRIAGE_META[level]
    return (
        f'<div class="triage-hero triage-{meta["slug"]}">'
        f'<div class="triage-hero-top">'
        f'<span class="triage-number">LEVEL {meta["number"]} · SATS</span>'
        f'<span class="triage-time">{SATS_WAKTU[level]}</span>'
        f'</div>'
        f'<div class="triage-hero-name">{SATS_FULL_LABELS[level]}</div>'
        f'<div class="triage-hero-subtitle">{meta["english"]}</div>'
        f'<div class="triage-action">{meta["action"]}</div>'
        f'</div>'
    )


def score_badge_html(score: int) -> str:
    s = min(max(int(score), 0), 3)
    label = ["Normal", "Perhatian", "Waspada", "Kritis"][s]
    return (
        f'<span class="score-badge score-{s}" '
        f'aria-label="Skor {score}, {label}">{score}</span>'
    )
