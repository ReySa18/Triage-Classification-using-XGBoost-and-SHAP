"""
ui/theme.py
Theme management — provides theme-aware colors for inline styles
"""


# ── Light theme colors ──
LIGHT = {
    'bg_app': '#f9f9ff',
    'bg_card': '#ffffff',
    'bg_surface': '#ecedf6',
    'bg_surface_high': '#e7e8f0',
    'bg_surface_low': '#f2f3fb',
    'border_main': '#c2c6d4',
    'border_light': '#e7e8f0',
    'text_primary': '#191c21',
    'text_secondary': '#424752',
    'text_tertiary': '#727783',
    'accent_primary': '#005eb8',
    'prob_track': '#e7e8f0',
    'shap_track': '#f2f3fb',
    'shap_center': '#c2c6d4',
    'legend_bg': '#ecedf6',
}

# ── Dark theme colors ──
DARK = {
    'bg_app': '#111318',
    'bg_card': '#1a1d24',
    'bg_surface': '#22252d',
    'bg_surface_high': '#2a2e38',
    'bg_surface_low': '#191c22',
    'border_main': '#3a3f4b',
    'border_light': '#2e3240',
    'text_primary': '#e4e5eb',
    'text_secondary': '#b0b4c0',
    'text_tertiary': '#8a8f9e',
    'accent_primary': '#6aadff',
    'prob_track': '#2a2e38',
    'shap_track': '#22252d',
    'shap_center': '#3a3f4b',
    'legend_bg': '#22252d',
}


def get_theme_colors(theme: str) -> dict:
    """Return color dict for current theme."""
    return DARK if theme == 'dark' else LIGHT


def is_dark(theme: str) -> bool:
    return theme == 'dark'
