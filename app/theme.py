"""Dark-theme styling for the Streamlit demo app.

Reads colors from ``cfg.ui.theme`` and emits a `<style>` block that
applies glassmorphism cards, panel backgrounds, and accent-cyan
typography. Call :func:`inject_theme_css` once at the top of the
Streamlit script.

This module avoids hardcoding colors — every value comes from config.
"""

from __future__ import annotations

from typing import Optional

from utils.config import Config, load_config


def _css(cfg: Config) -> str:
    t = cfg.ui.theme
    return f"""
    <style>
    .stApp {{
        background-color: {t.bg};
        color: {t.text};
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background-color: {t.bg};
    }}
    h1, h2, h3, h4 {{
        color: {t.accent};
        letter-spacing: 0.02em;
    }}
    .vcr-card {{
        background-color: {t.surface};
        background-color: rgba(16, 24, 48, {t.surface_alpha});
        border: 1px solid {t.accent}33;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 8px 0;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}
    .vcr-metric-label {{
        color: {t.text}cc;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .vcr-metric-value {{
        color: {t.accent};
        font-size: 1.6em;
        font-weight: 600;
    }}
    .vcr-status-warming {{
        color: #fbbf24;
        font-style: italic;
    }}
    </style>
    """


def inject_theme_css(cfg: Optional[Config] = None) -> str:
    """Return the `<style>` block; Streamlit caller wraps in st.markdown."""
    return _css(cfg if cfg is not None else load_config())
