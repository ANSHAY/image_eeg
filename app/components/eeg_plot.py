"""128-channel EEG waveform plot — Plotly Scattergl for the demo app.

Stacked traces with vertical offsets sized by per-channel std so the
plot is readable across signal scales. Each trace is colored by its
brain region (frontal / temporal / parietal / occipital) via
:mod:`app.region_map`. WebGL backend (Scattergl) keeps 128-trace
re-renders smooth in Streamlit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from app.region_map import channel_to_color
from utils.config import Config, load_config


def _rgba(hex6: str, alpha: float) -> str:
    """Plotly rejects #RRGGBBAA — convert (#RRGGBB, alpha) to rgba(...)."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_eeg_plot(
    trial: np.ndarray,
    cfg: Optional[Config] = None,
    height: int = 700,
) -> go.Figure:
    """Stacked 128-channel EEG waveform plot.

    Args:
        trial: shape ``(num_channels, num_samples)``. Already preprocessed
            (z-scored, cropped) — this is a presentation layer, not a
            transformer.
        cfg:   optional config override.
        height: figure pixel height; defaults to 700 which fits the
            split-screen layout at 1280×800.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart(fig, use_container_width=True)``.
    """
    c = cfg if cfg is not None else load_config()
    if trial.ndim != 2:
        raise ValueError(f"trial must be 2-D (channels, samples), got {trial.shape}")
    n_ch, n_samples = trial.shape

    t_axis = np.arange(n_samples) / c.eeg.sample_rate_hz
    per_ch_std = trial.std(axis=-1, keepdims=True)
    spacing = float(per_ch_std.max() * 3 + 1e-3)
    offsets = np.arange(n_ch)[:, None] * spacing
    stacked = trial + offsets

    colors = channel_to_color(c)
    fig = go.Figure()
    for ch in range(n_ch):
        fig.add_trace(
            go.Scattergl(
                x=t_axis,
                y=stacked[ch],
                mode="lines",
                line=dict(width=0.6, color=colors.get(ch, c.ui.theme.text)),
                hovertemplate=(
                    "ch %d<br>t=%%{x:.3f}s  v=%%{customdata:.3f}<extra></extra>"
                    % ch
                ),
                customdata=trial[ch],
                showlegend=False,
            ),
        )

    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=20, b=40),
        paper_bgcolor=c.ui.theme.bg,
        plot_bgcolor=c.ui.theme.bg,
        font=dict(color=c.ui.theme.text),
        xaxis=dict(
            title="time (s)",
            gridcolor=_rgba(c.ui.theme.text, 0.13),
            zerolinecolor=_rgba(c.ui.theme.text, 0.20),
        ),
        yaxis=dict(
            title="channel (stacked)",
            showticklabels=False,
            gridcolor=_rgba(c.ui.theme.text, 0.13),
        ),
    )
    return fig
