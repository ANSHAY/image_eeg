"""Frequency-spectrum plot for the demo app.

Computes Welch PSD per channel (scipy.signal.welch) and renders one
trace = channel-averaged PSD in dB. Useful for spotting whether the
notch filter is catching powerline interference and where the bulk
of the brain rhythms sit (alpha 8-13 Hz, beta 13-30 Hz, …).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy import signal as scipy_signal

from app.components.eeg_plot import _rgba
from utils.config import Config, load_config


def build_fft_plot(
    trial: np.ndarray,
    cfg: Optional[Config] = None,
    height: int = 240,
) -> go.Figure:
    """Channel-averaged Welch PSD in dB.

    Args:
        trial: shape ``(num_channels, num_samples)``.
        cfg:   optional config override.
        height: figure pixel height; defaults to 240 for the bottom strip.
    """
    c = cfg if cfg is not None else load_config()
    if trial.ndim != 2:
        raise ValueError(f"trial must be 2-D (channels, samples), got {trial.shape}")

    fs = c.eeg.sample_rate_hz
    nperseg = min(256, trial.shape[1])
    freqs, psd = scipy_signal.welch(trial, fs=fs, axis=-1, nperseg=nperseg)
    psd_mean = psd.mean(axis=0)
    psd_db = 10.0 * np.log10(np.maximum(psd_mean, 1e-12))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=freqs,
            y=psd_db,
            mode="lines",
            line=dict(color=c.ui.theme.accent, width=2),
            showlegend=False,
            hovertemplate="f=%{x:.1f} Hz<br>PSD=%{y:.1f} dB<extra></extra>",
        ),
    )
    # Powerline marker
    fig.add_vline(
        x=c.eeg.powerline_hz,
        line=dict(color=c.ui.region_colors.occipital, width=1, dash="dot"),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=15, b=35),
        paper_bgcolor=c.ui.theme.bg,
        plot_bgcolor=c.ui.theme.bg,
        font=dict(color=c.ui.theme.text),
        xaxis=dict(
            title="frequency (Hz)",
            gridcolor=_rgba(c.ui.theme.text, 0.13),
            range=[0, fs / 2],
        ),
        yaxis=dict(
            title="PSD (dB)",
            gridcolor=_rgba(c.ui.theme.text, 0.13),
        ),
    )
    return fig
