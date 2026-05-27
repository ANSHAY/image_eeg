"""EEG denoising primitives.

Pure functions that consume and return ``np.ndarray`` of shape
``(channels, samples)``. All filter parameters are sourced from
``cfg.preprocessing`` and ``cfg.eeg`` — no literal cutoffs in code.

Operates on a single trial at a time; composition into a fixed
pipeline lives in :class:`preprocessing.signal_processing.Pipeline`
(added in box 21).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from utils.config import Config, load_config

_ZSCORE_EPS = 1e-8


def _cfg(cfg: Optional[Config]) -> Config:
    return cfg if cfg is not None else load_config()


def bandpass(x: np.ndarray, cfg: Optional[Config] = None) -> np.ndarray:
    """Zero-phase Butterworth bandpass via SOS + ``sosfiltfilt``.

    Cutoffs and order from ``cfg.preprocessing.bandpass``; sample rate
    from ``cfg.eeg.sample_rate_hz``.
    """
    c = _cfg(cfg)
    bp = c.preprocessing.bandpass
    sos = scipy_signal.butter(
        N=bp.order,
        Wn=[bp.low_hz, bp.high_hz],
        btype="bandpass",
        fs=c.eeg.sample_rate_hz,
        output="sos",
    )
    return scipy_signal.sosfiltfilt(sos, x, axis=-1).astype(x.dtype, copy=False)


def notch(x: np.ndarray, cfg: Optional[Config] = None) -> np.ndarray:
    """Single-frequency notch via ``iirnotch`` + ``filtfilt``.

    Targets ``cfg.eeg.powerline_hz`` with quality factor from
    ``cfg.preprocessing.notch.quality_factor``.
    """
    c = _cfg(cfg)
    b, a = scipy_signal.iirnotch(
        w0=c.eeg.powerline_hz,
        Q=c.preprocessing.notch.quality_factor,
        fs=c.eeg.sample_rate_hz,
    )
    return scipy_signal.filtfilt(b, a, x, axis=-1).astype(x.dtype, copy=False)


def baseline_correct(x: np.ndarray) -> np.ndarray:
    """Channel-wise mean subtraction (zero-mean along the time axis)."""
    return x - x.mean(axis=-1, keepdims=True)


def zscore(x: np.ndarray) -> np.ndarray:
    """Channel-wise standardization to (μ=0, σ=1) along the time axis."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + _ZSCORE_EPS)


def temporal_crop(x: np.ndarray, cfg: Optional[Config] = None) -> np.ndarray:
    """Crop the time axis to the VEP window from ``cfg.preprocessing.crop``.

    Raises ``ValueError`` if the configured window exceeds the input
    length — wrong shape is far more useful as a hard failure than a
    silent slice that doesn't behave as documented.
    """
    c = _cfg(cfg)
    start, end = c.preprocessing.crop.start_sample, c.preprocessing.crop.end_sample
    if start < 0 or end > x.shape[-1] or start >= end:
        raise ValueError(
            f"crop window [{start}:{end}] is invalid for input length {x.shape[-1]}",
        )
    return x[..., start:end]
