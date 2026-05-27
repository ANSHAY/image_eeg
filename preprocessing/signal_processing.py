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
from scipy.stats import kurtosis

from utils.config import Config, load_config
from utils.logging import get_logger

_ZSCORE_EPS = 1e-8

_log = get_logger(__name__)


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


class ICACleaner:
    """ICA-based artifact removal — toggled by ``cfg.preprocessing.use_ica``.

    Fit once on a representative batch of trials (training subjects),
    then apply to single trials at inference time. Bad-component
    detection is heuristic: components whose source time-course
    kurtosis exceeds ``cfg.preprocessing.ica_kurtosis_threshold`` are
    excluded — high-kurtosis sources typically correspond to EOG (eye
    blinks) or EMG (muscle) bursts that the bandpass cannot reach.

    Lazy-imports ``mne`` to keep the dependency cost off the hot path
    when ICA is disabled.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = _cfg(cfg)
        self._ica: object | None = None
        self._info: object | None = None
        self._fitted: bool = False
        self.bad_components: list[int] = []

    def _make_info(self, n_channels: int):
        from mne import create_info

        return create_info(
            ch_names=[f"ch_{i}" for i in range(n_channels)],
            sfreq=float(self.cfg.eeg.sample_rate_hz),
            ch_types="eeg",
            verbose=False,
        )

    def fit(self, batch: np.ndarray) -> "ICACleaner":
        """Fit ICA on a (n_trials, n_channels, n_samples) batch.

        Returns self for chaining.
        """
        if batch.ndim != 3:
            raise ValueError(
                f"fit expects (trials, channels, samples), got shape {batch.shape}",
            )
        from mne import EpochsArray
        from mne.preprocessing import ICA

        self._info = self._make_info(batch.shape[1])
        epochs = EpochsArray(batch.astype(np.float64), self._info, verbose=False)
        self._ica = ICA(
            n_components=self.cfg.preprocessing.ica_components,
            random_state=self.cfg.project.seed,
            method="fastica",
            max_iter="auto",
            verbose=False,
        )
        self._ica.fit(epochs, verbose=False)
        self._fitted = True
        self.bad_components = self._find_bad_components(epochs)
        self._ica.exclude = list(self.bad_components)
        _log.info(
            "ICA fit on %d trials × %d ch; %d/%d components excluded "
            "(kurtosis > %.1f)",
            batch.shape[0], batch.shape[1],
            len(self.bad_components), self.cfg.preprocessing.ica_components,
            self.cfg.preprocessing.ica_kurtosis_threshold,
        )
        return self

    def _find_bad_components(self, epochs) -> list[int]:
        sources = self._ica.get_sources(epochs).get_data()
        # sources shape: (n_epochs, n_components, n_samples)
        # average |kurtosis| across epochs per component
        per_epoch = kurtosis(sources, axis=-1, fisher=True)
        mean_abs_kurt = np.abs(per_epoch).mean(axis=0)
        threshold = self.cfg.preprocessing.ica_kurtosis_threshold
        return [int(i) for i, k in enumerate(mean_abs_kurt) if k > threshold]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply trained ICA, removing excluded components. Accepts a
        single trial (channels, samples) or a batch (trials, ch, samples)."""
        if not self._fitted or self._ica is None:
            raise RuntimeError("ICACleaner.transform called before fit")
        from mne import EpochsArray

        single = x.ndim == 2
        if single:
            x = x[np.newaxis]
        elif x.ndim != 3:
            raise ValueError(
                f"transform expects (channels, samples) or (trials, ch, samples), got {x.shape}",
            )
        epochs = EpochsArray(x.astype(np.float64), self._info, verbose=False)
        cleaned = self._ica.apply(epochs.copy(), verbose=False).get_data()
        if single:
            cleaned = cleaned[0]
        return cleaned.astype(x.dtype if x.dtype == np.float32 else np.float32, copy=False)

    def fit_transform(self, batch: np.ndarray) -> np.ndarray:
        return self.fit(batch).transform(batch)
