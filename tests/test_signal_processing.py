"""Unit tests for preprocessing.signal_processing.

Validates each primitive against known-input/known-output cases:

  - bandpass passes mid-band, rejects DC drift and high-freq noise.
  - notch attenuates the configured powerline frequency by >30 dB.
  - baseline_correct zeros per-channel means.
  - zscore produces (μ=0, σ=1) and survives constant channels.
  - temporal_crop returns the right slice and rejects bad bounds.
  - Pipeline composes shape correctly and emits a serializable snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.signal_processing import (
    ICACleaner,
    Pipeline,
    bandpass,
    baseline_correct,
    notch,
    temporal_crop,
    zscore,
)
from utils.config import load_config


def _sine(freq_hz: float, fs: int, n_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """One-channel sine packed as (1, n_samples)."""
    t = np.arange(n_samples) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)[np.newaxis]


def _amplitude(x: np.ndarray) -> float:
    """RMS-based amplitude estimate, ignoring filter edge transients."""
    middle = x[..., x.shape[-1] // 4 : -x.shape[-1] // 4]
    return float(np.sqrt(np.mean(middle ** 2)))


def _attenuation_db(x_in: np.ndarray, x_out: np.ndarray) -> float:
    a_in = _amplitude(x_in)
    a_out = _amplitude(x_out)
    if a_out <= 0:
        return float("inf")
    return 20.0 * np.log10(a_in / a_out)


# -------------------- bandpass --------------------

def test_bandpass_passes_midband() -> None:
    cfg = load_config()
    x = _sine(30.0, cfg.eeg.sample_rate_hz, cfg.eeg.trial_length_samples)
    y = bandpass(x)
    assert _attenuation_db(x, y) < 3.0  # <3 dB loss in band


def test_bandpass_rejects_dc_drift() -> None:
    cfg = load_config()
    x = _sine(0.3, cfg.eeg.sample_rate_hz, cfg.eeg.trial_length_samples)
    y = bandpass(x)
    assert _attenuation_db(x, y) > 15.0


def test_bandpass_rejects_high_frequency() -> None:
    cfg = load_config()
    # 2000 samples gives the SOS filter enough length to reach steady-state;
    # at 500 samples the transient region dominates the middle window.
    x = _sine(200.0, cfg.eeg.sample_rate_hz, 2000)
    y = bandpass(x)
    assert _attenuation_db(x, y) > 12.0


def test_bandpass_preserves_shape_and_dtype() -> None:
    x = np.random.default_rng(0).standard_normal((128, 500)).astype(np.float32)
    y = bandpass(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


# -------------------- notch --------------------

def test_notch_suppresses_powerline_by_30dB() -> None:
    cfg = load_config()
    x = _sine(float(cfg.eeg.powerline_hz), cfg.eeg.sample_rate_hz, 2000)
    y = notch(x)
    assert _attenuation_db(x, y) > 30.0


def test_notch_preserves_nearby_frequencies() -> None:
    cfg = load_config()
    # 40 Hz sits below the powerline (50 Hz) — should pass roughly unaffected.
    x = _sine(40.0, cfg.eeg.sample_rate_hz, 2000)
    y = notch(x)
    assert _attenuation_db(x, y) < 6.0


# -------------------- baseline + zscore --------------------

def test_baseline_correct_zeros_per_channel_mean() -> None:
    x = np.random.default_rng(0).standard_normal((128, 500)).astype(np.float32) + 5.0
    y = baseline_correct(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-5)


def test_zscore_produces_unit_variance_per_channel() -> None:
    x = np.random.default_rng(0).standard_normal((32, 500)).astype(np.float32) * 7.0
    y = zscore(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-3)


def test_zscore_constant_channel_does_not_nan() -> None:
    """Eps floor in the divisor keeps constant channels finite."""
    x = np.zeros((4, 500), dtype=np.float32)
    y = zscore(x)
    assert np.all(np.isfinite(y))


# -------------------- temporal crop --------------------

def test_temporal_crop_correct_length() -> None:
    cfg = load_config()
    x = np.random.default_rng(0).standard_normal((128, cfg.eeg.trial_length_samples)).astype(np.float32)
    y = temporal_crop(x)
    expected_len = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    assert y.shape == (128, expected_len)


def test_temporal_crop_returns_correct_slice() -> None:
    cfg = load_config()
    n = cfg.eeg.trial_length_samples
    x = np.arange(n, dtype=np.float32)[np.newaxis]  # 0..n-1
    y = temporal_crop(x)
    s, e = cfg.preprocessing.crop.start_sample, cfg.preprocessing.crop.end_sample
    assert y[0, 0] == float(s)
    assert y[0, -1] == float(e - 1)


def test_temporal_crop_rejects_out_of_bounds() -> None:
    short = np.zeros((1, 10), dtype=np.float32)  # shorter than the crop window
    with pytest.raises(ValueError, match="crop window"):
        temporal_crop(short)


# -------------------- Pipeline --------------------

def test_pipeline_outputs_cropped_shape() -> None:
    cfg = load_config()
    x = np.random.default_rng(0).standard_normal((128, cfg.eeg.trial_length_samples)).astype(np.float32)
    expected_len = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    y = Pipeline()(x)
    assert y.shape == (128, expected_len)
    assert y.dtype == np.float32


def test_pipeline_to_dict_lists_all_steps() -> None:
    pipe = Pipeline()
    snap = pipe.to_dict()
    assert snap["order"] == list(Pipeline.STEP_ORDER)
    assert snap["bandpass"]["low_hz"] > 0
    assert snap["notch"]["freq_hz"] in (50, 60)
    assert snap["ica"]["enabled"] is False
    assert "bad_components" not in snap["ica"]


def test_pipeline_rejects_use_ica_without_cleaner() -> None:
    cfg = load_config()
    cfg_ica = cfg.model_copy(
        update={
            "preprocessing": cfg.preprocessing.model_copy(update={"use_ica": True}),
        },
    )
    with pytest.raises(ValueError, match="ICACleaner"):
        Pipeline(cfg=cfg_ica)


# -------------------- ICA --------------------

def test_ica_flags_high_kurtosis_component() -> None:
    """Inject a sparse spike pattern; expect ICA to flag at least one bad component."""
    cfg = load_config()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((8, 128, 500)).astype(np.float32)
    # spike in the same temporal location, same channel — easy for ICA to isolate
    batch[:, 0, 250] = 50.0
    # ICA expects bandpass-cleaned data
    from preprocessing.signal_processing import bandpass as _bp
    batch_bp = np.stack([_bp(b) for b in batch])
    cleaner = ICACleaner().fit(batch_bp)
    assert len(cleaner.bad_components) >= 1


def test_ica_transform_preserves_shape() -> None:
    cfg = load_config()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((4, 128, 500)).astype(np.float32) * 0.1
    from preprocessing.signal_processing import bandpass as _bp
    batch_bp = np.stack([_bp(b) for b in batch])
    cleaner = ICACleaner().fit(batch_bp)
    single = cleaner.transform(batch_bp[0])
    multi = cleaner.transform(batch_bp)
    assert single.shape == (128, 500)
    assert multi.shape == batch_bp.shape


def test_ica_transform_before_fit_raises() -> None:
    cleaner = ICACleaner()
    with pytest.raises(RuntimeError, match="before fit"):
        cleaner.transform(np.zeros((128, 500), dtype=np.float32))
