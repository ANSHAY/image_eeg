"""LSL receiver — validates the virtual EEG rig end-to-end.

Resolves the EEG and marker streams from :mod:`streaming.lsl_streamer`,
captures one full trial worth of samples, and saves a multi-channel
waveform plot to ``cfg.paths.results``. Intended as a smoke-test for the
Phase 1 streaming pipeline — not part of the real-time demo path.

Run on its own terminal:

  .venv/bin/python -m streaming.lsl_streamer --speed 2.0 &
  .venv/bin/python -m streaming.lsl_receiver
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — write PNG without a display server.
import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_byprop

from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_OUTPUT_FILENAME = "phase1_capture.png"


def _resolve(prop: str, value: str, timeout_s: float):
    log.info("resolving stream by %s=%s (timeout=%.1fs)", prop, value, timeout_s)
    streams = resolve_byprop(prop, value, timeout=timeout_s)
    if not streams:
        raise RuntimeError(f"no LSL stream found with {prop}={value!r} within {timeout_s}s")
    return streams[0]


def _pull_marker(inlet: StreamInlet, timeout_s: float) -> Optional[dict]:
    sample, ts = inlet.pull_sample(timeout=timeout_s)
    if sample is None:
        return None
    try:
        payload = json.loads(sample[0])
    except (json.JSONDecodeError, IndexError) as e:
        log.warning("could not decode marker %r: %s", sample, e)
        return None
    payload["lsl_timestamp"] = ts
    return payload


def _pull_trial(inlet: StreamInlet, n_samples: int, timeout_s: float) -> np.ndarray:
    chunk, _ = inlet.pull_chunk(timeout=timeout_s, max_samples=n_samples)
    return np.asarray(chunk, dtype=np.float32)


def _plot_trial(
    trial: np.ndarray,
    marker: Optional[dict],
    out_path: Path,
    cfg: Config,
) -> None:
    """Stacked 128-channel waveform plot. trial shape: (samples, channels)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples, n_channels = trial.shape
    t_axis = np.arange(n_samples) / cfg.eeg.sample_rate_hz

    # Offset each channel vertically so the stack is readable.
    offsets = np.arange(n_channels) * (np.abs(trial).max() * 2.5 + 1e-3)
    offset_trial = trial + offsets

    fig, ax = plt.subplots(figsize=(12, 14), dpi=120)
    ax.plot(t_axis, offset_trial, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (stacked)")
    title = f"{cfg.ui.strings.panel_eeg} — captured trial"
    if marker is not None:
        title += (
            f"  |  trial_id={marker.get('trial_id')} "
            f"subject={marker.get('subject_id')} "
            f"class={marker.get('class_name')!r}"
        )
    ax.set_title(title)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("saved capture plot to %s", out_path)


def main() -> int:
    setup_logging()
    cfg = load_config()

    try:
        eeg_info = _resolve(
            prop="name",
            value=cfg.streaming.outlet_name,
            timeout_s=cfg.streaming.resolve_timeout_s,
        )
        marker_info = _resolve(
            prop="name",
            value=cfg.streaming.marker_outlet_name,
            timeout_s=cfg.streaming.resolve_timeout_s,
        )
    except RuntimeError as e:
        log.error("%s", e)
        log.error("Is `python -m streaming.lsl_streamer` running on this network?")
        return 1

    eeg_inlet = StreamInlet(eeg_info, recover=False)
    marker_inlet = StreamInlet(marker_info, recover=False)

    log.info("inlets open; waiting for a marker to declare trial start")
    marker = _pull_marker(marker_inlet, timeout_s=cfg.streaming.resolve_timeout_s)
    if marker is None:
        log.warning("no marker received within timeout; capturing anyway")

    log.info("collecting %d samples", cfg.eeg.trial_length_samples)
    trial = _pull_trial(
        eeg_inlet,
        n_samples=cfg.eeg.trial_length_samples,
        timeout_s=cfg.streaming.resolve_timeout_s + 2.0,
    )

    if trial.size == 0:
        log.error("no EEG samples received; stream may have stalled")
        return 1
    if trial.shape[0] < cfg.eeg.trial_length_samples:
        log.warning(
            "underrun: got %d/%d samples",
            trial.shape[0], cfg.eeg.trial_length_samples,
        )
    if trial.shape[1] != cfg.eeg.num_channels:
        log.error(
            "channel-count mismatch: got %d expected %d",
            trial.shape[1], cfg.eeg.num_channels,
        )
        return 1

    out = Path(cfg.paths.results) / _OUTPUT_FILENAME
    _plot_trial(trial, marker, out, cfg)

    log.info(
        "OK — captured %d samples × %d channels; mean=%.3f std=%.3f",
        trial.shape[0], trial.shape[1], float(trial.mean()), float(trial.std()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
