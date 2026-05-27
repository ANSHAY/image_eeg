"""Virtual EEG rig — stream Spampinato trials over LSL in real time.

Creates two LSL outlets per ``config.yaml``:

  - EEG outlet: 128 ch × 1000 Hz float32, one sample per push.
  - Marker outlet: irregular-rate string channel, emits a JSON-encoded
    payload at every trial boundary carrying ``trial_id``, ``label``,
    ``subject_id``, ``class_name``, and ``image_path``. Downstream
    receivers align ground-truth labels and stimulus images against
    the continuous EEG stream via these markers.

CLI flags:

  --speed N      Playback multiplier (default from config). 1.0 = real time.
                 Higher values shorten the per-sample sleep proportionally;
                 0 or negative skips the sleep entirely (max throughput).
  --subject N    Restrict to a single subject id (LOSO-style replay).
  --loop         Restart from trial 0 after the last trial.
  --max-trials N Cap total trials emitted (debugging).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from pylsl import StreamInfo, StreamOutlet

from preprocessing.data_loader import SpampinatoLoader, Trial
from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_MARKER_RATE_IRREGULAR = 0.0
_MARKER_CHANNELS = 1
_MARKER_CHANNEL_FORMAT = "string"


def _build_eeg_outlet(cfg: Config) -> StreamOutlet:
    info = StreamInfo(
        name=cfg.streaming.outlet_name,
        type=cfg.streaming.outlet_type,
        channel_count=cfg.eeg.num_channels,
        nominal_srate=float(cfg.eeg.sample_rate_hz),
        channel_format=cfg.streaming.channel_format,
        source_id=cfg.streaming.source_id,
    )
    return StreamOutlet(info, chunk_size=cfg.streaming.chunk_size)


def _build_marker_outlet(cfg: Config) -> StreamOutlet:
    info = StreamInfo(
        name=cfg.streaming.marker_outlet_name,
        type=cfg.streaming.marker_outlet_type,
        channel_count=_MARKER_CHANNELS,
        nominal_srate=_MARKER_RATE_IRREGULAR,
        channel_format=_MARKER_CHANNEL_FORMAT,
        source_id=cfg.streaming.marker_source_id,
    )
    return StreamOutlet(info)


def _encode_marker(trial: Trial) -> str:
    return json.dumps(
        {
            "trial_id": trial.trial_id,
            "subject_id": trial.subject_id,
            "label": trial.label,
            "class_name": trial.class_name,
            "image_path": trial.image_path,
        },
    )


def _stream_trial(
    trial: Trial,
    eeg_outlet: StreamOutlet,
    marker_outlet: StreamOutlet,
    sample_period_s: float,
) -> None:
    marker_outlet.push_sample([_encode_marker(trial)])
    eeg = trial.eeg_data  # shape: (channels, samples)
    next_deadline = time.perf_counter() + sample_period_s
    for t_idx in range(eeg.shape[1]):
        eeg_outlet.push_sample(eeg[:, t_idx].tolist())
        if sample_period_s > 0:
            # perf_counter-based pacing — robust against time.sleep drift.
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            next_deadline += sample_period_s


def _parse_args(default_speed: float) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual EEG rig: stream Spampinato trials over LSL.")
    p.add_argument("--speed", type=float, default=default_speed,
                   help="Playback rate multiplier; 1.0 = real time. Default from config.")
    p.add_argument("--subject", type=int, default=None,
                   help="Restrict to a single subject id.")
    p.add_argument("--loop", action="store_true",
                   help="Loop forever after exhausting trials.")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Cap total trials emitted (debugging).")
    return p.parse_args()


def main() -> int:
    setup_logging()
    cfg = load_config()
    args = _parse_args(default_speed=cfg.streaming.default_speed)

    loader = SpampinatoLoader(cfg=cfg)
    try:
        trials = loader.load()
    except FileNotFoundError as e:
        log.error("dataset not available: %s", e)
        log.error("Run setup.sh or python -m data.download_dataset first.")
        return 1

    if args.subject is not None:
        trials = [t for t in trials if t.subject_id == args.subject]
        if not trials:
            log.error("no trials for subject_id=%d", args.subject)
            return 1
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    eeg_outlet = _build_eeg_outlet(cfg)
    marker_outlet = _build_marker_outlet(cfg)

    sample_period_s = (
        1.0 / cfg.eeg.sample_rate_hz / args.speed if args.speed > 0 else 0.0
    )
    log.info(
        "streaming %d trials at %.2fx (period=%.4f ms/sample)",
        len(trials), args.speed, sample_period_s * 1e3,
    )

    try:
        sent = 0
        while True:
            for trial in trials:
                _stream_trial(trial, eeg_outlet, marker_outlet, sample_period_s)
                sent += 1
                if sent % 10 == 0:
                    log.info("sent %d trials", sent)
            if not args.loop:
                break
    except KeyboardInterrupt:
        log.info("interrupted; %d trials sent", sent)

    return 0


if __name__ == "__main__":
    sys.exit(main())
