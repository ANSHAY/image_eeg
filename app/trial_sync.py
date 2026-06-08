"""Timestamp-aligned trial synchronizer for the Streamlit demo.

Solves the marker↔EEG desync problem: marker and EEG streams travel
on separate LSL outlets with independent buffers.  Naively pulling
the "next N samples" after a marker breaks under any timing jitter,
startup flush, or processing backpressure.

This module uses **LSL timestamps** to deterministically pair each
marker with exactly its corresponding EEG samples, guaranteeing that
the ground-truth image and the reconstruction always refer to the
same trial.

Architecture:
    Single background thread → non-blocking polls on both inlets →
    internal timestamped ring buffer → emit atomically-paired
    ``SyncedTrial`` objects to the output queue.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.config import Config

log = logging.getLogger(__name__)

# Tolerance for timestamp alignment (seconds).  Samples arriving up to
# this many seconds before the marker are considered part of the trial
# (accounts for sub-millisecond LSL push ordering jitter).
_TS_EPSILON_S = 0.005

# Maximum EEG samples to buffer internally (prevents unbounded memory
# growth if the marker stream stalls).
_EEG_BUFFER_MAX = 16_384

# Poll cadence when neither stream has data (seconds).
_IDLE_POLL_S = 0.005

# Timeout waiting for enough EEG samples after a marker (seconds).
_TRIAL_COLLECT_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class SyncedTrial:
    """An atomically paired (marker, EEG) trial — guaranteed synchronized."""

    marker: dict
    """JSON-decoded marker payload (trial_id, label, class_name, image_path, …)."""

    eeg: np.ndarray
    """Shape ``(channels, samples)``, float32."""

    seq: int
    """Monotonically increasing trial counter (for debugging)."""

    marker_ts: float
    """LSL timestamp of the marker event."""

    pairing_latency_ms: float
    """Wall-clock time from marker receipt to paired emission (ms)."""


class TrialSynchronizer:
    """Background thread that pairs LSL markers with their exact EEG data.

    Args:
        cfg:          Project config.
        output_queue: Bounded queue; when full the **oldest** trial is
                      dropped so the UI always shows the freshest data.
        stop_event:   Signal to terminate the background thread.
        queue_maxsize: Maximum items in *output_queue* (used only when
                      *output_queue* is ``None`` and we create our own).
    """

    def __init__(
        self,
        cfg: Config,
        output_queue: Optional[queue.Queue] = None,
        stop_event: Optional[threading.Event] = None,
        queue_maxsize: int = 32,
    ) -> None:
        self._cfg = cfg
        self._stop = stop_event or threading.Event()
        self._queue: queue.Queue[SyncedTrial] = (
            output_queue if output_queue is not None
            else queue.Queue(maxsize=queue_maxsize)
        )
        self._thread: Optional[threading.Thread] = None

        # Internal timestamped EEG ring buffer: deque of (lsl_ts, sample_vec)
        # where sample_vec is a list/array of length num_channels.
        self._eeg_buf: deque[tuple[float, list[float]]] = deque(
            maxlen=_EEG_BUFFER_MAX,
        )
        self._seq = 0

    # ------------------------------------------------------------------ public

    @property
    def queue(self) -> queue.Queue[SyncedTrial]:
        """The output queue containing synchronized trials."""
        return self._queue

    def start(self) -> threading.Thread:
        """Launch the background synchronizer thread (daemon)."""
        self._thread = threading.Thread(
            target=self._run,
            name="TrialSync",
            daemon=True,
        )
        self._thread.start()
        log.info("TrialSynchronizer started")
        return self._thread

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------ thread

    def _run(self) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError:
            log.error("pylsl not available — synchronizer cannot start")
            return

        cfg = self._cfg

        # Resolve both streams ------------------------------------------------
        eeg_streams = resolve_byprop(
            "name", cfg.streaming.outlet_name,
            timeout=cfg.streaming.resolve_timeout_s,
        )
        mk_streams = resolve_byprop(
            "name", cfg.streaming.marker_outlet_name,
            timeout=cfg.streaming.resolve_timeout_s,
        )
        if not eeg_streams or not mk_streams:
            log.error("LSL streams not found — synchronizer exiting")
            return

        eeg_inlet = StreamInlet(eeg_streams[0], recover=False)
        mk_inlet = StreamInlet(mk_streams[0], recover=False)

        # Let inlets settle, then flush stale data ----------------------------
        time.sleep(0.3)
        eeg_inlet.pull_chunk(timeout=0.0, max_samples=81_920)
        mk_inlet.pull_chunk(timeout=0.0)
        self._eeg_buf.clear()

        log.info("TrialSynchronizer: inlets connected, buffers flushed")

        # Main loop -----------------------------------------------------------
        required = cfg.eeg.trial_length_samples

        while not self._stop.is_set():
            # 1. Non-blocking drain of any available EEG samples into buffer.
            self._drain_eeg(eeg_inlet)

            # 2. Non-blocking check for a marker.
            marker_sample, marker_ts = mk_inlet.pull_sample(timeout=0.01)
            if marker_sample is None:
                continue

            # 3. Parse marker JSON.
            try:
                marker = json.loads(marker_sample[0])
            except (json.JSONDecodeError, IndexError):
                log.warning("malformed marker — skipping")
                continue

            # 4. Extract aligned trial using timestamps.
            t_pair_start = time.perf_counter()
            trial_eeg = self._extract_aligned(
                eeg_inlet, marker_ts, required,
            )
            if trial_eeg is None:
                log.warning(
                    "failed to collect %d aligned samples for marker "
                    "seq=%d (timeout) — dropping trial",
                    required, self._seq,
                )
                continue

            pairing_ms = (time.perf_counter() - t_pair_start) * 1e3

            # 5. Build the synced trial.
            synced = SyncedTrial(
                marker=marker,
                eeg=trial_eeg,
                seq=self._seq,
                marker_ts=marker_ts,
                pairing_latency_ms=pairing_ms,
            )
            self._seq += 1

            # 6. Enqueue (drop oldest if full).
            self._enqueue(synced)

    # ------------------------------------------------------------------ helpers

    def _drain_eeg(self, inlet) -> None:
        """Pull all available EEG samples into the internal timestamped buffer."""
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=4096)
        if chunk:
            for sample, ts in zip(chunk, timestamps):
                self._eeg_buf.append((ts, sample))

    def _extract_aligned(
        self,
        eeg_inlet,
        marker_ts: float,
        required: int,
    ) -> Optional[np.ndarray]:
        """Return (channels, samples) EEG aligned to *marker_ts*, or None.

        Algorithm:
            1. Discard buffered samples whose timestamp < marker_ts - ε
               (stale data from a prior trial).
            2. Collect from the buffer (and pull more from the inlet if
               needed) until we have *required* samples with ts ≥ threshold.
            3. Return the first *required* samples as (channels, samples).
        """
        threshold = marker_ts - _TS_EPSILON_S
        deadline = time.perf_counter() + _TRIAL_COLLECT_TIMEOUT_S
        aligned: list[list[float]] = []

        while len(aligned) < required and not self._stop.is_set():
            # Discard stale samples.
            while self._eeg_buf and self._eeg_buf[0][0] < threshold:
                self._eeg_buf.popleft()

            # Harvest aligned samples from the buffer.
            while self._eeg_buf and len(aligned) < required:
                ts, sample = self._eeg_buf[0]
                if ts < threshold:
                    self._eeg_buf.popleft()
                    continue
                self._eeg_buf.popleft()
                aligned.append(sample)

            if len(aligned) >= required:
                break

            # Not enough yet — pull more from the inlet.
            if time.perf_counter() > deadline:
                return None
            self._drain_eeg(eeg_inlet)
            if not self._eeg_buf:
                time.sleep(_IDLE_POLL_S)

        if len(aligned) < required:
            return None

        # (samples, channels) → (channels, samples)
        arr = np.asarray(aligned[:required], dtype=np.float32)
        return arr.T

    def _enqueue(self, trial: SyncedTrial) -> None:
        """Put *trial* on the output queue, dropping the oldest if full."""
        while True:
            try:
                self._queue.put_nowait(trial)
                return
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                    log.debug(
                        "output queue full — dropped oldest trial seq=%d",
                        dropped.seq,
                    )
                except queue.Empty:
                    pass
