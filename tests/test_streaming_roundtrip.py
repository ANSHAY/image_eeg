"""End-to-end smoke test for the Phase 1 streaming pipeline.

Spawns the lsl_streamer as a subprocess, opens an inlet in this
process, and verifies that one trial round-trips with the right
shape, channel count, and marker payload.

Tagged ``integration`` so it can be deselected in fast loops. Skips
gracefully if pylsl can't resolve the local outlets within the
timeout — LSL multicast can be blocked in sandboxed runners.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NUM_CHANNELS = 128
NUM_SAMPLES = 500
NUM_SUBJECTS = 2
NUM_CLASSES = 3
TRIALS_PER_SUBJECT = 2


@pytest.fixture
def tmp_dataset_and_config(tmp_path: Path) -> tuple[Path, dict]:
    """Build a tiny synthetic dataset and a tmp config.yaml that points at it.

    Stream names are suffixed with a uuid so concurrent test runs
    don't collide on the LSL bus.
    """
    spampinato_dir = tmp_path / "spampinato"
    stimuli_dir = tmp_path / "stimuli"
    spampinato_dir.mkdir()
    stimuli_dir.mkdir()
    (tmp_path / "results").mkdir()

    rng = np.random.default_rng(7)
    records = []
    for sid in range(NUM_SUBJECTS):
        for k in range(TRIALS_PER_SUBJECT):
            records.append(
                {
                    "eeg": torch.from_numpy(
                        rng.standard_normal((NUM_CHANNELS, NUM_SAMPLES)).astype(np.float32),
                    ),
                    "label": (sid + k) % NUM_CLASSES,
                    "image": (sid + k) % NUM_CLASSES,
                    "subject": sid,
                },
            )
    payload = {
        "dataset": records,
        "labels": [f"class_{i}" for i in range(NUM_CLASSES)],
        "images": [f"img_{i}.JPEG" for i in range(NUM_CLASSES)],
    }
    torch.save(payload, spampinato_dir / "eeg.pth")

    base = load_config()
    raw = base.model_dump()
    suffix = uuid.uuid4().hex[:8]
    raw["paths"]["spampinato"] = str(spampinato_dir)
    raw["paths"]["imagenet_stimuli"] = str(stimuli_dir)
    raw["paths"]["results"] = str(tmp_path / "results")
    raw["dataset"]["primary"]["expected_subjects"] = NUM_SUBJECTS
    raw["dataset"]["primary"]["expected_classes"] = NUM_CLASSES
    raw["streaming"]["outlet_name"] = f"EEG_TestRig_{suffix}"
    raw["streaming"]["source_id"] = f"test_rig_{suffix}"
    raw["streaming"]["marker_outlet_name"] = f"EEG_TestRig_Markers_{suffix}"
    raw["streaming"]["marker_source_id"] = f"test_rig_markers_{suffix}"

    tmp_yaml = tmp_path / "config.yaml"
    tmp_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_yaml, raw


@pytest.mark.integration
def test_subprocess_streamer_round_trips_one_trial(
    tmp_dataset_and_config: tuple[Path, dict],
) -> None:
    pylsl = pytest.importorskip("pylsl")
    tmp_yaml, raw_cfg = tmp_dataset_and_config

    env = {**os.environ, "VCR_CONFIG": str(tmp_yaml)}
    # --loop keeps the outlet alive long enough for the inlet to resolve;
    # we kill the subprocess in the finally block. speed=50 makes each
    # trial take ~10 ms so loop overhead is negligible.
    streamer = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streaming.lsl_streamer",
            "--speed", "50",
            "--loop",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        eeg_name = raw_cfg["streaming"]["outlet_name"]
        marker_name = raw_cfg["streaming"]["marker_outlet_name"]
        resolve_timeout = float(raw_cfg["streaming"]["resolve_timeout_s"])

        # Give the subprocess time to start, import, and open outlets.
        deadline = time.time() + resolve_timeout * 2
        eeg_streams = []
        marker_streams = []
        while time.time() < deadline and (not eeg_streams or not marker_streams):
            if not eeg_streams:
                eeg_streams = pylsl.resolve_byprop("name", eeg_name, timeout=0.5)
            if not marker_streams:
                marker_streams = pylsl.resolve_byprop("name", marker_name, timeout=0.5)

        if not eeg_streams or not marker_streams:
            stdout, stderr = streamer.communicate(timeout=5)
            pytest.skip(
                f"LSL resolve failed in this environment "
                f"(eeg={bool(eeg_streams)}, marker={bool(marker_streams)})\n"
                f"streamer stderr:\n{stderr.decode(errors='replace')}",
            )

        eeg_inlet = pylsl.StreamInlet(eeg_streams[0], recover=False)
        marker_inlet = pylsl.StreamInlet(marker_streams[0], recover=False)

        marker_sample, _ = marker_inlet.pull_sample(timeout=resolve_timeout)
        assert marker_sample is not None, "no marker received"
        marker_payload = json.loads(marker_sample[0])
        assert "trial_id" in marker_payload
        assert "subject_id" in marker_payload
        assert "label" in marker_payload
        assert "image_path" in marker_payload
        assert marker_payload["label"] in range(NUM_CLASSES)
        assert marker_payload["subject_id"] in range(NUM_SUBJECTS)

        chunk, _ = eeg_inlet.pull_chunk(
            timeout=resolve_timeout + 2.0,
            max_samples=NUM_SAMPLES,
        )
        arr = np.asarray(chunk, dtype=np.float32)
        # Allow a small underrun — multicast can drop one or two samples.
        assert arr.shape[0] >= NUM_SAMPLES - 5, (
            f"received {arr.shape[0]}/{NUM_SAMPLES} samples"
        )
        assert arr.shape[1] == NUM_CHANNELS, f"channels: got {arr.shape[1]}"
    finally:
        if streamer.poll() is None:
            streamer.terminate()
            try:
                streamer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamer.kill()
