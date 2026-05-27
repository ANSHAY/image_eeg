"""Headless end-to-end integration smoke for the demo pipeline.

Spawns the LSL streamer in a subprocess, drives the inference loop
(preprocess → encoder → retrieval) directly in the main process
without Streamlit, and asserts the per-trial end-to-end latency is
under the budget from spec § Phase 6 (< 2 s for retrieval mode).

Reuses the synthetic preprocessed dataset + CLIP bank machinery from
scripts/loso_synthetic.py so this works without the real Spampinato
download.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pylsl import StreamInlet, resolve_byprop

from generation.retrieval_generator import RetrievalGenerator
from models.eeg_encoder import EEGEncoder
from preprocessing.signal_processing import Pipeline
from utils.config import load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

NUM_TRIALS_TO_PROCESS = 5
LATENCY_BUDGET_MS = 2000.0  # retrieval-mode budget per spec
RESULTS_SUBDIR = "phase6_e2e"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_synthetic_world(tmp_root: Path) -> Path:
    """Mirror scripts/loso_synthetic.py's layout but smaller."""
    base = load_config()
    processed = tmp_root / "processed"
    results = tmp_root / "results"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(base.project.seed)
    C = base.eeg.num_channels
    T = base.eeg.trial_length_samples
    D = base.models.clip.embed_dim
    num_classes = 5

    anchors = rng.standard_normal((num_classes, D)).astype(np.float32)
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    np.save(processed / "clip_image_emb.npy", anchors)
    np.save(processed / "clip_image_labels.npy", np.arange(num_classes, dtype=np.int64))
    (processed / "clip_image_paths.json").write_text(
        json.dumps([f"img_{i}.JPEG" for i in range(num_classes)]),
        encoding="utf-8",
    )

    n = 20
    eeg = rng.standard_normal((n, C, T)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    targets = anchors[labels]
    subjects = np.zeros(n, dtype=np.int64)
    np.save(processed / "eeg_trials.npy", eeg)
    np.save(processed / "labels.npy", labels)
    np.save(processed / "image_emb_targets.npy", targets)
    np.save(processed / "subject_ids.npy", subjects)

    # Stub a Spampinato dataset file so the streamer's SpampinatoLoader can
    # iterate trials. The streamer reads from cfg.paths.spampinato/*.pth,
    # not the processed dir.
    spamp_dir = tmp_root / "spampinato"
    spamp_dir.mkdir()
    pth_records = [
        {
            "eeg": torch.from_numpy(eeg[i]),
            "label": int(labels[i]),
            "image": int(labels[i]),
            "subject": int(subjects[i]),
        }
        for i in range(n)
    ]
    torch.save(
        {
            "dataset": pth_records,
            "labels": [f"class_{i}" for i in range(num_classes)],
            "images": [f"img_{i}.JPEG" for i in range(num_classes)],
        },
        spamp_dir / "eeg.pth",
    )

    raw = base.model_dump()
    raw["paths"]["data_processed"] = str(processed)
    raw["paths"]["results"] = str(results)
    raw["paths"]["spampinato"] = str(spamp_dir)
    raw["paths"]["imagenet_stimuli"] = str(tmp_root / "stimuli")
    raw["dataset"]["primary"]["expected_subjects"] = 1
    raw["dataset"]["primary"]["expected_classes"] = num_classes
    raw["streaming"]["outlet_name"] = "EEG_E2E"
    raw["streaming"]["source_id"] = "e2e_src"
    raw["streaming"]["marker_outlet_name"] = "EEG_E2E_Markers"
    raw["streaming"]["marker_source_id"] = "e2e_markers"
    (tmp_root / "stimuli").mkdir()

    cfg_path = tmp_root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_path


def _spawn_streamer(cfg_path: Path) -> subprocess.Popen:
    env = {**os.environ, "VCR_CONFIG": str(cfg_path)}
    return subprocess.Popen(
        [
            sys.executable, "-m", "streaming.lsl_streamer",
            "--speed", "20", "--loop",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _process_trial(eeg: np.ndarray, pipeline, encoder, retriever) -> tuple[int, float, float]:
    """Returns (predicted_label, cosine_score, latency_ms)."""
    t0 = time.perf_counter()
    cleaned = pipeline(eeg)
    with torch.no_grad():
        z = encoder(torch.from_numpy(cleaned).unsqueeze(0)).cpu().numpy()[0]
    result = retriever.generate(z)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return int(result["label"]), float(result["score"]), elapsed_ms


def main() -> int:
    setup_logging()
    out_dir = Path(load_config().paths.results) / RESULTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    summary: dict = {}

    with tempfile.TemporaryDirectory(prefix="vcr_e2e_") as td:
        tmp_root = Path(td)
        cfg_path = _build_synthetic_world(tmp_root)
        from utils.config import load_config as _load
        cfg = _load(str(cfg_path))

        encoder = EEGEncoder(cfg=cfg).eval()
        retriever = RetrievalGenerator(cfg=cfg)
        pipeline = Pipeline(cfg=cfg)

        streamer = _spawn_streamer(cfg_path)
        try:
            log.info("resolving LSL streams")
            eeg_streams = []
            mk_streams = []
            deadline = time.time() + cfg.streaming.resolve_timeout_s * 3
            while time.time() < deadline and not (eeg_streams and mk_streams):
                if not eeg_streams:
                    eeg_streams = resolve_byprop("name", cfg.streaming.outlet_name, timeout=0.5)
                if not mk_streams:
                    mk_streams = resolve_byprop("name", cfg.streaming.marker_outlet_name, timeout=0.5)

            if not (eeg_streams and mk_streams):
                stderr_bytes, _ = streamer.communicate(timeout=5)
                log.error("could not resolve LSL streams in time")
                log.error("streamer stderr:\n%s", stderr_bytes.decode(errors="replace"))
                return 2

            eeg_inlet = StreamInlet(eeg_streams[0], recover=False)
            mk_inlet = StreamInlet(mk_streams[0], recover=False)

            processed = 0
            while processed < NUM_TRIALS_TO_PROCESS:
                marker_sample, _ = mk_inlet.pull_sample(timeout=cfg.streaming.resolve_timeout_s)
                if marker_sample is None:
                    continue
                try:
                    marker = json.loads(marker_sample[0])
                except (json.JSONDecodeError, IndexError):
                    continue
                chunk, _ = eeg_inlet.pull_chunk(
                    timeout=cfg.streaming.resolve_timeout_s,
                    max_samples=cfg.eeg.trial_length_samples,
                )
                arr = np.asarray(chunk, dtype=np.float32).T  # (channels, samples)
                if arr.shape != (cfg.eeg.num_channels, cfg.eeg.trial_length_samples):
                    log.warning("skip malformed trial shape=%s", arr.shape)
                    continue
                pred, score, latency_ms = _process_trial(arr, pipeline, encoder, retriever)
                latencies.append(latency_ms)
                processed += 1
                log.info(
                    "trial %d  true=%s  pred=%d  score=%.3f  latency=%.1f ms",
                    processed, marker.get("label"), pred, score, latency_ms,
                )
        finally:
            if streamer.poll() is None:
                streamer.terminate()
                try:
                    streamer.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamer.kill()

    median_ms = float(np.median(latencies)) if latencies else float("inf")
    p95_ms = float(np.percentile(latencies, 95)) if latencies else float("inf")
    summary = {
        "n_trials": len(latencies),
        "latencies_ms": latencies,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "budget_ms": LATENCY_BUDGET_MS,
        "passed": p95_ms < LATENCY_BUDGET_MS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "E2E latency  median %.1f ms  p95 %.1f ms  (budget %.0f ms)  → %s",
        median_ms, p95_ms, LATENCY_BUDGET_MS,
        "PASS" if summary["passed"] else "FAIL",
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
