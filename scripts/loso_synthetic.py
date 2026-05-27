"""Synthetic LOSO run — exercises the full Phase 3 training stack.

The real Spampinato dataset is not yet on disk, so the spec's `top-5
≥ 30 % on held-out subject` quality gate cannot be evaluated here.
What this script proves end-to-end:

  - Preprocess artifact format round-trips (run_pipeline outputs are
    readable by EEGDataset and the CLIP bank loader).
  - LOSO split indices cover and partition the dataset.
  - train.run_loso() iterates fold-by-fold, writes checkpoints,
    snapshots hparams, emits TensorBoard logs, and produces
    loso_summary.json.
  - Across-fold mean/std reduction in `_summarize` works.

Synthetic data: 3 subjects × 100 trials × 5 classes, EEG = class-keyed
sine patterns + small noise, CLIP targets = per-class unit-vector
anchors. The encoder *can* recover this if the loop is correct — but
absolute numbers are not predictive of real-data performance.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from models.train import run_loso
from utils.config import load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

NUM_SUBJECTS = 3
TRIALS_PER_SUBJECT = 100
NUM_CLASSES = 5
EPOCHS = 5  # quick smoke — not a quality run


def _build_synthetic_disk_layout(tmp_root: Path) -> Path:
    """Materialize preprocessed + CLIP-bank .npy + a tmp config.yaml."""
    base = load_config()
    processed = tmp_root / "processed"
    results = tmp_root / "results"
    runs = tmp_root / "runs"
    for p in (processed, results, runs):
        p.mkdir(parents=True, exist_ok=True)

    C = base.eeg.num_channels
    T = base.preprocessing.crop.end_sample - base.preprocessing.crop.start_sample
    D = base.models.clip.embed_dim
    rng = np.random.default_rng(base.project.seed)

    # Per-class CLIP-space anchors (image bank rows = exactly NUM_CLASSES rows).
    anchors = rng.standard_normal((NUM_CLASSES, D)).astype(np.float32)
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    np.save(processed / "clip_image_emb.npy", anchors)
    np.save(processed / "clip_image_labels.npy", np.arange(NUM_CLASSES, dtype=np.int64))
    (processed / "clip_image_paths.json").write_text(
        json.dumps([f"img_{i}.JPEG" for i in range(NUM_CLASSES)]),
        encoding="utf-8",
    )

    # Synthetic EEG: per-class frequency + per-subject phase shift.
    N = NUM_SUBJECTS * TRIALS_PER_SUBJECT
    eeg = np.zeros((N, C, T), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    targets = np.zeros((N, D), dtype=np.float32)
    subjects = np.zeros(N, dtype=np.int64)

    t_axis = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    i = 0
    for sid in range(NUM_SUBJECTS):
        subject_phase = rng.uniform(0, 2 * np.pi, size=C).astype(np.float32)
        for _ in range(TRIALS_PER_SUBJECT):
            lbl = int(rng.integers(0, NUM_CLASSES))
            freq = 1.0 + lbl * 0.5
            signal = np.sin(freq * t_axis[None, :] + subject_phase[:, None]).astype(np.float32)
            noise = 0.05 * rng.standard_normal((C, T)).astype(np.float32)
            eeg[i] = signal + noise
            labels[i] = lbl
            targets[i] = anchors[lbl]
            subjects[i] = sid
            i += 1
    np.save(processed / "eeg_trials.npy", eeg)
    np.save(processed / "labels.npy", labels)
    np.save(processed / "image_emb_targets.npy", targets)
    np.save(processed / "subject_ids.npy", subjects)

    raw = base.model_dump()
    raw["paths"]["data_processed"] = str(processed)
    raw["paths"]["results"] = str(results)
    raw["paths"]["runs"] = str(runs)
    raw["dataset"]["primary"]["expected_subjects"] = NUM_SUBJECTS
    raw["dataset"]["primary"]["expected_classes"] = NUM_CLASSES
    raw["training"]["epochs"] = EPOCHS
    raw["training"]["batch_size"] = 32
    raw["training"]["log_every_n_steps"] = 5

    tmp_cfg = tmp_root / "config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_cfg


def main() -> int:
    setup_logging()
    out_dir = Path(load_config().paths.results) / "loso_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vcr_loso_") as td:
        tmp_root = Path(td)
        tmp_cfg = _build_synthetic_disk_layout(tmp_root)

        # Bypass lru_cache on load_config — pass tmp_cfg explicitly.
        from utils.config import load_config as _load
        cfg = _load(str(tmp_cfg))

        summary = run_loso(
            cfg,
            held_out=None,
            max_epochs=EPOCHS,
            final_retrain=False,
        )

        # Pick up the produced run dir (newest under runs/)
        runs_dir = Path(cfg.paths.runs)
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        latest = run_dirs[-1]
        shutil.copytree(latest, out_dir / latest.name, dirs_exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "synthetic LOSO complete:  top1 %.3f ± %.3f  top5 %.3f ± %.3f  top10 %.3f ± %.3f",
        summary["top1_mean"], summary["top1_std"],
        summary["top5_mean"], summary["top5_std"],
        summary["top10_mean"], summary["top10_std"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
