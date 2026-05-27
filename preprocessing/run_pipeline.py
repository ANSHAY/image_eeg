"""Offline preprocessing driver.

End-to-end batch job that turns the on-disk Spampinato dataset into the
aligned tensors the EEG encoder consumes:

  - ``eeg_trials.npy``         float32 (N, num_channels, T)
  - ``labels.npy``              int64   (N,)
  - ``image_emb_targets.npy``   float32 (N, embed_dim), unit-norm

``T`` is the post-crop length from
``cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample``.
Each row in ``image_emb_targets`` is the CLIP embedding of the stimulus
the subject was viewing on that trial — looked up from the bank produced
by :mod:`preprocessing.clip_embeddings`.

ICA caveat (when ``cfg.preprocessing.use_ica`` is true):
This driver fits one ICACleaner on the full dataset before applying it
per-trial. ICA is unsupervised — it doesn't see labels — but its
decomposition does see all subjects' data, so cross-subject leakage is
non-zero. For strictly LOSO-clean preprocessing, fit per-fold inside
the training loop instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from preprocessing.clip_embeddings import (
    compute_clip_image_bank,
    load_image_bank,
    save_image_bank,
)
from preprocessing.data_loader import SpampinatoLoader
from preprocessing.signal_processing import ICACleaner, Pipeline, bandpass
from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_EEG_OUT = "eeg_trials.npy"
_LABELS_OUT = "labels.npy"
_TARGETS_OUT = "image_emb_targets.npy"
_SUBJECTS_OUT = "subject_ids.npy"
_HPARAMS_OUT = "preprocess_hparams.json"
_IMAGE_BANK_FILENAME = "clip_image_emb.npy"


def _ensure_image_bank(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute the CLIP image bank if absent, else load it."""
    bank_path = Path(cfg.paths.data_processed) / _IMAGE_BANK_FILENAME
    if bank_path.exists():
        log.info("reusing existing image bank at %s", bank_path)
        return load_image_bank(cfg)
    log.info("image bank not found — computing")
    emb, labels, paths = compute_clip_image_bank(cfg)
    save_image_bank(emb, labels, paths, cfg)
    return emb, labels, paths


def _maybe_fit_ica(
    eeg_stack: np.ndarray, cfg: Config,
) -> Optional[ICACleaner]:
    if not cfg.preprocessing.use_ica:
        return None
    log.warning(
        "fitting ICA on the full dataset (%d trials) — some cross-subject "
        "leakage in the decomposition. For strictly LOSO-clean preprocessing, "
        "disable here and fit per-fold inside the training loop.",
        eeg_stack.shape[0],
    )
    # ICA assumes bandpass-cleaned input; apply bandpass per-trial first.
    bp_stack = np.stack([bandpass(t, cfg) for t in eeg_stack])
    return ICACleaner(cfg).fit(bp_stack)


def run(cfg: Optional[Config] = None) -> dict:
    """Materialize the processed tensors. Returns a small summary dict."""
    cfg = cfg if cfg is not None else load_config()

    loader = SpampinatoLoader(cfg=cfg)
    trials = loader.load()
    log.info("loaded %d trials", len(trials))

    emb_bank, _, paths_bank = _ensure_image_bank(cfg)
    path_to_idx = {p: i for i, p in enumerate(paths_bank)}

    raw_stack = np.stack([t.eeg_data for t in trials]).astype(np.float32)
    ica_cleaner = _maybe_fit_ica(raw_stack, cfg)
    pipeline = Pipeline(cfg=cfg, ica_cleaner=ica_cleaner)

    eeg_out, labels_out, targets_out, subjects_out = [], [], [], []
    missing = 0
    for t in tqdm(trials, desc="preprocess"):
        try:
            target_idx = path_to_idx[t.image_path]
        except KeyError:
            missing += 1
            continue
        eeg_out.append(pipeline(t.eeg_data))
        labels_out.append(t.label)
        targets_out.append(emb_bank[target_idx])
        subjects_out.append(t.subject_id)

    if missing:
        log.warning("%d trials had no matching image in the CLIP bank — dropped", missing)

    eeg_arr = np.stack(eeg_out).astype(np.float32)
    labels_arr = np.asarray(labels_out, dtype=np.int64)
    targets_arr = np.stack(targets_out).astype(np.float32)
    subjects_arr = np.asarray(subjects_out, dtype=np.int64)

    out_dir = Path(cfg.paths.data_processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / _EEG_OUT, eeg_arr)
    np.save(out_dir / _LABELS_OUT, labels_arr)
    np.save(out_dir / _TARGETS_OUT, targets_arr)
    np.save(out_dir / _SUBJECTS_OUT, subjects_arr)

    import json
    (out_dir / _HPARAMS_OUT).write_text(
        json.dumps(pipeline.to_dict(), indent=2), encoding="utf-8",
    )

    summary = {
        "eeg_shape": list(eeg_arr.shape),
        "labels_shape": list(labels_arr.shape),
        "targets_shape": list(targets_arr.shape),
        "subjects_shape": list(subjects_arr.shape),
        "dropped_missing_image": missing,
        "output_dir": str(out_dir),
    }
    log.info("preprocess complete: %s", summary)
    return summary


def main() -> int:
    setup_logging()
    try:
        run()
        return 0
    except FileNotFoundError as e:
        log.error("preprocessing cannot run yet: %s", e)
        log.error("Run setup.sh and complete the dataset download first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
