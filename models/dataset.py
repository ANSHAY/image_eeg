"""PyTorch Dataset over the preprocessed Spampinato tensors.

Reads the four aligned arrays produced by
:mod:`preprocessing.run_pipeline`:

  eeg_trials.npy        float32 (N, num_channels, T)
  labels.npy            int64   (N,)
  image_emb_targets.npy float32 (N, embed_dim), unit-norm
  subject_ids.npy       int64   (N,)

A subset can be selected via ``indices=`` for LOSO splits without
re-loading the .npy files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.config import Config, load_config

_EEG_FILE = "eeg_trials.npy"
_LABELS_FILE = "labels.npy"
_TARGETS_FILE = "image_emb_targets.npy"
_SUBJECTS_FILE = "subject_ids.npy"


class EEGDataset(Dataset):
    """Memory-mapped EEG dataset for the encoder.

    Augmentations (when provided) are applied to ``x`` only on
    ``__getitem__`` — never to ``z_img`` (the target stays fixed) and
    never to validation/test samples (pass ``train_aug=None``).
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        indices: Optional[np.ndarray] = None,
        train_aug: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        c = cfg if cfg is not None else load_config()
        self.cfg = c
        proc = Path(c.paths.data_processed)
        # mmap_mode='r' so multiple folds can share the file without OOM
        self.eeg = np.load(proc / _EEG_FILE, mmap_mode="r")
        self.labels = np.load(proc / _LABELS_FILE)
        self.targets = np.load(proc / _TARGETS_FILE)
        self.subjects = np.load(proc / _SUBJECTS_FILE)
        n = self.eeg.shape[0]
        if not (n == len(self.labels) == len(self.targets) == len(self.subjects)):
            raise ValueError(
                f"preprocessed arrays misaligned: eeg={n} labels={len(self.labels)} "
                f"targets={len(self.targets)} subjects={len(self.subjects)}",
            )
        self.indices = np.arange(n) if indices is None else np.asarray(indices, dtype=np.int64)
        self.train_aug = train_aug

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = int(self.indices[idx])
        x = torch.from_numpy(np.array(self.eeg[i], dtype=np.float32))  # copy out of mmap
        if self.train_aug is not None:
            x = self.train_aug(x)
        return {
            "eeg": x,
            "label": int(self.labels[i]),
            "target": torch.from_numpy(self.targets[i].astype(np.float32)),
            "subject_id": int(self.subjects[i]),
        }


def loso_split_indices(
    subjects: np.ndarray, held_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices for a leave-one-subject-out fold.

    Returns:
        (train_idx, val_idx) — disjoint, together cover every trial.
    """
    val_idx = np.where(subjects == held_out)[0]
    train_idx = np.where(subjects != held_out)[0]
    if len(val_idx) == 0:
        raise ValueError(f"no trials for held-out subject {held_out}")
    return train_idx, val_idx


def shuffled_split_indices(
    total_samples: int, val_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Indices for a globally shuffled train/val split.
    
    Returns:
        (train_idx, val_idx)
    """
    indices = np.arange(total_samples)
    np.random.default_rng(seed).shuffle(indices)
    split = int(total_samples * (1.0 - val_ratio))
    return indices[:split], indices[split:]
