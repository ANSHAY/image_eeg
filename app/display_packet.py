"""Atomic display packet for the Streamlit demo.

Bundles cleaned EEG, retrieval result, ground-truth image path, marker
metadata, embedding vector, and latency into a single immutable object.
The UI pops one packet from the result queue and every field it displays
is guaranteed to belong to the same trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DisplayPacket:
    """One fully-processed trial — everything the UI needs in one place."""

    cleaned_eeg: np.ndarray
    """Shape ``(channels, samples)``, after DSP pipeline."""

    z: np.ndarray
    """Encoder embedding, shape ``(embed_dim,)``."""

    retrieval: Optional[dict]
    """Retrieval result dict (image, label, score, path, bank_index)."""

    marker: dict
    """Raw marker payload (seq, trial_id, label, class_name, image_path, …)."""

    gt_image_path: Optional[str]
    """Absolute path to the exact stimulus image shown during this trial."""

    true_label: int
    """Ground truth class label from marker."""

    pred_label: int
    """Predicted class label from retrieval."""

    cosine: float
    """Cosine similarity score from retrieval."""

    latency_ms: float
    """End-to-end processing latency in milliseconds."""

    correct: bool
    """Whether pred_label matches true_label."""
