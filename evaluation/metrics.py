"""Retrieval and alignment metrics for EEG ↔ CLIP embeddings.

All inputs are assumed unit-norm. Cosine similarity collapses to a dot
product; we never re-normalize inside these metrics because the
upstream contract (encoder ends in F.normalize, CLIP bank is built
unit-norm) is enforced elsewhere — silently re-normalizing here would
mask encoder bugs in evaluation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def top_k_retrieval(
    z_eeg: np.ndarray,
    z_img_db: np.ndarray,
    labels_db: np.ndarray,
    true_labels: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[int, float]:
    """Class-level top-K retrieval accuracy.

    For each EEG embedding, retrieve the K image-bank entries with the
    highest cosine similarity and check whether the query's true class
    appears among the K retrieved labels.

    Args:
        z_eeg:       (n_query, embed_dim) — EEG-derived embeddings.
        z_img_db:    (n_db, embed_dim)    — image bank.
        labels_db:   (n_db,)              — class labels in the bank.
        true_labels: (n_query,)           — ground-truth class per query.
        ks:          iterable of K values to evaluate.

    Returns:
        ``{k: accuracy}``. Accuracy is the fraction of queries whose
        true label is in the top-K retrieved labels.
    """
    ks_list = sorted(set(int(k) for k in ks))
    if not ks_list:
        return {}
    max_k = ks_list[-1]
    if max_k > z_img_db.shape[0]:
        raise ValueError(
            f"max k={max_k} exceeds image-bank size {z_img_db.shape[0]}",
        )

    sims = z_eeg @ z_img_db.T  # (n_query, n_db); unit-norm ⇒ cosine
    # argpartition is O(n) for the top-K cut; argsort the slice for stable order.
    top_idx_unsorted = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
    top_sims = np.take_along_axis(sims, top_idx_unsorted, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)  # (n_query, max_k)
    top_labels = labels_db[top_idx]

    out: dict[int, float] = {}
    for k in ks_list:
        hits = (top_labels[:, :k] == true_labels[:, None]).any(axis=1)
        out[k] = float(hits.mean())
    return out


def cosine_similarity_pairs(
    z_eeg: np.ndarray,
    z_img: np.ndarray,
) -> np.ndarray:
    """Per-row cosine similarity between aligned EEG/image pairs.

    Both arrays must have shape ``(N, embed_dim)`` and be row-paired —
    the i-th EEG embedding aligned to the i-th image target.

    Returns:
        ``(N,)`` array of cosine similarities. Useful as a histogram
        of matched-pair alignment.
    """
    if z_eeg.shape != z_img.shape:
        raise ValueError(f"shape mismatch: {z_eeg.shape} vs {z_img.shape}")
    return (z_eeg * z_img).sum(axis=1)


def class_centroid_purity(
    z_eeg: np.ndarray,
    true_labels: np.ndarray,
    z_img_bank: np.ndarray,
    labels_bank: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fraction of EEG embeddings closest to the centroid of their true class.

    Args:
        z_eeg:       (n_query, embed_dim) — EEG embeddings.
        true_labels: (n_query,)
        z_img_bank:  (n_db, embed_dim)
        labels_bank: (n_db,)

    Returns:
        ``(purity, predicted_labels)`` where ``predicted_labels`` is
        the nearest-centroid prediction per query — useful for building
        confusion matrices downstream.
    """
    classes = np.unique(labels_bank)
    centroids = np.stack(
        [z_img_bank[labels_bank == c].mean(axis=0) for c in classes],
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    sims = z_eeg @ centroids.T  # (n_query, n_classes)
    predicted = classes[sims.argmax(axis=1)]
    purity = float((predicted == true_labels).mean())
    return purity, predicted
