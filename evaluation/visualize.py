"""Evaluation plots — UMAP/t-SNE scatters, confusion matrices, similarity histograms.

Pure functions that build matplotlib Figures. Callers save them; we
don't write to disk in here so the plots compose cleanly into the
Streamlit app's session_state. The exception is :func:`save_figure`,
a small helper that fixes a sensible DPI / tight-layout default.

All plots use :mod:`matplotlib`'s Agg-friendly API — no Tk/GTK
dependency, safe to call from headless training runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 120) -> Path:
    """Save with tight layout and parent-mkdir; returns the resolved path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def umap_scatter(
    z_eeg: np.ndarray,
    true_labels: np.ndarray,
    z_img_bank: np.ndarray,
    labels_bank: np.ndarray,
    seed: int = 42,
    title: str = "EEG embeddings vs CLIP class centroids (UMAP)",
) -> plt.Figure:
    """2-D UMAP of CLIP class centroids overlaid with EEG embeddings.

    UMAP is fit on the centroids only (small, ~40 points → fast) and
    EEG embeddings are projected via ``transform`` so adding new EEG
    points doesn't shift the centroid layout.
    """
    import umap

    classes = np.unique(labels_bank)
    centroids = np.stack(
        [z_img_bank[labels_bank == c].mean(axis=0) for c in classes],
    )
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    n_neighbors = min(15, max(2, len(classes) - 1))
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors)
    centroid_2d = reducer.fit_transform(centroids)
    eeg_2d = reducer.transform(z_eeg)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    label_to_color = {int(c): palette[i] for i, c in enumerate(classes)}

    for c_i, c in enumerate(classes):
        mask = true_labels == c
        if mask.any():
            ax.scatter(
                eeg_2d[mask, 0], eeg_2d[mask, 1],
                s=10, alpha=0.4, color=palette[c_i],
            )
    ax.scatter(
        centroid_2d[:, 0], centroid_2d[:, 1],
        s=200, marker="X", edgecolors="black", linewidths=1.5,
        c=[label_to_color[int(c)] for c in classes],
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return fig


def tsne_scatter(
    z: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    title: str = "EEG embeddings (t-SNE)",
) -> plt.Figure:
    """2-D t-SNE of a single embedding set — useful for sanity check
    before UMAP if downstream wants a fast verification view."""
    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(5.0, (len(z) - 1) / 3.0))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    z_2d = tsne.fit_transform(z)

    fig, ax = plt.subplots(figsize=(10, 8))
    classes = np.unique(labels)
    palette = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    for c_i, c in enumerate(classes):
        mask = labels == c
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=12, alpha=0.7, color=palette[c_i])
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    return fig


def confusion_matrix_plot(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Confusion matrix",
) -> plt.Figure:
    """Confusion matrix heatmap. Rows = true, cols = predicted, row-normalized."""
    classes = np.unique(np.concatenate([true_labels, predicted_labels]))
    idx = {int(c): i for i, c in enumerate(classes)}
    n = len(classes)
    mat = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true_labels, predicted_labels):
        mat[idx[int(t)], idx[int(p)]] += 1
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    normed = mat / row_sums

    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), max(5, n * 0.3)))
    im = ax.imshow(normed, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if class_names is not None and len(class_names) >= n:
        labels = [class_names[int(c)] for c in classes]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def cosine_sim_histogram(
    sims: np.ndarray,
    bins: int = 50,
    title: str = "Cosine similarity (EEG ↔ matching image)",
) -> plt.Figure:
    """Histogram of per-pair cosine similarities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=bins, color="#3b82f6", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(float(np.mean(sims)), linestyle="--", color="black", linewidth=1.0,
               label=f"mean={float(np.mean(sims)):.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return fig
