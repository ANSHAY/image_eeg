"""2-D UMAP scatter of CLIP class centroids + a trailing EEG-embedding cursor.

Workflow at app start:

  uplot = UMAPProjection(cfg, image_bank, labels_bank)
    Fits UMAP once on the per-class centroids — fast (~40 points),
    deterministic via cfg.project.seed, doesn't move when new EEG
    points arrive.

Then per trial:

  fig = uplot.figure(history)
    history is a list of ``(z_eeg, true_label, predicted_label)``
    tuples. Returns a fresh Plotly Figure with the centroids drawn
    once and the EEG points overlaid as a fading trail.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from app.components.eeg_plot import _rgba
from utils.config import Config, load_config


class UMAPProjection:
    """Frozen UMAP fit on CLIP class centroids; projects EEG embeddings."""

    def __init__(
        self,
        cfg: Optional[Config],
        image_bank: np.ndarray,
        labels_bank: np.ndarray,
    ) -> None:
        import umap

        self.cfg = cfg if cfg is not None else load_config()
        self.classes = np.unique(labels_bank)
        centroids = np.stack(
            [image_bank[labels_bank == c].mean(axis=0) for c in self.classes],
        )
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self.centroids = centroids / np.maximum(norms, 1e-12)

        n_neighbors = min(15, max(2, len(self.classes) - 1))
        self._reducer = umap.UMAP(
            n_components=2,
            random_state=self.cfg.project.seed,
            n_neighbors=n_neighbors,
        )
        self.centroids_2d = self._reducer.fit_transform(self.centroids)

    def project(self, z_eeg: np.ndarray) -> np.ndarray:
        """Transform new EEG embeddings into the frozen UMAP layout."""
        z = z_eeg if z_eeg.ndim == 2 else z_eeg[np.newaxis]
        return self._reducer.transform(z.astype(np.float32))

    def figure(
        self,
        history: Sequence[tuple[np.ndarray, int, int]],
        height: int = 360,
    ) -> go.Figure:
        """Plotly figure of centroids + EEG trail.

        Args:
            history: most-recent-last list of ``(z_eeg, true_label, pred_label)``.
        """
        c = self.cfg
        fig = go.Figure()

        # Centroids (fixed background layer)
        fig.add_trace(
            go.Scattergl(
                x=self.centroids_2d[:, 0],
                y=self.centroids_2d[:, 1],
                mode="markers+text",
                marker=dict(
                    size=14, color=c.ui.theme.accent,
                    line=dict(width=1.5, color=c.ui.theme.text),
                    symbol="x-thin-open",
                ),
                text=[f"c{int(cl)}" for cl in self.classes],
                textposition="top center",
                textfont=dict(color=c.ui.theme.text, size=9),
                hovertemplate="centroid class=%{text}<extra></extra>",
                name="class centroids",
            ),
        )

        if history:
            projected = self.project(np.stack([z for z, _, _ in history]))
            # Fade older points: alpha grows from 0.2 → 1.0 with recency.
            n = len(history)
            alphas = np.linspace(0.2, 1.0, n)
            colors = [
                _rgba(c.ui.theme.accent, float(a)) for a in alphas
            ]
            true_lbls = [t for _, t, _ in history]
            pred_lbls = [p for _, _, p in history]
            matches = ["✓" if t == p else "✗" for t, p in zip(true_lbls, pred_lbls)]
            fig.add_trace(
                go.Scattergl(
                    x=projected[:, 0],
                    y=projected[:, 1],
                    mode="markers",
                    marker=dict(size=8, color=colors,
                                line=dict(width=0.5, color=c.ui.theme.text)),
                    hovertemplate=(
                        "EEG embed<br>true=%{customdata[0]} "
                        "pred=%{customdata[1]} %{customdata[2]}<extra></extra>"
                    ),
                    customdata=np.column_stack([true_lbls, pred_lbls, matches]),
                    name="EEG trail",
                ),
            )

        fig.update_layout(
            height=height,
            margin=dict(l=30, r=10, t=15, b=30),
            paper_bgcolor=c.ui.theme.bg,
            plot_bgcolor=c.ui.theme.bg,
            font=dict(color=c.ui.theme.text),
            showlegend=False,
            xaxis=dict(title="UMAP-1", gridcolor=_rgba(c.ui.theme.text, 0.13)),
            yaxis=dict(title="UMAP-2", gridcolor=_rgba(c.ui.theme.text, 0.13)),
        )
        return fig
