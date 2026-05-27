"""CLIP-nearest-neighbor retrieval generator (Phase 4A baseline).

Reads the precomputed CLIP image bank produced by
:mod:`preprocessing.clip_embeddings`, builds a FAISS ``IndexFlatIP``
over the embeddings (inner-product on unit-norm rows ≡ cosine), and
serves nearest-image retrieval at inference.

This path is fast (<100 ms per query on the bank size in spec, CPU
only) and is the always-works fallback for the demo. The generative
SD-Turbo path is built separately in :mod:`generation.sd_generator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from PIL import Image, ImageDraw

from preprocessing.clip_embeddings import load_image_bank
from utils.config import Config, load_config
from utils.logging import get_logger

log = get_logger(__name__)

_PLACEHOLDER_SIZE = 224  # ViT-B/32's image_size; replaced by config below when constructed.
_INDEX_FLAT_IP = "IndexFlatIP"


class RetrievalGenerator:
    """Nearest-neighbor retriever over the precomputed CLIP image bank."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.embeddings, self.labels, self.paths = load_image_bank(self.cfg)
        index_type = self.cfg.generation.retrieval.index_type
        if index_type != _INDEX_FLAT_IP:
            raise ValueError(
                f"unsupported FAISS index type: {index_type!r} "
                f"(only {_INDEX_FLAT_IP} is implemented)",
            )
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
        log.info(
            "retrieval index built: %d entries × %d-dim (%s)",
            self.embeddings.shape[0], self.embeddings.shape[1], index_type,
        )

    # --- query API ----------------------------------------------------

    def generate(self, z_eeg: np.ndarray) -> dict:
        """Top-1 retrieval. Returns ``{image, label, score, path}``."""
        results = self.generate_topk(z_eeg, k=1)
        return results[0]

    def generate_topk(
        self,
        z_eeg: np.ndarray,
        k: Optional[int] = None,
    ) -> list[dict]:
        """Top-K retrieval as a list of result dicts sorted by descending score.

        Args:
            z_eeg: a single EEG-derived embedding of shape ``(D,)`` or
                ``(1, D)``. The vector should already be L2-normalized.
            k:     defaults to ``cfg.generation.retrieval.top_k``.
        """
        z = self._as_query(z_eeg)
        top_k = k if k is not None else self.cfg.generation.retrieval.top_k
        if top_k <= 0:
            raise ValueError(f"k must be positive, got {top_k}")
        if top_k > self.embeddings.shape[0]:
            log.warning(
                "k=%d exceeds bank size %d — clamping",
                top_k, self.embeddings.shape[0],
            )
            top_k = self.embeddings.shape[0]
        scores, indices = self.index.search(z, top_k)
        scores, indices = scores[0], indices[0]
        results: list[dict] = []
        for score, idx in zip(scores, indices):
            path = self.paths[int(idx)]
            results.append(
                {
                    "image": self._load_image(path),
                    "label": int(self.labels[int(idx)]),
                    "score": float(score),
                    "path": path,
                    "bank_index": int(idx),
                },
            )
        return results

    # --- helpers ------------------------------------------------------

    def _as_query(self, z_eeg: np.ndarray) -> np.ndarray:
        if z_eeg.ndim == 1:
            z = z_eeg[np.newaxis]
        elif z_eeg.ndim == 2 and z_eeg.shape[0] == 1:
            z = z_eeg
        else:
            raise ValueError(
                f"z_eeg must be 1-D or (1, D), got shape {z_eeg.shape}",
            )
        if z.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"z_eeg dim {z.shape[1]} != bank dim {self.embeddings.shape[1]}",
            )
        return z.astype(np.float32, copy=False)

    def _load_image(self, path: str) -> Image.Image:
        p = Path(path)
        if p.is_file():
            return Image.open(p).convert("RGB")
        # Stimulus image not on disk — emit a labeled placeholder so the
        # demo and tests can run before the user has fetched the
        # ImageNet stimulus subset.
        return self._placeholder(p.name)

    def _placeholder(self, label_text: str) -> Image.Image:
        size = self.cfg.models.clip.image_size or _PLACEHOLDER_SIZE
        img = Image.new("RGB", (size, size), color=(20, 25, 45))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(2, 2), (size - 2, size - 2)], outline=(0, 212, 255), width=2)
        draw.text((10, size // 2 - 10), f"missing\n{label_text}", fill=(230, 240, 255))
        return img
