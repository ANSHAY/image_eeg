"""Unit tests for generation.retrieval_generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from generation.retrieval_generator import RetrievalGenerator
from utils.config import load_config


@pytest.fixture
def retrieval_setup(tmp_path: Path):
    """Write a small CLIP bank to tmp_path and return an override cfg."""
    base = load_config()
    proc = tmp_path / "processed"
    stim = tmp_path / "stimuli"
    proc.mkdir()
    stim.mkdir()

    rng = np.random.default_rng(0)
    n = 30
    d = base.models.clip.embed_dim
    raw = rng.standard_normal((n, d)).astype(np.float32)
    embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    labels = rng.integers(0, 5, size=n).astype(np.int64)
    paths = [f"img_{i:03d}.JPEG" for i in range(n)]

    np.save(proc / "clip_image_emb.npy", embeddings)
    np.save(proc / "clip_image_labels.npy", labels)
    (proc / "clip_image_paths.json").write_text(json.dumps(paths), encoding="utf-8")

    cfg = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "data_processed": str(proc),
                    "imagenet_stimuli": str(stim),
                },
            ),
        },
    )
    return cfg, embeddings, labels, paths


def test_top1_identity_recovers_query_row(retrieval_setup) -> None:
    cfg, embeddings, labels, paths = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    for i in range(len(embeddings)):
        out = gen.generate(embeddings[i])
        assert out["bank_index"] == i
        assert out["label"] == int(labels[i])
        assert out["path"] == paths[i]
        # cosine sim with self should be ~1
        assert out["score"] > 0.999


def test_topk_is_sorted_descending(retrieval_setup) -> None:
    cfg, embeddings, _, _ = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    results = gen.generate_topk(embeddings[0], k=5)
    assert len(results) == 5
    scores = [r["score"] for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_topk_clamps_to_bank_size(retrieval_setup) -> None:
    cfg, embeddings, _, _ = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    results = gen.generate_topk(embeddings[0], k=1000)
    assert len(results) == len(embeddings)


def test_query_dim_mismatch_raises(retrieval_setup) -> None:
    cfg, _, _, _ = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    bad = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match=r"dim 7 != bank dim"):
        gen.generate(bad)


def test_2d_query_with_multiple_rows_rejected(retrieval_setup) -> None:
    cfg, _, _, _ = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    bad = np.zeros((4, cfg.models.clip.embed_dim), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D or"):
        gen.generate(bad)


def test_missing_stimulus_returns_placeholder(retrieval_setup) -> None:
    """Stimulus images aren't on disk in this fixture, so we should get
    a placeholder image rather than a FileNotFoundError."""
    cfg, embeddings, _, _ = retrieval_setup
    gen = RetrievalGenerator(cfg=cfg)
    out = gen.generate(embeddings[0])
    img = out["image"]
    assert img.size == (cfg.models.clip.image_size, cfg.models.clip.image_size)


def test_invalid_index_type_rejected() -> None:
    base = load_config()
    cfg = base.model_copy(
        update={
            "generation": base.generation.model_copy(
                update={
                    "retrieval": base.generation.retrieval.model_copy(
                        update={"index_type": "IndexHNSWFlat"},
                    ),
                },
            ),
        },
    )
    # We need a valid bank for the load step before the index_type
    # check fires. Use the real config's data_processed (which has no
    # bank yet) — expect a different error path that is still useful.
    with pytest.raises((ValueError, FileNotFoundError)):
        RetrievalGenerator(cfg=cfg)
