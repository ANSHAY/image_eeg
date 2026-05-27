"""Unit tests for evaluation.metrics on hand-built toy data."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    class_centroid_purity,
    cosine_similarity_pairs,
    top_k_retrieval,
)


def _unit_norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# -------------------- top_k_retrieval --------------------

def test_top_k_perfect_recall_when_query_matches_db_row() -> None:
    """If z_eeg ≡ z_img_db rows, top-1 should hit 100%."""
    rng = np.random.default_rng(0)
    z_db = _unit_norm(rng.standard_normal((20, 64)).astype(np.float32))
    labels_db = rng.integers(0, 5, size=20)
    out = top_k_retrieval(z_db, z_db, labels_db, labels_db, ks=(1, 3, 5))
    assert out[1] == 1.0
    assert out[3] == 1.0
    assert out[5] == 1.0


def test_top_k_is_monotonic_non_decreasing() -> None:
    rng = np.random.default_rng(1)
    z_db = _unit_norm(rng.standard_normal((30, 64)).astype(np.float32))
    z_q = _unit_norm(rng.standard_normal((10, 64)).astype(np.float32))
    labels_db = rng.integers(0, 5, size=30)
    true_labels = rng.integers(0, 5, size=10)
    out = top_k_retrieval(z_q, z_db, labels_db, true_labels, ks=(1, 3, 5, 10))
    assert out[1] <= out[3] <= out[5] <= out[10]


def test_top_k_random_queries_near_chance() -> None:
    """5 classes uniformly distributed in the bank ⇒ chance ≈ 0.2 at top-1."""
    rng = np.random.default_rng(2)
    n_db = 500
    n_q = 200
    n_cls = 5
    z_db = _unit_norm(rng.standard_normal((n_db, 64)).astype(np.float32))
    z_q = _unit_norm(rng.standard_normal((n_q, 64)).astype(np.float32))
    labels_db = rng.integers(0, n_cls, size=n_db)
    true_labels = rng.integers(0, n_cls, size=n_q)
    out = top_k_retrieval(z_q, z_db, labels_db, true_labels, ks=(1,))
    # Allow generous slack — finite sample size around 1/n_cls.
    assert abs(out[1] - (1 / n_cls)) < 0.10


def test_top_k_rejects_k_larger_than_db() -> None:
    z_db = np.eye(4, dtype=np.float32)
    labels_db = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="exceeds image-bank size"):
        top_k_retrieval(z_db, z_db, labels_db, labels_db, ks=(10,))


def test_top_k_empty_ks_returns_empty_dict() -> None:
    z_db = np.eye(4, dtype=np.float32)
    labels_db = np.array([0, 1, 2, 3])
    assert top_k_retrieval(z_db, z_db, labels_db, labels_db, ks=()) == {}


# -------------------- cosine_similarity_pairs --------------------

def test_cosine_pairs_identical_is_one() -> None:
    z = _unit_norm(np.random.default_rng(0).standard_normal((10, 8)))
    sims = cosine_similarity_pairs(z, z)
    np.testing.assert_allclose(sims, 1.0, atol=1e-6)


def test_cosine_pairs_orthogonal_is_zero() -> None:
    a = np.zeros((1, 4), dtype=np.float32); a[0, 0] = 1.0
    b = np.zeros((1, 4), dtype=np.float32); b[0, 1] = 1.0
    assert abs(cosine_similarity_pairs(a, b)[0]) < 1e-6


def test_cosine_pairs_rejects_shape_mismatch() -> None:
    a = np.zeros((4, 8))
    b = np.zeros((3, 8))
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity_pairs(a, b)


# -------------------- class_centroid_purity --------------------

def test_centroid_purity_perfect_when_classes_well_separated() -> None:
    """Two classes living on opposite poles of a 2-D unit circle."""
    z_db = np.array([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],   # class 0
        [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0],  # class 1
    ], dtype=np.float32)
    labels_db = np.array([0, 0, 0, 1, 1, 1])
    z_eeg = np.array([
        [1.0, 0.0], [0.9, 0.1],   # near class 0
        [-1.0, 0.0], [-0.9, 0.1], # near class 1
    ], dtype=np.float32)
    z_eeg = z_eeg / np.linalg.norm(z_eeg, axis=1, keepdims=True)
    true_labels = np.array([0, 0, 1, 1])
    purity, predicted = class_centroid_purity(z_eeg, true_labels, z_db, labels_db)
    assert purity == 1.0
    np.testing.assert_array_equal(predicted, true_labels)


def test_centroid_purity_predicted_shape_matches_queries() -> None:
    rng = np.random.default_rng(0)
    z_db = _unit_norm(rng.standard_normal((20, 16)).astype(np.float32))
    labels_db = rng.integers(0, 4, size=20)
    z_q = _unit_norm(rng.standard_normal((7, 16)).astype(np.float32))
    true_labels = rng.integers(0, 4, size=7)
    purity, predicted = class_centroid_purity(z_q, true_labels, z_db, labels_db)
    assert predicted.shape == (7,)
    assert 0.0 <= purity <= 1.0
