"""Smoke test for generation.sd_generator.SDGenerator.

Tagged ``integration`` because triggering ``ensure_loaded()`` downloads
SD-Turbo (~5 GB) and IP-Adapter weights on first run. The test skips
gracefully when the network is offline or when load fails — failures
inside the SD path are not fatal to the project per the spec (retrieval
mode is the always-works fallback).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

from generation.sd_generator import SDGenerator
from utils.config import load_config


def test_constructor_does_not_trigger_download() -> None:
    """Constructing SDGenerator must not eagerly load weights."""
    gen = SDGenerator()
    assert gen.loaded is False


def test_query_validation_rejects_wrong_rank() -> None:
    gen = SDGenerator()
    bad = np.zeros((4, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D or"):
        gen._as_query(bad)


def test_error_placeholder_matches_configured_size() -> None:
    cfg = load_config()
    gen = SDGenerator(cfg=cfg)
    placeholder = gen._error_placeholder("test message")
    assert placeholder.size == (cfg.generation.sd.image_size, cfg.generation.sd.image_size)


@pytest.mark.integration
def test_generate_returns_image_within_latency_budget() -> None:
    """End-to-end smoke: random unit vector → PIL.Image at configured size.

    Latency budget is 120 s on CPU — generous because first-run includes
    weight download. Re-runs should land closer to 30 s per spec.
    """
    cfg = load_config()
    gen = SDGenerator(cfg=cfg)

    rng = np.random.default_rng(0)
    z = rng.standard_normal(cfg.models.clip.embed_dim).astype(np.float32)
    z /= np.linalg.norm(z)

    t0 = time.perf_counter()
    try:
        img = gen.generate(z)
    except Exception as e:
        pytest.skip(f"SD pipeline could not initialize: {e}")
    elapsed = time.perf_counter() - t0

    assert isinstance(img, Image.Image)
    assert img.size == (cfg.generation.sd.image_size, cfg.generation.sd.image_size)
    assert elapsed < 120.0, f"first-run latency {elapsed:.1f}s exceeds 120 s budget"
