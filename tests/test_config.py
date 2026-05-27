"""Verify config.yaml loads, validates, and behaves as a frozen Config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from utils.config import Config, load_config


def test_load_config_returns_config_instance() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_project_metadata() -> None:
    cfg = load_config()
    assert cfg.project.name
    assert isinstance(cfg.project.seed, int)


def test_eeg_shapes_match_spec() -> None:
    cfg = load_config()
    assert cfg.eeg.num_channels == 128
    assert cfg.eeg.sample_rate_hz == 1000
    assert cfg.eeg.trial_length_samples == 500
    assert cfg.eeg.powerline_hz in (50, 60)


def test_clip_embedding_dim_is_512() -> None:
    """ViT-B/32 outputs 512-dim. Locked decision in the implementation plan."""
    cfg = load_config()
    assert cfg.models.clip.embed_dim == 512


def test_generation_mode_is_enum() -> None:
    cfg = load_config()
    assert cfg.generation.mode in ("retrieval", "sd_turbo")


def test_inference_backend_is_enum() -> None:
    cfg = load_config()
    assert cfg.inference.backend in ("torch", "openvino")


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.project.name = "mutated"  # type: ignore[misc]


def test_paths_are_non_empty_strings() -> None:
    cfg = load_config()
    for field_name in type(cfg.paths).model_fields:
        value = getattr(cfg.paths, field_name)
        assert isinstance(value, str)
        assert value, f"cfg.paths.{field_name} is empty"


def test_ui_strings_are_non_empty() -> None:
    cfg = load_config()
    for field_name in type(cfg.ui.strings).model_fields:
        value = getattr(cfg.ui.strings, field_name)
        assert isinstance(value, str)
        assert value, f"cfg.ui.strings.{field_name} is empty"


def test_training_weights_sum_to_one() -> None:
    """Loss weights are convex (or close enough that the user-set values are intentional)."""
    cfg = load_config()
    total = cfg.training.loss.info_nce_weight + cfg.training.loss.mse_weight
    assert abs(total - 1.0) < 1e-6, f"loss weights sum to {total}, expected 1.0"


def test_lru_cache_returns_same_instance() -> None:
    """Two calls with the same (None) path return the cached Config — important
    so downstream code can compare identity if it ever needs to."""
    assert load_config() is load_config()
