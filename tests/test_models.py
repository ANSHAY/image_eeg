"""Unit tests for the EEG encoder and contrastive alignment loss."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from models.eeg_encoder import EEGEncoder
from models.losses import ContrastiveAlignmentLoss
from utils.config import load_config


# ----------------------- encoder ----------------------------

def test_encoder_forward_shape() -> None:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg)
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(4, cfg.eeg.num_channels, T)
    y = model(x)
    assert y.shape == (4, cfg.models.clip.embed_dim)


def test_encoder_output_is_unit_norm() -> None:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg).eval()
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(8, cfg.eeg.num_channels, T)
    with torch.no_grad():
        y = model(x)
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(8), atol=1e-5, rtol=1e-5)


def test_encoder_gradients_reach_all_parameters() -> None:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg)
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(2, cfg.eeg.num_channels, T)
    y = model(x)
    y.sum().backward()
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"parameters without gradient: {no_grad}"


def test_encoder_handles_different_batch_sizes() -> None:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg).eval()
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    for B in (1, 4, 16):
        with torch.no_grad():
            y = model(torch.randn(B, cfg.eeg.num_channels, T))
        assert y.shape == (B, cfg.models.clip.embed_dim)


def test_encoder_param_budget_under_5M() -> None:
    """Spec budget is ~3-5 M for CPU training feasibility."""
    cfg = load_config()
    n = EEGEncoder(cfg=cfg).num_parameters()
    assert n < 5_000_000, f"encoder has {n:,} parameters; budget exceeded"


# ----------------------- losses -----------------------------

def _random_pair(B: int = 8, D: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    a = F.normalize(torch.randn(B, D), dim=-1)
    b = F.normalize(torch.randn(B, D), dim=-1)
    return a, b


def test_initial_temperature_matches_clip_convention() -> None:
    cfg = load_config()
    loss_fn = ContrastiveAlignmentLoss(cfg=cfg)
    expected = 1.0 / cfg.training.loss.temperature_init
    assert math.isclose(loss_fn.temperature().item(), expected, rel_tol=1e-5)


def test_temperature_is_clamped_above() -> None:
    cfg = load_config()
    loss_fn = ContrastiveAlignmentLoss(cfg=cfg)
    with torch.no_grad():
        loss_fn.log_temperature.fill_(1000.0)  # well above the clamp
    expected_max = cfg.training.loss.temperature_max
    # float32 precision: exp(log(100)) ≈ 100.000008 — allow modest slack.
    assert loss_fn.temperature().item() <= expected_max + 1e-3


def test_identical_pairs_yield_zero_loss() -> None:
    loss_fn = ContrastiveAlignmentLoss()
    a, _ = _random_pair()
    total, comp = loss_fn(a, a)
    assert float(comp["mse"]) < 1e-6
    # InfoNCE with identical (positive on diagonal) and random off-diagonal:
    # large positive logit on diag, similar logits off-diag (random pairs).
    # Loss won't be exactly zero unless temperature → ∞, but it will be small.
    assert float(comp["info_nce"]) < math.log(a.size(0))


def test_info_nce_is_symmetric() -> None:
    loss_fn = ContrastiveAlignmentLoss()
    a, b = _random_pair()
    forward = loss_fn.info_nce(a, b)
    swapped = loss_fn.info_nce(b, a)
    torch.testing.assert_close(forward, swapped, atol=1e-5, rtol=1e-5)


def test_info_nce_rejects_shape_mismatch() -> None:
    loss_fn = ContrastiveAlignmentLoss()
    a = F.normalize(torch.randn(8, 512), dim=-1)
    b = F.normalize(torch.randn(4, 512), dim=-1)
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn.info_nce(a, b)


def test_gradient_flows_through_temperature() -> None:
    loss_fn = ContrastiveAlignmentLoss()
    a, b = _random_pair()
    total, _ = loss_fn(a, b)
    total.backward()
    assert loss_fn.log_temperature.grad is not None
    assert torch.isfinite(loss_fn.log_temperature.grad).all()


def test_total_loss_is_convex_combination_of_components() -> None:
    cfg = load_config()
    loss_fn = ContrastiveAlignmentLoss(cfg=cfg)
    a, b = _random_pair()
    total, comp = loss_fn(a, b)
    expected = (
        cfg.training.loss.info_nce_weight * comp["info_nce"]
        + cfg.training.loss.mse_weight * comp["mse"]
    )
    torch.testing.assert_close(total.detach(), expected, atol=1e-6, rtol=1e-6)
