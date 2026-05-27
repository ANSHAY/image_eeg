"""ONNX-export + OpenVINO-compile parity tests for EEGEncoder.

Verifies the Phase 5 inference path produces bit-equivalent outputs to
the reference torch model (within ``atol=1e-3`` per the plan's gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.eeg_encoder import EEGEncoder
from models.openvino_export import (
    OpenVINOEncoder,
    compile_openvino,
    export_to_onnx,
)
from utils.config import load_config


@pytest.fixture(scope="module")
def trained_encoder() -> EEGEncoder:
    """An EEGEncoder with a fixed seed — deterministic init replaces
    'trained weights' for parity testing (the parity property doesn't
    depend on the model being trained)."""
    torch.manual_seed(0)
    model = EEGEncoder()
    model.eval()
    return model


@pytest.fixture(scope="module")
def ov_pair(tmp_path_factory, trained_encoder):
    """Build the matching OpenVINOEncoder once per test module."""
    tmp = tmp_path_factory.mktemp("ov")
    onnx_path = tmp / "encoder.onnx"
    export_to_onnx(trained_encoder, onnx_path)
    compiled = compile_openvino(onnx_path)
    return trained_encoder, OpenVINOEncoder(compiled), onnx_path


def test_onnx_file_is_written(ov_pair) -> None:
    _, _, onnx_path = ov_pair
    assert Path(onnx_path).is_file()
    assert Path(onnx_path).stat().st_size > 0


def test_bit_exactness_single_trial(ov_pair) -> None:
    cfg = load_config()
    torch_model, ov_model, _ = ov_pair
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(1, cfg.eeg.num_channels, T)
    with torch.no_grad():
        y_torch = torch_model(x)
    y_ov = ov_model(x)
    assert y_torch.shape == y_ov.shape
    torch.testing.assert_close(y_torch, y_ov, atol=1e-3, rtol=1e-3)


def test_bit_exactness_over_100_trials(ov_pair) -> None:
    """Phase 5 acceptance: bit-exact within atol=1e-3 on 100 trials."""
    cfg = load_config()
    torch_model, ov_model, _ = ov_pair
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    rng = np.random.default_rng(0)

    diffs: list[float] = []
    for _ in range(100):
        x = torch.from_numpy(
            rng.standard_normal((1, cfg.eeg.num_channels, T)).astype(np.float32),
        )
        with torch.no_grad():
            y_torch = torch_model(x).numpy()
        y_ov = ov_model(x).numpy()
        diffs.append(float(np.max(np.abs(y_torch - y_ov))))
    max_diff = max(diffs)
    assert max_diff < 1e-3, f"max abs diff over 100 trials = {max_diff:.2e}"


def test_dynamic_batch_axis_accepts_different_sizes(ov_pair) -> None:
    cfg = load_config()
    torch_model, ov_model, _ = ov_pair
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    for B in (1, 4, 16):
        x = torch.randn(B, cfg.eeg.num_channels, T)
        with torch.no_grad():
            y_torch = torch_model(x)
        y_ov = ov_model(x)
        assert y_ov.shape == y_torch.shape == (B, cfg.models.clip.embed_dim)


def test_output_is_unit_norm_after_ov_pipeline(ov_pair) -> None:
    """The L2-normalize layer must survive the export → compile chain."""
    cfg = load_config()
    _, ov_model, _ = ov_pair
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(8, cfg.eeg.num_channels, T)
    y = ov_model(x)
    torch.testing.assert_close(
        y.norm(dim=-1), torch.ones(8), atol=1e-4, rtol=1e-4,
    )
