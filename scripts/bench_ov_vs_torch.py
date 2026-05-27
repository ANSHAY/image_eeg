"""Benchmark torch vs OpenVINO inference throughput for EEGEncoder.

Writes ``results/phase5_bench.json`` with per-backend median/p95
per-trial latency and the OV-over-torch speedup. The plan's gate is
2–5 × on this Intel Core Ultra hardware; this script is the evidence.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from models.eeg_encoder import EEGEncoder
from models.openvino_export import (
    OpenVINOEncoder,
    compile_openvino,
    export_to_onnx,
)
from utils.config import load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_NUM_TRIALS = 1000
_WARMUP = 50
_OUTPUT_FILENAME = "phase5_bench.json"


def _time_run(fn, x: torch.Tensor, n: int) -> list[float]:
    """Returns per-call wall-clock times in ms."""
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fn(x)
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _summarize(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "n": len(s),
        "mean_ms": float(statistics.fmean(s)),
        "median_ms": float(s[len(s) // 2]),
        "p95_ms": float(s[int(len(s) * 0.95)]),
        "min_ms": float(s[0]),
        "max_ms": float(s[-1]),
    }


def main() -> int:
    setup_logging()
    cfg = load_config()
    log.info(
        "benchmarking torch vs OpenVINO on %d trials (+%d warmup) "
        "with batch=1 single-trial inference",
        _NUM_TRIALS, _WARMUP,
    )

    torch.manual_seed(cfg.project.seed)
    torch.set_num_threads((__import__("os").cpu_count() or 1))

    model = EEGEncoder(cfg=cfg).eval()
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    x = torch.randn(1, cfg.eeg.num_channels, T)

    # Bench: torch
    torch_fn = lambda inp: model(inp).detach()  # noqa: E731
    _time_run(torch_fn, x, _WARMUP)
    torch_samples = _time_run(torch_fn, x, _NUM_TRIALS)

    # Build OV path
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "encoder.onnx"
        export_to_onnx(model, onnx_path, cfg=cfg)
        compiled = compile_openvino(onnx_path, cfg=cfg)
    ov_encoder = OpenVINOEncoder(compiled, cfg=cfg)

    # Bench: OV
    _time_run(ov_encoder, x, _WARMUP)
    ov_samples = _time_run(ov_encoder, x, _NUM_TRIALS)

    torch_stats = _summarize(torch_samples)
    ov_stats = _summarize(ov_samples)
    speedup = torch_stats["median_ms"] / max(ov_stats["median_ms"], 1e-6)

    summary = {
        "num_trials": _NUM_TRIALS,
        "warmup": _WARMUP,
        "input_shape": list(x.shape),
        "openvino_device": cfg.inference.openvino_device,
        "torch": torch_stats,
        "openvino": ov_stats,
        "speedup_median": speedup,
        "gate_target_min": 2.0,
        "gate_target_max": 5.0,
        "gate_met": speedup >= 2.0,
    }

    out_dir = Path(cfg.paths.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / _OUTPUT_FILENAME
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info(
        "torch:    median %.2f ms  p95 %.2f ms", torch_stats["median_ms"], torch_stats["p95_ms"],
    )
    log.info(
        "openvino: median %.2f ms  p95 %.2f ms", ov_stats["median_ms"], ov_stats["p95_ms"],
    )
    log.info("speedup (torch median / openvino median): %.2fx", speedup)
    log.info("results saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
