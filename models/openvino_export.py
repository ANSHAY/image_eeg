"""Export EEGEncoder to ONNX, compile to OpenVINO IR, and wrap for inference.

Phase 5 of the plan — post-training optimization. The trained PyTorch
encoder is the reference; OpenVINO is the fast path for the demo's
real-time inference on the Intel Core Ultra NPU (with iGPU and CPU
fallback via the ``AUTO:NPU,GPU,CPU`` device string).

Public API:

  export_to_onnx(model, onnx_path, cfg)
      torch.onnx.export with a dynamic batch axis at the configured
      opset version. Bit-exact to torch within fp32 ULPs (verified by
      the bit-exactness test in box 52).

  compile_openvino(onnx_path, cfg) -> ov.CompiledModel
      ov.Core().compile_model on the device string from
      cfg.inference.openvino_device. Falls back through the AUTO
      hierarchy automatically when the NPU driver is absent.

  OpenVINOEncoder(cfg, compiled_model)
      Drop-in replacement for EEGEncoder.forward at inference time —
      same ``__call__(x) -> torch.Tensor`` contract so the app and
      tests don't need to branch by backend. Selected by
      cfg.inference.backend = "openvino".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.eeg_encoder import EEGEncoder
from utils.config import Config, load_config
from utils.logging import get_logger

log = get_logger(__name__)


def export_to_onnx(
    model: EEGEncoder,
    onnx_path: Path,
    cfg: Optional[Config] = None,
) -> Path:
    """Write the trained encoder to ONNX with a dynamic batch dim.

    Args:
        model: a trained (or initialized) EEGEncoder; switched to eval()
            inside this function — caller doesn't need to.
        onnx_path: destination path. Created/overwritten.
        cfg: optional config override.
    """
    c = cfg if cfg is not None else load_config()
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    T = c.preprocessing.crop.end_sample - c.preprocessing.crop.start_sample
    dummy = torch.zeros(1, c.eeg.num_channels, T, dtype=torch.float32)

    log.info(
        "exporting EEGEncoder → %s (input shape %s, opset %d)",
        onnx_path, tuple(dummy.shape), c.inference.onnx_opset,
    )
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["eeg"],
        output_names=["embedding"],
        opset_version=c.inference.onnx_opset,
        dynamic_axes={"eeg": {0: "batch"}, "embedding": {0: "batch"}},
        do_constant_folding=True,
    )
    return onnx_path


def compile_openvino(onnx_path: Path, cfg: Optional[Config] = None):
    """Compile an ONNX file with OpenVINO on cfg.inference.openvino_device."""
    import openvino as ov

    c = cfg if cfg is not None else load_config()
    core = ov.Core()
    log.info(
        "available OpenVINO devices: %s; requesting %s",
        core.available_devices, c.inference.openvino_device,
    )
    compiled = core.compile_model(
        model=str(onnx_path),
        device_name=c.inference.openvino_device,
    )
    log.info("compiled OK on device family: %s", compiled.get_property("EXECUTION_DEVICES"))
    return compiled


class OpenVINOEncoder:
    """Inference-time wrapper that matches EEGEncoder's __call__ contract."""

    def __init__(self, compiled_model, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self._compiled = compiled_model
        # Use the first input / output port; ONNX export has exactly one of each.
        self._input_port = self._compiled.inputs[0]
        self._output_port = self._compiled.outputs[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self._compiled([arr])[self._output_port]
        return torch.from_numpy(np.asarray(out, dtype=np.float32))

    @classmethod
    def from_torch(
        cls,
        model: EEGEncoder,
        onnx_path: Path,
        cfg: Optional[Config] = None,
    ) -> "OpenVINOEncoder":
        """Convenience: export the torch model and immediately compile it.

        Used by Phase 6's app + the bench script when no pre-exported IR
        is on disk.
        """
        export_to_onnx(model, onnx_path, cfg=cfg)
        compiled = compile_openvino(onnx_path, cfg=cfg)
        return cls(compiled, cfg=cfg)
