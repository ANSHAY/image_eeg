"""Generative image synthesis via SD-Turbo + IP-Adapter.

The Phase 4B "wow" path. Where retrieval pulls an existing image from
the bank, this path conditions a single-step SD-Turbo forward pass on
the EEG-derived CLIP-space embedding via IP-Adapter — so we get a
synthesized image rather than a database hit.

Memory discipline (spec §Critical Note 4):
The SD pipeline is loaded **lazily** on the first :meth:`generate`
call. The 32 GB-RAM laptop can't comfortably hold CLIP + SD-Turbo
simultaneously, so the calling code (or the orchestrator) should
:func:`gc.collect` away any CLIP model references before this loader
fires. :meth:`unload` drops the pipeline cleanly when the caller wants
to switch back to retrieval mode.

IP-Adapter dim compatibility:
Stock ``ip-adapter_sd15.bin`` (h94/IP-Adapter) was trained against a
CLIP-vision model with a 1024-dim image embedding. Our EEG encoder
emits ``cfg.models.clip.embed_dim`` (512 for ViT-B/32). The pipeline
will refuse to consume a mismatching embedding shape. When that
happens, :meth:`generate` logs a clear ERROR and returns a placeholder
so the demo flow doesn't crash — the user can either swap the CLIP
backbone, train a small projection head, or leave generation off.
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.config import Config, load_config
from utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_PROMPT = ""  # IP-Adapter carries all conditioning
_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

class MappingPrior(torch.nn.Module):
    """Translates 512-dim OpenAI CLIP vectors to 1024-dim LAION CLIP vectors."""
    def __init__(self, in_dim=512, out_dim=1024, hidden_dim=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class SDGenerator:
    """Lazy-loading SD-Turbo + IP-Adapter image generator."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self._pipeline: object | None = None
        self._loaded: bool = False
        self._expected_embed_dim: Optional[int] = None
        self._prior: Optional[MappingPrior] = None

    # --- lifecycle ----------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Idempotent: instantiate the pipeline + IP-Adapter on first call."""
        if self._loaded:
            return
        # Give the GC a chance to free anything (notably CLIP) the caller
        # has already dereferenced. The 32 GB ceiling can't hold both.
        gc.collect()

        from diffusers import StableDiffusionPipeline

        sd = self.cfg.generation.sd
        dtype = _DTYPE_MAP.get(sd.torch_dtype, torch.float32)
        log.info(
            "loading SD-Turbo: %s (dtype=%s)", sd.hf_id, sd.torch_dtype,
        )
        kwargs = {"torch_dtype": dtype}
        if sd.revision:
            kwargs["revision"] = sd.revision
        self._pipeline = StableDiffusionPipeline.from_pretrained(sd.hf_id, **kwargs)

        log.info(
            "loading IP-Adapter: %s/%s/%s",
            sd.ip_adapter_hf_id, sd.ip_adapter_subfolder, sd.ip_adapter_weight_name,
        )
        self._pipeline.load_ip_adapter(
            sd.ip_adapter_hf_id,
            subfolder=sd.ip_adapter_subfolder,
            weight_name=sd.ip_adapter_weight_name,
        )
        
        # Load the Mapping Prior if we are using 512-dim EEG features
        if self.cfg.models.clip.embed_dim == 512:
            import os
            prior_path = "weights/mapping_prior.pt"
            if os.path.exists(prior_path):
                log.info("loading Mapping Prior to bridge 512-dim -> 1024-dim")
                self._prior = MappingPrior()
                self._prior.load_state_dict(torch.load(prior_path, map_location="cpu"))
                self._prior.eval()
                self._prior.to(self._pipeline.device, dtype=dtype)
            else:
                log.warning("Mapping Prior not found at %s! 512-dim generation will fail.", prior_path)

        self._pipeline.set_progress_bar_config(disable=True)
        self._loaded = True

    def unload(self) -> None:
        """Free the pipeline + adapter. Lets the caller swap back to retrieval."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._prior is not None:
            del self._prior
            self._prior = None
        self._loaded = False
        gc.collect()
        log.info("SD pipeline unloaded")

    # --- generation API -----------------------------------------------

    def generate(self, z_eeg: np.ndarray) -> Image.Image:
        """Generate one image conditioned on a single EEG embedding.

        Args:
            z_eeg: shape ``(D,)`` or ``(1, D)``, unit-norm.

        Returns:
            PIL.Image of size ``cfg.generation.sd.image_size``.
        """
        z = self._as_query(z_eeg)
        try:
            self.ensure_loaded()
        except Exception as e:
            log.error("SD pipeline failed to load: %s", e)
            return self._error_placeholder(f"load error: {e}")

        sd = self.cfg.generation.sd
        try:
            tensor = torch.from_numpy(z).to(_DTYPE_MAP.get(sd.torch_dtype, torch.float32)).to(self._pipeline.device)
            
            # If using 512-dim, pipe it through our Mapping Prior to get the 1024-dim IP-Adapter vector
            if self.cfg.models.clip.embed_dim == 512 and self._prior is not None:
                with torch.no_grad():
                    tensor = self._prior(tensor)
            
            result = self._pipeline(
                prompt=_DEFAULT_PROMPT,
                num_inference_steps=sd.num_inference_steps,
                guidance_scale=sd.guidance_scale,
                height=sd.image_size,
                width=sd.image_size,
                ip_adapter_image_embeds=[tensor.unsqueeze(0)],
            )
            return result.images[0]
        except Exception as e:
            log.error("SD generation failed: %s", e)
            return self._error_placeholder(str(e)[:60])

    # --- helpers ------------------------------------------------------

    def _as_query(self, z_eeg: np.ndarray) -> np.ndarray:
        z = z_eeg
        if z.ndim == 1:
            z = z[np.newaxis]
        if z.ndim != 2 or z.shape[0] != 1:
            raise ValueError(
                f"z_eeg must be 1-D or (1, D), got shape {z_eeg.shape}",
            )
        return z.astype(np.float32, copy=False)

    def _error_placeholder(self, msg: str) -> Image.Image:
        size = self.cfg.generation.sd.image_size
        img = Image.new("RGB", (size, size), color=(20, 25, 45))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(2, 2), (size - 2, size - 2)], outline=(239, 68, 68), width=3)
        draw.text((10, 10), "SD-Turbo unavailable", fill=(255, 255, 255))
        draw.text((10, 40), msg, fill=(230, 240, 255))
        return img
