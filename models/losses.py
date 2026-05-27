"""Hybrid alignment loss for EEG ↔ CLIP.

Total = ``info_nce_weight`` × symmetric InfoNCE + ``mse_weight`` × MSE

  - **Symmetric InfoNCE**: CLIP-style — given a batch of (z_eeg, z_img)
    matching pairs, build the full ``(B, B)`` similarity matrix and
    apply cross-entropy along both axes (EEG→image *and* image→EEG).
    The temperature is the learnable ``logit_scale = log(1/T)`` from
    CLIP, initialized to ``log(1 / temperature_init)`` and clamped at
    ``log(temperature_max)`` so the softmax can't collapse.
  - **MSE**: direct L2 between the EEG and image embeddings. Because
    both sides are unit-norm, ``MSE = 2 - 2·cos_sim``, so this is
    effectively a cosine-distance regularizer that helps when the
    contrastive batch is small.

Inputs are assumed unit-norm — the EEG encoder ends in F.normalize,
and the CLIP image bank is built with L2 normalization. The loss does
not re-normalize because that would mask a bug in the encoder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from utils.config import Config, load_config


class ContrastiveAlignmentLoss(nn.Module):
    """InfoNCE + MSE with a learnable temperature.

    Exposes ``temperature()`` as a tensor (post-clamp, exp'd) so it can
    be logged each step.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        super().__init__()
        c = cfg if cfg is not None else load_config()
        loss_cfg = c.training.loss
        self.info_nce_weight = float(loss_cfg.info_nce_weight)
        self.mse_weight = float(loss_cfg.mse_weight)
        # CLIP convention: log_temperature is the logit *scale* — large
        # values sharpen the softmax. Initialized so exp(log_T) = 1/T0.
        init = math.log(1.0 / loss_cfg.temperature_init)
        self.log_temperature = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.log_temperature_max = math.log(loss_cfg.temperature_max)

    def temperature(self) -> torch.Tensor:
        """Post-clamp scalar multiplier applied to the similarity matrix."""
        return self.log_temperature.clamp(max=self.log_temperature_max).exp()

    def info_nce(self, z_eeg: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE on a (B, B) similarity matrix."""
        if z_eeg.shape != z_img.shape:
            raise ValueError(
                f"shape mismatch: z_eeg {z_eeg.shape} vs z_img {z_img.shape}",
            )
        logits = z_eeg @ z_img.t() * self.temperature()
        targets = torch.arange(z_eeg.size(0), device=z_eeg.device)
        loss_e2i = F.cross_entropy(logits, targets)
        loss_i2e = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_e2i + loss_i2e)

    def mse(self, z_eeg: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_eeg, z_img)

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_img: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nce = self.info_nce(z_eeg, z_img)
        mse = self.mse(z_eeg, z_img)
        total = self.info_nce_weight * nce + self.mse_weight * mse
        components = {
            "info_nce": nce.detach(),
            "mse": mse.detach(),
            "temperature": self.temperature().detach(),
            "total": total.detach(),
        }
        return total, components
