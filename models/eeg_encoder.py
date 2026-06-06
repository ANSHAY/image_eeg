"""1-D CNN that maps a preprocessed EEG trial to a CLIP-aligned embedding.

Architecture (driven by ``cfg.models.encoder`` + ``cfg.models.clip.embed_dim``):

  input (B, C, T)                                  C = cfg.eeg.num_channels
      │
      ▼  SpatialConv1d k=1, no temporal mixing
  (B, spatial_out, T)
      │
      ▼  N temporal blocks (Conv1d k=temporal_kernel s=temporal_stride
      │  → BN → GELU → Dropout). N = len(cfg.models.encoder.temporal_channels).
  (B, temporal_channels[-1], T / stride**N)
      │
      ▼  AdaptiveAvgPool1d(pool_size)
  (B, temporal_channels[-1], pool_size)
      │
      ▼  Flatten → Linear → GELU → Dropout → Linear
  (B, embed_dim)
      │
      ▼  F.normalize  (rows live on the unit hypersphere)
  output (B, embed_dim)

Every dimension flows from config — no shape literals in code. The final
``F.normalize`` is **load-bearing**: the InfoNCE loss in :mod:`models.losses`
relies on unit-norm rows so that cosine similarity reduces to a dot product
and the encoder can't cheat by inflating magnitudes.

CLIP is never instantiated here. The encoder only emits vectors in the
same space CLIP defines; the alignment target is supplied externally
during training (precomputed CLIP image embeddings).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from utils.config import Config, load_config


class _TemporalBlock(nn.Sequential):
    """Conv1d → BatchNorm1d → GELU → Dropout. Strided convolutions halve
    the time axis when ``stride > 1``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,  # BatchNorm absorbs the bias
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )


class EEGEncoder(nn.Module):
    """1D-CNN EEG → CLIP-space encoder.

    Output is L2-normalized; every layer dimension comes from config.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        super().__init__()
        c = cfg if cfg is not None else load_config()
        self.cfg = c

        enc = c.models.encoder
        embed_dim = c.models.clip.embed_dim

        # Pointwise channel-mixer — learns linear combinations of the
        # raw 128 electrodes before any temporal feature extraction.
        self.spatial = nn.Sequential(
            nn.Conv1d(
                in_channels=c.eeg.num_channels,
                out_channels=enc.spatial_out,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(enc.spatial_out),
            nn.GELU(),
            nn.Dropout(enc.dropout),
        )

        blocks: list[nn.Module] = []
        in_ch = enc.spatial_out
        for out_ch in enc.temporal_channels:
            blocks.append(
                _TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=enc.temporal_kernel,
                    stride=enc.temporal_stride,
                    dropout=enc.dropout,
                ),
            )
            in_ch = out_ch
        self.temporal = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(enc.pool_size)

        flatten_dim = in_ch * enc.pool_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, enc.proj_hidden),
            nn.GELU(),
            nn.Dropout(enc.head_dropout),
            nn.Linear(enc.proj_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of trials.

        Args:
            x: shape ``(batch, num_channels, T)``.

        Returns:
            ``(batch, embed_dim)`` with rows on the unit hypersphere.
        """
        h = self.spatial(x)
        h = self.temporal(h)
        h = self.pool(h)
        h = self.head(h)
        return F.normalize(h, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
