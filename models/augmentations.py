"""Training-time EEG augmentations.

Three composable transforms, all driven by ``cfg.training.augment``:

  - ``GaussianNoise(sigma)``      — additive Gaussian per element.
  - ``TemporalJitter(max_shift)`` — circular shift in time by a random
                                    offset in ``[-max_shift, +max_shift]``.
  - ``ChannelDropout(p)``         — zero out a random subset of channels
                                    with independent Bernoulli(p) per trial.

All operate in-place-friendly on torch tensors of shape
``(channels, samples)`` or ``(batch, channels, samples)``. Apply only
to training batches — never on validation/test inputs, since augmentation
masks the model's actual robustness.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from utils.config import Config, load_config


class GaussianNoise:
    """Add Gaussian noise with standard deviation ``sigma``."""

    def __init__(self, sigma: float) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0:
            return x
        return x + torch.randn_like(x) * self.sigma


class TemporalJitter:
    """Circular shift along the time axis by a random integer in
    ``[-max_shift_samples, +max_shift_samples]``."""

    def __init__(self, max_shift_samples: int) -> None:
        if max_shift_samples < 0:
            raise ValueError(
                f"max_shift_samples must be non-negative, got {max_shift_samples}",
            )
        self.max_shift = int(max_shift_samples)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift == 0:
            return x
        # Single shift for the whole tensor — preserves channel alignment.
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(x, shifts=shift, dims=-1)


class ChannelDropout:
    """Independently zero out each channel with probability ``p``.

    For a 3-D batch input, the dropout mask is drawn fresh per trial
    so different trials have different channel subsets dropped —
    matches the behavior described in the spec ('zero out 10% of
    channels per trial').
    """

    def __init__(self, p: float) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return x
        if x.ndim == 2:
            mask = (torch.rand(x.shape[0], device=x.device) >= self.p).to(x.dtype)
            return x * mask.unsqueeze(-1)
        if x.ndim == 3:
            mask = (torch.rand(x.shape[0], x.shape[1], device=x.device) >= self.p).to(x.dtype)
            return x * mask.unsqueeze(-1)
        raise ValueError(
            f"ChannelDropout expects 2-D or 3-D input, got shape {tuple(x.shape)}",
        )


class Compose:
    """Sequentially apply a list of callables."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def make_train_augmentations(cfg: Optional[Config] = None) -> Compose:
    """Build the canonical training-time augmentation chain from config."""
    c = cfg if cfg is not None else load_config()
    aug = c.training.augment
    return Compose(
        [
            GaussianNoise(aug.noise_sigma),
            TemporalJitter(aug.jitter_samples),
            ChannelDropout(aug.channel_dropout_p),
        ],
    )
