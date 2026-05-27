"""Channel-to-brain-region mapping for the EEG plot.

Each 128-channel EEG headset (e.g., Biosemi BioSemi ABC layout) has a
canonical electrode arrangement. The Spampinato setup follows the
international 10-20 system extended to 128 sites. Region assignment
here is *coarse* — sufficient for the demo's color-coding (frontal,
temporal, parietal, occipital) but not for source localization.

Mapping comes from ``cfg.ui.region_to_channels`` so the user can
override per dataset. When the config has empty lists (the default
pre-Phase 6), this module falls back to an even quartile split:
channels 0..n/4 → frontal, n/4..n/2 → temporal, n/2..3n/4 → parietal,
3n/4..n → occipital. The fallback isn't physiologically accurate but
keeps the plot color-coded until the user fills in the real layout.
"""

from __future__ import annotations

from typing import Optional

from utils.config import Config, load_config

_REGION_ORDER = ("frontal", "temporal", "parietal", "occipital")


def region_channels(cfg: Optional[Config] = None) -> dict[str, list[int]]:
    """Return the channel-list per brain region.

    Uses ``cfg.ui.region_to_channels`` when populated; otherwise emits
    an even-quartile fallback over ``cfg.eeg.num_channels``.
    """
    c = cfg if cfg is not None else load_config()
    explicit = {r: list(getattr(c.ui.region_to_channels, r)) for r in _REGION_ORDER}
    total_specified = sum(len(v) for v in explicit.values())
    if total_specified > 0:
        return explicit
    # Fallback: contiguous quartiles
    n = c.eeg.num_channels
    q = n // 4
    return {
        "frontal": list(range(0, q)),
        "temporal": list(range(q, 2 * q)),
        "parietal": list(range(2 * q, 3 * q)),
        "occipital": list(range(3 * q, n)),
    }


def channel_to_region(cfg: Optional[Config] = None) -> dict[int, str]:
    """Inverse map: channel index → region name."""
    m: dict[int, str] = {}
    for region, channels in region_channels(cfg).items():
        for ch in channels:
            m[ch] = region
    return m


def channel_to_color(cfg: Optional[Config] = None) -> dict[int, str]:
    """Channel index → hex color string from cfg.ui.region_colors."""
    c = cfg if cfg is not None else load_config()
    palette = {
        "frontal": c.ui.region_colors.frontal,
        "temporal": c.ui.region_colors.temporal,
        "parietal": c.ui.region_colors.parietal,
        "occipital": c.ui.region_colors.occipital,
    }
    return {ch: palette[region] for ch, region in channel_to_region(c).items()}
