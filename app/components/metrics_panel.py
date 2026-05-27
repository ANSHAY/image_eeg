"""Rolling-metrics panel for the demo header strip.

Maintains a fixed-length history of per-trial outcomes and renders
glassmorphism cards with the rolling mean of:

  - top-1 accuracy   (fraction matching ground-truth class)
  - cosine similarity to the matching CLIP image embedding
  - end-to-end latency (ms)

Uses cfg.ui.strings.* for every user-visible label so the panel
stays config-driven.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Optional

from utils.config import Config, load_config

_HISTORY_LEN = 20


@dataclass
class _Sample:
    correct: bool
    cosine: float
    latency_ms: float


class MetricsPanel:
    """Tracks the last ``history_len`` trials and renders the metrics strip."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        history_len: int = _HISTORY_LEN,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self._history: deque[_Sample] = deque(maxlen=history_len)

    def record(self, *, correct: bool, cosine: float, latency_ms: float) -> None:
        self._history.append(_Sample(correct=correct, cosine=cosine, latency_ms=latency_ms))

    @property
    def rolling(self) -> dict[str, float]:
        if not self._history:
            return {"top1": 0.0, "cosine": 0.0, "latency_ms": 0.0, "n": 0}
        return {
            "top1": fmean(1.0 if s.correct else 0.0 for s in self._history),
            "cosine": fmean(s.cosine for s in self._history),
            "latency_ms": fmean(s.latency_ms for s in self._history),
            "n": len(self._history),
        }

    def render_html(self) -> str:
        """Build the four-card metrics strip for st.markdown."""
        c = self.cfg
        s = self.cfg.ui.strings
        m = self.rolling

        def _card(label: str, value: str) -> str:
            return (
                '<div class="vcr-card" style="display:inline-block;'
                'min-width:160px;margin-right:10px;text-align:center;">'
                f'<div class="vcr-metric-label">{label}</div>'
                f'<div class="vcr-metric-value">{value}</div>'
                "</div>"
            )

        n_text = f" (rolling over {m['n']} trials)" if m["n"] > 0 else " (waiting…)"
        cards = "".join(
            [
                _card(s.metric_top1, f"{m['top1']*100:.1f}%"),
                _card(s.metric_cosine, f"{m['cosine']:.3f}"),
                _card(s.metric_latency, f"{m['latency_ms']:.0f} ms"),
            ],
        )
        return f'<div style="margin:8px 0;">{cards}<span style="opacity:0.7;">{n_text}</span></div>'
