"""pytest config — adds the project root to sys.path so `from utils.config …` works,
and registers custom markers."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires external resources (subprocess, network, LSL bus)",
    )
