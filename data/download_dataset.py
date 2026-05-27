"""Download EEG dataset (Spampinato primary, THINGS-EEG2 fallback).

STUB — Phase 1 box 11 (`feat(data): add download_dataset.py …`) implements
the actual fetch logic. This stub exists so `setup.sh` is runnable today
without a chicken-and-egg failure.
"""

from __future__ import annotations

from pathlib import Path

from utils.config import load_config
from utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    cfg = load_config()

    dest = Path(cfg.paths.spampinato)
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        log.info("dataset destination %s already populated — skipping", dest)
        return

    log.warning(
        "download_dataset.py is a stub (Phase 1 box 11). "
        "Will fetch %s into %s.",
        cfg.dataset.primary.url,
        dest,
    )


if __name__ == "__main__":
    main()
