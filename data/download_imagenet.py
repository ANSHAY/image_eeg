"""Download 40-class ImageNet stimulus subset used by the EEG dataset.

STUB — Phase 1 box 12 (`feat(data): add download_imagenet.py …`) implements
the actual fetch logic. This stub exists so `setup.sh` is runnable today.
"""

from __future__ import annotations

from pathlib import Path

from utils.config import load_config
from utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    log = get_logger(__name__)
    cfg = load_config()

    dest = Path(cfg.paths.imagenet_stimuli)
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        log.info("imagenet stimuli destination %s already populated — skipping", dest)
        return

    log.warning(
        "download_imagenet.py is a stub (Phase 1 box 12). "
        "Will fetch stimuli into %s, class list from %s.",
        dest,
        cfg.paths.imagenet_classes_file,
    )


if __name__ == "__main__":
    main()
