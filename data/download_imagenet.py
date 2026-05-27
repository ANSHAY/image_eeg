"""Place the 40-class ImageNet stimuli used by the EEG dataset.

The Spampinato release usually ships its stimuli as an ``images.zip``
adjacent to the ``.pth`` file. This script:

  1. Skips if the stimuli destination already contains image files.
  2. Looks for ``cfg.dataset.imagenet_stimuli.bundled_archive_name`` in
     the Spampinato dataset directory; if found, extracts it.
  3. Falls back to a direct HTTPS download when ``download_url`` is set.
  4. Otherwise prints precise manual instructions and exits 1.

Re-runnable safely after a manual extract or download completes.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20


def _has_stimuli(stimuli_dir: Path, expected_extension: str) -> bool:
    ext = expected_extension.lower()
    return any(
        p.is_file() and p.suffix.lower() == ext
        for p in stimuli_dir.rglob("*")
    )


def _extract_bundle(archive_path: Path, dest_dir: Path) -> bool:
    if not archive_path.is_file():
        return False
    try:
        log.info("extracting %s -> %s", archive_path, dest_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        return True
    except (zipfile.BadZipFile, OSError) as e:
        log.warning("extract failed for %s: %s", archive_path, e)
        return False


def _fetch(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        log.info("downloading %s -> %s", url, dest_path)
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.rename(dest_path)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("download failed for %s: %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def _manual_instructions(cfg: Config) -> str:
    stim = cfg.dataset.imagenet_stimuli
    stimuli_dir = Path(cfg.paths.imagenet_stimuli)
    spampinato_dir = Path(cfg.paths.spampinato)
    return (
        f"ImageNet stimuli not available. Manual options:\n\n"
        f"A. Extract from the Spampinato bundle (preferred):\n"
        f"   - The dataset distribution typically includes an images archive.\n"
        f"   - Place '{stim.bundled_archive_name}' next to the .pth file at:\n"
        f"     {spampinato_dir.resolve()}\n"
        f"   - Re-run: .venv/bin/python -m data.download_imagenet\n\n"
        f"B. Download the 40-class subset from ImageNet:\n"
        f"   1. Register at {stim.landing_url} (academic email recommended)\n"
        f"   2. Download the 40 wnid synsets used by Spampinato — see\n"
        f"      {cfg.paths.imagenet_classes_file} once Phase 3 generates it.\n"
        f"   3. Extract the JPEGs under: {stimuli_dir.resolve()}/\n"
        f"   4. Re-run the script.\n"
    )


def main() -> int:
    setup_logging()
    cfg = load_config()

    stim_cfg = cfg.dataset.imagenet_stimuli
    stimuli_dir = Path(cfg.paths.imagenet_stimuli)
    spampinato_dir = Path(cfg.paths.spampinato)
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    if _has_stimuli(stimuli_dir, stim_cfg.expected_extension):
        log.info("%s already populated with %s files — skipping", stimuli_dir, stim_cfg.expected_extension)
        return 0

    bundle = spampinato_dir / stim_cfg.bundled_archive_name
    if _extract_bundle(bundle, stimuli_dir):
        if _has_stimuli(stimuli_dir, stim_cfg.expected_extension):
            log.info("extracted bundled stimuli from %s", bundle)
            return 0
        log.warning("extracted %s but no %s files found inside", bundle, stim_cfg.expected_extension)

    if stim_cfg.download_url is not None:
        archive_dest = stimuli_dir.parent / stim_cfg.bundled_archive_name
        if _fetch(stim_cfg.download_url, archive_dest) and _extract_bundle(archive_dest, stimuli_dir):
            return 0

    sys.stderr.write(_manual_instructions(cfg))
    return 1


if __name__ == "__main__":
    sys.exit(main())
