"""Download the EEG dataset — Spampinato primary, THINGS-EEG2 fallback.

Idempotent: a populated destination dir short-circuits the script.

Many EEG dataset hosts (perceivelab Google Drive, OSF projects) cannot
be fetched programmatically without site cookies or a manual permission
acknowledgment. The script therefore:

  1. Skips when ``<dataset_dir>/<expected_filename>`` already exists.
  2. Attempts an automatic fetch from ``cfg.dataset.<source>.download_url``
     when that field is set (direct HTTPS only — no scraping).
  3. Falls back to the secondary source if the primary fails.
  4. Prints precise manual-fetch instructions and exits with a non-zero
     status when neither source is reachable. The exit code lets
     ``setup.sh`` continue (the smoke check still runs) while making
     the deficit visible to the user.

Re-running the script after a manual download completes the bootstrap
without re-fetching.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from utils.config import Config, DatasetSource, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB


def _has_expected_file(dest_dir: Path, expected_filename: str) -> bool:
    return (dest_dir / expected_filename).is_file()


def _fetch(url: str, dest_path: Path) -> bool:
    """Stream a direct HTTPS download to ``dest_path``. Returns True on success."""
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
        log.info("downloaded %s (%.1f MiB)", dest_path.name, dest_path.stat().st_size / (1 << 20))
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("download failed for %s: %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def _try_source(source: DatasetSource, dest_dir: Path) -> bool:
    """Attempt to satisfy this source — skip-if-present, then auto-fetch."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _has_expected_file(dest_dir, source.expected_filename):
        log.info("[%s] %s already present in %s — skipping", source.name, source.expected_filename, dest_dir)
        return True
    if source.download_url is None:
        log.warning(
            "[%s] no direct download_url configured; manual fetch required from %s",
            source.name, source.landing_url,
        )
        return False
    return _fetch(source.download_url, dest_dir / source.expected_filename)


def _manual_instructions(cfg: Config) -> str:
    primary = cfg.dataset.primary
    fallback = cfg.dataset.fallback
    spampinato_dir = Path(cfg.paths.spampinato)
    things_dir = Path(cfg.paths.things_eeg2)
    return (
        f"Neither dataset could be auto-downloaded.\n\n"
        f"Manual fetch — primary ({primary.name}):\n"
        f"  1. Visit {primary.landing_url}\n"
        f"  2. Follow the README to obtain '{primary.expected_filename}'\n"
        f"     (typically a Google Drive share — auth required).\n"
        f"  3. Place the file under: {spampinato_dir.resolve()}/\n"
        f"  4. Re-run: .venv/bin/python -m data.download_dataset\n\n"
        f"Manual fetch — fallback ({fallback.name}):\n"
        f"  1. Visit {fallback.landing_url}\n"
        f"  2. Download '{fallback.expected_filename}'.\n"
        f"  3. Place it under: {things_dir.resolve()}/\n"
        f"  4. Re-run the downloader.\n\n"
        f"To enable automated fetch in the future, set the corresponding\n"
        f"`dataset.<source>.download_url` in config.yaml to a direct HTTPS URL.\n"
    )


def main() -> int:
    setup_logging()
    cfg = load_config()

    primary_dir = Path(cfg.paths.spampinato)
    fallback_dir = Path(cfg.paths.things_eeg2)

    if _try_source(cfg.dataset.primary, primary_dir):
        return 0

    log.warning("[%s] unavailable; trying fallback [%s]", cfg.dataset.primary.name, cfg.dataset.fallback.name)
    if _try_source(cfg.dataset.fallback, fallback_dir):
        return 0

    sys.stderr.write(_manual_instructions(cfg))
    return 1


if __name__ == "__main__":
    sys.exit(main())
