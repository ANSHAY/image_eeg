"""Load EEG-ImageNet trials from the Spampinato dataset.

Supports both on-disk formats the published releases use:

  - .pth  — perceivelab/eeg_visual_classification release; torch.save of a
            dict with keys ``dataset`` (list of per-trial dicts with
            ``eeg``, ``image``, ``label``, ``subject``), ``labels`` (class
            names), and ``images`` (stimulus filenames indexed by
            ``trial['image']``).
  - .mat  — original Spampinato 2017 release; ``scipy.io.loadmat`` produces
            cell arrays carrying the same fields under MATLAB-style names.

The loader returns an iterable of immutable :class:`Trial` dataclasses with
strict shape validation against ``cfg.eeg``. Mismatches fail loudly — we'd
rather catch a wrong dataset on import than silently pump bad shapes into
the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from utils.config import Config, load_config
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Trial:
    """One EEG trial aligned to a single stimulus image.

    Frozen so trials can be safely passed to threads / multiprocessing
    workers downstream without aliasing surprises.
    """

    subject_id: int
    trial_id: int
    eeg_data: np.ndarray
    label: int
    image_path: str
    class_name: str = field(default="")

    def __post_init__(self) -> None:
        if self.eeg_data.ndim != 2:
            raise ValueError(
                f"trial {self.trial_id}: eeg_data must be 2-D "
                f"(channels, samples), got shape {self.eeg_data.shape}",
            )
        if self.eeg_data.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"trial {self.trial_id}: eeg_data dtype must be float32/float64, "
                f"got {self.eeg_data.dtype}",
            )
        if self.label < 0:
            raise ValueError(f"trial {self.trial_id}: label must be non-negative, got {self.label}")


class SpampinatoLoader:
    """Lazy loader for the Spampinato EEG-ImageNet dataset.

    Locates a single dataset file (``.pth`` preferred, ``.mat`` fallback)
    under ``cfg.paths.spampinato`` and yields :class:`Trial` records on
    iteration. Stimulus image paths are resolved relative to
    ``cfg.paths.imagenet_stimuli``.

    The loader does not preprocess the signal — that is the job of
    :mod:`preprocessing.signal_processing`. Trials are returned as the
    raw float arrays present in the file.
    """

    PTH_SUFFIX = ".pth"
    MAT_SUFFIX = ".mat"

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()
        self.dataset_dir = Path(self.cfg.paths.spampinato)
        self.imagenet_dir = Path(self.cfg.paths.imagenet_stimuli)
        self._cached_trials: Optional[list[Trial]] = None

    # --- discovery -----------------------------------------------------

    def find_dataset_file(self) -> Path:
        """Return the path to the first ``.pth`` or ``.mat`` under the dataset dir.

        Raises:
            FileNotFoundError: dataset dir missing or no compatible file found.
        """
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"dataset directory {self.dataset_dir} does not exist. "
                "Run setup.sh or python -m data.download_dataset first.",
            )
        for suffix in (self.PTH_SUFFIX, self.MAT_SUFFIX):
            matches = sorted(self.dataset_dir.glob(f"*{suffix}"))
            if matches:
                if len(matches) > 1:
                    log.warning(
                        "multiple %s files in %s; using %s",
                        suffix, self.dataset_dir, matches[0].name,
                    )
                return matches[0]
        raise FileNotFoundError(
            f"no .pth or .mat dataset file found under {self.dataset_dir}",
        )

    # --- format-specific readers --------------------------------------

    def _load_pth(self, path: Path) -> list[Trial]:
        import torch

        log.info("loading Spampinato .pth from %s", path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "dataset" not in blob:
            raise ValueError(f"{path}: expected dict with key 'dataset', got {type(blob).__name__}")

        records: list[dict[str, Any]] = blob["dataset"]
        class_names: list[str] = list(blob.get("labels", []))
        image_files: list[str] = list(blob.get("images", []))

        trials: list[Trial] = []
        for tid, rec in enumerate(records):
            eeg = rec["eeg"]
            if hasattr(eeg, "numpy"):
                eeg = eeg.numpy()
            eeg = np.asarray(eeg, dtype=np.float32)
            
            expected_samples = self.cfg.eeg.trial_length_samples
            if eeg.shape[1] > expected_samples:
                eeg = eeg[:, :expected_samples]
            elif eeg.shape[1] < expected_samples:
                pad_width = expected_samples - eeg.shape[1]
                eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode="constant")
                
            label_idx = int(rec["label"])
            image_idx = int(rec["image"])
            image_file = image_files[image_idx] if image_files else f"img_{image_idx}.jpg"
            trial = Trial(
                subject_id=int(rec.get("subject", 0)),
                trial_id=tid,
                eeg_data=eeg,
                label=label_idx,
                image_path=str(self.imagenet_dir / image_file),
                class_name=class_names[label_idx] if class_names else "",
            )
            trials.append(trial)
        return trials

    def _load_mat(self, path: Path) -> list[Trial]:
        from scipy.io import loadmat

        log.info("loading Spampinato .mat from %s", path)
        blob = loadmat(path, squeeze_me=True, struct_as_record=False)
        # The 2017 release stores the dataset under the variable name
        # `dataset` (a struct array) and class names under `labels`.
        if "dataset" not in blob:
            raise ValueError(f"{path}: expected MATLAB variable 'dataset' not found")
        records = blob["dataset"]
        class_names = list(blob.get("labels", []))
        image_files = list(blob.get("images", []))

        # MATLAB struct arrays come back as 0-d objects with field access.
        # Normalize to a 1-D Python list for uniform iteration.
        records = list(np.atleast_1d(records))

        trials: list[Trial] = []
        for tid, rec in enumerate(records):
            eeg = np.asarray(rec.eeg, dtype=np.float32)
            
            expected_samples = self.cfg.eeg.trial_length_samples
            if eeg.shape[1] > expected_samples:
                eeg = eeg[:, :expected_samples]
            elif eeg.shape[1] < expected_samples:
                pad_width = expected_samples - eeg.shape[1]
                eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode="constant")
                
            label_idx = int(rec.label)
            image_idx = int(rec.image)
            image_file = (
                str(image_files[image_idx]) if image_files else f"img_{image_idx}.jpg"
            )
            trial = Trial(
                subject_id=int(getattr(rec, "subject", 0)),
                trial_id=tid,
                eeg_data=eeg,
                label=label_idx,
                image_path=str(self.imagenet_dir / image_file),
                class_name=str(class_names[label_idx]) if class_names else "",
            )
            trials.append(trial)
        return trials

    # --- validation ---------------------------------------------------

    def _validate(self, trials: list[Trial]) -> None:
        if not trials:
            raise ValueError("dataset contained zero trials")

        expected_channels = self.cfg.eeg.num_channels
        expected_samples = self.cfg.eeg.trial_length_samples

        bad_shapes = [
            (t.trial_id, t.eeg_data.shape)
            for t in trials
            if t.eeg_data.shape != (expected_channels, expected_samples)
        ]
        if bad_shapes:
            sample = bad_shapes[:5]
            raise ValueError(
                f"{len(bad_shapes)} trials have shape != "
                f"({expected_channels}, {expected_samples}). "
                f"First five: {sample}",
            )

        subject_ids = {t.subject_id for t in trials}
        expected_subjects = self.cfg.dataset.primary.expected_subjects
        if len(subject_ids) != expected_subjects:
            log.warning(
                "subject count mismatch: dataset has %d, config expects %d. "
                "If you switched to a fallback dataset, update "
                "cfg.dataset.primary.expected_subjects.",
                len(subject_ids), expected_subjects,
            )

        labels = {t.label for t in trials}
        expected_classes = self.cfg.dataset.primary.expected_classes
        if len(labels) != expected_classes:
            log.warning(
                "class count mismatch: dataset has %d labels, config expects %d.",
                len(labels), expected_classes,
            )

    # --- public API ---------------------------------------------------

    def load(self) -> list[Trial]:
        """Return all trials, validated, cached for subsequent calls."""
        if self._cached_trials is not None:
            return self._cached_trials
        path = self.find_dataset_file()
        if path.suffix == self.PTH_SUFFIX:
            trials = self._load_pth(path)
        elif path.suffix == self.MAT_SUFFIX:
            trials = self._load_mat(path)
        else:  # find_dataset_file restricts to .pth/.mat — defensive only
            raise ValueError(f"unsupported dataset suffix: {path.suffix}")
        self._validate(trials)
        log.info(
            "loaded %d trials, %d subjects, %d classes",
            len(trials),
            len({t.subject_id for t in trials}),
            len({t.label for t in trials}),
        )
        self._cached_trials = trials
        return trials

    def __iter__(self) -> Iterator[Trial]:
        yield from self.load()

    def __len__(self) -> int:
        return len(self.load())

    def by_subject(self, subject_id: int) -> list[Trial]:
        return [t for t in self.load() if t.subject_id == subject_id]
