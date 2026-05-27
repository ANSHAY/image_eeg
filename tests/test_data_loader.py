"""Unit tests for preprocessing.data_loader.

Builds a synthetic .pth fixture (no .mat path tested — the perceivelab
release uses .pth; .mat is a documented fallback that would need its
own integration check against real MATLAB data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from preprocessing.data_loader import SpampinatoLoader, Trial
from utils.config import Config, load_config


NUM_CHANNELS = 128
NUM_SAMPLES = 500
NUM_SUBJECTS = 3
NUM_CLASSES = 4
TRIALS_PER_SUBJECT = 4
TOTAL_TRIALS = NUM_SUBJECTS * TRIALS_PER_SUBJECT


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny synthetic .pth that matches the perceivelab format."""
    dataset_dir = tmp_path / "spampinato"
    dataset_dir.mkdir()

    rng = np.random.default_rng(0)
    records = []
    for sid in range(NUM_SUBJECTS):
        for k in range(TRIALS_PER_SUBJECT):
            label = (sid + k) % NUM_CLASSES
            records.append(
                {
                    "eeg": torch.from_numpy(
                        rng.standard_normal((NUM_CHANNELS, NUM_SAMPLES)).astype(np.float32),
                    ),
                    "label": label,
                    "image": label,
                    "subject": sid,
                },
            )
    payload = {
        "dataset": records,
        "labels": [f"class_{i}" for i in range(NUM_CLASSES)],
        "images": [f"img_{i}.JPEG" for i in range(NUM_CLASSES)],
    }
    torch.save(payload, dataset_dir / "eeg_signals_raw_with_mean_std.pth")
    return dataset_dir


@pytest.fixture
def stimuli_dir(tmp_path: Path) -> Path:
    d = tmp_path / "imagenet_stimuli"
    d.mkdir()
    return d


def _make_cfg(spampinato_dir: Path, imagenet_dir: Path) -> Config:
    base = load_config()
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "spampinato": str(spampinato_dir),
                    "imagenet_stimuli": str(imagenet_dir),
                },
            ),
            "dataset": base.dataset.model_copy(
                update={
                    "primary": base.dataset.primary.model_copy(
                        update={
                            "expected_subjects": NUM_SUBJECTS,
                            "expected_classes": NUM_CLASSES,
                        },
                    ),
                },
            ),
        },
    )


def test_trial_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Trial(
            subject_id=0,
            trial_id=0,
            eeg_data=np.zeros(100, dtype=np.float32),
            label=0,
            image_path="x",
        )


def test_trial_rejects_negative_label() -> None:
    with pytest.raises(ValueError, match="label"):
        Trial(
            subject_id=0,
            trial_id=0,
            eeg_data=np.zeros((128, 500), dtype=np.float32),
            label=-1,
            image_path="x",
        )


def test_trial_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        Trial(
            subject_id=0,
            trial_id=0,
            eeg_data=np.zeros((128, 500), dtype=np.int32),
            label=0,
            image_path="x",
        )


def test_loader_finds_pth_in_dataset_dir(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    path = loader.find_dataset_file()
    assert path.suffix == ".pth"
    assert path.parent == synthetic_dataset_dir


def test_loader_raises_when_directory_missing(tmp_path: Path, stimuli_dir: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    cfg = _make_cfg(nonexistent, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        loader.find_dataset_file()


def test_loader_raises_when_no_supported_files(tmp_path: Path, stimuli_dir: Path) -> None:
    empty = tmp_path / "spampinato_empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("nothing here")
    cfg = _make_cfg(empty, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    with pytest.raises(FileNotFoundError, match="no .pth or .mat"):
        loader.find_dataset_file()


def test_loader_returns_expected_trial_count(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    trials = loader.load()
    assert len(trials) == TOTAL_TRIALS
    assert len(loader) == TOTAL_TRIALS


def test_loader_returns_correctly_shaped_trials(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    trials = SpampinatoLoader(cfg=cfg).load()
    for t in trials:
        assert t.eeg_data.shape == (NUM_CHANNELS, NUM_SAMPLES)
        assert t.eeg_data.dtype == np.float32
        assert 0 <= t.label < NUM_CLASSES
        assert 0 <= t.subject_id < NUM_SUBJECTS
        assert t.class_name.startswith("class_")
        assert t.image_path.endswith(".JPEG")
        assert str(stimuli_dir) in t.image_path


def test_loader_caches_trials_across_calls(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    a = loader.load()
    b = loader.load()
    assert a is b


def test_loader_iter_yields_all_trials(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    iterated = list(SpampinatoLoader(cfg=cfg))
    assert len(iterated) == TOTAL_TRIALS


def test_by_subject_filters_to_one_subject(
    synthetic_dataset_dir: Path, stimuli_dir: Path,
) -> None:
    cfg = _make_cfg(synthetic_dataset_dir, stimuli_dir)
    loader = SpampinatoLoader(cfg=cfg)
    subject_0 = loader.by_subject(0)
    assert len(subject_0) == TRIALS_PER_SUBJECT
    assert all(t.subject_id == 0 for t in subject_0)


def test_loader_rejects_wrong_channel_count(
    tmp_path: Path, stimuli_dir: Path,
) -> None:
    bad_dir = tmp_path / "spampinato_bad"
    bad_dir.mkdir()
    payload = {
        "dataset": [
            {
                "eeg": torch.zeros(64, NUM_SAMPLES),  # wrong channels
                "label": 0,
                "image": 0,
                "subject": 0,
            },
        ],
        "labels": ["c"],
        "images": ["i.JPEG"],
    }
    torch.save(payload, bad_dir / "eeg.pth")
    cfg = _make_cfg(bad_dir, stimuli_dir)
    with pytest.raises(ValueError, match=r"\(128, 500\)"):
        SpampinatoLoader(cfg=cfg).load()
