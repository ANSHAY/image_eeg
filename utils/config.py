"""Frozen, validated configuration loaded once from `config.yaml`.

All runtime parameters live in YAML; code references `cfg.section.key`
rather than literals (enforced by `tests/test_no_magic_strings.py`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

_CONFIG_ENV_VAR = "VCR_CONFIG"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Project(_Frozen):
    name: str
    seed: int


class Paths(_Frozen):
    data_raw: str
    data_processed: str
    imagenet_stimuli: str
    spampinato: str
    things_eeg2: str
    imagenet_classes_file: str
    checkpoints: str
    results: str
    runs: str
    logs: str
    tests_fixtures: str


class DatasetSource(_Frozen):
    name: str
    landing_url: str
    download_url: Optional[str] = None
    expected_filename: str
    expected_subjects: int
    expected_classes: int


class ImagenetStimuli(_Frozen):
    bundled_archive_name: str
    landing_url: str
    download_url: Optional[str] = None
    expected_extension: str


class Dataset(_Frozen):
    primary: DatasetSource
    fallback: DatasetSource
    imagenet_stimuli: ImagenetStimuli


class EEG(_Frozen):
    sample_rate_hz: int
    num_channels: int
    trial_length_samples: int
    powerline_hz: int


class Bandpass(_Frozen):
    low_hz: float
    high_hz: float
    order: int


class Notch(_Frozen):
    quality_factor: float


class Crop(_Frozen):
    start_sample: int
    end_sample: int


class Preprocessing(_Frozen):
    bandpass: Bandpass
    notch: Notch
    crop: Crop
    use_ica: bool
    ica_components: int
    ica_kurtosis_threshold: float


class Streaming(_Frozen):
    outlet_name: str
    outlet_type: str
    source_id: str
    marker_outlet_name: str
    marker_outlet_type: str
    marker_source_id: str
    channel_format: str
    default_speed: float
    chunk_size: int
    resolve_timeout_s: float


class CLIPModel(_Frozen):
    hf_id: str
    revision: Optional[str] = None
    embed_dim: int
    image_size: int
    embed_batch_size: int


class EncoderArch(_Frozen):
    spatial_out: int
    temporal_channels: list[int]
    temporal_kernel: int
    temporal_stride: int
    dropout: float
    head_dropout: float
    pool_size: int
    proj_hidden: int


class Models(_Frozen):
    clip: CLIPModel
    encoder: EncoderArch


class Optimizer(_Frozen):
    name: str
    lr: float
    weight_decay: float
    betas: list[float]


class Scheduler(_Frozen):
    name: str
    T_0: int
    T_mult: int


class Loss(_Frozen):
    info_nce_weight: float
    mse_weight: float
    temperature_init: float
    temperature_max: float


class Augment(_Frozen):
    noise_sigma: float
    jitter_samples: int
    channel_dropout_p: float


class CV(_Frozen):
    scheme: str
    num_subjects: int


class Checkpoint(_Frozen):
    keep_best_metric: str
    keep_last: bool


class Training(_Frozen):
    batch_size: int
    epochs: int
    num_workers: int
    optimizer: Optimizer
    scheduler: Scheduler
    loss: Loss
    augment: Augment
    cv: CV
    checkpoint: Checkpoint
    log_every_n_steps: int
    eval_every_n_epochs: int


class SDGeneration(_Frozen):
    hf_id: str
    revision: Optional[str] = None
    torch_dtype: str
    ip_adapter_hf_id: str
    ip_adapter_subfolder: str
    ip_adapter_weight_name: str
    num_inference_steps: int
    guidance_scale: float
    image_size: int


class RetrievalGeneration(_Frozen):
    index_type: str
    top_k: int
    return_thumbnails: bool


class Generation(_Frozen):
    mode: str = Field(pattern="^(retrieval|sd_turbo)$")
    sd: SDGeneration
    retrieval: RetrievalGeneration


class Inference(_Frozen):
    backend: str = Field(pattern="^(torch|openvino)$")
    openvino_device: str
    onnx_opset: int


class Theme(_Frozen):
    bg: str
    accent: str
    text: str
    surface: str
    surface_alpha: float


class RegionToChannels(_Frozen):
    frontal: list[int]
    temporal: list[int]
    parietal: list[int]
    occipital: list[int]


class RegionColors(_Frozen):
    frontal: str
    temporal: str
    parietal: str
    occipital: str


class UIStrings(_Frozen):
    app_title: str
    panel_eeg: str
    panel_recon: str
    panel_truth: str
    panel_umap: str
    panel_metrics: str
    panel_fft: str
    warming_up: str
    no_stream: str
    generation_mode_label: str
    mode_retrieval: str
    mode_sd_turbo: str
    metric_top1: str
    metric_cosine: str
    metric_latency: str
    metric_confidence: str
    metric_predicted_class: str


class UI(_Frozen):
    framework: str
    refresh_interval_ms: int
    theme: Theme
    region_colors: RegionColors
    region_to_channels: RegionToChannels
    strings: UIStrings


class Config(_Frozen):
    project: Project
    paths: Paths
    dataset: Dataset
    eeg: EEG
    preprocessing: Preprocessing
    streaming: Streaming
    models: Models
    training: Training
    generation: Generation
    inference: Inference
    ui: UI


def _project_root() -> Path:
    """Resolve project root as the directory containing this utils/ package."""
    return Path(__file__).resolve().parent.parent


def _default_config_path() -> Path:
    return _project_root() / "config.yaml"


@lru_cache(maxsize=8)
def load_config(path: Optional[str] = None) -> Config:
    """Load and validate `config.yaml`. Cached by path string.

    Resolution order: explicit ``path`` argument > ``VCR_CONFIG`` env
    variable > ``<project_root>/config.yaml``. The env var lets tests
    point subprocesses at a tmp config without modifying argv.

    Returns:
        Frozen, type-checked Config instance.
    """
    if path is None:
        path = os.environ.get(_CONFIG_ENV_VAR)
    p = Path(path) if path is not None else _default_config_path()
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
