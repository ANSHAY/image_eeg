# Visual Cortex Reconstructor

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-114%20passing-brightgreen.svg)](#quality-gates-and-current-results)

Reconstruct the image a person was looking at from their EEG signal — fully offline, on a consumer laptop with no discrete GPU.

A 0.5-second, 128-channel EEG trial is streamed over [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer), encoded into the CLIP image-embedding space by a 1-D CNN, and then either retrieved against a bank of CLIP image embeddings or handed to SD-Turbo + IP-Adapter to synthesise a picture. A Streamlit split-screen demo shows the live brainwaves on one side and the reconstruction on the other.

> **About this project.** This is a personal research and portfolio project. The full pipeline runs end-to-end, but several of the quality gates below are validated on *synthetic* EEG because the real dataset ([Spampinato et al.](https://github.com/perceivelab/eeg_visual_classification)) is gated behind a manual download I haven't completed. The tables are explicit about what is real and what is synthetic — I'd rather under-claim than overclaim. This is not a medical device and not a scientific result; it's an engineering exercise in running a modern multi-model pipeline offline on modest hardware.

## How it works

```mermaid
flowchart LR
    A[EEG dataset<br/>128 ch, 1000 Hz] --> B[data loader]
    B --> C[LSL streamer]
    C -->|Lab Streaming Layer| D[receiver / app inlet]
    D --> E[signal processing<br/>bandpass · notch · baseline · z-score · optional ICA]
    E --> F[EEG encoder<br/>1-D CNN → CLIP space]
    F --> G{generation mode}
    G -->|retrieval| H[FAISS k-NN over a<br/>precomputed CLIP image bank]
    G -->|generative| I[SD-Turbo + IP-Adapter]
    H --> J[Streamlit UI<br/>live EEG · reconstruction · UMAP]
    I --> J
    K[CLIP image encoder<br/>frozen] -.training targets.-> F
    K -.image bank.-> H

    subgraph training [Offline training]
      E --> L[InfoNCE + MSE loss<br/>leave-one-subject-out CV]
      L --> F
    end

    subgraph edge [Edge inference]
      F --> M[ONNX export] --> N[OpenVINO IR<br/>CPU · iGPU · NPU] --> J
    end
```

The CLIP image encoder is **frozen** — it supplies the training targets for the EEG
encoder *and* builds the retrieval bank; it is never fine-tuned. At inference the
torch encoder can be swapped for an OpenVINO IR via a single config flag.

## Hardware target

- Intel Core Ultra 5 225H (CPU + integrated NPU + integrated graphics)
- 32 GB DDR5, no discrete GPU
- Linux (Arch-based)
- ~100 GB free disk

Inference runs fully offline. CLIP and SD-Turbo are downloaded once during setup, then frozen.

## Setup

The project expects a Python 3.14 virtual environment at `./.venv/`. If you don't have one:

```bash
python3.14 -m venv .venv
```

Then run the idempotent bootstrap:

```bash
./setup.sh
```

This installs the pinned dependencies (CPU-only torch, transformers 4.x, diffusers 0.38,
OpenVINO 2026.1, …), attempts to download the EEG dataset (Spampinato primary,
THINGS-EEG2 fallback) and the 40-class ImageNet stimulus subset, then runs a smoke-import check.

Both dataset downloads exit gracefully when the source requires manual acknowledgment
(Google Drive / OSF). When that happens, follow the printed instructions to fetch the
bundle manually and re-run `setup.sh`.

On Arch Linux, `pylsl` benefits from a system `liblsl`:

```bash
sudo pacman -S liblsl
```

If it's absent, `pylsl` falls back to its wheel-bundled library — that works, but the system package is preferred.

## Run

```bash
# Virtual EEG rig (LSL streamer + receiver)
.venv/bin/python -m streaming.lsl_streamer --speed 1.0
.venv/bin/python -m streaming.lsl_receiver

# Offline preprocessing (requires the Spampinato bundle)
.venv/bin/python -m preprocessing.run_pipeline

# Train the EEG → CLIP encoder (requires preprocessed data)
.venv/bin/python -m models.train

# Retrieval generator spot-check on held-out trials
.venv/bin/python -m scripts.spot_check_retrieval --ckpt runs/<run-id>/fold_<sid>/best.ckpt

# OpenVINO vs. torch benchmark
.venv/bin/python -m scripts.bench_ov_vs_torch

# Demo app (two terminals). The app auto-discovers the latest checkpoint
# under runs/*/fold_*/best.ckpt, and runs with a random encoder if none exists.
.venv/bin/python -m streaming.lsl_streamer --speed 1.0 &
.venv/bin/streamlit run app/streamlit_app.py
```

## Quality gates and current results

Results are split into what has been **validated end-to-end** and what is **still gated**
on the real dataset. Synthetic runs exercise every code path (encoder, loss, LOSO loop,
eval, checkpointing) but do not stand in for a real accuracy number.

| Gate | Target | Current | Source |
|---|---|---|---|
| `pytest -m "not integration"` | all pass | **114 / 114 pass** | continuous |
| Overfit-100 sanity | train top-1 ≥ 0.95 | **1.000** | [results/sanity/summary.json](results/sanity/summary.json) |
| Synthetic LOSO end-to-end | clean run, no crashes | **3 folds × 5 epochs, completed** | [results/loso_synthetic/summary.json](results/loso_synthetic/summary.json) |
| OpenVINO bit-exactness vs. torch | atol 1e-3 over 100 trials | **passes** (max diff 2.98e-8) | [tests/test_openvino_export.py](tests/test_openvino_export.py) |
| OpenVINO speedup (median/median) | 2–5× | **1.89× CPU-only** (NPU driver absent on dev box) | [results/phase5_bench.json](results/phase5_bench.json) |
| End-to-end latency (retrieval mode) | p95 < 2 s | **p95 = 167.7 ms, median 51.9 ms** | [results/phase6_e2e/summary.json](results/phase6_e2e/summary.json) |
| Real-data LOSO top-5 ≥ 30 % | spec gate | ⏸ awaiting Spampinato dataset | — |
| End-to-end latency (SD-Turbo mode) | p95 < 60 s | ⏸ awaiting SD-Turbo weights | — |

## Roadmap

The engineering for every phase is in place; what remains is real-data and
capture work that needs artifacts not shipped in the repo:

1. **Fetch the Spampinato dataset** (manual Google Drive download). The downloaders print
   step-by-step instructions; re-run `setup.sh` after placing the `.pth` file under
   `data/raw/spampinato/`.
2. **Full leave-one-subject-out training** on real data, to fill in the top-5 accuracy gate:
   `.venv/bin/python -m models.train`.
3. **SD-Turbo generative path**, once the weights are cached:
   `pytest -m integration tests/test_sd_generator.py`.
4. **Spot-check and comparison grids** on real held-out trials via
   `scripts/spot_check_retrieval.py` and `scripts/compare_generators.py`.
5. **Record a short demo** of the Streamlit app for the README.

## Configuration

Every runtime parameter — paths, model IDs, hyperparameters, UI strings, colors, filter
cutoffs — lives in [config.yaml](config.yaml). Code never references literal values; the
[`utils.config`](utils/config.py) loader returns a frozen, pydantic-validated `Config`
object, and `tests/test_no_magic_strings.py` enforces this. To run with an alternate config,
set `VCR_CONFIG=/path/to/alt.yaml` or pass `path=` to `load_config()`.

## Tech stack

- **EEG / signal processing:** MNE, SciPy, `pylsl` (Lab Streaming Layer)
- **Representation learning:** PyTorch (CPU), CLIP (ViT-B/32) via `transformers`
- **Retrieval:** FAISS (`IndexFlatIP` over unit-norm embeddings)
- **Generation:** diffusers — SD-Turbo + IP-Adapter
- **Edge inference:** ONNX + OpenVINO (CPU / iGPU / NPU)
- **App:** Streamlit, Plotly, UMAP
- **Foundations:** pydantic config, pytest, TensorBoard

## Repository layout

```
.
├── config.yaml          # single source of truth for every parameter
├── requirements.txt     # pinned deps (CPU torch via the PyTorch CPU index)
├── setup.sh             # idempotent bootstrap (uses existing .venv/)
├── utils/               # config loader (pydantic), seeding, logging
├── data/                # dataset + ImageNet stimulus downloaders
├── preprocessing/       # data loader, filters, ICA, CLIP-bank precompute
├── streaming/           # virtual LSL EEG rig + receiver
├── models/              # encoder, losses, dataset, training, ONNX/OpenVINO export
├── generation/          # retrieval (FAISS) + SD-Turbo + IP-Adapter
├── evaluation/          # top-K, cosine, centroid purity, UMAP/t-SNE/confusion matrix
├── app/                 # Streamlit demo + components + theme
├── scripts/             # sanity / bench / spot-check / e2e drivers
├── tests/               # pytest suite + integration tests
└── results/             # tracked JSON evidence (checkpoints gitignored)
```

## Data & acknowledgements

- **EEG dataset:** [Spampinato et al., *Deep Learning Human Mind for Automated Visual Classification*](https://github.com/perceivelab/eeg_visual_classification) (primary), with [THINGS-EEG2](https://osf.io/anp5v/) as a fallback. Datasets are not redistributed here; download them from the original sources under their own licenses.
- **CLIP:** `openai/clip-vit-base-patch32` via Hugging Face `transformers`.
- **Generation:** `stabilityai/sd-turbo` with IP-Adapter, via `diffusers`.
- **Streaming:** [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer).

Please cite the dataset authors if you use their data.

## License

Released under the [MIT License](LICENSE).
