"""Phase 4B comparison — retrieval vs. SD-Turbo vs. ground truth.

For each of N held-out trials, this script:

  1. Runs the trained EEGEncoder over the trial.
  2. Retrieves top-1 via :class:`generation.retrieval_generator.RetrievalGenerator`.
  3. Generates via :class:`generation.sd_generator.SDGenerator`
     (lazy-loaded; the script unloads CLIP before triggering load to
     stay under the 32 GB RAM ceiling).
  4. Tiles ``(ground_truth | retrieval | sd_turbo)`` rows into a grid.

Outputs to ``results/phase4b/comparison.png`` + per-trial scores JSON.
Blocked by the same dependencies as Phase 3 box 10 (a real trained
encoder) and Phase 4B box 3 (SD-Turbo weights cached) — the script is
ready to run once both are in place.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from generation.retrieval_generator import RetrievalGenerator
from generation.sd_generator import SDGenerator
from models.dataset import EEGDataset, loso_split_indices
from models.eeg_encoder import EEGEncoder
from utils.config import load_config
from utils.logging import get_logger, setup_logging
from utils.seed import set_seed

log = get_logger(__name__)

_DEFAULT_N = 10
_OUTPUT_DIR = "phase4b"
_GRID_FILENAME = "comparison.png"
_METRICS_FILENAME = "metrics.json"
_TILE_PADDING = 6
_LABEL_HEIGHT = 28


def _load_encoder(ckpt_path: Path) -> EEGEncoder:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    log.info("encoder loaded from %s", ckpt_path)
    return model


def _label_strip(text: str, width: int, height: int = _LABEL_HEIGHT) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(16, 24, 48))
    ImageDraw.Draw(img).text((8, 6), text, fill=(230, 240, 255))
    return img


def _stack_row(images: list[Image.Image], label: str) -> Image.Image:
    size = max(im.size[0] for im in images)
    resized = [im.resize((size, size)) for im in images]
    row_w = size * len(images) + _TILE_PADDING * (len(images) + 1)
    row_h = size + _LABEL_HEIGHT + _TILE_PADDING * 2
    canvas = Image.new("RGB", (row_w, row_h), color=(10, 15, 30))
    for i, im in enumerate(resized):
        x = _TILE_PADDING * (i + 1) + size * i
        canvas.paste(im, (x, _TILE_PADDING))
    canvas.paste(_label_strip(label, row_w), (0, row_h - _LABEL_HEIGHT))
    return canvas


def _grid(rows: list[Image.Image]) -> Image.Image:
    rw, rh = rows[0].size
    grid = Image.new("RGB", (rw, rh * len(rows)), color=(8, 12, 24))
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * rh))
    return grid


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=_DEFAULT_N)
    p.add_argument("--held-out", type=int, default=None)
    args = p.parse_args()

    cfg = load_config()
    set_seed(cfg.project.seed)

    encoder = _load_encoder(Path(args.ckpt))
    retriever = RetrievalGenerator(cfg=cfg)

    ds = EEGDataset(cfg=cfg)
    rng = np.random.default_rng(cfg.project.seed)
    if args.held_out is not None:
        _, val_idx = loso_split_indices(ds.subjects, args.held_out)
        chosen = rng.choice(val_idx, size=min(args.n, len(val_idx)), replace=False)
    else:
        chosen = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    chosen = np.sort(chosen)

    # Step 1 — encode all queries while CLIP/encoder are in memory.
    queries: list[np.ndarray] = []
    truths: list[int] = []
    for i in chosen:
        sample = ds[int(np.where(ds.indices == i)[0][0])]
        with torch.no_grad():
            z = encoder(sample["eeg"].unsqueeze(0)).cpu().numpy()[0]
        queries.append(z)
        truths.append(sample["label"])

    # Step 2 — retrieval (still cheap, no SD yet).
    retrieved = [retriever.generate(z) for z in queries]

    # Step 3 — drop encoder before pulling in SD-Turbo (memory discipline).
    del encoder
    gc.collect()
    sd = SDGenerator(cfg=cfg)
    sd_images = [sd.generate(z) for z in queries]
    sd.unload()

    # Step 4 — tile rows.
    rows = []
    records = []
    for trial_idx, true_label, ret, sd_img in zip(chosen, truths, retrieved, sd_images):
        gt_idx = int(np.where(retriever.labels == true_label)[0][0])
        gt = retriever._load_image(retriever.paths[gt_idx])
        label = (
            f"[trial {trial_idx}]  true={true_label}  ret={ret['label']}  "
            f"sim={ret['score']:.3f}"
        )
        rows.append(_stack_row([gt, ret["image"], sd_img], label))
        records.append(
            {
                "trial_index": int(trial_idx),
                "true_label": int(true_label),
                "retrieval_label": ret["label"],
                "retrieval_score": ret["score"],
                "retrieval_match": ret["label"] == true_label,
            },
        )

    out_dir = Path(cfg.paths.results) / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _grid(rows).save(out_dir / _GRID_FILENAME)
    (out_dir / _METRICS_FILENAME).write_text(
        json.dumps(
            {
                "n_trials": len(records),
                "retrieval_match_rate": sum(r["retrieval_match"] for r in records) / max(1, len(records)),
                "held_out_subject": args.held_out,
                "checkpoint": str(args.ckpt),
                "per_trial": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("comparison grid → %s", out_dir / _GRID_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
