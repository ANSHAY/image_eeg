"""Phase 4A spot-check — side-by-side grid for held-out trials.

For each of N held-out EEG trials, this script:

  1. Encodes the EEG through a trained EEGEncoder checkpoint.
  2. Retrieves the top-1 image via RetrievalGenerator.
  3. Tiles ``(ground_truth | retrieved | label_strip)`` into a grid.

Produces ``results/phase4a/spot_check.png`` + a per-trial metrics JSON.

Running with the synthetic LOSO checkpoint (Phase 3's plumbing-only
result) gives a layout demo, not a quality reading. Re-run after a
real-data Phase 3 training run for the true Phase 4A evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from generation.retrieval_generator import RetrievalGenerator
from models.dataset import EEGDataset, loso_split_indices
from models.eeg_encoder import EEGEncoder
from utils.config import load_config
from utils.logging import get_logger, setup_logging
from utils.seed import set_seed

log = get_logger(__name__)

_DEFAULT_N = 20
_TILE_PADDING = 6
_LABEL_HEIGHT = 28
_OUTPUT_DIR = "phase4a"
_OUTPUT_FILENAME = "spot_check.png"
_METRICS_FILENAME = "metrics.json"


def _load_encoder(ckpt_path: Path) -> EEGEncoder:
    cfg = load_config()
    model = EEGEncoder(cfg=cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    log.info("loaded encoder from %s", ckpt_path)
    return model


def _pick_indices(n_trials: int, held_out: Optional[int], cfg, n: int) -> np.ndarray:
    """Pick n indices from the held-out subject if specified, else random."""
    ds = EEGDataset(cfg=cfg)
    rng = np.random.default_rng(cfg.project.seed)
    if held_out is not None:
        _, val_idx = loso_split_indices(ds.subjects, held_out)
        chosen = rng.choice(val_idx, size=min(n, len(val_idx)), replace=False)
    else:
        chosen = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return np.sort(chosen)


def _label_strip(text: str, width: int, height: int = _LABEL_HEIGHT) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(16, 24, 48))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), text, fill=(230, 240, 255))
    return img


def _stack_pair(gt: Image.Image, pred: Image.Image, label: str) -> Image.Image:
    size = max(gt.size[0], pred.size[0])
    gt_r = gt.resize((size, size))
    pred_r = pred.resize((size, size))
    pair_w = size * 2 + _TILE_PADDING * 3
    pair_h = size + _LABEL_HEIGHT + _TILE_PADDING * 2
    canvas = Image.new("RGB", (pair_w, pair_h), color=(10, 15, 30))
    canvas.paste(gt_r, (_TILE_PADDING, _TILE_PADDING))
    canvas.paste(pred_r, (_TILE_PADDING * 2 + size, _TILE_PADDING))
    canvas.paste(_label_strip(label, pair_w), (0, pair_h - _LABEL_HEIGHT))
    return canvas


def _grid(pairs: list[Image.Image], cols: int = 4) -> Image.Image:
    rows = (len(pairs) + cols - 1) // cols
    pw, ph = pairs[0].size
    grid = Image.new("RGB", (pw * cols, ph * rows), color=(8, 12, 24))
    for i, pair in enumerate(pairs):
        r, c = divmod(i, cols)
        grid.paste(pair, (c * pw, r * ph))
    return grid


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a best.ckpt produced by models.train")
    p.add_argument("--n", type=int, default=_DEFAULT_N)
    p.add_argument("--held-out", type=int, default=None,
                   help="Restrict to one held-out subject's val trials.")
    args = p.parse_args()

    cfg = load_config()
    set_seed(cfg.project.seed)

    model = _load_encoder(Path(args.ckpt))
    retriever = RetrievalGenerator(cfg=cfg)
    ds = EEGDataset(cfg=cfg)

    chosen = _pick_indices(len(ds), args.held_out, cfg, args.n)
    log.info("evaluating %d trials (held_out=%s)", len(chosen), args.held_out)

    pairs: list[Image.Image] = []
    records: list[dict] = []
    hits = 0
    for i in chosen:
        sample = ds[int(np.where(ds.indices == i)[0][0])]
        eeg = sample["eeg"].unsqueeze(0)
        true_label = sample["label"]
        with torch.no_grad():
            z = model(eeg).cpu().numpy()[0]
        top = retriever.generate(z)
        gt_idx = int(np.where(retriever.labels == true_label)[0][0])
        gt_img = retriever._load_image(retriever.paths[gt_idx])
        match = (top["label"] == true_label)
        hits += int(match)
        records.append(
            {
                "trial_index": int(i),
                "true_label": true_label,
                "predicted_label": top["label"],
                "score": top["score"],
                "match": bool(match),
            },
        )
        label = (
            f"[trial {i}]  true={true_label}  pred={top['label']}  "
            f"score={top['score']:.3f}  {'OK' if match else 'X'}"
        )
        pairs.append(_stack_pair(gt_img, top["image"], label))

    out_dir = Path(cfg.paths.results) / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _grid(pairs).save(out_dir / _OUTPUT_FILENAME)
    summary = {
        "n_trials": len(chosen),
        "n_matches": hits,
        "match_rate": hits / max(1, len(chosen)),
        "held_out_subject": args.held_out,
        "checkpoint": str(args.ckpt),
        "per_trial": records,
    }
    (out_dir / _METRICS_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "spot-check complete: %d/%d match (%.0f%%); grid → %s",
        hits, len(chosen), 100 * summary["match_rate"], out_dir / _OUTPUT_FILENAME,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
