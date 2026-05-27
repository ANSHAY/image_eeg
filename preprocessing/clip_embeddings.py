"""Precompute the frozen CLIP image bank.

Loads ``cfg.models.clip.hf_id`` once, encodes every unique stimulus
image referenced by the Spampinato trials, L2-normalizes the resulting
features, and writes three aligned artifacts under
``cfg.paths.data_processed``:

  - ``clip_image_emb.npy``    — float32, shape ``(N, embed_dim)``,
                                rows unit-norm (cosine ⇔ dot product).
  - ``clip_image_labels.npy`` — int64, shape ``(N,)``, class index.
  - ``clip_image_paths.json`` — list of N image paths, same order.

The CLIP model is the **frozen target distribution** for the EEG
encoder (Phase 3) and the retrieval database for the Phase 4 generator.
Never fine-tune it — that would shift the alignment target out from
under the encoder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from preprocessing.data_loader import SpampinatoLoader
from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

# Output filenames are protocol-level (the EEG encoder's training loop reads them
# by exact name); module-level constants keep them out of the user-tunable config.
_EMB_FILENAME = "clip_image_emb.npy"
_LABELS_FILENAME = "clip_image_labels.npy"
_PATHS_FILENAME = "clip_image_paths.json"


def _load_clip(cfg: Config) -> tuple[CLIPModel, CLIPProcessor]:
    """Load model + processor onto CPU in eval mode. Returns (model, processor)."""
    log.info("loading CLIP: %s", cfg.models.clip.hf_id)
    kwargs = {"revision": cfg.models.clip.revision} if cfg.models.clip.revision else {}
    model = CLIPModel.from_pretrained(cfg.models.clip.hf_id, **kwargs)
    processor = CLIPProcessor.from_pretrained(cfg.models.clip.hf_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def _encode_batch(
    paths: list[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> np.ndarray:
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def compute_clip_image_bank(
    cfg: Optional[Config] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute (embeddings, labels, paths) for every unique stimulus image.

    Image paths are deduped across trials — the same picture viewed by
    multiple subjects/trials is encoded once. Labels follow whichever
    trial's label was seen first for that path (they should agree by
    construction of the dataset).
    """
    cfg = cfg if cfg is not None else load_config()

    loader = SpampinatoLoader(cfg=cfg)
    trials = loader.load()

    # Dedup preserving insertion order.
    unique: dict[str, int] = {}
    for t in trials:
        if t.image_path not in unique:
            unique[t.image_path] = t.label
    paths = list(unique.keys())
    labels = np.array([unique[p] for p in paths], dtype=np.int64)
    log.info("encoding %d unique stimulus images", len(paths))

    model, processor = _load_clip(cfg)

    batch_size = cfg.models.clip.embed_batch_size
    embeds: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-embed"):
        batch_paths = [Path(p) for p in paths[i : i + batch_size]]
        embeds.append(_encode_batch(batch_paths, model, processor))

    embeddings = np.concatenate(embeds, axis=0)
    if embeddings.shape[1] != cfg.models.clip.embed_dim:
        raise ValueError(
            f"CLIP output dim {embeddings.shape[1]} != cfg.models.clip.embed_dim "
            f"{cfg.models.clip.embed_dim}. Update config or pick a matching model.",
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        max_dev = float(np.max(np.abs(norms - 1.0)))
        raise RuntimeError(f"CLIP embeddings not unit-norm; max |norm-1|={max_dev:.2e}")
    return embeddings, labels, paths


def save_image_bank(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    cfg: Optional[Config] = None,
) -> Path:
    """Write embeddings/labels/paths into ``cfg.paths.data_processed``."""
    cfg = cfg if cfg is not None else load_config()
    out_dir = Path(cfg.paths.data_processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / _EMB_FILENAME, embeddings)
    np.save(out_dir / _LABELS_FILENAME, labels)
    (out_dir / _PATHS_FILENAME).write_text(
        json.dumps(paths, indent=2), encoding="utf-8",
    )
    log.info(
        "wrote image bank: %d embeddings (dim=%d) to %s",
        len(embeddings), embeddings.shape[1], out_dir,
    )
    return out_dir


def load_image_bank(
    cfg: Optional[Config] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read back a previously-saved image bank. Validates unit-norm."""
    cfg = cfg if cfg is not None else load_config()
    out_dir = Path(cfg.paths.data_processed)
    emb = np.load(out_dir / _EMB_FILENAME)
    labels = np.load(out_dir / _LABELS_FILENAME)
    paths = json.loads((out_dir / _PATHS_FILENAME).read_text(encoding="utf-8"))
    if len(emb) != len(labels) or len(emb) != len(paths):
        raise ValueError(
            f"image bank misaligned: emb={len(emb)} labels={len(labels)} paths={len(paths)}",
        )
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("loaded image bank is not unit-norm; recompute it")
    return emb, labels, paths


def main() -> int:
    setup_logging()
    cfg = load_config()
    try:
        embeddings, labels, paths = compute_clip_image_bank(cfg)
    except FileNotFoundError as e:
        log.error("cannot compute image bank: %s", e)
        log.error("Run setup.sh to download the dataset first.")
        return 1
    save_image_bank(embeddings, labels, paths, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
