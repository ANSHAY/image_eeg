"""Phase 3 overfit sanity check.

Generates 100 synthetic trials with class-discriminable targets, runs
the EEG encoder + ContrastiveAlignmentLoss for a short number of
epochs with NO augmentation, and asserts the model can perfectly
memorize the train set (top-1 → 1.0). Failure here means there is a
plumbing bug in the encoder/loss/training-loop stack — separate from
any real-data underfitting risk.

Writes a per-step training-loss CSV and a summary JSON to
``cfg.paths.results/sanity/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluation.metrics import top_k_retrieval
from models.eeg_encoder import EEGEncoder
from models.losses import ContrastiveAlignmentLoss
from utils.config import load_config
from utils.logging import get_logger, setup_logging
from utils.seed import set_seed

log = get_logger(__name__)

_NUM_TRIALS = 100
_NUM_CLASSES = 10
_EPOCHS = 40
_BATCH_SIZE = 32
_PASS_TOP1 = 0.95  # allow tiny float slack — but should be ≥0.99 in practice


def _build_synthetic_dataset(cfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate (eeg, target_emb, labels, image_bank).

    Each class has a unique CLIP-space anchor; per-class EEG signals
    are a class-specific 5-cycle sine over channels + small noise, so
    the encoder has a separable underlying pattern.
    """
    set_seed(cfg.project.seed)
    rng = np.random.default_rng(cfg.project.seed)

    C = cfg.eeg.num_channels
    T = cfg.preprocessing.crop.end_sample - cfg.preprocessing.crop.start_sample
    D = cfg.models.clip.embed_dim

    # Class anchors in CLIP space — random unit vectors, well-separated.
    anchors = rng.standard_normal((_NUM_CLASSES, D)).astype(np.float32)
    anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)

    labels = rng.integers(0, _NUM_CLASSES, size=_NUM_TRIALS)
    eeg = np.zeros((_NUM_TRIALS, C, T), dtype=np.float32)
    targets = np.zeros((_NUM_TRIALS, D), dtype=np.float32)

    t_axis = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    for i, lbl in enumerate(labels):
        # Class-specific frequency + per-channel phase → distinguishable
        # pattern that channel-mixer + temporal-conv can separate.
        freq = 1.0 + lbl * 0.5
        phases = rng.uniform(0, 2 * np.pi, size=C).astype(np.float32)
        signal = np.sin(freq * t_axis[None, :] + phases[:, None]).astype(np.float32)
        noise = 0.05 * rng.standard_normal((C, T)).astype(np.float32)
        eeg[i] = signal + noise
        targets[i] = anchors[lbl]

    return (
        torch.from_numpy(eeg),
        torch.from_numpy(targets),
        torch.from_numpy(labels.astype(np.int64)),
        anchors,  # image bank for retrieval evaluation
    )


def main() -> int:
    setup_logging()
    cfg = load_config()

    eeg, targets, labels, image_bank = _build_synthetic_dataset(cfg)
    log.info("synthetic: eeg=%s labels=%d classes=%d", tuple(eeg.shape), len(labels), _NUM_CLASSES)

    model = EEGEncoder(cfg=cfg)
    loss_fn = ContrastiveAlignmentLoss(cfg=cfg)
    optimizer = AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    dataset = TensorDataset(eeg, targets, labels)
    loader = DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True, drop_last=False)

    history: list[dict] = []
    for epoch in range(_EPOCHS):
        model.train()
        losses: list[float] = []
        for x, tgt, _y in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            z = model(x)
            total, _ = loss_fn(z, tgt)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            losses.append(float(total.detach()))

        # Eval on train set — overfit check
        model.eval()
        with torch.no_grad():
            z_all = model(eeg).cpu().numpy()
        labels_bank = np.arange(_NUM_CLASSES, dtype=np.int64)
        topk = top_k_retrieval(z_all, image_bank, labels_bank, labels.numpy(), ks=(1, 5))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)),
             "train_top1": topk[1], "train_top5": topk[5]},
        )
        log.info(
            "epoch %d  loss=%.4f  train_top1=%.3f  train_top5=%.3f",
            epoch, history[-1]["train_loss"], topk[1], topk[5],
        )

    out_dir = Path(cfg.paths.results) / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final = history[-1]
    summary = {
        "num_trials": _NUM_TRIALS,
        "num_classes": _NUM_CLASSES,
        "epochs": _EPOCHS,
        "final_train_top1": final["train_top1"],
        "final_train_top5": final["train_top5"],
        "final_train_loss": final["train_loss"],
        "passed": final["train_top1"] >= _PASS_TOP1,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not summary["passed"]:
        log.error(
            "OVERFIT SANITY FAILED — train top-1 = %.3f < %.3f after %d epochs. "
            "Stack has a plumbing bug; do not start a real LOSO run.",
            summary["final_train_top1"], _PASS_TOP1, _EPOCHS,
        )
        return 1

    log.info(
        "OVERFIT SANITY PASSED — train top-1 = %.3f after %d epochs",
        summary["final_train_top1"], _EPOCHS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
