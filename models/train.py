"""Training loop for the EEG encoder with LOSO cross-validation.

Per the implementation plan (§ Phase 3), we hold out one subject at a
time as the validation fold. For Spampinato that's 6 folds; metrics
are reported mean ± std across folds, and a final model is retrained
on all subjects after CV.

CLI:
  python -m models.train                       # full LOSO CV + final retrain
  python -m models.train --held-out 0          # one specific fold
  python -m models.train --max-epochs 5        # quick smoke
  python -m models.train --resume runs/<id>    # resume from checkpoint

Outputs per fold under ``cfg.paths.runs / <run_id> / fold_<sid>``:
  best.ckpt           — checkpoint with highest val top-5 so far
  last.ckpt           — most recent checkpoint (resume target)
  hparams.json        — full config snapshot + git SHA + pip freeze
  fold_metrics.json   — appended after each eval
  tensorboard/        — SummaryWriter logs

The final cross-fold table is written to ``<run_id>/loso_summary.json``
so README / docs can render the mean±std without rerunning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import (
    class_centroid_purity,
    cosine_similarity_pairs,
    top_k_retrieval,
)
from models.augmentations import make_train_augmentations
from models.dataset import EEGDataset, loso_split_indices
from models.eeg_encoder import EEGEncoder
from models.losses import ContrastiveAlignmentLoss
from preprocessing.clip_embeddings import load_image_bank
from utils.config import Config, load_config
from utils.logging import get_logger, setup_logging
from utils.seed import set_seed

log = get_logger(__name__)

_BEST_CKPT = "best.ckpt"
_LAST_CKPT = "last.ckpt"
_HPARAMS_FILE = "hparams.json"
_METRICS_FILE = "fold_metrics.json"
_LOSO_SUMMARY = "loso_summary.json"
_PIP_FREEZE_FILE = "pip_freeze.txt"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze(out_path: Path) -> None:
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL,
        ).decode()
        out_path.write_text(freeze, encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        out_path.write_text("# pip freeze unavailable\n", encoding="utf-8")


def _snapshot_hparams(cfg: Config, fold_dir: Path, extra: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.model_dump(),
        "git_sha": _git_sha(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    (fold_dir / _HPARAMS_FILE).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    _pip_freeze(fold_dir / _PIP_FREEZE_FILE)


@dataclass
class FoldResult:
    held_out_subject: Optional[int]
    best_val_top1: float
    best_val_top5: float
    best_val_top10: float
    best_val_cosine_mean: float
    best_val_centroid_purity: float
    best_epoch: int


def _build_optimizer(model: torch.nn.Module, loss_fn: torch.nn.Module, cfg: Config) -> AdamW:
    opt = cfg.training.optimizer
    params = list(model.parameters()) + list(loss_fn.parameters())
    return AdamW(
        params,
        lr=opt.lr,
        betas=tuple(opt.betas),
        weight_decay=opt.weight_decay,
    )


def _build_scheduler(optimizer: AdamW, cfg: Config) -> CosineAnnealingWarmRestarts:
    sch = cfg.training.scheduler
    return CosineAnnealingWarmRestarts(optimizer, T_0=sch.T_0, T_mult=sch.T_mult)


def _evaluate(
    model: EEGEncoder,
    val_loader: DataLoader,
    image_bank: np.ndarray,
    labels_bank: np.ndarray,
) -> dict[str, float]:
    """Top-K + cosine + centroid purity on the validation set."""
    model.eval()
    eeg_embs: list[np.ndarray] = []
    target_embs: list[np.ndarray] = []
    true_labels: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            z = model(batch["eeg"])
            eeg_embs.append(z.cpu().numpy())
            target_embs.append(batch["target"].cpu().numpy())
            true_labels.extend(batch["label"].tolist())
    z_eeg = np.concatenate(eeg_embs, axis=0).astype(np.float32)
    z_tgt = np.concatenate(target_embs, axis=0).astype(np.float32)
    y = np.asarray(true_labels, dtype=np.int64)
    # Clamp ks so the metric works on small banks (synthetic tests use
    # one row per class; real data has 2000+ rows so this is a no-op).
    bank_n = image_bank.shape[0]
    requested_ks = (1, 5, 10)
    feasible_ks = tuple(k for k in requested_ks if k <= bank_n)
    top_k = top_k_retrieval(z_eeg, image_bank, labels_bank, y, ks=feasible_ks)
    sims = cosine_similarity_pairs(z_eeg, z_tgt)
    purity, _ = class_centroid_purity(z_eeg, y, image_bank, labels_bank)
    return {
        "top1": top_k.get(1, float("nan")),
        "top5": top_k.get(5, float("nan")),
        "top10": top_k.get(10, float("nan")),
        "cosine_mean": float(sims.mean()),
        "cosine_median": float(np.median(sims)),
        "centroid_purity": purity,
    }


def _train_one_fold(
    cfg: Config,
    run_dir: Path,
    held_out: Optional[int],
    train_ds: EEGDataset,
    val_ds: EEGDataset,
    image_bank: np.ndarray,
    labels_bank: np.ndarray,
    max_epochs: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> FoldResult:
    fold_tag = f"fold_{held_out}" if held_out is not None else "final"
    fold_dir = run_dir / fold_tag
    fold_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_hparams(cfg, fold_dir, extra={"held_out_subject": held_out})
    writer = SummaryWriter(log_dir=str(fold_dir / "tensorboard"))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = EEGEncoder(cfg=cfg)
    loss_fn = ContrastiveAlignmentLoss(cfg=cfg)
    optimizer = _build_optimizer(model, loss_fn, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    start_epoch = 0
    best_top5 = -1.0
    best_record: dict = {}

    if resume_from is not None and resume_from.is_file():
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_top5 = float(ckpt.get("best_top5", -1.0))
        log.info("resumed from %s at epoch %d (best top5=%.4f)",
                 resume_from, start_epoch, best_top5)

    epochs = max_epochs if max_epochs is not None else cfg.training.epochs
    global_step = 0
    fold_metrics: list[dict] = []

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"[{fold_tag}] epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["eeg"]
            target = batch["target"]
            z = model(x)
            total, comp = loss_fn(z, target)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + (global_step / max(1, len(train_loader))))

            epoch_losses.append(float(total.detach()))
            if global_step % cfg.training.log_every_n_steps == 0:
                writer.add_scalar("train/total", float(total.detach()), global_step)
                writer.add_scalar("train/info_nce", float(comp["info_nce"]), global_step)
                writer.add_scalar("train/mse", float(comp["mse"]), global_step)
                writer.add_scalar("train/temperature", float(comp["temperature"]), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        if (epoch + 1) % cfg.training.eval_every_n_epochs == 0:
            metrics = _evaluate(model, val_loader, image_bank, labels_bank)
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            log.info(
                "[%s] epoch %d train_loss=%.4f  top1=%.3f top5=%.3f top10=%.3f cos=%.3f purity=%.3f",
                fold_tag, epoch, train_loss,
                metrics["top1"], metrics["top5"], metrics["top10"],
                metrics["cosine_mean"], metrics["centroid_purity"],
            )
            fold_metrics.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            (fold_dir / _METRICS_FILE).write_text(
                json.dumps(fold_metrics, indent=2), encoding="utf-8",
            )

            if metrics["top5"] > best_top5:
                best_top5 = metrics["top5"]
                best_record = {**metrics, "epoch": epoch}
                _save_ckpt(fold_dir / _BEST_CKPT, model, loss_fn, optimizer, scheduler, epoch, best_top5)

        # always write last.ckpt for resume
        _save_ckpt(fold_dir / _LAST_CKPT, model, loss_fn, optimizer, scheduler, epoch, best_top5)

    writer.close()
    return FoldResult(
        held_out_subject=held_out,
        best_val_top1=float(best_record.get("top1", 0.0)),
        best_val_top5=float(best_record.get("top5", 0.0)),
        best_val_top10=float(best_record.get("top10", 0.0)),
        best_val_cosine_mean=float(best_record.get("cosine_mean", 0.0)),
        best_val_centroid_purity=float(best_record.get("centroid_purity", 0.0)),
        best_epoch=int(best_record.get("epoch", -1)),
    )


def _save_ckpt(
    path: Path,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
    epoch: int,
    best_top5: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "loss": loss_fn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_top5": best_top5,
        },
        path,
    )


def run_loso(
    cfg: Config,
    held_out: Optional[int] = None,
    max_epochs: Optional[int] = None,
    resume_from: Optional[Path] = None,
    final_retrain: bool = True,
) -> dict:
    """Drive the LOSO cross-validation. Returns the summary dict."""
    set_seed(cfg.project.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    full_ds = EEGDataset(cfg=cfg)
    subjects_all = full_ds.subjects
    unique_subjects = sorted(int(s) for s in np.unique(subjects_all))
    log.info("subjects in dataset: %s", unique_subjects)

    image_bank, labels_bank, _ = load_image_bank(cfg)

    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(cfg.paths.runs) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run dir: %s", run_dir)

    folds_to_run = [held_out] if held_out is not None else unique_subjects
    fold_results: list[FoldResult] = []
    for sid in folds_to_run:
        train_idx, val_idx = loso_split_indices(subjects_all, sid)
        train_ds = EEGDataset(cfg=cfg, indices=train_idx, train_aug=make_train_augmentations(cfg))
        val_ds = EEGDataset(cfg=cfg, indices=val_idx)
        result = _train_one_fold(
            cfg, run_dir, sid, train_ds, val_ds,
            image_bank, labels_bank, max_epochs=max_epochs, resume_from=resume_from,
        )
        fold_results.append(result)

    summary = _summarize(fold_results)
    (run_dir / _LOSO_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("LOSO summary: top1 mean=%.3f±%.3f top5 mean=%.3f±%.3f",
             summary["top1_mean"], summary["top1_std"],
             summary["top5_mean"], summary["top5_std"])

    if final_retrain and held_out is None:
        log.info("training final model on all subjects")
        final_train_ds = EEGDataset(cfg=cfg, train_aug=make_train_augmentations(cfg))
        # validation = all data (just for monitoring) — no held-out subject
        final_val_ds = EEGDataset(cfg=cfg)
        _train_one_fold(
            cfg, run_dir, None, final_train_ds, final_val_ds,
            image_bank, labels_bank, max_epochs=max_epochs,
        )

    return summary


def _summarize(fold_results: list[FoldResult]) -> dict:
    def stats(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    top1_m, top1_s = stats([r.best_val_top1 for r in fold_results])
    top5_m, top5_s = stats([r.best_val_top5 for r in fold_results])
    top10_m, top10_s = stats([r.best_val_top10 for r in fold_results])
    cos_m, cos_s = stats([r.best_val_cosine_mean for r in fold_results])
    pur_m, pur_s = stats([r.best_val_centroid_purity for r in fold_results])
    return {
        "folds": [r.__dict__ for r in fold_results],
        "top1_mean": top1_m, "top1_std": top1_s,
        "top5_mean": top5_m, "top5_std": top5_s,
        "top10_mean": top10_m, "top10_std": top10_s,
        "cosine_mean": cos_m, "cosine_std": cos_s,
        "centroid_purity_mean": pur_m, "centroid_purity_std": pur_s,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG encoder with LOSO CV.")
    p.add_argument("--held-out", type=int, default=None,
                   help="Run only this subject as the held-out fold.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override cfg.training.epochs (smoke runs).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--skip-final", action="store_true",
                   help="Skip the final all-subjects retrain.")
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = _parse_args()
    cfg = load_config()
    try:
        run_loso(
            cfg,
            held_out=args.held_out,
            max_epochs=args.max_epochs,
            resume_from=Path(args.resume) if args.resume else None,
            final_retrain=not args.skip_final,
        )
        return 0
    except FileNotFoundError as e:
        log.error("training cannot start: %s", e)
        log.error("Run preprocessing.run_pipeline first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
