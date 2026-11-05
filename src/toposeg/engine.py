from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .metrics import compute_metrics, mean_metrics
from .utils import atomic_save_torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _optional(batch: dict[str, Any], key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    return value.to(device, non_blocking=True) if value is not None else None


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return image, label, _optional(batch, "centerline", device), _optional(batch, "root_skeleton", device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    max_grad_norm: float = 1.0,
    global_step: int = 0,
    max_steps: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    amp_enabled = amp and device.type == "cuda"
    for batch in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        image, target, centerline, root_skeleton = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(image)
            loss, details = criterion(outputs, target, centerline, global_step, root_skeleton=root_skeleton)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for key, value in details.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        global_step += 1
    return ({key: value / max(1, count) for key, value in totals.items()}, global_step)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    global_step: int = 0,
    amp: bool = True,
) -> dict[str, float]:
    model.eval()
    loss_rows, metric_rows = [], []
    amp_enabled = amp and device.type == "cuda"
    for batch in loader:
        image, target, centerline, root_skeleton = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(image)
            _, details = criterion(outputs, target, centerline, global_step, root_skeleton=root_skeleton)
        loss_rows.append(details)
        predictions = outputs["region"].argmax(1).cpu().numpy()
        targets = target.cpu().numpy()
        metric_rows.extend(compute_metrics(pred, gt) for pred, gt in zip(predictions, targets))
    result = {f"val_{key}": value for key, value in mean_metrics(metric_rows).items()}
    result.update({f"val_{key}": float(np.mean([row[key] for row in loss_rows])) for key in loss_rows[0]} if loss_rows else {})
    return result


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: dict[str, Any],
    seed: int,
    epoch: int,
    global_step: int,
    best_score: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    atomic_save_torch({
        "config": config, "seed": seed, "epoch": epoch, "global_step": global_step,
        "best_score": best_score, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scaler: torch.amp.GradScaler | None = None, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: dict[str, Any],
    output_dir: str | Path,
    seed: int,
    resume: str | Path | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    amp = bool(config.get("training", {}).get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    start_epoch, global_step, best_score = 0, 0, -float("inf")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scaler, device)
        start_epoch, global_step, best_score = state["epoch"] + 1, state["global_step"], state.get("best_score", best_score)
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
    writer = SummaryWriter(output_dir / "tensorboard") if SummaryWriter is not None else None
    log_path = output_dir / "training.csv"
    best_path = output_dir / "best.pt"
    max_steps = int(config.get("training", {}).get("max_steps", 2000))
    patience = int(config.get("training", {}).get("patience", 20))
    epochs = int(config.get("training", {}).get("epochs", 9999))
    if global_step >= max_steps:
        if writer:
            writer.close()
        return best_path if best_path.is_file() else output_dir / "last.pt"
    stale = 0
    for epoch in range(start_epoch, epochs):
        started = time.time()
        train_stats, global_step = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp, float(config.get("training", {}).get("grad_clip", 1.0)), global_step, max_steps, scheduler)
        val_stats = validate(model, val_loader, criterion, device, global_step, amp)
        row = {"epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started, **train_stats, **val_stats}
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer_csv.writeheader()
            writer_csv.writerow(row)
        if writer:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_step)
        score = float(val_stats.get("val_complete_root_cldice", -val_stats.get("val_loss", 0.0)))
        if score > best_score:
            best_score, stale = score, 0
            save_checkpoint(best_path, model, optimizer, scaler, config, seed, epoch, global_step, best_score, scheduler)
        else:
            stale += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, scaler, config, seed, epoch, global_step, best_score, scheduler)
        if global_step >= max_steps or stale >= patience:
            break
    if writer:
        writer.close()
    return best_path
