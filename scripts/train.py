from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from toposeg.constants import TOPOLOGY_CLASSES
from toposeg.dataset import SeedlingDataset, load_label
from toposeg.engine import fit
from toposeg.losses import TopologyAwareLoss
from toposeg.models import CompactUNet
from toposeg.utils import load_config, seed_everything


def compute_organ_weights(split_csv: str | Path, cache: str | Path) -> dict[str, float]:
    cache = Path(cache)
    if cache.is_file():
        return {int(k): float(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    frame = pd.read_csv(split_csv)
    frame = frame[frame["split"] == "train"] if "split" in frame else frame
    ratios: dict[int, list[float]] = {cls: [] for cls in TOPOLOGY_CLASSES}
    for path in frame["label_path"]:
        label = load_label(path)
        for cls in TOPOLOGY_CLASSES:
            mask = label == cls
            area = int(mask.sum())
            if area > 0:
                ratios[cls].append(float(skeletonize(mask).sum()) / area)
    slimness = {cls: float(np.median(values)) for cls, values in ratios.items() if values}
    if len(slimness) != len(TOPOLOGY_CLASSES):
        raise ValueError(f"Missing topology class in training split: {sorted(set(TOPOLOGY_CLASSES) - set(slimness))}")
    mean_slimness = float(np.mean(list(slimness.values())))
    adjusted = {cls: float(np.clip(value / mean_slimness, 0.5, 2.0)) for cls, value in slimness.items()}
    total = sum(adjusted.values())
    weights = {cls: value / total for cls, value in adjusted.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(cls): weight for cls, weight in weights.items()}), encoding="utf-8")
    return weights


def build_scheduler(optimizer: torch.optim.Optimizer, max_steps: int, warmup: int = 1000, min_lr_ratio: float = 1e-6 / 3e-4):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return 0.5 * (1 + np.cos(np.pi * progress)) * (1 - min_lr_ratio) + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seedling segmentation model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--final", action="store_true", help="Use training.final_steps")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.experiment:
        config["experiment"] = args.experiment
    if args.final:
        config["training"]["max_steps"] = config["training"]["final_steps"]
    if args.resume is not None and not args.resume.is_file():
        print(f"resume checkpoint missing, starting from scratch: {args.resume}")
        args.resume = None
    seed_everything(args.seed)
    data = config["data"]
    training = config["training"]

    loss_config = dict(config["loss"])
    if loss_config.get("organ_weights") == "adaptive":
        loss_config["organ_weights"] = compute_organ_weights(data["split_csv"], Path("manifests") / "organ_weights.json")

    train_set = SeedlingDataset(
        data["split_csv"], "train", training=True, patch_size=data["patch_size"],
        sampling=data.get("sampling", "mixed"), augment=data.get("augment", True), seed=args.seed,
    )
    val_set = SeedlingDataset(
        data["split_csv"], "val", training=True, patch_size=data["patch_size"],
        sampling="class", augment=False, seed=args.seed, deterministic=True,
    )
    loader_args = {"num_workers": data.get("workers", 0), "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, batch_size=data.get("batch_size", 2), shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=data.get("batch_size", 2), shuffle=False, **loader_args)
    model = CompactUNet(base_channels=config["model"].get("base_channels", 24), skeleton_head=config["model"].get("skeleton_head", True))
    criterion = TopologyAwareLoss(total_steps=training["max_steps"], **loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.get("learning_rate", 3e-4),
        weight_decay=training.get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, training["max_steps"], training.get("warmup_steps", 1000))
    experiment = config.get("experiment", args.config.stem)
    output_dir = Path("outputs") / experiment / f"seed_{args.seed}"
    best = fit(model, train_loader, val_loader, optimizer, criterion, config, output_dir, args.seed, args.resume, scheduler)
    print(best)


if __name__ == "__main__":
    main()
