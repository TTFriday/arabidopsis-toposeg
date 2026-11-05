from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from toposeg.dataset import SeedlingDataset
from toposeg.engine import load_checkpoint
from toposeg.inference import sliding_window_inference
from toposeg.metrics import compute_metrics, mean_metrics
from toposeg.models import CompactUNet
from toposeg.phenotyping import phenotype_error_table
from toposeg.utils import load_config, atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-image sliding-window evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=["val", "test", "ood", "nominal"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images (smoke tests)")
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", args.config.stem)
    checkpoint = args.checkpoint or Path("outputs") / experiment / f"seed_{args.seed}" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = CompactUNet(base_channels=config["model"].get("base_channels", 24), skeleton_head=config["model"].get("skeleton_head", True))
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    split_csv = config["data"]["split_csv"]
    if args.split in {"ood", "nominal"}:
        split_csv = str(Path(split_csv).with_name(f"{args.split}.csv"))
        split_filter = args.split
    else:
        split_filter = args.split
    dataset = SeedlingDataset(split_csv, split_filter, training=False)
    output = args.output or Path("outputs") / experiment / f"seed_{args.seed}" / f"evaluation_{args.split}"
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions, targets, image_ids = [], [], [], []
    limit = args.limit if args.limit else len(dataset)
    for index in range(min(limit, len(dataset))):
        sample = dataset.full_image(index)
        prediction, _ = sliding_window_inference(model, sample["image"], config["inference"]["patch_size"], config["inference"].get("overlap", 0.5), device=device)
        pred_np, target_np = prediction.numpy().astype(np.uint8), sample["label"].numpy().astype(np.uint8)
        image_id = str(sample["image_id"])
        Image.fromarray(pred_np).save(prediction_dir / f"{image_id}.png")
        row = {"image_id": image_id, "experiment": experiment, "seed": args.seed, **compute_metrics(pred_np, target_np)}
        source_row = dataset.frame.iloc[index]
        row["event_id"] = str(source_row.event_id)
        row["condition"] = str(source_row.condition)
        rows.append(row)
        predictions.append(pred_np); targets.append(target_np); image_ids.append(image_id)
    if not rows:
        raise RuntimeError(f"No images found for split {args.split}")
    pd.DataFrame(rows).to_csv(output / "per_image_metrics.csv", index=False)
    phenotypes = phenotype_error_table(predictions, targets, image_ids)
    metadata = pd.DataFrame(rows)[["image_id", "event_id", "condition"]].drop_duplicates("image_id")
    phenotypes = phenotypes.assign(experiment=experiment, seed=args.seed).merge(metadata, on="image_id", how="left", validate="one_to_one")
    phenotypes.to_csv(output / "phenotype_errors.csv", index=False)
    atomic_write_json(mean_metrics(rows), output / "summary.json")
    print(output)


if __name__ == "__main__":
    main()
