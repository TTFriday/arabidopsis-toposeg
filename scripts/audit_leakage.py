from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from toposeg.manifest import parse_folder_metadata


def thumbnail_correlation(a_path: Path, b_path: Path, size: tuple[int, int] = (96, 128)) -> float:
    with Image.open(a_path) as a, Image.open(b_path) as b:
        a_arr = np.asarray(a.convert("L").resize(size), dtype=np.float32)
        b_arr = np.asarray(b.convert("L").resize(size), dtype=np.float32)
    a_arr -= a_arr.mean()
    b_arr -= b_arr.mean()
    denom = np.sqrt((a_arr**2).sum() * (b_arr**2).sum())
    return float((a_arr * b_arr).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify leakage in the distributed ChronoRoot2 split")
    parser.add_argument("--data", type=Path, default=Path("../data/08_image_seedling_segmentation"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("manifests/leakage_audit.json"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    distributed = json.loads((args.data / "splits_final.json").read_text(encoding="utf-8"))[0]
    distributed_split: dict[str, str] = {}
    for split, images in distributed.items():
        for image in images:
            distributed_split[image] = split

    duplicates = 0
    for image_id in distributed.get("test", []):
        train_copy = args.data / "imagesTr" / f"{image_id}_0000.png"
        test_copy = args.data / "imagesTs" / f"{image_id}_0000.png"
        if train_copy.is_file() and test_copy.is_file():
            duplicates += int(train_copy.read_bytes() == test_copy.read_bytes())

    mapped = manifest[manifest["mapped"]]
    event_of_folder = {folder: parse_folder_metadata(folder)["event_id"] for folder in mapped["folder"].unique()}
    mapped = mapped.assign(event=mapped["folder"].map(event_of_folder))
    mapped = mapped.assign(distributed=mapped["image_id"].map(distributed_split).fillna("unmapped"))
    crossing_events = 0
    crossing_images = 0
    for event, group in mapped.groupby("event"):
        splits = set(group["distributed"])
        if len(splits - {"unmapped"}) > 1:
            crossing_events += 1
            crossing_images += len(group)

    adjacent = []
    for folder, group in mapped.groupby("folder"):
        ordered = group.sort_values(
            "image_id", key=lambda s: s.str.extract(r"(\d+)$")[0].astype(int)
        )
        paths = [Path(p) for p in ordered["image_path"]]
        ids = list(ordered["image_id"])
        events = list(ordered["event"])
        for index, (first, second) in enumerate(zip(paths, paths[1:])):
            value = thumbnail_correlation(first, second)
            adjacent.append({"folder": folder, "event_id": events[index], "image_a": ids[index], "image_b": ids[index + 1], "correlation": value})
    adjacent_frame = pd.DataFrame(adjacent)
    adjacent_values = adjacent_frame["correlation"].to_numpy()
    adjacent_path = args.output.with_name("adjacent_frame_correlation.csv")
    adjacent_frame.to_csv(adjacent_path, index=False)

    report = {
        "unique_images": int(len(manifest)),
        "mapped_images": int(len(mapped)),
        "distributed_test_images": len(distributed.get("test", [])),
        "distributed_test_images_identical_to_training_directory": duplicates,
        "acquisition_events_total": int(mapped["event"].nunique()),
        "events_crossing_distributed_train_val_test": crossing_events,
        "images_in_crossing_events": crossing_images,
        "images_in_crossing_events_share": round(crossing_images / len(mapped), 4),
        "adjacent_frame_pairs_measured": len(adjacent_values),
        "adjacent_frame_correlation_median": round(float(np.median(adjacent_values)), 4),
        "adjacent_frame_correlation_p90": round(float(np.percentile(adjacent_values, 90)), 4),
        "adjacent_frame_correlation_csv": adjacent_path.name,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
