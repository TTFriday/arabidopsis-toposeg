from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from toposeg.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training images and labels")
    parser.add_argument("--data", type=Path, default=Path("../data/08_image_seedling_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("manifests"))
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data, args.output / "manifest.csv", compute_hashes=not args.skip_hashes)
    class_columns = [column for column in manifest if column.startswith("has_class_")]
    condition_stats = manifest.groupby(["condition", "mapped"], dropna=False).agg(images=("image_id", "count"), events=("event_id", "nunique"), **{column: (column, "sum") for column in class_columns}).reset_index()
    condition_stats.to_csv(args.output / "audit_by_condition.csv", index=False)
    summary = {
        "images": int(len(manifest)),
        "unique_images": int(manifest.image_id.nunique()),
        "events": int(manifest.event_id.nunique()),
        "mapped": int(manifest.mapped.sum()),
        "unmapped": int((~manifest.mapped).sum()),
        "multiple_vids": int(manifest.is_multiple_vids.sum()),
        "image_hash_duplicates": int(manifest.image_sha256.duplicated().sum()) if not args.skip_hashes else None,
        "label_hash_duplicates": int(manifest.label_sha256.duplicated().sum()) if not args.skip_hashes else None,
        "class_positive_images": {column: int(manifest[column].sum()) for column in class_columns},
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
