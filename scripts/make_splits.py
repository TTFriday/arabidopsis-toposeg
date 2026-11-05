from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from toposeg.constants import DEFAULT_SEED
from toposeg.splits import make_group_splits, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic event-grouped splits")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("manifests/splits"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    main_split, ood = make_group_splits(manifest, args.seed)
    write_splits(main_split, ood, args.output)
    class_columns = [column for column in main_split if column.startswith("has_class_")]
    report = main_split.groupby("split").agg(images=("image_id", "count"), events=("event_id", "nunique"), conditions=("condition", "nunique"), **{column: (column, "sum") for column in class_columns})
    print(report.to_string())
    print(f"ood: images={len(ood)}, events={ood.event_id.nunique()}")


if __name__ == "__main__":
    main()
