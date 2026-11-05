"""Build the diagnostic split that reproduces the published ChronoRoot test assignment.

The 156 files in ``imagesTs`` are byte-identical copies of files in ``imagesTr``, so the published
protocol evaluates models on images that are present in the training directory. ``nominal.csv`` lets
our checkpoints be scored on exactly that image set, to measure how much the published assignment
inflates reported accuracy. It is a diagnostic artifact, not a protocol split: no checkpoint is
selected on it, and its numbers are never reported as generalisation performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the published nominal test assignment")
    parser.add_argument("--data", type=Path, required=True, help="dataset root containing imagesTr and imagesTs")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest.csv built from imagesTr")
    parser.add_argument("--splits", type=Path, required=True, help="frozen main.csv used to attribute each image")
    parser.add_argument("--output", type=Path, required=True, help="directory for nominal.csv and the report")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["file_name"] = manifest.image_path.map(lambda p: Path(p).name)
    test_files = sorted(Path(p).name for p in args.data.glob("imagesTs/*"))
    if not test_files:
        raise RuntimeError(f"No imagesTs files found under {args.data}")

    identical, differing = [], []
    for name in test_files:
        published = args.data / "imagesTs" / name
        source = args.data / "imagesTr" / name
        if not source.is_file() or sha256(published) != sha256(source):
            differing.append(name)
        else:
            identical.append(name)
    if differing:
        raise RuntimeError(f"{len(differing)} imagesTs files are not copies of imagesTr files; example {differing[:3]}")

    selected = manifest[manifest.file_name.isin(identical)].copy()
    if len(selected) != len(identical):
        raise RuntimeError(f"Matched {len(selected)} manifest rows for {len(identical)} files")

    attribution = pd.concat([pd.read_csv(args.splits), pd.read_csv(args.splits.with_name("ood.csv"))], ignore_index=True)
    attribution["file_name"] = attribution.image_path.map(lambda p: Path(p).name)
    selected = selected.merge(attribution[["file_name", "split"]].rename(columns={"split": "frozen_split"}), on="file_name", validate="one_to_one")
    selected["split"] = "nominal"

    args.output.mkdir(parents=True, exist_ok=True)
    selected.drop(columns=["file_name"]).to_csv(args.output / "nominal.csv", index=False)

    report = {
        "published_test_images": len(test_files),
        "byte_identical_to_training_directory": len(identical),
        "frozen_protocol_attribution": selected.frozen_split.value_counts().to_dict(),
        "acquisition_events_in_nominal_set": int(selected.event_id.nunique()),
        "note": "nominal.csv reuses imagesTr paths because the published imagesTs bytes are identical",
    }
    (args.output / "nominal_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
