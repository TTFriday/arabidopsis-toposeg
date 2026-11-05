"""Write publishable split manifests: root-relative paths and opaque source-folder ids.

The working manifests store the absolute paths of this machine and the original source folder names,
which embed a third party's personal directory. Released manifests carry paths relative to the dataset
root (resolved through TOPOSEG_DATA_ROOT or SeedlingDataset(data_root=...)) and a folder_id whose
meaning is carried by the derived columns of folder_map.csv instead of by the raw name.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def folder_id(folder: object) -> str:
    if not isinstance(folder, str) or not folder:
        return "UNMAPPED"
    return "F" + hashlib.sha1(folder.encode("utf-8")).hexdigest()[:8].upper()


def relative(path: object, root: Path) -> str:
    resolved = Path(str(path))
    if not resolved.is_absolute():
        return str(resolved).replace("\\", "/")
    try:
        return str(resolved.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError as error:
        raise RuntimeError(f"{path} is not inside the dataset root {root}") from error


def export(splits: Path, data_root: Path, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, dict[str, object]] = {}
    for path in sorted(splits.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["image_path"] = frame.image_path.map(lambda p: relative(p, data_root))
        frame["label_path"] = frame.label_path.map(lambda p: relative(p, data_root))
        frame["folder_id"] = frame.folder.map(folder_id)
        for row in frame.itertuples(index=False):
            mappings[row.folder_id] = {
                "folder_id": row.folder_id,
                "condition": row.condition,
                "rpi": row.rpi,
                "timestamp": row.timestamp,
                "camera": row.camera,
                "is_multiple_vids": row.is_multiple_vids,
            }
        frame.drop(columns=["folder"]).to_csv(output / path.name, index=False)
    table = pd.DataFrame(sorted(mappings.values(), key=lambda entry: entry["folder_id"]))
    table.to_csv(output / "folder_map.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Write scrubbed split manifests for release")
    parser.add_argument("--splits", type=Path, default=Path("manifests/splits"))
    parser.add_argument("--data-root", type=Path, default=Path("../data/08_image_seedling_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("release/manifests"))
    args = parser.parse_args()
    table = export(args.splits, args.data_root, args.output)
    leaked = [path.name for path in args.output.glob("*.csv") if "Users" in path.read_text(encoding="utf-8")]
    if leaked:
        raise RuntimeError(f"personal directory names survived the scrub in {leaked}")
    print(f"{args.output}: {len(table)} source folders, paths relative to {args.data_root}")


if __name__ == "__main__":
    main()
