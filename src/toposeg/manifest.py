from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import NUM_CLASSES

_CONDITION_RE = re.compile(r"ArabidopsisDataset_(?P<condition>[^_]+)")
_RPI_RE = re.compile(r"(?:^|_)(?P<rpi>rpi\d+|test)(?:_|$)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"(?P<date>20\d{2}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2})")
_CAMERA_RE = re.compile(r"_(?:cam)?(?P<camera>[1-4])$", re.IGNORECASE)
_IMAGE_ID_RE = re.compile(r"^(image_\d+)(?:_0000)?$")


def parse_folder_metadata(folder: str) -> dict[str, str | bool]:
    condition_match = _CONDITION_RE.search(folder)
    rpi_match = _RPI_RE.search(folder)
    timestamp_match = _TIMESTAMP_RE.search(folder)
    camera_match = _CAMERA_RE.search(folder)
    condition = condition_match.group("condition") if condition_match else "unknown"
    rpi = rpi_match.group("rpi").lower() if rpi_match else "unknown"
    timestamp = (
        f"{timestamp_match.group('date')}_{timestamp_match.group('time')}"
        if timestamp_match
        else "unknown"
    )
    is_multiple = "multiplevideos" in folder.lower() or "multiplevids" in folder.lower()
    camera = camera_match.group("camera") if camera_match else "unknown"
    if is_multiple:
        event_id = f"{condition}|MultipleVids|{folder.rsplit('_', 1)[-1]}"
    elif rpi != "unknown" or timestamp != "unknown":
        event_id = f"{condition}|{rpi}|{timestamp}"
    else:
        event_id = _CAMERA_RE.sub("", folder)
    return {
        "condition": condition,
        "rpi": rpi,
        "timestamp": timestamp,
        "camera": camera,
        "event_id": event_id,
        "is_multiple_vids": is_multiple,
    }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(path: Path) -> str:
    match = _IMAGE_ID_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected image name: {path.name}")
    return match.group(1)


def build_manifest(
    data_dir: str | Path,
    output_csv: str | Path | None = None,
    expected_count: int | None = 957,
    compute_hashes: bool = True,
) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"
    map_path = data_dir / "folder_image_map.json"
    with map_path.open("r", encoding="utf-8") as f:
        folder_map = json.load(f)

    id_to_folder: dict[str, str] = {}
    duplicate_mappings: set[str] = set()
    for folder, image_ids in folder_map.items():
        for image_id in image_ids:
            canonical = _canonical_id(Path(image_id))
            if canonical in id_to_folder and id_to_folder[canonical] != folder:
                duplicate_mappings.add(canonical)
            id_to_folder[canonical] = folder
    if duplicate_mappings:
        raise ValueError(f"Images mapped to multiple folders: {sorted(duplicate_mappings)[:5]}")

    image_paths = sorted(image_dir.glob("*.png"), key=lambda p: int(_canonical_id(p).split("_")[1]))
    if expected_count is not None and len(image_paths) != expected_count:
        raise AssertionError(f"Expected {expected_count} training images, found {len(image_paths)}")
    rows = []
    for image_path in image_paths:
        image_id = _canonical_id(image_path)
        label_path = label_dir / f"{image_id}.png"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        folder = id_to_folder.get(image_id, "")
        mapped = bool(folder)
        metadata = parse_folder_metadata(folder) if mapped else {
            "condition": "unmapped", "rpi": "unknown", "timestamp": "unknown",
            "camera": "unknown", "event_id": f"unmapped|{image_id}", "is_multiple_vids": False,
        }
        with Image.open(image_path) as im:
            width, height = im.size
        with Image.open(label_path) as label_im:
            label = np.asarray(label_im)
        values = set(np.unique(label).tolist())
        if not values.issubset(set(range(NUM_CLASSES + 1))):
            raise ValueError(f"Invalid labels in {label_path}: {sorted(values)}")
        row = {
            "image_id": image_id,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "folder": folder,
            "mapped": mapped,
            **metadata,
            "width": width,
            "height": height,
            "image_sha256": sha256_file(image_path) if compute_hashes else "",
            "label_sha256": sha256_file(label_path) if compute_hashes else "",
        }
        row.update({f"has_class_{c}": int(c in values) for c in range(1, NUM_CLASSES)})
        rows.append(row)
    manifest = pd.DataFrame(rows)
    if manifest["image_id"].duplicated().any():
        raise AssertionError("Manifest image IDs are not unique")
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_csv, index=False)
    return manifest


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_id", "image_path", "label_path", "event_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return frame
