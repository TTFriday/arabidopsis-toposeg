"""Paths are machine-local in the working manifests and root-relative in the released ones."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.export_release_manifests import export
from toposeg.dataset import resolve_paths


@pytest.fixture
def working_manifest(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "image_id": ["image_1", "image_2"],
            "image_path": [str(root / "imagesTr" / "image_1_0000.png"), str(root / "imagesTr" / "image_2_0001.png")],
            "label_path": [str(root / "labelsTr" / "image_1.png"), str(root / "labelsTr" / "image_2.png")],
            "folder": ["Manual_Annotation__Users_operator_Desktop_Roots_rpi101_2024-01-01_10-00_1",
                       "Manual_Annotation__Users_operator_Desktop_Roots_rpi101_2024-01-01_10-00_2"],
            "condition": ["Roots", "Roots"],
            "rpi": ["rpi101", "rpi101"],
            "timestamp": ["2024-01-01_10-00", "2024-01-01_10-00"],
            "camera": ["1", "2"],
            "is_multiple_vids": [False, False],
            "event_id": ["Roots|rpi101|2024-01-01_10-00", "Roots|rpi101|2024-01-01_10-00"],
            "split": ["train", "test"],
        }
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    frame.to_csv(splits / "main.csv", index=False)
    return splits, root, frame


def test_release_scrubs_paths_and_folder_names(working_manifest, tmp_path):
    splits, root, _ = working_manifest
    output = tmp_path / "release"
    table = export(splits, root, output)
    released = pd.read_csv(output / "main.csv")
    assert "folder" not in released.columns
    assert not any(str(value).startswith(str(root)) for value in released.image_path)
    assert released.image_path.tolist() == ["imagesTr/image_1_0000.png", "imagesTr/image_2_0001.png"]
    assert released.label_path.tolist() == ["labelsTr/image_1.png", "labelsTr/image_2.png"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.glob("*.csv"))
    assert "operator" not in text and "Users_" not in text and str(root) not in text
    mapping = pd.read_csv(output / "folder_map.csv")
    assert len(mapping) == 2 and mapping.folder_id.str.startswith("F").all()
    assert mapping.condition.tolist() == ["Roots", "Roots"]


def test_relative_paths_resolve_against_the_data_root(working_manifest, tmp_path):
    splits, root, frame = working_manifest
    export(splits, root, tmp_path / "release")
    joined = resolve_paths(frame, root)
    assert joined.image_path.iloc[0].replace("\\", "/").endswith("dataset/imagesTr/image_1_0000.png")
    absolute = resolve_paths(frame, tmp_path)
    assert absolute.image_path.iloc[0] == str(root / "imagesTr" / "image_1_0000.png")


def test_without_a_data_root_paths_are_left_alone():
    frame = pd.DataFrame({"image_path": ["imagesTr/image_1_0000.png"]})
    assert resolve_paths(frame, None).image_path.iloc[0] == "imagesTr/image_1_0000.png"
