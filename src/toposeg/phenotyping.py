from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize

from .constants import ROOT_CLASSES


def skeleton_length(mask: np.ndarray) -> float:
    skeleton = skeletonize(mask.astype(bool))
    horizontal = np.sum(skeleton[:, 1:] & skeleton[:, :-1])
    vertical = np.sum(skeleton[1:, :] & skeleton[:-1, :])
    diagonal_a = np.sum(skeleton[1:, 1:] & skeleton[:-1, :-1])
    diagonal_b = np.sum(skeleton[1:, :-1] & skeleton[:-1, 1:])
    return float(horizontal + vertical + np.sqrt(2.0) * (diagonal_a + diagonal_b))


def skeleton_points(mask: np.ndarray) -> tuple[int, int]:
    skeleton = skeletonize(mask.astype(bool))
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = int(np.sum(skeleton & (neighbors == 1)))
    branches = int(ndimage.label(skeleton & (neighbors >= 3), structure=np.ones((3, 3)))[1])
    return endpoints, branches


def phenotype(mask: np.ndarray) -> dict[str, float | int]:
    main = mask == ROOT_CLASSES[0]
    lateral = mask == ROOT_CLASSES[1]
    root = main | lateral
    endpoints, branches = skeleton_points(root)
    return {
        "main_root_length_px": skeleton_length(main),
        "lateral_root_length_px": skeleton_length(lateral),
        "complete_root_length_px": skeleton_length(root),
        "root_endpoints": endpoints,
        "root_branch_points": branches,
    }


def phenotype_error_table(predictions: list[np.ndarray], targets: list[np.ndarray], image_ids: list[str] | None = None) -> pd.DataFrame:
    if len(predictions) != len(targets):
        raise ValueError("Prediction/target length mismatch")
    image_ids = image_ids or [str(i) for i in range(len(predictions))]
    rows = []
    for image_id, prediction, target in zip(image_ids, predictions, targets):
        pred_values, gt_values = phenotype(prediction), phenotype(target)
        row: dict[str, float | str] = {"image_id": image_id}
        for key in gt_values:
            row[f"pred_{key}"] = pred_values[key]
            row[f"gt_{key}"] = gt_values[key]
            row[f"error_{key}"] = float(pred_values[key]) - float(gt_values[key])
            row[f"absolute_error_{key}"] = abs(float(row[f"error_{key}"]))
        rows.append(row)
    return pd.DataFrame(rows)
