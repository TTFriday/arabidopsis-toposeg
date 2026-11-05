from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

from .constants import IGNORE_INDEX, NUM_CLASSES, ROOT_CLASSES
from .phenotyping import skeleton_length


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes) & (prediction >= 0) & (prediction < num_classes)
    indices = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def class_metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for cls in range(matrix.shape[0]):
        tp = float(matrix[cls, cls])
        fp = float(matrix[:, cls].sum() - tp)
        fn = float(matrix[cls, :].sum() - tp)
        gt_empty, pred_empty = tp + fn == 0, tp + fp == 0
        dice = 1.0 if gt_empty and pred_empty else 2 * tp / max(1.0, 2 * tp + fp + fn)
        iou = 1.0 if gt_empty and pred_empty else tp / max(1.0, tp + fp + fn)
        precision = 1.0 if pred_empty and gt_empty else tp / max(1.0, tp + fp)
        recall = 1.0 if gt_empty else tp / max(1.0, tp + fn)
        values.update({f"dice_class_{cls}": dice, f"iou_class_{cls}": iou, f"precision_class_{cls}": precision, f"recall_class_{cls}": recall})
    return values


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def cldice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    pred_skeleton, target_skeleton = skeletonize(prediction), skeletonize(target)
    precision = np.logical_and(pred_skeleton, target).sum() / max(1, pred_skeleton.sum())
    sensitivity = np.logical_and(target_skeleton, prediction).sum() / max(1, target_skeleton.sum())
    return float(2 * precision * sensitivity / max(1e-12, precision + sensitivity))


def connected_component_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, pred_count = ndimage.label(prediction.astype(bool), structure)
    target_labels, target_count = ndimage.label(target.astype(bool), structure)
    component_error = float(abs(pred_count - target_count))
    if target_count == 0:
        largest_recall = 1.0 if pred_count == 0 else 0.0
    else:
        sizes = np.bincount(target_labels.ravel())[1:]
        largest_id = int(np.argmax(sizes)) + 1
        largest = target_labels == largest_id
        largest_recall = float(np.logical_and(prediction, largest).sum() / max(1, largest.sum()))
    return {"component_count_error": component_error, "largest_component_recall": largest_recall}


def _point_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skeleton = skeletonize(mask.astype(bool))
    count = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = skeleton & (count == 1)
    branches = skeleton & (count >= 3)
    branches = ndimage.label(branches, np.ones((3, 3), dtype=np.uint8))[0]
    branch_centers = np.zeros_like(skeleton)
    for index in range(1, branches.max() + 1):
        coordinates = np.argwhere(branches == index)
        if len(coordinates):
            branch_centers[tuple(np.rint(coordinates.mean(0)).astype(int))] = True
    return endpoints, branch_centers


def tolerant_point_metrics(predicted_points: np.ndarray, target_points: np.ndarray, tolerance: float = 5.0) -> dict[str, float]:
    predicted, target = np.argwhere(predicted_points), np.argwhere(target_points)
    if not len(predicted) and not len(target):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "count_error": 0.0}
    if not len(predicted) or not len(target):
        return {"precision": 0.0 if len(predicted) else 1.0, "recall": 0.0 if len(target) else 1.0, "f1": 0.0, "count_error": float(abs(len(predicted) - len(target)))}
    distances = cdist(predicted, target)
    rows, columns = linear_sum_assignment(distances)
    true_positive = int(np.sum(distances[rows, columns] <= tolerance))
    precision = true_positive / len(predicted)
    recall = true_positive / len(target)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "count_error": float(abs(len(predicted) - len(target)))}


def surface_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return math.nan, math.nan
    structure = np.ones((3, 3), dtype=bool)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    target_distance = ndimage.distance_transform_edt(~target_surface)
    pred_distance = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def compute_metrics(prediction: np.ndarray, target: np.ndarray, point_tolerance: float = 5.0) -> dict[str, float]:
    prediction, target = np.asarray(prediction).copy(), np.asarray(target)
    prediction[target == IGNORE_INDEX] = 0
    valid_target = target.copy()
    matrix = confusion_matrix(prediction, valid_target)
    result = class_metrics_from_confusion(matrix)
    valid = target != IGNORE_INDEX
    root_pred = np.isin(prediction, ROOT_CLASSES) & valid
    root_target = np.isin(target, ROOT_CLASSES) & valid
    result["complete_root_dice"] = binary_dice(root_pred, root_target)
    result["complete_root_cldice"] = cldice_score(root_pred, root_target)
    result.update(connected_component_metrics(root_pred, root_target))
    target_length = skeleton_length(root_target)
    result["skeleton_length_error_px"] = float(skeleton_length(root_pred) - target_length)
    result["skeleton_length_absolute_error_px"] = abs(result["skeleton_length_error_px"])
    pred_end, pred_branch = _point_masks(root_pred)
    target_end, target_branch = _point_masks(root_target)
    for name, pred_points, target_points in (("endpoint", pred_end, target_end), ("branchpoint", pred_branch, target_branch)):
        for metric, value in tolerant_point_metrics(pred_points, target_points, point_tolerance).items():
            result[f"{name}_{metric}"] = value
    result["root_hd95_px"], result["root_asd_px"] = surface_distances(root_pred, root_target)
    return result


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {key: float(np.nanmean([row.get(key, np.nan) for row in rows])) for key in numeric}
