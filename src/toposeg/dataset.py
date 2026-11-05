from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX, ROOT_CLASSES, TOPOLOGY_CLASSES

Sampling = Literal["random", "class", "skeleton", "boundary", "mixed"]
MIXED_PROBABILITIES = {"random": 0.10, "class": 0.30, "skeleton": 0.40, "boundary": 0.20}
DATA_ROOT_ENV = "TOPOSEG_DATA_ROOT"


def resolve_paths(frame: pd.DataFrame, data_root: str | Path | None) -> pd.DataFrame:
    """Join stored paths with the dataset root, leaving absolute paths untouched.

    Released manifests carry root-relative paths so that no local directory structure is published.
    """
    root = Path(os.environ.get(DATA_ROOT_ENV, "")) if data_root is None else Path(data_root)
    if not data_root and not os.environ.get(DATA_ROOT_ENV):
        return frame
    out = frame.copy()
    for column in ("image_path", "label_path"):
        if column in out:
            out[column] = out[column].map(lambda p, root=root: p if Path(str(p)).is_absolute() else str(root / str(p)))
    return out


def percentile_normalize(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, (low, high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def load_label(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.uint8)


def centerline_target(label: torch.Tensor) -> torch.Tensor:
    masks = [skeletonize((label == c).numpy()).astype(np.float32) for c in TOPOLOGY_CLASSES]
    return torch.from_numpy(np.stack(masks))


TUBE_STRUCTURE = ndimage.generate_binary_structure(2, 1)


def tubed_root_skeleton(label: torch.Tensor) -> torch.Tensor:
    """Skeleton of the primary+lateral root union, dilated by one pixel (Skeleton Recall target)."""
    roots = np.isin(label.numpy(), ROOT_CLASSES)
    tube = ndimage.binary_dilation(skeletonize(roots), structure=TUBE_STRUCTURE) if roots.any() else roots
    return torch.from_numpy(tube.astype(np.float32))[None]


def _boundary_pixels(label: np.ndarray) -> np.ndarray:
    mask = np.isin(label, range(1, 7))
    if not mask.any():
        return np.empty((0, 2), dtype=int)
    eroded = ndimage.binary_erosion(mask, iterations=3)
    ring = mask & ~eroded
    if not ring.any():
        ring = mask
    return np.argwhere(ring)


def _sample_center(label: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[int, int]:
    h, w = label.shape
    candidates = np.empty((0, 2), dtype=int)
    if mode == "class":
        present = [c for c in range(1, 7) if np.any(label == c)]
        if present:
            chosen = int(rng.choice(present))
            candidates = np.argwhere(label == chosen)
    elif mode == "skeleton":
        mask = np.isin(label, TOPOLOGY_CLASSES)
        if mask.any():
            candidates = np.argwhere(skeletonize(mask))
    elif mode == "boundary":
        candidates = _boundary_pixels(label)
    if len(candidates):
        return tuple(candidates[int(rng.integers(len(candidates)))])
    return int(rng.integers(h)), int(rng.integers(w))


def _crop_with_padding(image: np.ndarray, label: np.ndarray, center: tuple[int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = center
    top, left = y - size // 2, x - size // 2
    bottom, right = top + size, left + size
    pad = (max(0, -top), max(0, bottom - image.shape[0]), max(0, -left), max(0, right - image.shape[1]))
    if any(pad):
        image = np.pad(image, ((pad[0], pad[1]), (pad[2], pad[3])), mode="reflect")
        label = np.pad(label, ((pad[0], pad[1]), (pad[2], pad[3])), mode="constant", constant_values=IGNORE_INDEX)
        top += pad[0]
        left += pad[2]
    return image[top:top + size, left:left + size], label[top:top + size, left:left + size]


def _augment(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        image, label = image.flip(-1), label.flip(-1)
    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.95, 1.05)
    theta = torch.tensor([[
        [scale * np.cos(np.deg2rad(angle)), -scale * np.sin(np.deg2rad(angle)), 0.0],
        [scale * np.sin(np.deg2rad(angle)), scale * np.cos(np.deg2rad(angle)), 0.0],
    ]], dtype=torch.float32)
    grid = F.affine_grid(theta, image[None].shape, align_corners=False)
    image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
    label_sample = F.grid_sample(label[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0].long()
    in_bounds = F.grid_sample(torch.ones_like(label)[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0] > 0.5
    label = label_sample.masked_fill(~in_bounds, IGNORE_INDEX)
    if random.random() < 0.8:
        image = image * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
    if random.random() < 0.5:
        image = image.clamp(0, 1).pow(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        image = image + torch.randn_like(image) * random.uniform(0.0, 0.03)
    return image.clamp(0, 1), label


class SeedlingDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        training: bool = False,
        patch_size: int = 512,
        sampling: Sampling = "random",
        augment: bool = True,
        seed: int = 20261016,
        deterministic: bool = False,
        data_root: str | Path | None = None,
    ) -> None:
        self.frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.frame = resolve_paths(self.frame, data_root)
        if split is not None and "split" in self.frame:
            self.frame = self.frame[self.frame["split"] == split]
        self.frame = self.frame.reset_index(drop=True)
        self.training, self.patch_size = training, patch_size
        self.sampling, self.augment = sampling, augment
        self.seed, self.deterministic = seed, deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image = percentile_normalize(load_grayscale(row.image_path))
        label = load_label(row.label_path)
        if self.training:
            rng = np.random.default_rng([self.seed, index]) if self.deterministic else self.rng
            mode = self.sampling
            if mode == "mixed":
                modes = list(MIXED_PROBABILITIES)
                mode = str(rng.choice(modes, p=list(MIXED_PROBABILITIES.values())))
            center = _sample_center(label, mode, rng)
            image, label = _crop_with_padding(image, label, center, self.patch_size)
        image_t = torch.from_numpy(np.ascontiguousarray(image))[None].float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
        if self.training and self.augment:
            image_t, label_t = _augment(image_t, label_t)
        return {
            "image": image_t,
            "label": label_t,
            "centerline": centerline_target(label_t),
            "root_skeleton": tubed_root_skeleton(label_t),
            "image_id": str(row.image_id),
        }

    def full_image(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image = torch.from_numpy(percentile_normalize(load_grayscale(row.image_path)).copy())[None].float()
        label = torch.from_numpy(load_label(row.label_path).copy()).long()
        return {"image": image, "label": label, "image_id": str(row.image_id)}
