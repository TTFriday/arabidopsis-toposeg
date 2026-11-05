from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_window(size: int, sigma_scale: float = 0.125, device: torch.device | str = "cpu") -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    gaussian = torch.exp(-0.5 * (coords / max(1.0, size * sigma_scale)) ** 2)
    window = torch.outer(gaussian, gaussian)
    return (window / window.max()).clamp_min(1e-3)


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor | np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    num_classes: int = 7,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected HW, CHW, or single-image BCHW input")
    device = torch.device(device or next(model.parameters()).device)
    image = image.float()
    original_h, original_w = image.shape[-2:]
    pad_h, pad_w = max(0, patch_size - original_h), max(0, patch_size - original_w)
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect" if original_h > 1 and original_w > 1 else "replicate")
    h, w = padded.shape[-2:]
    stride = max(1, int(patch_size * (1 - overlap)))
    weight = gaussian_window(patch_size, device=device)[None, None]
    probability_sum = torch.zeros((1, num_classes, h, w), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    for y in _starts(h, patch_size, stride):
        for x in _starts(w, patch_size, stride):
            patch = padded[..., y:y + patch_size, x:x + patch_size].to(device)
            output = model(patch)
            logits = output["region"] if isinstance(output, dict) else output
            probabilities = logits.float().softmax(1)
            probability_sum[..., y:y + patch_size, x:x + patch_size] += probabilities * weight
            weight_sum[..., y:y + patch_size, x:x + patch_size] += weight
    probabilities = (probability_sum / weight_sum.clamp_min(1e-6))[..., :original_h, :original_w]
    prediction = probabilities.argmax(1)
    model.train(was_training)
    return prediction[0].cpu(), probabilities[0].cpu()
