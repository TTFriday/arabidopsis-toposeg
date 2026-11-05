from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .constants import IGNORE_INDEX, NUM_CLASSES, ROOT_CLASSES, TOPOLOGY_CLASSES


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("Expected BCHW tensor")
    eroded_h = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    eroded_w = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_h, eroded_w)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = image.float()
        opened = soft_open(image)
        skeleton = F.relu(image - opened)
        for _ in range(iterations):
            image = soft_erode(image)
            delta = F.relu(image - soft_open(image))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton


def soft_dice(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if mask is not None:
        prob, target = prob * mask, target * mask
    intersection = (prob * target).sum(dim=(-2, -1))
    denominator = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (2 * intersection + eps) / (denominator + eps)


def soft_cldice(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None, iterations: int = 10) -> torch.Tensor:
    with torch.autocast(device_type=prob.device.type, enabled=False):
        prob, target = prob.float(), target.float()
        if valid is not None:
            prob, target = prob * valid.float(), target * valid.float()
        skel_prob = soft_skeletonize(prob, iterations)
        skel_target = soft_skeletonize(target, iterations)
        topology_precision = (skel_prob * target).sum((-2, -1)) / (skel_prob.sum((-2, -1)) + 1e-6)
        topology_sensitivity = (skel_target * prob).sum((-2, -1)) / (skel_target.sum((-2, -1)) + 1e-6)
        return (2 * topology_precision * topology_sensitivity + 1e-6) / (topology_precision + topology_sensitivity + 1e-6)


def soft_skeleton_recall(prob: torch.Tensor, tubed_skeleton: torch.Tensor, valid: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Skeleton Recall (Kirchhoff et al., ECCV 2024): fraction of the tubed GT skeleton covered by prob."""
    with torch.autocast(device_type=prob.device.type, enabled=False):
        prob, tubed_skeleton = prob.float(), tubed_skeleton.float()
        if valid is not None:
            prob, tubed_skeleton = prob * valid.float(), tubed_skeleton * valid.float()
        return ((prob * tubed_skeleton).sum((-2, -1)) + eps) / (tubed_skeleton.sum((-2, -1)) + eps)


def _boundary_mask(target: torch.Tensor, border: int) -> torch.Tensor:
    valid = (target != IGNORE_INDEX).float().unsqueeze(1)
    if border > 0:
        valid[..., :border, :] = 0
        valid[..., -border:, :] = 0
        valid[..., :, :border] = 0
        valid[..., :, -border:] = 0
    return valid


def ramp_schedule(progress: float, start: float = 0.10, end: float = 0.25) -> float:
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / (end - start)


class TopologyAwareLoss(nn.Module):
    """L = 0.5 CE + 0.5 Dice + g(tau) * (w_topo * L_topo + w_aux * L_skel).

    L_topo = joint_root * (1 - clDice_root) + (1 - joint_root) * sum_c alpha_c (1 - clDice_c)
    with per-organ alpha renormalised over classes present in the batch.
    organ_weights: "adaptive" uses supplied weights, "uniform" uses 1/n, "root_only"
    reduces L_topo to the complete-root clDice (standard clDice baseline).
    topology_form="skeleton_recall" (root_only only) replaces 1 - clDice_root by
    1 - SkeletonRecall_root on the tubed skeleton of the root union.
    """

    def __init__(
        self,
        region_ce_weight: float = 0.5,
        region_dice_weight: float = 0.5,
        topology_weight: float = 0.20,
        auxiliary_weight: float = 0.10,
        joint_root_weight: float = 0.3,
        organ_weights: dict[int, float] | str = "uniform",
        topology_ramp_start: float = 0.10,
        topology_ramp_end: float = 0.25,
        skeleton_iterations: int = 10,
        boundary: int = 32,
        total_steps: int = 60000,
        topology_form: str = "cldice",
    ) -> None:
        super().__init__()
        if topology_form not in ("cldice", "skeleton_recall"):
            raise ValueError(f"Unknown topology_form: {topology_form}")
        if topology_form == "skeleton_recall" and organ_weights != "root_only":
            raise ValueError("skeleton_recall is defined on the complete root only; use organ_weights=root_only")
        self.topology_form = topology_form
        if isinstance(organ_weights, str):
            if organ_weights not in ("uniform", "adaptive", "root_only"):
                raise ValueError(f"Unknown organ_weights mode: {organ_weights}")
            self.organ_weights_mode = organ_weights
            self.organ_weights: dict[int, float] = {}
        else:
            self.organ_weights_mode = "adaptive"
            self.organ_weights = {int(k): float(v) for k, v in organ_weights.items()}
            missing = set(TOPOLOGY_CLASSES) - set(self.organ_weights)
            if missing:
                raise ValueError(f"organ_weights missing classes: {sorted(missing)}")
        self.region_ce_weight = region_ce_weight
        self.region_dice_weight = region_dice_weight
        self.topology_weight = topology_weight
        self.auxiliary_weight = auxiliary_weight
        self.joint_root_weight = joint_root_weight
        self.topology_ramp_start = topology_ramp_start
        self.topology_ramp_end = topology_ramp_end
        self.skeleton_iterations = skeleton_iterations
        self.boundary = boundary
        self.total_steps = total_steps

    def topology_scale(self, step: int) -> float:
        progress = min(1.0, max(0.0, step / max(1, self.total_steps)))
        return ramp_schedule(progress, self.topology_ramp_start, self.topology_ramp_end)

    def _organ_loss(self, probs: torch.Tensor, one_hot: torch.Tensor, topology_valid: torch.Tensor, adaptive: bool) -> torch.Tensor:
        per_class: list[tuple[float, torch.Tensor]] = []
        for cls in TOPOLOGY_CLASSES:
            gt = one_hot[:, cls : cls + 1]
            present = gt.sum((-2, -1)) > 0
            if not present.any():
                continue
            score = soft_cldice(probs[:, cls : cls + 1], gt, topology_valid, self.skeleton_iterations)[present].mean()
            if adaptive and self.organ_weights_mode != "uniform":
                weight = self.organ_weights.get(cls, 1.0)
            else:
                weight = 1.0
            per_class.append((weight, 1 - score))
        if not per_class:
            return probs.sum() * 0
        total_weight = sum(weight for weight, _ in per_class)
        return sum(weight * term for weight, term in per_class) / total_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        target: torch.Tensor,
        centerline_target: torch.Tensor | None = None,
        step: int = 0,
        root_skeleton: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["region"]
        assert logits is not None
        valid = (target != IGNORE_INDEX).float()
        ce_map = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="none")
        ce = (ce_map * valid).sum() / valid.sum().clamp_min(1.0)
        probs = logits.softmax(1)
        safe_target = target.masked_fill(target == IGNORE_INDEX, 0)
        one_hot = F.one_hot(safe_target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        present_dice = []
        for cls in range(1, NUM_CLASSES):
            gt = one_hot[:, cls : cls + 1]
            present = gt.sum((-2, -1)) > 0
            if present.any():
                score = soft_dice(probs[:, cls : cls + 1], gt, valid[:, None])
                present_dice.append(score[present].mean())
        dice_loss = 1 - torch.stack(present_dice).mean() if present_dice else probs.sum() * 0

        topology_valid = _boundary_mask(target, self.boundary)
        root_prob = probs[:, ROOT_CLASSES].sum(1, keepdim=True)
        root_gt = one_hot[:, ROOT_CLASSES].sum(1, keepdim=True).clamp_max(1)
        root_present = root_gt.sum((-2, -1)) > 0
        root_cldice = (
            soft_cldice(root_prob, root_gt, topology_valid, self.skeleton_iterations)[root_present].mean()
            if root_present.any()
            else probs.sum() * 0 + 1
        )
        if self.topology_form == "skeleton_recall":
            if root_skeleton is None:
                raise ValueError("skeleton_recall needs the tubed root skeleton from the dataset")
            recall = soft_skeleton_recall(root_prob, root_skeleton, topology_valid)
            has_skeleton = (root_skeleton * topology_valid).sum((-2, -1)) > 0
            topology_loss = 1 - recall[has_skeleton].mean() if has_skeleton.any() else probs.sum() * 0
        elif self.organ_weights_mode == "root_only":
            topology_loss = 1 - root_cldice
        else:
            organ_term = self._organ_loss(probs, one_hot, topology_valid, adaptive=True)
            if self.joint_root_weight > 0:
                topology_loss = self.joint_root_weight * (1 - root_cldice) + (1 - self.joint_root_weight) * organ_term
            else:
                topology_loss = organ_term

        aux_logits = outputs.get("centerline")
        aux_loss = probs.sum() * 0
        if aux_logits is not None and centerline_target is not None:
            aux_probs = aux_logits.sigmoid()
            present = centerline_target.sum((-2, -1)) > 0
            if present.any():
                scores = soft_dice(aux_probs, centerline_target.float(), valid[:, None])
                aux_loss = 1 - scores[present].mean()

        scale = self.topology_scale(step) if self.topology_weight > 0 or self.auxiliary_weight > 0 else 0.0
        total = (
            self.region_ce_weight * ce
            + self.region_dice_weight * dice_loss
            + scale * (self.topology_weight * topology_loss + self.auxiliary_weight * aux_loss)
        )
        details = {
            "loss": float(total.detach()),
            "ce": float(ce.detach()),
            "dice": float(dice_loss.detach()),
            "cldice": float(topology_loss.detach()),
            "joint_root": float(root_cldice.detach()),
            "aux": float(aux_loss.detach()),
            "topology_scale": scale,
        }
        return total, details
