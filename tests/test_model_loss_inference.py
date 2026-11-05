import torch
from torch import nn

from toposeg.losses import TopologyAwareLoss, ramp_schedule
from toposeg.models import CompactUNet
from toposeg.inference import sliding_window_inference


def test_model_shapes():
    model = CompactUNet(base_channels=4)
    output = model(torch.randn(2, 1, 64, 80))
    assert output["region"].shape == (2, 7, 64, 80)
    assert output["centerline"].shape == (2, 4, 64, 80)


def _synthetic_target():
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[:, 15:50, 30:34] = 1
    target[:, 25:28, 34:50] = 2
    target[:, 60:63, 5:60] = 4
    return target


def test_loss_finite_and_backward():
    model = CompactUNet(base_channels=4)
    target = _synthetic_target()
    from toposeg.dataset import centerline_target
    aux = centerline_target(target[0])[None]
    loss, details = TopologyAwareLoss(
        skeleton_iterations=2, boundary=4,
        organ_weights={1: 0.3, 2: 0.4, 4: 0.2, 6: 0.1},
        total_steps=100,
    )(model(torch.randn(1, 1, 64, 64)), target, aux, step=50)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in details.values())
    assert details["topology_scale"] == 1.0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ramp_schedule():
    assert ramp_schedule(0.0) == 0.0
    assert ramp_schedule(0.10) == 0.0
    assert abs(ramp_schedule(0.175) - 0.5) < 1e-9
    assert ramp_schedule(0.25) == 1.0
    assert ramp_schedule(0.9) == 1.0


def test_topology_disabled_before_ramp():
    criterion = TopologyAwareLoss(skeleton_iterations=1, boundary=0, organ_weights="uniform")
    target = _synthetic_target()
    logits = torch.randn(1, 7, 64, 64, requires_grad=True)
    _, details = criterion({"region": logits, "centerline": None}, target, None, step=0)
    assert details["topology_scale"] == 0.0


def test_ignore_pixels_do_not_affect_loss():
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 4:12, 7:9] = 1
    target[:, 0, 0] = 7
    logits_a = torch.randn(1, 7, 16, 16, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    logits_b.data[:, :, 0, 0] = torch.tensor([100.0, -100.0, 50.0, -50.0, 25.0, -25.0, 75.0])
    criterion = TopologyAwareLoss(skeleton_iterations=1, boundary=0, auxiliary_weight=0, organ_weights="uniform")
    loss_a, _ = criterion({"region": logits_a, "centerline": None}, target)
    loss_b, _ = criterion({"region": logits_b, "centerline": None}, target)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_organ_weight_modes():
    target = _synthetic_target()
    logits = torch.randn(1, 7, 64, 64, requires_grad=True)
    uniform = TopologyAwareLoss(skeleton_iterations=1, boundary=0, organ_weights="uniform")
    adaptive = TopologyAwareLoss(
        skeleton_iterations=1, boundary=0, organ_weights={1: 0.7, 2: 0.1, 4: 0.1, 6: 0.1},
    )
    root_only = TopologyAwareLoss(skeleton_iterations=1, boundary=0, organ_weights="root_only")
    outputs = {"region": logits, "centerline": None}
    for criterion in (uniform, adaptive, root_only):
        loss, _ = criterion(outputs, target, None, step=100000)
        assert torch.isfinite(loss)


def test_skeleton_recall_loss():
    import pytest
    from toposeg.dataset import tubed_root_skeleton
    target = _synthetic_target()
    tube = tubed_root_skeleton(target[0])[None]
    assert tube.shape == (1, 1, 64, 64)
    assert tube.sum() > 0
    criterion = TopologyAwareLoss(
        skeleton_iterations=1, boundary=0, organ_weights="root_only",
        topology_form="skeleton_recall", auxiliary_weight=0.0,
    )
    logits = torch.randn(1, 7, 64, 64, requires_grad=True)
    outputs = {"region": logits, "centerline": None}
    loss, details = criterion(outputs, target, None, step=100000, root_skeleton=tube)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        criterion(outputs, target, None, step=100000)
    with pytest.raises(ValueError):
        TopologyAwareLoss(organ_weights="uniform", topology_form="skeleton_recall")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 7, 1)

    def forward(self, image):
        return {"region": self.conv(image), "centerline": None}


def test_sliding_window_output_shape():
    prediction, probabilities = sliding_window_inference(TinyModel(), torch.rand(1, 23, 19), patch_size=16, overlap=0.5)
    assert prediction.shape == (23, 19)
    assert probabilities.shape == (7, 23, 19)
    assert torch.allclose(probabilities.sum(0), torch.ones(23, 19), atol=1e-5)
