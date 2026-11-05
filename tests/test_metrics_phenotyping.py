import numpy as np

from toposeg.metrics import compute_metrics
from toposeg.phenotyping import phenotype, skeleton_length


def test_perfect_prediction_metrics_and_ignore():
    target = np.zeros((32, 32), dtype=np.uint8)
    target[5:25, 15] = 1
    target[15, 16:25] = 2
    target[0, 0] = 7
    prediction = target.copy()
    prediction[0, 0] = 6
    metrics = compute_metrics(prediction, target)
    assert metrics["complete_root_dice"] == 1.0
    assert metrics["complete_root_cldice"] == 1.0
    assert metrics["root_hd95_px"] == 0.0
    assert metrics["root_asd_px"] == 0.0
    assert metrics["endpoint_f1"] == 1.0
    assert metrics["branchpoint_f1"] == 1.0


def test_phenotype_pixel_length():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[5, 2:7] = 1
    assert skeleton_length(mask == 1) == 4.0
    values = phenotype(mask)
    assert values["main_root_length_px"] == 4.0
    assert values["lateral_root_length_px"] == 0.0
    assert values["complete_root_length_px"] == 4.0
    assert values["root_endpoints"] == 2
