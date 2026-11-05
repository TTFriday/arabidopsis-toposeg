import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import scripts.analyze_phenotypes as phenotype_analysis
import scripts.analyze_protocol as protocol_analysis
import scripts.analyze_results as result_analysis

ORGANS = (1, 2, 3, 4, 5, 6)
SEEDS = (17, 29, 43)
EVENTS = 6
EXPERIMENTS = ("baseline_region", "full")


def metric_value(experiment: str, gain: float, base_cldice: float, noise: float) -> tuple[float, float]:
    """Return (complete-root clDice, complete-root length absolute error); error falls with topology gain."""
    if experiment == "baseline_region":
        return base_cldice, 200.0 + noise
    return base_cldice + gain, max(1.0, 200.0 - 1500.0 * gain + noise)


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    metrics, phenotypes, split_rows = [], [], []
    for event in range(EVENTS):
        event_id = f"event_{event}"
        for image in range(2):
            image_id = f"image_{event}_{image}"
            present = {cls: int((event + cls + image) % 3 != 0) for cls in ORGANS}
            split_rows.append({"image_id": image_id, "event_id": event_id, **{f"has_class_{cls}": value for cls, value in present.items()}})
            ground_truth = 2000.0 if image == 0 else 0.0
            for experiment in EXPERIMENTS:
                for seed in SEEDS:
                    gain = 0.01 * (event + 1) + 0.004 * rng.normal(0, 1)
                    base_cldice = 0.40 + 0.001 * event
                    cldice, length_error = metric_value(experiment, gain, base_cldice, float(rng.normal(0, 2)))
                    root_dice = 0.45 + 0.001 * event + (0.2 * gain + 0.003 * rng.normal(0, 1) if experiment == "full" else 0.0)
                    dice = {cls: (0.5 if present[cls] else 1.0) for cls in ORGANS}
                    metrics.append({
                        "image_id": image_id, "experiment": experiment, "seed": seed, "event_id": event_id, "condition": "Roots",
                        **{f"dice_class_{cls}": dice[cls] for cls in ORGANS},
                        **{f"iou_class_{cls}": dice[cls] / 2 for cls in ORGANS},
                        **{f"precision_class_{cls}": dice[cls] for cls in ORGANS},
                        **{f"recall_class_{cls}": dice[cls] for cls in ORGANS},
                        "complete_root_dice": root_dice,
                        "complete_root_cldice": cldice,
                        "endpoint_f1": 0.2 + (0.05 if experiment == "full" else 0.0),
                        "branchpoint_f1": 0.1 + (0.05 if experiment == "full" else 0.0),
                        "skeleton_length_absolute_error_px": length_error,
                        "root_hd95_px": 60.0, "root_asd_px": 12.0,
                        "largest_component_recall": 0.8, "component_count_error": 3.0,
                    })
                    endpoints = (12, 20) if experiment == "baseline_region" else (12, 14)
                    branches = (4, 7) if experiment == "baseline_region" else (4, 5)
                    phenotypes.append({
                        "image_id": image_id, "experiment": experiment, "seed": seed, "event_id": event_id,
                        "pred_main_root_length_px": 700.0 + length_error, "gt_main_root_length_px": 700.0,
                        "error_main_root_length_px": length_error, "absolute_error_main_root_length_px": length_error,
                        "pred_lateral_root_length_px": 300.0 + length_error, "gt_lateral_root_length_px": 300.0 if image == 0 else 0.0,
                        "error_lateral_root_length_px": length_error, "absolute_error_lateral_root_length_px": length_error,
                        "pred_complete_root_length_px": ground_truth + length_error, "gt_complete_root_length_px": ground_truth,
                        "error_complete_root_length_px": length_error, "absolute_error_complete_root_length_px": length_error,
                        "pred_root_endpoints": endpoints[1], "gt_root_endpoints": endpoints[0],
                        "error_root_endpoints": endpoints[1] - endpoints[0], "absolute_error_root_endpoints": abs(endpoints[1] - endpoints[0]),
                        "pred_root_branch_points": branches[1], "gt_root_branch_points": branches[0],
                        "error_root_branch_points": branches[1] - branches[0], "absolute_error_root_branch_points": abs(branches[1] - branches[0]),
                    })
    return pd.DataFrame(metrics), pd.DataFrame(phenotypes), pd.DataFrame(split_rows)


def write_runs(root, frame: pd.DataFrame, filename: str) -> None:
    for (experiment, seed), group in frame.groupby(["experiment", "seed"]):
        directory = root / experiment / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / filename, index=False)


@pytest.fixture
def prepared(tmp_path):
    metrics, phenotypes, split = make_frames()
    split.to_csv(tmp_path / "main.csv", index=False)
    write_runs(tmp_path, metrics, "per_image_metrics.csv")
    write_runs(tmp_path, phenotypes, "phenotype_errors.csv")
    return tmp_path, metrics, phenotypes, split


def test_macro_positive_dice_ignores_absent_organs(prepared):
    tmp_path, _, _, split = prepared
    frame = result_analysis.load_metrics([tmp_path / "full" / "seed_17" / "per_image_metrics.csv"], tmp_path / "main.csv")
    for index in (0, 3, 7):
        row = frame.iloc[index]
        source = split[split.image_id == row.image_id].iloc[0]
        expected = [row[f"dice_class_{cls}"] for cls in ORGANS if source[f"has_class_{cls}"] == 1]
        assert np.isclose(row.macro_positive_dice, np.mean(expected))
    assert (frame.macro_all_dice >= frame.macro_positive_dice).all()


def test_analyze_results_writes_tables_and_paired_tests(prepared, monkeypatch):
    tmp_path, output = prepared[0], prepared[0] / "analysis"
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(tmp_path), "--output", str(output),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500",
    ])
    result_analysis.main()
    tests = pd.read_csv(output / "paired_tests.csv")
    assert set(tests.experiment) == {"full"}
    assert {"holm_p", "ci95_lower", "ci95_upper", "noninferior_at_margin"} <= set(tests)
    topology = tests[tests.metric == "complete_root_cldice"]
    assert (topology.mean_difference > 0).all() and (topology.paired_events == EVENTS).all()
    noninferior = tests[tests.metric == "macro_positive_dice"].noninferior_at_margin
    assert noninferior.notna().all() and set(noninferior) <= {0.0, 1.0}
    region = tests[tests.metric == "macro_positive_dice"].iloc[0]
    assert region.noninferior_at_margin == float(region.ci95_lower > -0.005)
    errors = tests[tests.metric == "skeleton_length_absolute_error_px"]
    assert (errors.direction == "lower").all() and (errors.favors == "experiment").all()
    for name in ("summary.csv", "organ_table.csv", "seed_means.csv", "per_image_all_runs.csv", "organ_dice.png"):
        assert (output / name).is_file(), name
    organs = pd.read_csv(output / "organ_table.csv")
    assert organs.inflation_from_empty_classes.ge(0).all()


def test_named_evaluation_excludes_other_protocols(prepared, monkeypatch):
    """A published-protocol diagnostic stored beside the test run must not be pooled into it."""
    tmp_path, metrics, phenotypes, _ = prepared
    root = tmp_path / "runs"
    write_runs(root, metrics, "per_image_metrics.csv")
    write_runs(root, phenotypes, "phenotype_errors.csv")
    for directory in sorted(root.glob("*/*/per_image_metrics.csv")) + sorted(root.glob("*/*/phenotype_errors.csv")):
        nominal = directory.parent / "evaluation_nominal"
        nominal.mkdir(exist_ok=True)
        directory.rename(nominal / directory.name)
    for directory in sorted(root.glob("*/*/evaluation_nominal")):
        test = directory.parent / "evaluation_test"
        test.mkdir()
        for name in ("per_image_metrics.csv", "phenotype_errors.csv"):
            (test / name).write_bytes((directory / name).read_bytes())
    expected = len(metrics)
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(root), "--output", str(tmp_path / "test_only"),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500",
    ])
    result_analysis.main()
    assert len(pd.read_csv(tmp_path / "test_only" / "per_image_all_runs.csv")) == expected
    monkeypatch.setattr("sys.argv", [
        "analyze_results", "--input", str(root), "--output", str(tmp_path / "nominal_only"),
        "--split-csv", str(tmp_path / "main.csv"), "--iterations", "500", "--evaluation", "evaluation_nominal",
    ])
    result_analysis.main()
    assert len(pd.read_csv(tmp_path / "nominal_only" / "per_image_all_runs.csv")) == expected


def test_incomplete_seed_coverage_is_rejected(prepared, monkeypatch):
    tmp_path, metrics, _, _ = prepared
    partial = tmp_path / "partial"
    partial.mkdir()
    metrics[metrics.seed == 17].to_csv(partial / "per_image_metrics.csv", index=False)
    monkeypatch.setattr("sys.argv", ["x", "--input", str(partial), "--output", str(tmp_path / "out"), "--split-csv", str(tmp_path / "main.csv")])
    with pytest.raises(RuntimeError, match="incomplete"):
        result_analysis.main()
    monkeypatch.setattr("sys.argv", ["x", "--input", str(partial), "--output", str(tmp_path / "out"), "--split-csv", str(tmp_path / "main.csv"), "--debug-partial"])
    result_analysis.main()



def test_phenotype_summary_and_association(prepared, monkeypatch):
    tmp_path, output = prepared[0], prepared[0] / "pheno"
    monkeypatch.setattr("sys.argv", ["analyze_phenotypes", "--input", str(tmp_path), "--output", str(output), "--iterations", "500"])
    phenotype_analysis.main()
    summary = pd.read_csv(output / "phenotype_summary.csv")
    assert set(summary.phenotype) == set(phenotype_analysis.PHENO_COLUMNS)
    complete = summary[summary.phenotype == "complete_root_length"].set_index("experiment")
    assert complete.loc["full", "images_without_ground_truth"] == 6 * len(SEEDS)
    assert complete.loc["full", "phantom_rate"] == 1.0
    assert complete.loc["full", "mae"] < complete.loc["baseline_region", "mae"]
    paired = pd.read_csv(output / "phenotype_paired.csv")
    length = paired[paired.metric == "absolute_error_complete_root_length_px"]
    assert (length.mean_difference < 0).all() and (length.paired_events == EVENTS).all()
    link = pd.read_csv(output / "topology_phenotype_association.csv")
    fitted = link[link.status == "ok"]
    assert not fitted.empty, link.status.unique().tolist()
    assert "joint" in set(fitted.model)
    topology = fitted[fitted.term == "delta_cldice"]
    image_unit = topology[topology.unit == "seed_x_image"]
    assert (image_unit.estimate < 0).all() and (image_unit.ci95_upper < 0).any()
    event_unit = topology[topology.unit == "event"]
    assert not event_unit.empty and np.isfinite(event_unit.estimate).all()
    assert list(output.glob("phenotype_*.png"))


def write_evaluation(root: Path, frame: pd.DataFrame, filename: str, evaluation: str) -> None:
    """Write one result file per experiment and seed inside a named evaluation directory."""
    for (experiment, seed), group in frame.groupby(["experiment", "seed"]):
        directory = root / experiment / f"seed_{seed}" / evaluation
        directory.mkdir(parents=True, exist_ok=True)
        group.to_csv(directory / filename, index=False)


def test_protocol_separates_memorised_from_unseen_images(prepared, monkeypatch, capsys):
    tmp_path, metrics, _, split = prepared
    runs = tmp_path / "protocol_runs"
    roles = {"event_0": "train", "event_1": "train", "event_2": "val", "event_3": "val", "event_4": "test", "event_5": "test"}
    split.assign(frozen_split=split.event_id.map(roles)).to_csv(tmp_path / "nominal.csv", index=False)
    unseen_ids = split[split.event_id.isin(["event_4", "event_5"])].image_id
    split[split.image_id.isin(unseen_ids)].to_csv(tmp_path / "main.csv", index=False)
    write_evaluation(runs, metrics, "per_image_metrics.csv", "evaluation_nominal")
    write_evaluation(runs, metrics[metrics.image_id.isin(unseen_ids)], "per_image_metrics.csv", "evaluation_test")
    output = tmp_path / "analysis_protocol"
    monkeypatch.setattr("sys.argv", [
        "analyze_protocol", "--input", str(runs), "--splits", str(tmp_path), "--output", str(output), "--iterations", "500",
    ])
    protocol_analysis.main()
    groups = pd.read_csv(output / "protocol_groups.csv")
    assert set(groups.group) == {"nominal_all", "nominal_trained", "nominal_validation", "nominal_unseen", "leak_free_test"}
    counts = groups[groups.metric == "complete_root_cldice"].set_index("group").images
    assert counts.to_dict() == {"nominal_all": 36, "nominal_trained": 12, "nominal_validation": 12, "nominal_unseen": 12, "leak_free_test": 12}
    assert groups.set_index("group").events.loc[["nominal_trained", "nominal_validation", "nominal_unseen", "leak_free_test"]].eq(2).all()
    trained = groups[(groups.experiment == "full") & (groups.group == "nominal_trained") & (groups.metric == "complete_root_cldice")]["mean"].iloc[0]
    reference = metrics[(metrics.experiment == "full") & metrics.event_id.isin(["event_0", "event_1"])]
    assert np.isclose(trained, reference.complete_root_cldice.mean())
    contrasts = pd.read_csv(output / "protocol_contrasts.csv")
    assert set(contrasts.contrast) == {"nominal_trained - nominal_unseen", "nominal_validation - nominal_unseen"}
    assert set(contrasts.events_a) == {2} and set(contrasts.events_b) == {2}
    errors = contrasts[contrasts.metric == "skeleton_length_absolute_error_px"]
    assert (errors.better_group == np.where(errors.mean_difference > 0, "nominal_unseen", "nominal_trained")).all()
    assert "evaluation_ood" in capsys.readouterr().out
