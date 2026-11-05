from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toposeg import figstyle
from toposeg.stats import (
    LOWER_IS_BETTER, ORGANS, POST_HOC_EXPERIMENTS, REQUIRED_SEEDS, favour_label, load_metrics, paired_by_event, select_evaluation_paths,
)

TOPOLOGY_METRICS = ["complete_root_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1", "skeleton_length_absolute_error_px"]
REGION_METRICS = ["macro_positive_dice", "root_hd95_px", "root_asd_px", "largest_component_recall", "component_count_error"]


def check_seeds(data: pd.DataFrame, baseline: str, allow_partial: bool) -> None:
    incomplete = {
        experiment: sorted(REQUIRED_SEEDS - set(group.seed))
        for experiment, group in data.groupby("experiment")
        if set(group.seed) != REQUIRED_SEEDS
    }
    if incomplete and not allow_partial:
        raise RuntimeError(f"Three-seed results are incomplete: {incomplete}")
    if allow_partial:
        print(f"WARNING: debugging mode, partial seeds accepted: {incomplete}")
    if baseline not in set(data.experiment):
        raise RuntimeError(f"Baseline {baseline!r} not found")


def organ_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, seed), group in data.groupby(["experiment", "seed"]):
        for cls, name in ORGANS.items():
            present = group[group[f"has_class_{cls}"] == 1]
            rows.append({
                "experiment": experiment, "seed": seed, "organ": name, "positive_images": len(present),
                "dice_positive": present[f"dice_class_{cls}"].mean(),
                "iou_positive": present[f"iou_class_{cls}"].mean(),
                "precision_positive": present[f"precision_class_{cls}"].mean(),
                "recall_positive": present[f"recall_class_{cls}"].mean(),
                "dice_all_images": group[f"dice_class_{cls}"].mean(),
            })
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["experiment", "organ"]).agg(
        positive_images=("positive_images", "mean"),
        dice_positive_mean=("dice_positive", "mean"), dice_positive_sd=("dice_positive", "std"),
        iou_positive_mean=("iou_positive", "mean"), iou_positive_sd=("iou_positive", "std"),
        precision_positive_mean=("precision_positive", "mean"), recall_positive_mean=("recall_positive", "mean"),
        dice_all_images_mean=("dice_all_images", "mean"),
    ).reset_index()
    grouped["inflation_from_empty_classes"] = grouped["dice_all_images_mean"] - grouped["dice_positive_mean"]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed evaluations and paired event-level statistics")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--split-csv", type=Path, default=Path("manifests/splits/main.csv"))
    parser.add_argument("--baseline", default="baseline_region")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--evaluation", default="evaluation_test",
                        help="Only read results from evaluation directories with this name, so that the nominal and out-of-domain diagnostics cannot mix into the protocol analysis")
    parser.add_argument("--exclude", nargs="*", default=list(POST_HOC_EXPERIMENTS),
                        help="Experiments kept out of this analysis; the post-hoc comparator has its own family")
    parser.add_argument("--debug-partial", action="store_true", help="Accept fewer than three seeds (debugging only)")
    args = parser.parse_args()

    paths = select_evaluation_paths(sorted(args.input.glob("**/per_image_metrics.csv")), args.evaluation)
    data = load_metrics(paths, args.split_csv)
    data = data[~data.experiment.isin(args.exclude)]
    check_seeds(data, args.baseline, args.debug_partial)
    metrics = [metric for metric in TOPOLOGY_METRICS + REGION_METRICS if metric in data]
    if not metrics:
        raise RuntimeError("No supported metric columns found")

    seed_means = data.groupby(["experiment", "seed"])[metrics].mean().reset_index()
    summary = seed_means.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    events = data.groupby(["experiment", "seed", "event_id"])[metrics].mean().reset_index()
    rng = np.random.default_rng(20261016)
    tests = pd.concat([paired_by_event(events, args.baseline, metric, rng, args.iterations) for metric in metrics], ignore_index=True)
    tests["direction"] = np.where(tests["metric"].isin(LOWER_IS_BETTER), "lower", "higher")
    tests["favors"] = favour_label(tests, LOWER_IS_BETTER)
    tests["noninferior_at_margin"] = np.nan
    region = tests["metric"] == "macro_positive_dice"
    tests.loc[region, "noninferior_at_margin"] = (tests.loc[region, "ci95_lower"] > -args.noninferiority_margin).astype(float)

    organs = organ_table(data)

    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "per_image_all_runs.csv", index=False)
    seed_means.to_csv(args.output / "seed_means.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    organs.to_csv(args.output / "organ_table.csv", index=False)
    tests.to_csv(args.output / "paired_tests.csv", index=False)

    # Ten metrics on one wide row printed at 1.3 pt; as two rows of five at the printed width they
    # stay at the sizes figstyle sets, and the panels are numbered in the row order of the summary table.
    grid = [("topology metrics", [m for m in TOPOLOGY_METRICS if m in metrics]),
            ("region metrics", [m for m in REGION_METRICS if m in metrics])]
    ncols = max(len(row_metrics) for _, row_metrics in grid)
    order = [experiment for experiment in figstyle.EXPERIMENT_ORDER if experiment in set(summary.experiment)]
    positions = np.arange(len(order))
    with plt.rc_context(figstyle.BASE):
        figure, axes = plt.subplots(2, ncols, figsize=(figstyle.PRINT_WIDTH_IN, 4.3),
                                    constrained_layout=True, squeeze=False)
        for row, (row_label, row_metrics) in enumerate(grid):
            for column in range(ncols):
                axis = axes[row][column]
                if column >= len(row_metrics):
                    axis.set_axis_off()
                    continue
                metric = row_metrics[column]
                cells = summary.set_index("experiment")
                axis.errorbar(positions, cells.loc[order, f"{metric}_mean"],
                              yerr=cells.loc[order, f"{metric}_std"].fillna(0),
                              fmt="o", capsize=2, ms=3)
                axis.set_xticks(positions, [str(index + 1) for index in positions])
                axis.set_title(figstyle.METRIC_TITLE.get(metric, metric))
                axis.grid(alpha=0.25, lw=0.4)
                axis.tick_params(labelleft=column == 0)
            axes[row][0].set_ylabel(row_label)
        for column in range(ncols):
            axes[1][column].set_xlabel("configuration")
    figstyle.save(figure, args.output / "three_seed_summary")

    pivot = organs.pivot(index="organ", columns="experiment", values="dice_positive_mean")
    pivot = pivot.reindex(columns=order).rename(columns=figstyle.EXPERIMENT_SHORT)
    pivot.index = [name.replace("_", " ") for name in pivot.index]
    with plt.rc_context(figstyle.BASE):
        figure, axis = plt.subplots(figsize=(figstyle.PRINT_WIDTH_IN, 2.9), constrained_layout=True)
        pivot.plot.bar(ax=axis, rot=30, width=0.8, legend=False)
        axis.set_ylabel("Dice where the organ is present")
        axis.set_ylim(0, 1)
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25, lw=0.4)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False)
    figstyle.save(figure, args.output / "organ_dice")
    print(args.output)


if __name__ == "__main__":
    main()
