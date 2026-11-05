from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from toposeg import figstyle
from toposeg.dataset import resolve_paths

ORGANS = {1: "main root", 2: "lateral root", 3: "seed", 4: "hypocotyl", 5: "leaf", 6: "petiole"}
HEX = {1: "#d62728", 2: "#ff7f0e", 3: "#9467bd", 4: "#2ca02c", 5: "#1f77b4", 6: "#8c564b"}
COLORS_RGB = {cls: tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)) for cls, value in HEX.items()}

# Column headings for the qualitative grid: the full names do not fit a one-inch column.
COLUMN_TITLE = {
    "input": "input", "label": "label",
    "baseline_region": "region-only", "baseline_cldice": "joint-root",
    "full": "full", "ablation_uniform": "uniform",
    "ablation_no_joint_root": "no joint root", "ablation_no_skeleton_head": "no head",
}


def leakage_figure(audit: Path, correlations: Path, output: Path) -> None:
    report = json.loads(audit.read_text(encoding="utf-8"))
    pairs = pd.read_csv(correlations)
    crossing = report["images_in_crossing_events_share"] * 100
    with plt.rc_context(figstyle.BASE):
        figure, axes = plt.subplots(1, 2, figsize=(figstyle.PRINT_WIDTH_IN, 2.3), constrained_layout=True)
        axes[0].hist(pairs["correlation"], bins=40, color="#4472a4")
        axes[0].axvline(report["adjacent_frame_correlation_median"], color="black", ls="--", lw=1)
        axes[0].set_xlabel("adjacent-frame correlation")
        axes[0].set_ylabel("frame pairs")
        axes[0].set_title(f"median {report['adjacent_frame_correlation_median']:.4f} "
                          f"over {report['adjacent_frame_pairs_measured']} pairs")
        axes[1].bar(["distributed\nsplit", "frozen event\nsplit"], [crossing, 0.0], color=["#c44e52", "#55a868"])
        for index, value in enumerate([crossing, 0.0]):
            axes[1].text(index, value + 2, f"{value:.1f}%", ha="center")
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("% of mapped images\ncrossing subsets")
        axes[1].set_title(f"{report['events_crossing_distributed_train_val_test']}"
                          f"/{report['acquisition_events_total']} events cross subsets")
    figstyle.save(figure, output / "fig1_leakage")


def overlay(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=float)
    for cls in ORGANS:
        rgb[mask == cls] = COLORS_RGB[cls]
    return rgb


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def crop_window(mask: np.ndarray, size: int) -> tuple[slice, slice]:
    """Centre a full-resolution crop on the root and hypocotyl pixels so thin organs stay visible."""
    rows, columns = np.nonzero(np.isin(mask, (1, 2, 4)))
    centre = (mask.shape[0] // 2, mask.shape[1] // 2) if len(rows) == 0 else (int(np.median(rows)), int(np.median(columns)))
    window = []
    for index, (position, extent) in enumerate(zip(centre, mask.shape)):
        start = int(np.clip(position - size // 2, 0, max(0, extent - size)))
        window.append(slice(start, start + min(size, extent)))
    return window[0], window[1]


def qualitative_figure(
    outputs: Path,
    experiments: list[str],
    images: list[str],
    split_csv: Path,
    output: Path,
    seed: int = 17,
    crop: int = 700,
) -> list[str]:
    """Full-resolution crops: roots here are one to five pixels wide and vanish under downsampling."""
    split = resolve_paths(pd.read_csv(split_csv), None)
    columns = ["input", "label"] + list(experiments)
    row_height = figstyle.PRINT_WIDTH_IN / len(columns)
    with plt.rc_context(figstyle.BASE):
        figure, axes = plt.subplots(
            len(images), len(columns),
            figsize=(figstyle.PRINT_WIDTH_IN, row_height * len(images) + 0.9),
            constrained_layout=True, squeeze=False)
        missing = []
        for row, image_id in enumerate(images):
            record = split[split.image_id == image_id]
            if record.empty:
                raise KeyError(f"{image_id} is not in {split_csv}")
            record = record.iloc[0]
            label = read_mask(record.label_path)
            window = crop_window(label, crop)
            with Image.open(record.image_path) as handle:
                source = np.asarray(handle).astype(np.float32)
            source = source[..., 0] if source.ndim == 3 else source
            low, high = np.percentile(source, (0.5, 99.5))
            panels = [np.clip((source[window] - low) / max(1e-6, high - low), 0, 1), label[window]]
            for experiment in experiments:
                path = outputs / experiment / f"seed_{seed}" / "evaluation_test" / "predictions" / f"{image_id}.png"
                if path.is_file():
                    panels.append(read_mask(path)[window])
                else:
                    panels.append(None)
                    missing.append(f"{experiment}/{image_id}")
            for column, panel in enumerate(panels):
                axis = axes[row][column]
                if panel is None:
                    axis.text(0.5, 0.5, "not run", ha="center", va="center")
                elif column == 0:
                    axis.imshow(panel, cmap="gray")
                else:
                    axis.imshow(overlay(panel))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(COLUMN_TITLE[columns[column]])
            axes[row][0].set_ylabel(f"{image_id}\n{record.condition}")
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS_RGB[cls]) for cls in ORGANS]
        figure.legend(handles, ORGANS.values(), loc="outside lower center",
                      ncol=len(ORGANS), frameon=False)
    figstyle.save(figure, output / "fig2_qualitative")
    return missing


FOREST_PANELS = [
    ("paired_tests.csv", "complete_root_cldice", "complete-root clDice ↑"),
    ("paired_tests.csv", "endpoint_f1", "endpoint F1 ↑"),
    ("paired_tests.csv", "macro_positive_dice", "macro-Dice ↑"),
    ("paired_tests.csv", "component_count_error", "component-count error ↓"),
    ("phenotype_paired.csv", "absolute_error_root_endpoints", "root-tip count error ↓"),
    ("phenotype_paired.csv", "absolute_error_complete_root_length_px", "root length error (px) ↓"),
]


def forest_figure(analysis: Path, comparator: Path, output: Path, margin: float = 0.005) -> None:
    """Event-level paired differences against the region-only baseline with 95% bootstrap intervals.

    The post-hoc comparator comes from its own Holm family and is drawn in a separate colour below
    a rule, so it is never read as part of the pre-specified contrasts.
    """
    arms = [e for e in figstyle.EXPERIMENT_ORDER if e != "baseline_region"]
    extra = {"paired_tests.csv": comparator / "comparator_paired.csv",
             "phenotype_paired.csv": comparator / "comparator_phenotype_paired.csv"}
    with plt.rc_context(figstyle.BASE):
        figure, axes = plt.subplots(2, 3, figsize=(figstyle.PRINT_WIDTH_IN, 3.6), constrained_layout=True, sharey=True)
        for axis, (source, metric, title) in zip(axes.flat, FOREST_PANELS):
            tests = pd.read_csv(analysis / source)
            rows = [tests[(tests.metric == metric) & (tests.experiment == arm)].iloc[0] for arm in arms]
            labels = [figstyle.EXPERIMENT_SHORT[arm] for arm in arms]
            colours = ["#2f4b7c"] * len(rows)
            if extra[source].is_file():
                post_hoc = pd.read_csv(extra[source])
                post_hoc = post_hoc[(post_hoc.metric == metric) & (post_hoc.reference == "baseline_region")]
                if not post_hoc.empty:
                    rows.append(post_hoc.iloc[0])
                    labels.append(figstyle.EXPERIMENT_SHORT["baseline_skelrecall"] + " (post hoc)")
                    colours.append("#c05a1a")
            positions = np.arange(len(rows))[::-1]
            for position, row, colour in zip(positions, rows, colours):
                significant = row.holm_p < 0.05
                axis.plot([row.ci95_lower, row.ci95_upper], [position, position], color=colour, lw=1.0)
                axis.plot(row.mean_difference, position, "o", ms=3.5, color=colour,
                          markerfacecolor=colour if significant else "white", markeredgewidth=0.8)
            axis.axvline(0, color="black", lw=0.6)
            if metric == "macro_positive_dice":
                axis.axvline(-margin, color="grey", lw=0.6, ls="--")
            if len(rows) > len(arms):
                axis.axhline(positions[len(arms) - 1] - 0.5, color="grey", lw=0.4, ls=":")
            axis.set_title(title)
            axis.set_yticks(positions, labels)
            axis.grid(axis="x", alpha=0.25, lw=0.4)
            axis.tick_params(axis="x", labelsize=6)
        figure.supxlabel("event-level paired difference from the region-only baseline (95% CI)", fontsize=7)
    figstyle.save(figure, output / "fig4_paired_effects")


def derived_figures(analysis: Path, output: Path) -> list[str]:
    """Name the analysis figures; the effects figure is drawn by forest_figure."""
    mapping = {
        "three_seed_summary": "fig3_seed_summary",
        "phenotype_baseline_region": "fig5_phenotype_baseline_region",
        "phenotype_ablation_uniform": "fig6_phenotype_ablation_uniform",
    }
    missing = []
    for source, target in mapping.items():
        for suffix in (".pdf", ".png"):
            if (analysis / f"{source}{suffix}").is_file():
                shutil.copyfile(analysis / f"{source}{suffix}", output / f"{target}{suffix}")
            else:
                missing.append(f"{source}{suffix}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analysis figures")
    parser.add_argument("--audit", type=Path, default=Path("manifests/leakage_audit.json"))
    parser.add_argument("--correlations", type=Path, default=Path("manifests/adjacent_frame_correlation.csv"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--split-csv", type=Path, default=Path("manifests/splits/main.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/analysis"),
                        help="where analyze_results.py and analyze_phenotypes.py wrote the derived figures")
    parser.add_argument("--comparator", type=Path, default=Path("outputs/analysis_comparator"),
                        help="post-hoc comparator analysis; drawn in the effects figure when present")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--experiments", nargs="*", default=["baseline_region", "baseline_cldice", "full"])
    parser.add_argument("--images", nargs="*", default=[], help="test images; defaults to one per condition by event order")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    leakage_figure(args.audit, args.correlations, args.output)
    images = args.images
    if not images:
        test = pd.read_csv(args.split_csv).query("split == 'test'").sort_values(["condition", "event_id", "image_id"])
        images = list(test.groupby("condition").image_id.first()[:4])
    missing = qualitative_figure(args.outputs, args.experiments, images, args.split_csv, args.output, args.seed)
    forest_figure(args.analysis, args.comparator, args.output)
    missing += derived_figures(args.analysis, args.output)
    print(args.output)
    if missing:
        print(f"{len(missing)} figure inputs missing: {missing[:6]}")


if __name__ == "__main__":
    main()
