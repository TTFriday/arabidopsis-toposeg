"""Post-hoc external comparator: the Skeleton Recall loss (Kirchhoff et al., ECCV 2024).

The comparator was added after the six-arm analysis was fixed, so it is kept out of that Holm family
and tested in its own: three event-level paired contrasts per metric, Holm-corrected within each
metric. Seed-level summaries are written for the leak-free test set and, when present, the unmapped
out-of-domain images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from toposeg.stats import LOWER_IS_BETTER, REQUIRED_SEEDS, holm_adjust, load_metrics, paired_test, select_evaluation_paths

COMPARATOR = "baseline_skelrecall"
REFERENCES = ["baseline_region", "baseline_cldice", "full"]
CONTRASTS = [(COMPARATOR, "baseline_region"), (COMPARATOR, "baseline_cldice"), ("full", COMPARATOR)]
METRICS = ["macro_positive_dice", "complete_root_dice", "complete_root_cldice", "endpoint_f1", "branchpoint_f1",
           "skeleton_length_absolute_error_px", "root_hd95_px", "root_asd_px", "largest_component_recall",
           "component_count_error"]
PHENOTYPES = ["complete_root_length_px", "main_root_length_px", "lateral_root_length_px", "root_endpoints",
              "root_branch_points"]


def require_seeds(frame: pd.DataFrame, experiments: list[str], label: str) -> pd.DataFrame:
    frame = frame[frame.experiment.isin(experiments)]
    for experiment in experiments:
        seeds = set(frame[frame.experiment == experiment].seed)
        if seeds != REQUIRED_SEEDS:
            raise RuntimeError(f"{label}: {experiment} has seeds {sorted(seeds)}, expected {sorted(REQUIRED_SEEDS)}")
    return frame


def paired_contrasts(events: pd.DataFrame, metrics: list[str], rng: np.random.Generator, iterations: int) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        block = []
        for experiment, reference in CONTRASTS:
            left = events[events.experiment == experiment][["seed", "event_id", metric]].rename(columns={metric: "a"})
            right = events[events.experiment == reference][["seed", "event_id", metric]].rename(columns={metric: "b"})
            paired = left.merge(right, on=["seed", "event_id"], validate="one_to_one").dropna()
            differences = (paired.a - paired.b).groupby(paired.event_id).mean().to_numpy()
            lower, upper, p_value = paired_test(differences, rng, iterations)
            block.append({"experiment": experiment, "reference": reference, "metric": metric,
                          "paired_events": len(differences), "mean_difference": float(differences.mean()),
                          "ci95_lower": lower, "ci95_upper": upper, "permutation_p": p_value})
        block = pd.DataFrame(block)
        block["holm_p"] = holm_adjust(block["permutation_p"])
        rows.append(block)
    frame = pd.concat(rows, ignore_index=True)
    lower_better = frame.metric.isin(LOWER_IS_BETTER) | frame.metric.str.startswith("absolute_error_")
    favourable = (frame.mean_difference > 0) != lower_better
    frame["favors"] = np.where(frame.mean_difference == 0, "tie", np.where(favourable, frame.experiment, frame.reference))
    return frame


def seed_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    seeds = frame.groupby(["experiment", "seed"])[metrics].mean().reset_index()
    summary = seeds.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def phenotype_summary(data: pd.DataFrame) -> pd.DataFrame:
    """MAE over images with a non-zero reference, per seed, then mean and SD over the three seeds."""
    rows = []
    for (experiment, seed), group in data.groupby(["experiment", "seed"]):
        row = {"experiment": experiment, "seed": seed}
        for stem in PHENOTYPES:
            visible = group[group[f"gt_{stem}"] > 0]
            row[f"mae_{stem}"] = visible[f"absolute_error_{stem}"].mean()
            row[f"bias_{stem}"] = visible[f"error_{stem}"].mean()
        rows.append(row)
    seeds = pd.DataFrame(rows)
    columns = [column for column in seeds if column not in ("experiment", "seed")]
    summary = seeds.groupby("experiment")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeleton Recall comparator as a separate post-hoc comparison family")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis_comparator"))
    parser.add_argument("--splits", type=Path, default=Path("manifests/splits"))
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    experiments = [COMPARATOR, *REFERENCES]

    paths = select_evaluation_paths(sorted(args.input.glob("**/per_image_metrics.csv")), "evaluation_test")
    data = require_seeds(load_metrics(paths, args.splits / "main.csv"), experiments, "test metrics")
    metrics = [metric for metric in METRICS if metric in data]
    events = data.groupby(["experiment", "seed", "event_id"])[metrics].mean().reset_index()

    phenotype_paths = select_evaluation_paths(sorted(args.input.glob("**/phenotype_errors.csv")), "evaluation_test")
    phenotypes = pd.concat([pd.read_csv(path) for path in phenotype_paths], ignore_index=True)
    phenotypes["seed"] = phenotypes["seed"].astype(int)
    phenotypes = require_seeds(phenotypes, experiments, "test phenotypes")
    error_columns = [f"absolute_error_{stem}" for stem in PHENOTYPES]
    phenotype_events = phenotypes.groupby(["experiment", "seed", "event_id"])[error_columns].mean().reset_index()

    rng = np.random.default_rng(20261016)
    tests = paired_contrasts(events, metrics, rng, args.iterations)
    phenotype_tests = paired_contrasts(phenotype_events, error_columns, rng, args.iterations)

    args.output.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.output / "comparator_paired.csv", index=False)
    phenotype_tests.to_csv(args.output / "comparator_phenotype_paired.csv", index=False)
    seed_summary(data, metrics).to_csv(args.output / "comparator_summary.csv", index=False)
    phenotype_summary(phenotypes).to_csv(args.output / "comparator_phenotype_summary.csv", index=False)

    ood_paths = [path for path in sorted(args.input.glob("**/evaluation_ood/per_image_metrics.csv"))
                 if path.parts[-4] in experiments]
    if ood_paths:
        ood = require_seeds(load_metrics(ood_paths, args.splits / "ood.csv"), experiments, "ood metrics")
        seed_summary(ood, metrics).to_csv(args.output / "comparator_ood_summary.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
