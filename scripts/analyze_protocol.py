"""Aggregate the published-split and out-of-domain protocol diagnostics.

The published ChronoRoot assignment evaluates the 156 ``imagesTs`` files as an independent test set.
The audit shows those files are byte-identical copies of ``imagesTr`` files, so under the leak-free
split 78 of them were trained on, 37 drove early stopping and only 41 were never seen. This script
measures how much that overlap inflates a reported score, and records absolute levels on the 46
images whose acquisition metadata could not be mapped.

Both sets are protocol diagnostics only. Neither is used for model selection, and neither is
presented as an independent test of generalisation: the trained-image columns measure memorisation
by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from toposeg.stats import (
    LOWER_IS_BETTER, POST_HOC_EXPERIMENTS, REQUIRED_SEEDS, event_bootstrap_difference, load_metrics, select_evaluation_paths,
)

METRICS = ["macro_positive_dice", "complete_root_dice", "complete_root_cldice", "endpoint_f1",
           "branchpoint_f1", "skeleton_length_absolute_error_px", "root_hd95_px", "component_count_error"]
PROTOCOLS = {
    "leak_free_test": ("evaluation_test", "main.csv", {}),
    "nominal": ("evaluation_nominal", "nominal.csv", {"train": "nominal_trained", "val": "nominal_validation", "test": "nominal_unseen"}),
    "unmapped_ood": ("evaluation_ood", "ood.csv", {}),
}
# The nominal images are the same image_ids the leak-free split already assigns to train, val and
# test, so only the trained/validation/unseen parts of the nominal set have disjoint events and can
# be tested as two independent cluster samples. Comparisons against leak_free_test reuse images and
# events and are reported as descriptive columns in protocol_groups.csv instead.
CONTRASTS = [
    ("nominal_trained", "nominal_unseen"),
    ("nominal_validation", "nominal_unseen"),
    ("unmapped_ood", "leak_free_test"),
]


def group_labels(splits: Path, manifest: str, by_frozen: dict[str, str], default: str) -> pd.DataFrame:
    """Map each image of one protocol onto its leak-free role: trained, validation, or unseen."""
    frame = pd.read_csv(splits / manifest)
    if not by_frozen:
        return frame[["image_id"]].assign(group=default).drop_duplicates("image_id")
    frame = frame[["image_id", "frozen_split"]].drop_duplicates("image_id")
    frame["group"] = frame.pop("frozen_split").map(by_frozen)
    if frame.group.isna().any():
        raise RuntimeError(f"{manifest}: images without a leak-free split label")
    return frame


def protocol_groups(root: Path, splits: Path, allow_partial: bool) -> tuple[pd.DataFrame, list[str]]:
    """Stack every available protocol evaluation and label each image by its leak-free role."""
    frames, pending = [], []
    for default_group, (evaluation, manifest, by_frozen) in PROTOCOLS.items():
        paths = sorted(root.glob(f"**/{evaluation}/per_image_metrics.csv"))
        if not paths:
            pending.append(evaluation)
            continue
        data = load_metrics(select_evaluation_paths(paths, evaluation), splits / manifest)
        data = data[~data.experiment.isin(POST_HOC_EXPERIMENTS)]
        data = data.merge(group_labels(splits, manifest, by_frozen, default_group), on="image_id", validate="many_to_one")
        if data.group.isna().any():
            raise RuntimeError(f"{evaluation}: result images missing from {manifest}")
        incomplete = {
            experiment: sorted(REQUIRED_SEEDS - set(group.seed))
            for experiment, group in data.groupby("experiment")
            if set(group.seed) != REQUIRED_SEEDS
        }
        if incomplete and not allow_partial:
            print(f"skipping {evaluation}: incomplete seeds {incomplete}")
            pending.append(evaluation)
            continue
        if by_frozen:
            # The published protocol pools all 156 images, so keep that level alongside its parts.
            frames.append(data.assign(group="nominal_all"))
        frames.append(data)
    if not frames:
        raise RuntimeError(f"No complete protocol evaluation found under {root}")
    return pd.concat(frames, ignore_index=True), pending


def seed_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per-arm, per-group, per-metric mean with the spread over the three seeds."""
    rows = []
    for (experiment, group), subset in data.groupby(["experiment", "group"]):
        for metric in (m for m in METRICS if m in subset):
            per_seed = subset.groupby("seed")[metric].mean()
            image = subset[metric]
            rows.append({
                "experiment": experiment, "group": group, "metric": metric,
                "images": image.notna().sum(), "events": subset.event_id.nunique(),
                "mean": float(per_seed.mean()), "sd_over_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                "min_seed": float(per_seed.min()), "max_seed": float(per_seed.max()),
            })
    return pd.DataFrame(rows)


def event_values(data: pd.DataFrame, group: str, metric: str) -> np.ndarray:
    """One value per acquisition event: images averaged, then the three seeds averaged."""
    subset = data[(data.group == group) & data[metric].notna()]
    if subset.empty:
        return np.array([], dtype=float)
    by_seed = subset.groupby(["event_id", "seed"])[metric].mean()
    return by_seed.groupby(level="event_id").mean().to_numpy()


def contrast_table(data: pd.DataFrame, rng: np.random.Generator, iterations: int) -> pd.DataFrame:
    rows = []
    for experiment in sorted(set(data.experiment)):
        subset = data[data.experiment == experiment]
        for group_a, group_b in CONTRASTS:
            for metric in (m for m in METRICS if m in subset):
                a, b = event_values(subset, group_a, metric), event_values(subset, group_b, metric)
                if len(a) < 2 or len(b) < 2:
                    continue
                stats = event_bootstrap_difference(a, b, rng, iterations)
                stats["better_group"] = group_a if (stats["mean_difference"] > 0) != (metric in LOWER_IS_BETTER) else group_b
                rows.append({"experiment": experiment, "metric": metric, "contrast": f"{group_a} - {group_b}", **stats})
    return pd.DataFrame(rows)


def rank_table(groups: pd.DataFrame) -> pd.DataFrame:
    """Rank agreement of the six configurations between the two protocols, per metric.

    Whether the configuration order survives the protocol change is a number, so it is computed
    here from the group levels, together with the configuration that scores worst under each
    protocol.
    """
    rows = []
    for metric in sorted(set(groups.metric)):
        subset = groups[groups.metric == metric]
        leak = subset[subset.group == "leak_free_test"].set_index("experiment")["mean"]
        nominal = subset[subset.group == "nominal_all"].set_index("experiment")["mean"]
        common = sorted(set(leak.index) & set(nominal.index))
        if len(common) < 3:
            continue
        worst = (lambda values: values.idxmax()) if metric in LOWER_IS_BETTER else (lambda values: values.idxmin())
        rho = spearmanr(leak[common], nominal[common]).statistic
        rows.append({
            "metric": metric, "configurations": len(common), "spearman_leak_free_vs_nominal": float(rho),
            "worst_leak_free_test": worst(leak[common]), "worst_nominal_all": worst(nominal[common]),
        })
    return pd.DataFrame(rows)


def coverage_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, group), subset in data.groupby(["experiment", "group"]):
        rows.append({
            "experiment": experiment, "group": group, "seeds": subset.seed.nunique(),
            "images": subset.image_id.nunique(), "events": subset.event_id.nunique(),
            "conditions": "|".join(sorted(set(subset.condition))),
        })
    return pd.DataFrame(rows).sort_values(["group", "experiment"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify published-protocol inflation and OOD levels")
    parser.add_argument("--input", type=Path, default=Path("outputs"), help="Tree holding evaluation_* directories")
    parser.add_argument("--splits", type=Path, default=Path("manifests/splits"), help="Directory with the split manifests")
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis_protocol"))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261016)
    parser.add_argument("--debug-partial", action="store_true", help="Accept runs without all three seeds (never publish these)")
    args = parser.parse_args()

    data, pending = protocol_groups(args.input, args.splits, args.debug_partial)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = seed_summary(data)
    groups.to_csv(args.output / "protocol_groups.csv", index=False)
    contrast_table(data, np.random.default_rng(args.bootstrap_seed), args.iterations).to_csv(args.output / "protocol_contrasts.csv", index=False)
    coverage = coverage_table(data)
    coverage.to_csv(args.output / "protocol_coverage.csv", index=False)
    ranks = rank_table(groups)
    ranks.to_csv(args.output / "protocol_rank_correlation.csv", index=False)
    print(ranks.to_string(index=False))
    print(coverage.to_string(index=False))
    if pending:
        print(f"PENDING protocols (not aggregated): {pending}")


if __name__ == "__main__":
    main()
