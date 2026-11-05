from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def paired_test(differences: np.ndarray, rng: np.random.Generator, iterations: int = 10000) -> tuple[float, float, float]:
    """Bootstrap CI and sign-flip permutation p-value for paired event-level differences."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        raise RuntimeError("At least two paired events are required")
    samples = rng.integers(0, len(differences), size=(iterations, len(differences)))
    boot = differences[samples].mean(1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(differences)))
    permuted = (signs * differences).mean(1)
    p_value = (1 + np.sum(np.abs(permuted) >= abs(differences.mean()))) / (iterations + 1)
    return float(lower), float(upper), float(p_value)


def event_bootstrap_difference(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, iterations: int = 10000) -> dict[str, float]:
    """Cluster-bootstrap the difference of two unpaired means of per-event values (A minus B).

    Acquisition events, not images, are the independent unit, so events are resampled with
    replacement inside each group. A percentile p-value is reported instead of a sign-flip test
    because the groups share no events and the null is not symmetric.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise RuntimeError("At least two events are required in each group")
    draws = a.mean() - b.mean()
    samples_a = rng.integers(0, len(a), size=(iterations, len(a)))
    samples_b = rng.integers(0, len(b), size=(iterations, len(b)))
    boot = a[samples_a].mean(1) - b[samples_b].mean(1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    below = float(np.mean(boot <= 0.0))
    above = float(np.mean(boot >= 0.0))
    return {
        "mean_difference": float(draws),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_p": float(min(1.0, 2.0 * min(below, above))),
        "events_a": len(a),
        "events_b": len(b),
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values.iloc[index]))
        running = max(running, value)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def favour_label(frame: pd.DataFrame, lower_is_better: set[str]) -> pd.Series:
    """Label which arm each paired difference favours, given the metric's better direction."""
    favorable = (frame["mean_difference"] > 0) != frame["metric"].isin(lower_is_better)
    return pd.Series(np.where(frame["mean_difference"] == 0, "tie", np.where(favorable, "experiment", "baseline")), index=frame.index)


def paired_by_event(
    values: pd.DataFrame,
    baseline: str,
    metric: str,
    rng: np.random.Generator,
    iterations: int = 10000,
) -> pd.DataFrame:
    """Event-level paired differences of a metric between each experiment and a baseline.

    ``values`` needs the columns experiment, seed, event_id and ``metric``.
    Per-event means over seeds are paired, then tested against zero.
    """
    if metric not in values:
        raise KeyError(f"Metric {metric!r} not present in the evaluation frame")
    rows = []
    for experiment in sorted(set(values.experiment) - {baseline}):
        reference = values[values.experiment == baseline][["seed", "event_id", metric]].rename(columns={metric: "baseline"})
        method = values[values.experiment == experiment][["seed", "event_id", metric]].rename(columns={metric: "method"})
        paired = reference.merge(method, on=["seed", "event_id"], validate="one_to_one").dropna()
        if paired.empty:
            continue
        paired = paired.assign(difference=paired["method"] - paired["baseline"])
        differences = paired.groupby("event_id")["difference"].mean().to_numpy()
        lower, upper, p_value = paired_test(differences, rng, iterations)
        rows.append({
            "baseline": baseline, "experiment": experiment, "metric": metric,
            "paired_events": len(differences),
            "mean_difference": float(differences.mean()),
            "ci95_lower": lower, "ci95_upper": upper, "permutation_p": p_value,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["holm_p"] = holm_adjust(frame["permutation_p"])
    return frame


ORGANS = {1: "main_root", 2: "lateral_root", 3: "seed", 4: "hypocotyl", 5: "leaf", 6: "petiole"}
REQUIRED_SEEDS = {17, 29, 43}
# Added after the pre-specified six-arm analysis was fixed; analysed as its own comparison family.
POST_HOC_EXPERIMENTS = ("baseline_skelrecall",)
LOWER_IS_BETTER = {"root_hd95_px", "root_asd_px", "component_count_error", "skeleton_length_absolute_error_px"}


def load_metrics(paths: list[Path], split_csv: Path) -> pd.DataFrame:
    """Concatenate per-image result files and add the present-organ-only macro Dice.

    An organ absent from the ground truth scores Dice 1.0 on a correct empty prediction, which
    inflates a plain macro average, so ``macro_positive_dice`` averages only organs the split
    manifest marks as present in that image.
    """
    if not paths:
        raise RuntimeError("No per_image_metrics.csv files found; refusing to create synthetic results")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"experiment", "seed", "event_id", "image_id"}
    if missing := required - set(data):
        raise RuntimeError(f"Result files missing columns: {sorted(missing)}")
    data["seed"] = data["seed"].astype(int)
    split = pd.read_csv(split_csv)
    positive = split[["image_id"] + [f"has_class_{cls}" for cls in ORGANS]].drop_duplicates("image_id")
    data = data.merge(positive, on="image_id", how="left")
    dice = {cls: data[f"dice_class_{cls}"].to_numpy() for cls in ORGANS}
    present = {cls: data[f"has_class_{cls}"].fillna(0).to_numpy() for cls in ORGANS}
    counts = sum(present.values())
    data["macro_positive_dice"] = np.where(counts > 0, sum(dice[cls] * present[cls] for cls in ORGANS) / np.maximum(counts, 1), np.nan)
    data["macro_all_dice"] = np.mean([dice[cls] for cls in ORGANS], axis=0)
    return data


def select_evaluation_paths(paths: list[Path], evaluation: str) -> list[Path]:
    """Keep only result files from the named evaluation directory.

    Runs are stored as experiment/seed_N/<evaluation>/results.csv, so a tree holding several
    protocols (test, nominal, ood) would otherwise pool images from all of them into one analysis.
    Paths with no evaluation directory, as written by flat test fixtures, pass through unchanged.
    """
    kept = [path for path in paths
            if all(not part.startswith("evaluation_") or part == evaluation for part in path.parts)]
    if not kept:
        raise RuntimeError(f"No result files found under an '{evaluation}' directory")
    return kept
