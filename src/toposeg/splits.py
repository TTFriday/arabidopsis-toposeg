from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_SEED, NUM_CLASSES

SPLIT_RATIOS = {"train": 0.65, "val": 0.15, "test": 0.20}


def _group_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    conditions = sorted(frame["condition"].fillna("unknown").astype(str).unique())
    columns = [f"condition={x}" for x in conditions] + [f"class={c}" for c in range(1, NUM_CLASSES)]
    condition_counts = frame["condition"].fillna("unknown").astype(str).value_counts()
    features = []
    for _, group in frame.groupby("event_id", sort=False):
        condition = str(group["condition"].iloc[0])
        cond = [len(group) if condition == x else 0 for x in conditions]
        classes = [group.get(f"has_class_{c}", pd.Series(0, index=group.index)).sum() for c in range(1, NUM_CLASSES)]
        rarity = 1.0 / max(1, condition_counts[condition])
        features.append([len(group), rarity, *cond, *classes])
    return np.asarray(features, dtype=float), columns


def make_group_splits(
    manifest: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratios = ratios or SPLIT_RATIOS
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must sum to one")
    mapped = manifest[manifest.get("mapped", True).astype(bool)].copy()
    ood = manifest[~manifest.get("mapped", True).astype(bool)].copy()
    ood["split"] = "ood"
    if mapped.empty:
        raise ValueError("No mapped images available for main split")

    groups = [(str(event), group.copy()) for event, group in mapped.groupby("event_id", sort=True)]
    rng = np.random.default_rng(seed)
    tie = {event: value for event, value in zip([g[0] for g in groups], rng.random(len(groups)))}
    condition_frequency = Counter(mapped["condition"].astype(str))
    groups.sort(key=lambda x: (-len(x[1]), condition_frequency[str(x[1]["condition"].iloc[0])], tie[x[0]]))

    names = list(ratios)
    target_n = {s: ratios[s] * len(mapped) for s in names}
    class_cols = [f"has_class_{c}" for c in range(1, NUM_CLASSES) if f"has_class_{c}" in mapped]
    totals = {c: float(mapped[c].sum()) for c in class_cols}
    assigned_n = {s: 0 for s in names}
    assigned_cls = {s: Counter() for s in names}
    assigned_conditions = {s: Counter() for s in names}
    assignments: dict[str, str] = {}

    for index, (event, group) in enumerate(groups):
        condition = str(group["condition"].iloc[0])
        class_counts = {c: float(group[c].sum()) for c in class_cols}
        candidates = names if index >= len(names) else [names[index]]
        best_split, best_score = None, float("inf")
        for split in candidates:
            n_before = assigned_n[split]
            n_after = n_before + len(group)
            size_score = 10.0 * (
                (n_after - target_n[split]) ** 2 - (n_before - target_n[split]) ** 2
            ) / max(1.0, len(mapped) ** 2)
            class_score = sum(
                (
                    (assigned_cls[split][c] + count - ratios[split] * totals[c]) ** 2
                    - (assigned_cls[split][c] - ratios[split] * totals[c]) ** 2
                ) / max(1.0, totals[c] ** 2)
                for c, count in class_counts.items()
            )
            condition_target = ratios[split] * condition_frequency[condition]
            condition_before = assigned_conditions[split][condition]
            condition_score = (
                (condition_before + len(group) - condition_target) ** 2
                - (condition_before - condition_target) ** 2
            ) / max(1.0, condition_frequency[condition] ** 2)
            overfill = 10.0 * max(0.0, (n_after - target_n[split]) / max(1.0, target_n[split])) ** 2
            score = size_score + class_score + condition_score + overfill
            if score < best_score:
                best_split, best_score = split, score
        assert best_split is not None
        assignments[event] = best_split
        assigned_n[best_split] += len(group)
        assigned_cls[best_split].update(class_counts)
        assigned_conditions[best_split][condition] += len(group)

    mapped["split"] = mapped["event_id"].astype(str).map(assignments)
    assert not mapped["split"].isna().any()
    assert mapped.groupby("event_id")["split"].nunique().max() == 1
    mapped = _rebalance_conditions(mapped)
    validate_splits(manifest, mapped, ood)
    return mapped.sort_values("image_id"), ood.sort_values("image_id")


def _rebalance_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Move the smallest event of any condition that is absent from val/test there.

    Only applies when the condition still keeps at least one event in train,
    preserving the event-level grouping guarantee.
    """
    for target in ("test", "val"):
        present = set(frame.loc[frame["split"] == target, "condition"])
        for condition in sorted(frame["condition"].astype(str).unique()):
            if condition in present:
                continue
            events = frame[(frame["condition"] == condition) & (frame["split"] == "train")].groupby("event_id").size()
            if len(events) < 2:
                continue
            smallest = events.idxmin()
            frame.loc[frame["event_id"] == smallest, "split"] = target
    return frame


def validate_splits(source: pd.DataFrame, main: pd.DataFrame, ood: pd.DataFrame) -> None:
    split_ids = {s: set(main.loc[main["split"] == s, "image_id"]) for s in SPLIT_RATIOS}
    for left in split_ids:
        for right in split_ids:
            if left < right:
                assert split_ids[left].isdisjoint(split_ids[right]), f"Overlap: {left}/{right}"
    assert set(main["image_id"]).isdisjoint(set(ood["image_id"]))
    assert set(main["image_id"]) | set(ood["image_id"]) == set(source["image_id"])
    for hash_col in ("image_sha256", "label_sha256"):
        if hash_col in source and source[hash_col].astype(str).str.len().gt(0).any():
            hash_splits = main.groupby(hash_col)["split"].nunique()
            assert hash_splits.max() <= 1, f"Duplicate {hash_col} crosses splits"


def write_splits(main: pd.DataFrame, ood: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(output_dir / "main.csv", index=False)
    ood.to_csv(output_dir / "ood.csv", index=False)
