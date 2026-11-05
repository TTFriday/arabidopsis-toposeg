from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from toposeg import figstyle
from toposeg.stats import POST_HOC_EXPERIMENTS, favour_label, paired_by_event, select_evaluation_paths

REQUIRED_SEEDS = {17, 29, 43}
PHENO_COLUMNS = {
    "main_root_length": "main_root_length_px",
    "lateral_root_length": "lateral_root_length_px",
    "complete_root_length": "complete_root_length_px",
    "root_endpoints": "root_endpoints",
    "root_branch_points": "root_branch_points",
}
UNITS = {"main_root_length_px": "px", "lateral_root_length_px": "px", "complete_root_length_px": "px", "root_endpoints": "count", "root_branch_points": "count"}


def load_phenotypes(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise RuntimeError("No phenotype_errors.csv files found; refusing to create synthetic results")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"experiment", "seed", "event_id", "image_id"}
    if missing := required - set(data):
        raise RuntimeError(f"Phenotype files missing columns: {sorted(missing)}")
    data["seed"] = data["seed"].astype(int)
    incomplete = {
        experiment: sorted(REQUIRED_SEEDS - set(group.seed))
        for experiment, group in data.groupby("experiment")
        if set(group.seed) != REQUIRED_SEEDS
    }
    if incomplete:
        raise RuntimeError(f"Three-seed phenotype results are incomplete: {incomplete}")
    return data


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, per_experiment in data.groupby("experiment"):
        for label, stem in PHENO_COLUMNS.items():
            gt, error = per_experiment[f"gt_{stem}"], per_experiment[f"absolute_error_{stem}"]
            visible = per_experiment[gt > 0]
            invisible = per_experiment[gt == 0]
            bias = visible[f"error_{stem}"]
            variable = len(visible) > 2 and visible[f"pred_{stem}"].std() > 0 and visible[f"gt_{stem}"].std() > 0
            rho = spearmanr(visible[f"pred_{stem}"], visible[f"gt_{stem}"]).statistic if variable else np.nan
            phantom = invisible.loc[invisible[f"pred_{stem}"] > 0, f"pred_{stem}"]
            rows.append({
                "experiment": experiment,
                "phenotype": label,
                "unit": UNITS[stem],
                "images_with_ground_truth": len(visible),
                "images_without_ground_truth": len(invisible),
                "mae": visible[f"absolute_error_{stem}"].mean(),
                "median_absolute_error": visible[f"absolute_error_{stem}"].median(),
                "normalised_mae": visible[f"absolute_error_{stem}"].sum() / max(1e-9, visible[f"gt_{stem}"].sum()),
                "bias": bias.mean(),
                "limits_lower": bias.mean() - 1.96 * bias.std(),
                "limits_upper": bias.mean() + 1.96 * bias.std(),
                "spearman": rho,
                "phantom_rate": float((invisible[f"pred_{stem}"] > 0).mean()) if len(invisible) else np.nan,
                "phantom_mean_value": phantom.mean(),
            })
    return pd.DataFrame(rows)


def _mixed_fit(merged: pd.DataFrame, predictors: list[str]) -> tuple[object, Exception | None]:
    """Fit a random-intercept model, retrying with less demanding estimators.

    A fit is accepted only when the optimiser reports convergence and both the coefficients and
    their standard errors are finite; otherwise the failure is returned for reporting.
    """
    formula = "delta_phenotype_error ~ " + " + ".join(predictors)
    last: Exception | None = None
    for options in ({"reml": True}, {"reml": False}, {"reml": False, "method": "nm"}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = smf.mixedlm(formula, merged, groups=merged["event_id"]).fit(disp=False, **options)
            values = fitted.params[predictors].to_numpy(dtype=float)
            errors = fitted.bse[predictors].to_numpy(dtype=float)
            if bool(fitted.converged) and np.isfinite(values).all() and np.isfinite(errors).all():
                return fitted, None
            last = RuntimeError(f"not converged (converged={fitted.converged}, options={options})")
        except Exception as error:
            last = error
    return None, last


def _ols_fit(frame: pd.DataFrame, predictors: list[str]) -> tuple[object, Exception | None]:
    """Ordinary least squares for the event-level unit, where each row is already one event."""
    formula = "delta_phenotype_error ~ " + " + ".join(predictors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.ols(formula, frame).fit()
        values = fitted.params[predictors].to_numpy(dtype=float)
        errors = fitted.bse[predictors].to_numpy(dtype=float)
        if np.isfinite(values).all() and np.isfinite(errors).all():
            return fitted, None
        return None, RuntimeError(f"non-finite coefficients (options=ols, predictors={predictors})")
    except Exception as error:
        return None, error


def association(data: pd.DataFrame, metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Do per-image topology gains explain per-image phenotype error gains beyond region gains?

    Two units are reported. ``seed_x_image`` pools every test image and run: within an image a better
    mask mechanically has a smaller measurement error, so that slope is descriptive and overstates the
    effect of changing configuration. ``event`` averages each configuration within an acquisition event
    first, so its slope compares events; with few events it stays exploratory either way.

    Region and topology gains are usually correlated across configurations, so a joint model is
    reported only when the two predictors are not near-collinear; marginal models are always reported.
    """
    keep = ["experiment", "seed", "event_id", "image_id", "complete_root_cldice", "complete_root_dice"]
    metrics = metrics[keep]
    rows = []
    for experiment in sorted(set(data.experiment) - {baseline}):
        for label in ("complete_root_length", "root_endpoints", "root_branch_points"):
            stem = PHENO_COLUMNS[label]
            base_metric = metrics[metrics.experiment == baseline].set_index(["seed", "image_id"])[["complete_root_cldice", "complete_root_dice"]]
            method_metric = metrics[metrics.experiment == experiment].set_index(["seed", "image_id"])[["complete_root_cldice", "complete_root_dice"]]
            gains = (method_metric - base_metric).rename(columns={"complete_root_cldice": "delta_cldice", "complete_root_dice": "delta_dice"}).reset_index()
            base_pheno = data[data.experiment == baseline].set_index(["seed", "image_id"])[f"absolute_error_{stem}"]
            method_pheno = data[data.experiment == experiment].set_index(["seed", "image_id"])[f"absolute_error_{stem}"]
            delta_pheno = (method_pheno - base_pheno).rename("delta_phenotype_error").reset_index()
            merged = gains.merge(delta_pheno, on=["seed", "image_id"]).merge(
                metrics[["seed", "image_id", "event_id"]].drop_duplicates(), on=["seed", "image_id"]
            ).dropna()
            varying = [term for term in ("delta_cldice", "delta_dice") if merged[term].std() > 0]
            correlation = float(merged[["delta_cldice", "delta_dice"]].corr().iloc[0, 1]) if len(varying) == 2 else np.nan
            if len(merged) < 30 or not varying:
                rows.append({"experiment": experiment, "phenotype": label, "unit": "seed_x_image",
                             "status": "insufficient variation", "images": len(merged)})
                continue
            units = [("seed_x_image", merged)]
            cell = merged.groupby("event_id", as_index=False)[["delta_cldice", "delta_dice", "delta_phenotype_error"]].mean()
            if len(cell) >= 6:
                units.append(("event", cell))
            for unit, frame in units:
                active = [term for term in varying if frame[term].std() > 0]
                if not active:
                    continue
                models = [("marginal", [term]) for term in active]
                if len(active) == 2 and abs(frame[["delta_cldice", "delta_dice"]].corr().iloc[0, 1]) < 0.95:
                    models.append(("joint", active))
                fit = _mixed_fit if unit == "seed_x_image" else _ols_fit
                for kind, predictors in models:
                    model, error = fit(frame, predictors)
                    if model is None:
                        rows.append({"experiment": experiment, "phenotype": label, "unit": unit, "model": kind,
                                     "status": f"failed: {type(error).__name__}: {error}", "images": len(merged),
                                     "observations": len(frame), "predictor_correlation": correlation})
                        continue
                    conf = model.conf_int()
                    for term in predictors:
                        rows.append({
                            "experiment": experiment, "phenotype": label, "unit": unit, "model": kind, "status": "ok",
                            "images": int(len(merged)), "observations": int(model.nobs),
                            "events": int(frame.event_id.nunique()), "predictor_correlation": correlation,
                            "term": term, "estimate": float(model.params[term]), "std_error": float(model.bse[term]),
                            "ci95_lower": float(conf.loc[term, 0]), "ci95_upper": float(conf.loc[term, 1]),
                            "p_value": float(model.pvalues[term]),
                        })
    return pd.DataFrame(rows)


def figures(data: pd.DataFrame, metrics: pd.DataFrame, baseline: str, output: Path) -> None:
    for experiment in sorted(set(data.experiment)):
        panels = 1 if experiment == baseline else 2
        with plt.rc_context(figstyle.BASE):
            figure, axes = plt.subplots(1, panels, figsize=(figstyle.PRINT_WIDTH_IN, 2.6), constrained_layout=True)
            axes = np.atleast_1d(axes)
            palette = plt.get_cmap("tab10").colors
            for index, (seed, group) in enumerate(data[data.experiment == experiment].groupby("seed")):
                colour = palette[index % 10]
                visible = group[group.gt_complete_root_length_px > 0]
                mean = (visible.pred_complete_root_length_px + visible.gt_complete_root_length_px) / 2
                difference = visible.pred_complete_root_length_px - visible.gt_complete_root_length_px
                axes[0].scatter(mean, difference, s=3, alpha=0.4, color=colour, label=f"seed {seed}")
                axes[0].axhline(difference.mean(), ls="--", lw=0.8, color=colour)
                axes[0].axhline(difference.mean() + 1.96 * difference.std(), ls=":", lw=0.8, color=colour)
                axes[0].axhline(difference.mean() - 1.96 * difference.std(), ls=":", lw=0.8, color=colour)
            axes[0].set_xscale("log")
            axes[0].set_xlabel("mean length, prediction and truth (px)")
            axes[0].set_ylabel("prediction − ground truth (px)")
            axes[0].set_title(f"Bland–Altman: {figstyle.EXPERIMENT_SHORT.get(experiment, experiment)}")
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
            if experiment != baseline:
                base = metrics[metrics.experiment == baseline].set_index(["seed", "image_id"])["complete_root_cldice"]
                method = metrics[metrics.experiment == experiment].set_index(["seed", "image_id"])["complete_root_cldice"]
                gains = (method - base).rename("delta_cldice").reset_index()
                errors = data[data.experiment == experiment].set_index(["seed", "image_id"])["absolute_error_complete_root_length_px"]
                reference = data[data.experiment == baseline].set_index(["seed", "image_id"])["absolute_error_complete_root_length_px"]
                delta = (errors - reference).rename("delta_error").reset_index()
                points = gains.merge(delta, on=["seed", "image_id"])
                axes[1].scatter(points.delta_cldice, points.delta_error, s=3, alpha=0.4)
                slope, intercept = np.polyfit(points.delta_cldice, points.delta_error, 1)
                xs = np.linspace(points.delta_cldice.min(), points.delta_cldice.max(), 50)
                axes[1].plot(xs, slope * xs + intercept, lw=1.2)
                axes[1].set_xlabel(f"Δ complete-root clDice vs "
                                   f"{figstyle.EXPERIMENT_SHORT.get(baseline, baseline)}")
                axes[1].set_ylabel("Δ length absolute error (px)")
                axes[1].set_title(f"pooled image-level slope\n{slope:.0f} px per clDice unit")
        figstyle.save(figure, output / f"phenotype_{experiment}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-seed root phenotyping accuracy and the topology–phenotype association")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--baseline", default="baseline_region")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--exclude", nargs="*", default=list(POST_HOC_EXPERIMENTS))
    parser.add_argument("--evaluation", default="evaluation_test",
                        help="Only read results from evaluation directories with this name, so that the nominal and out-of-domain diagnostics cannot mix into the protocol analysis")
    args = parser.parse_args()

    data = load_phenotypes(select_evaluation_paths(sorted(args.input.glob("**/phenotype_errors.csv")), args.evaluation))
    metric_paths = select_evaluation_paths(sorted(args.input.glob("**/per_image_metrics.csv")), args.evaluation)
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics["seed"] = metrics["seed"].astype(int)
    data = data[~data.experiment.isin(args.exclude)]
    metrics = metrics[~metrics.experiment.isin(args.exclude)]
    if args.baseline not in set(data.experiment):
        raise RuntimeError(f"Baseline {args.baseline!r} not found in phenotype results")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarise(data)
    summary.to_csv(args.output / "phenotype_summary.csv", index=False)

    events = data.groupby(["experiment", "seed", "event_id"])[[f"absolute_error_{stem}" for stem in PHENO_COLUMNS.values()]].mean().reset_index()
    rng = np.random.default_rng(20261016)
    paired = pd.concat(
        [paired_by_event(events, args.baseline, f"absolute_error_{stem}", rng, args.iterations) for stem in PHENO_COLUMNS.values()], ignore_index=True
    )
    paired["direction"] = "lower"
    paired["favors"] = favour_label(paired, set(paired.metric))
    paired.to_csv(args.output / "phenotype_paired.csv", index=False)

    link = association(data, metrics, args.baseline)
    link.to_csv(args.output / "topology_phenotype_association.csv", index=False)
    figures(data, metrics, args.baseline, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
