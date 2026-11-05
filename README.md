# Arabidopsis TopoSeg

Topology-aware semantic segmentation of *Arabidopsis thaliana* seedlings, with a leakage-free,
acquisition-event-grouped benchmark built on the ChronoRoot 2.0 release.

The repository contains the audit, split construction, training, evaluation and statistical
pipeline, the frozen split manifests, the archived result tables, and the seven figures.

## Why the protocol matters

The distributed ChronoRoot 2.0 split groups frames by camera folder. All 156 files in its test
directory are byte-identical copies of files in the training directory, and 29 of 47 acquisition
events (579 of 911 mapped images, 63.6%) cross its train/validation/test subsets. Metrics computed
under that assignment therefore estimate performance on near-copies of training data.

This repository ships a frozen, event-disjoint split of the same release:

| subset | images | events |
|---|---|---|
| train | 550 | 24 |
| val | 160 | 11 |
| test | 201 | 12 |

plus 46 images whose acquisition metadata could not be mapped, kept as an out-of-domain stress set
and never used for model selection.

## Method

A compact dual-head U-Net predicts six organs (main root, lateral root, seed, hypocotyl, leaf,
petiole) with an auxiliary centerline head for the thin organs. The loss combines region terms with
a symmetric cluster-wise Dice (clDice) term over per-organ masks and a joint complete-root term.
Six configurations are trained with three seeds each (17, 29, 43) for 30,000 steps: a region-only
baseline, a complete-root clDice baseline, the full loss, and three ablations (uniform organ
weights, no joint complete-root term, no centerline head).

A seventh arm, added after the six-arm analysis, trains the Skeleton Recall loss on the complete
root under the same budget, ramp and seeds (`configs/baseline_skelrecall.yaml`). It is analysed as
its own comparison family by `scripts/analyze_comparator.py`, and the six-arm analysis scripts
exclude it by default.

Evaluation covers region metrics (Dice, IoU, HD95, ASD), topology metrics (clDice, endpoint and
branch-point F1, component count, largest-component recall) and phenotype readouts (root length and
tip counts). Comparisons use event-level paired differences, bootstrap confidence intervals,
sign-flip permutation tests and Holm correction within each metric. The pre-specified primary
comparison, full loss against the region-only baseline, did not show a difference; the archived
tables report every contrast, including that one.

## Repository layout

```
configs/                 six experiment configurations, the Skeleton Recall arm, base.yaml
data/                    data placement instructions (images are not redistributed)
figures/                 the overview schematic and the six result figures, as vector PDFs
manifests/
  splits/main.csv        frozen event-grouped split (550/160/201 images)
  splits/nominal.csv     published test assignment, for the protocol diagnostic
  splits/ood.csv         the 46 unmapped images
  folder_map.csv         opaque folder ids with derived acquisition fields
  leakage_audit.json     leakage-audit summary
  adjacent_frame_correlation.csv
  organ_weights.json     per-organ weights computed from the training labels
results/
  analysis/              three-seed summaries, paired tests, phenotype tables
  analysis_vs_full/      ablation contrasts against the full configuration
  analysis_protocol/     published-split and out-of-domain diagnostics
  analysis_comparator/   Skeleton Recall contrasts (separate Holm family)
  cost/                  parameters, MACs, CPU and GPU latency, and the host record
  runs/                  per-run training logs and test summaries
src/toposeg/             dataset, model, losses, metrics, inference, statistics
scripts/                 audit, split, train, evaluate and analysis entry points
tests/                   CPU tests for data handling, metrics, losses and release paths
```

## Dataset

The images are not redistributed here. Download the ChronoRoot 2.0 dataset from the GigaScience
Database (DOI [10.5524/102804](https://doi.org/10.5524/102804), CC0) and place the Arabidopsis
seedling segmentation subset in nnU-Net format:

```
<data root>/
  imagesTr/
  labelsTr/
  imagesTs/
```

Point the pipeline at it with an absolute path:

```bash
export TOPOSEG_DATA_ROOT=/absolute/path/to/08_image_seedling_segmentation
```

The split manifests store paths relative to this root, so no machine-specific directory structure
is published. `imagesTs/` is used only by the protocol diagnostics; the frozen split is built from
`imagesTr/` and `labelsTr/`.

## Installation

```bash
conda env create -f environment.yml
conda activate toposeg
```

or, with an existing Python 3.10 environment:

```bash
pip install -e .
```

## Running the pipeline

```bash
# 1. audit the release and measure the leakage
python scripts/audit_data.py
python scripts/audit_leakage.py

# 2. rebuild the split (the frozen manifests are already in manifests/splits/)
python scripts/make_splits.py

# 3. train and evaluate one configuration
python scripts/train.py --config configs/full.yaml --seed 17 --final
python scripts/evaluate.py --config configs/full.yaml --seed 17 --split test

# 4. aggregate the three seeds and run the paired statistics
python scripts/analyze_results.py --input outputs --output outputs/analysis \
    --split-csv manifests/splits/main.csv
python scripts/analyze_phenotypes.py --input outputs --output outputs/analysis
python scripts/analyze_results.py --baseline full --output outputs/analysis_vs_full
python scripts/analyze_phenotypes.py --baseline full --output outputs/analysis_vs_full

# 5. protocol diagnostics against the published assignment and the unmapped images
python scripts/analyze_protocol.py --input outputs --splits manifests/splits \
    --output outputs/analysis_protocol

# 6. Skeleton Recall comparator, as a separate family
python scripts/analyze_comparator.py --output outputs/analysis_comparator

# 7. figures: the six result figures, then the overview schematic
python scripts/make_figures.py --analysis outputs/analysis
python scripts/make_overview_figure.py --out figures/fig0_overview.pdf

# 8. computational cost: parameters, MACs, latency on both devices
python scripts/measure_cost.py --device cpu --reps 30 --image-reps 5
python scripts/measure_cost.py --device cuda --reps 30 --image-reps 5
```

`measure_cost.py` cross-checks each parameter count against the `best.pt` of the three seeds under
`--runs` (default `outputs/`); the weights are not redistributed here, so on a fresh clone the
check is skipped with a note and the count comes from the model code and configuration alone.
Take the timings with nothing else running: latency is the one column that a loaded machine
changes.

`scripts/run_all.bat` runs the complete six-configuration, three-seed queue and skips runs whose
test summary already exists; `scripts/run_protocol_eval.bat` runs the two diagnostic protocols and
`scripts/run_skelrecall.bat` the three Skeleton Recall seeds. The six-arm matrix is 18 runs of
30,000 steps (about 3.2 days on one GTX 1080); the comparator adds three more.

## Results

The archived tables that back the figures and the reported numbers are in `results/analysis/`,
`results/analysis_vs_full/`, `results/analysis_protocol/` and `results/analysis_comparator/`.
Per-run `training.csv` logs and test summaries are in `results/runs/`. The complexity table is read
from `results/cost/`: one CSV per device, each row carrying its `patch_reps`, `warmup_reps` and
`frame_reps`, plus the `hardware_cpu.json` and `hardware_cuda.json` records of the machine they ran
on. Re-running the analysis scripts on these files reproduces the tables without retraining.

## Tests

```bash
pytest
```

The suite runs on CPU and covers dataset geometry and ignore handling, the topology losses
(including Skeleton Recall) and metrics, inference, the statistical helpers, and the
path-scrubbing used for the released manifests.

## Citation

If you use this code, please cite the accompanying article (under review) and this repository.
Machine-readable metadata is in `CITATION.cff`.

## License

MIT; see `LICENSE`.

## Contact

Qi Peng, School of Physics and Electronic Information Engineering, Hubei Engineering University,
Xiaogan, Hubei, China. petersky0316@hbeu.edu.cn
