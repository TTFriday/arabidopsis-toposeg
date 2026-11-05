"""Figure styling that keeps the printed point size constant across figures.

Figures are authored at the width they are displayed at, :data:`PRINT_WIDTH_IN`, so the sizes in
:data:`BASE` are the sizes a reader sees. Authoring wider shrinks the text in proportion: an
earlier ten-panel summary figure was authored 40 inches wide and printed its 7 pt text at 1.3 pt.
Every figure here is therefore authored at :data:`PRINT_WIDTH_IN`, and :func:`check_width` fails
the run if one is not.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt

PRINT_WIDTH_IN = 0.98 * 372.0 / 72.0  # 5.063 in

BASE = {
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.8,
    "figure.dpi": 200,
    # A panel is a fifth of the text block: an offset multiplier does not fit beside the title.
    "axes.formatter.useoffset": False,
}

# Row order used by the summary figure and the result tables.
EXPERIMENT_ORDER = [
    "baseline_region",
    "baseline_cldice",
    "full",
    "ablation_uniform",
    "ablation_no_joint_root",
    "ablation_no_skeleton_head",
]

# Short labels for a one-inch column or a six-entry legend.
EXPERIMENT_SHORT = {
    "baseline_region": "region-only",
    "baseline_cldice": "joint-root",
    "full": "full",
    "ablation_uniform": "uniform",
    "ablation_no_joint_root": "no joint root",
    "ablation_no_skeleton_head": "no centreline head",
    "baseline_skelrecall": "Skeleton Recall",
}

# Panel titles, wrapped so that no title is wider than a fifth of the text block. The two
# longest names need three lines: at two lines they ran into the title of the neighbouring
# panel, because a fifth of the text block is only about 61 pt wide.
METRIC_TITLE = {
    "complete_root_dice": "complete-root\nDice",
    "complete_root_cldice": "complete-root\nclDice",
    "endpoint_f1": "endpoint F1",
    "branchpoint_f1": "branch-point F1",
    "skeleton_length_absolute_error_px": "skeleton-length\nerror (px)",
    "macro_positive_dice": "macro-Dice",
    "root_hd95_px": "root HD95 (px)",
    "root_asd_px": "root ASD (px)",
    "largest_component_recall": "largest\ncomponent\nrecall",
    "component_count_error": "component\ncount\nerror",
}


def check_width(figure: matplotlib.figure.Figure) -> None:
    width = float(figure.get_size_inches()[0])
    if abs(width - PRINT_WIDTH_IN) > 0.01:
        raise RuntimeError(
            f"figure is {width:.3f} in wide, not the {PRINT_WIDTH_IN:.3f} in it is printed at; "
            "its text would be scaled by the ratio"
        )


def save(figure: matplotlib.figure.Figure, stem) -> None:
    """Write the PNG preview and the vector PDF, refusing a rescaled figure."""
    check_width(figure)
    figure.savefig(stem.with_suffix(".png"), dpi=BASE["figure.dpi"])
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)