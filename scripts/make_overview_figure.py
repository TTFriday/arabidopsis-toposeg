"""Method overview schematic: protocol, network, objective, evaluation.

Every quantity drawn here is a configuration parameter stated in the code and the
configs; nothing in the schematic is a measured result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from toposeg import figstyle

EDGE = "#30475e"
FILL = {"protocol": "#e8eef6", "net": "#eef6ea", "loss": "#fbf1e4", "eval": "#f3ecf6"}
ACCENT = {"train": "#3b6ea8", "val": "#7aa6d6", "test": "#c44e52", "ood": "#8c8c8c"}


def box(ax, x, y, w, h, fill, text=None, size=6.3, weight="normal", ha="center", lw=0.6, ls="-"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       fc=fill, ec=EDGE, lw=lw, ls=ls)
    )
    if text is not None:
        tx = x + w / 2 if ha == "center" else x + 0.008
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=lw, color=EDGE))


def panel_title(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=6.8, weight="bold", color=EDGE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update(figstyle.BASE)
    fig = plt.figure(figsize=(figstyle.PRINT_WIDTH_IN, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top = 0.90
    # (a) protocol
    xa, wa = 0.005, 0.255
    panel_title(ax, xa, top + 0.01, "(a) Audit and split")
    box(ax, xa, 0.05, wa, top - 0.05, FILL["protocol"], lw=0.5)
    box(ax, xa + 0.012, 0.74, wa - 0.024, 0.12, "white",
        "957 unique annotated images\n911 mapped to source folders", size=6.0)
    arrow(ax, xa + wa / 2, 0.74, xa + wa / 2, 0.705)
    box(ax, xa + 0.012, 0.585, wa - 0.024, 0.12, "white",
        "47 acquisition events\n(condition, device, timestamp)", size=6.0)
    arrow(ax, xa + wa / 2, 0.585, xa + wa / 2, 0.55)
    ax.text(xa + wa / 2, 0.525, "event-grouped greedy assignment", ha="center", va="center", fontsize=5.8, style="italic")
    sub = [("train", "550 img\n24 ev"), ("val", "160 img\n11 ev"), ("test", "201 img\n12 ev")]
    sw = (wa - 0.024 - 2 * 0.008) / 3
    for i, (name, label) in enumerate(sub):
        sx = xa + 0.012 + i * (sw + 0.008)
        ax.add_patch(FancyBboxPatch((sx, 0.335), sw, 0.155, boxstyle="round,pad=0.003,rounding_size=0.01",
                                    fc=ACCENT[name], ec="none", alpha=0.9))
        ax.text(sx + sw / 2, 0.455, name, ha="center", va="center", fontsize=6.0, weight="bold", color="white")
        ax.text(sx + sw / 2, 0.383, label, ha="center", va="center", fontsize=5.6, color="white", linespacing=1.15)
    box(ax, xa + 0.012, 0.185, wa - 0.024, 0.115, "white",
        "46 unmapped images: out-of-domain\nstress set (no paired contrast)", size=5.7, ls="--")
    box(ax, xa + 0.012, 0.07, wa - 0.024, 0.095, "white",
        "diagnostic: 156 published test images", size=5.7, ls=":")

    # (b) network
    xb, wb = 0.275, 0.215
    panel_title(ax, xb, top + 0.01, "(b) Dual-head U-Net")
    box(ax, xb, 0.05, wb, top - 0.05, FILL["net"], lw=0.5)
    box(ax, xb + 0.012, 0.76, wb - 0.024, 0.10, "white", "512$\\times$512 patch, 16-bit gray", size=6.0)
    # encoder/decoder sketch
    levels = [0.105, 0.085, 0.065, 0.045, 0.028]
    cx = xb + wb / 2
    y = 0.70
    for i, w in enumerate(levels):
        hgt = 0.055
        yy = y - i * 0.068
        ax.add_patch(FancyBboxPatch((cx - 0.008 - w * 0.62, yy - hgt), w * 0.55, hgt * 0.8,
                                    boxstyle="square,pad=0", fc="#9cc59a", ec=EDGE, lw=0.4))
        ax.add_patch(FancyBboxPatch((cx + 0.008 + w * 0.07, yy - hgt), w * 0.55, hgt * 0.8,
                                    boxstyle="square,pad=0", fc="#cfe5cc", ec=EDGE, lw=0.4))
        if i < len(levels) - 1:
            ax.plot([cx - 0.008 - w * 0.07, cx + 0.008 + w * 0.07], [yy - hgt * 0.6] * 2, lw=0.4, color=EDGE, ls=":")
    ax.text(xb + 0.014, 0.64, "encoder", fontsize=5.4, rotation=90, va="center", ha="left")
    ax.text(xb + wb - 0.014, 0.64, "decoder", fontsize=5.4, rotation=90, va="center", ha="right")
    ax.text(cx, 0.335, "5 scales, 24–256 channels\n3.31 M parameters, 26.1 GMAC", ha="center", va="center", fontsize=5.8)
    hw = (wb - 0.024 - 0.01) / 2
    box(ax, xb + 0.012, 0.085, hw, 0.17, "white", "region head\n7 logits\n(inference)", size=5.7)
    box(ax, xb + 0.022 + hw, 0.085, hw, 0.17, "white", "centerline head\n4 channels\n(training only)", size=5.7, ls="--")
    arrow(ax, cx - 0.03, 0.30, xb + 0.012 + hw / 2, 0.258)
    arrow(ax, cx + 0.03, 0.30, xb + 0.022 + 1.5 * hw, 0.258)

    # (c) objective
    xc, wc = 0.505, 0.27
    panel_title(ax, xc, top + 0.01, "(c) Training objectives")
    box(ax, xc, 0.05, wc, top - 0.05, FILL["loss"], lw=0.5)
    box(ax, xc + 0.012, 0.73, wc - 0.024, 0.13, "white",
        "$L = L_{\\mathrm{reg}} + g(\\tau)\\,(0.20\\,L_{\\mathrm{topo}} + 0.10\\,L_{\\mathrm{skel}})$\n"
        "$L_{\\mathrm{reg}} = 0.5\\,L_{\\mathrm{CE}} + 0.5\\,L_{\\mathrm{Dice}}$", size=6.0)
    box(ax, xc + 0.012, 0.565, wc - 0.024, 0.14, "white",
        "$L_{\\mathrm{topo}} = 0.3\\,(1-\\mathrm{clDice}_{\\mathrm{root}})"
        " + 0.7\\sum_c \\hat{\\alpha}_c (1-\\mathrm{clDice}_c)$\n"
        "joint root term  |  slimness weights $\\hat{\\alpha}_c$", size=5.1)
    arms = [
        "region-only: $L_{\\mathrm{reg}}$",
        "joint-root: + root clDice only",
        "full: complete objective",
        "uniform: $\\hat{\\alpha}_c = 1/n$",
        "no joint root: drop root term",
        "no centerline head: drop $L_{\\mathrm{skel}}$",
        "post hoc: arm 2 with Skeleton Recall",
    ]
    for i, a in enumerate(arms):
        ax.text(xc + 0.018, 0.515 - i * 0.063, f"{i + 1}  {a}", fontsize=5.7, va="center",
                style="italic" if i == 6 else "normal")
    ax.text(xc + wc - 0.012, 0.08, "ramp $g$: 0 until 10%, 1 from 25% of steps",
            fontsize=5.3, ha="right", va="center", color="#555555")

    # (d) evaluation
    xd, wd = 0.79, 0.205
    panel_title(ax, xd, top + 0.01, "(d) Evaluation")
    box(ax, xd, 0.05, wd, top - 0.05, FILL["eval"], lw=0.5)
    steps = [
        ("whole-frame sliding window\n50% overlap, Gaussian blend", 0.745),
        ("regional: Dice, HD95, ASD\ntopological: clDice, endpoint F1", 0.565),
        ("phenotype: root length,\ntip and branch counts", 0.385),
        ("event-level paired contrasts\n12 events, 3 seeds, Holm", 0.205),
    ]
    for i, (t, yy) in enumerate(steps):
        box(ax, xd + 0.012, yy, wd - 0.024, 0.13, "white", t, size=5.7)
        if i < len(steps) - 1:
            arrow(ax, xd + wd / 2, yy, xd + wd / 2, yy - 0.048)
    ax.text(xd + wd / 2, 0.11, "published split scored\nas a diagnostic only",
            fontsize=5.2, ha="center", va="center", color="#555555")

    for x0, x1 in [(xa + wa, xb), (xb + wb, xc), (xc + wc, xd)]:
        arrow(ax, x0 + 0.001, 0.5, x1 - 0.001, 0.5, lw=1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
