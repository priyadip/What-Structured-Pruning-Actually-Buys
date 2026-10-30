#!/usr/bin/env python3
"""Figure 1: the unit, the geometry of Theorem 1, and the measured consequence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import (FancyArrowPatch, Polygon, Rectangle,   # noqa: E402
                                FancyBboxPatch)

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 7.0, "figure.dpi": 220, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "legend.frameon": False,
                     "mathtext.fontset": "cm"})

KEEP = "#2166ac"
DROP = "#b2182b"
SPAN = "#d1e5f0"
GOOD = "#3f7f2f"


def panel_a(ax):
    ax.set_xlim(0, 10.6); ax.set_ylim(-0.2, 7.2); ax.axis("off")
    ax.set_title("(a) the compressible unit", fontsize=7.4, pad=3)

    ys = [5.9, 5.05, 4.2, 3.35, 2.5, 1.65]
    kept = [True, False, True, False, True, False]
    for y, k in zip(ys, kept):
        ax.add_patch(Rectangle((1.35, y - 0.24), 1.35, 0.48,
                               fc=KEEP if k else "white",
                               ec=KEEP if k else DROP, lw=1.0,
                               ls="-" if k else (0, (2, 1.4)), alpha=0.9))
    ax.text(2.02, 6.62, r"producer $P_\ell$", ha="center", fontsize=7.0)
    ax.text(2.02, 0.95, r"$N$ units", ha="center", fontsize=6.3, color="0.4")
    ax.text(0.02, 4.6, "keep", color=KEEP, fontsize=6.5, ha="left")
    ax.text(0.02, 4.05, r"$|S|=M$", color=KEEP, fontsize=6.2, ha="left")
    ax.text(0.02, 3.1, "drop", color=DROP, fontsize=6.5, ha="left")

    ax.add_patch(FancyBboxPatch((3.65, 1.9), 1.35, 3.9,
                                boxstyle="round,pad=0.04", fc="0.965",
                                ec="0.45", lw=0.9))
    ax.text(4.32, 6.05, r"$U$", ha="center", fontsize=8.0)
    ax.text(4.32, 1.45, r"$n\!\times\!N$", ha="center", fontsize=6.3, color="0.4")
    for i, k in enumerate(kept):
        ax.add_patch(Rectangle((3.76 + i * 0.2, 2.05), 0.14, 3.6,
                               fc=KEEP if k else DROP, ec="none",
                               alpha=0.6 if k else 0.4))

    ax.add_patch(FancyBboxPatch((6.15, 2.65), 1.15, 2.4,
                                boxstyle="round,pad=0.04", fc="white",
                                ec="0.25", lw=1.1))
    ax.text(6.72, 5.35, r"consumer", ha="center", fontsize=6.8)
    ax.text(6.72, 3.75, r"$V$", ha="center", fontsize=8.0)

    ax.add_patch(FancyBboxPatch((8.55, 2.65), 1.15, 2.4,
                                boxstyle="round,pad=0.04", fc="0.965",
                                ec="0.45", lw=0.9))
    ax.text(9.12, 5.35, r"target", ha="center", fontsize=6.8)
    ax.text(9.12, 3.75, r"$UV^{\!\top}$", ha="center", fontsize=7.4)

    for x0, x1 in ((2.78, 3.58), (5.06, 6.08), (7.37, 8.48)):
        ax.add_patch(FancyArrowPatch((x0, 3.85), (x1, 3.85), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color="0.35"))
    ax.text(6.72, 1.95, "correct:", ha="center", fontsize=6.6, color=GOOD)
    ax.text(6.72, 1.35, r"re-solve $V$", ha="center", fontsize=6.6, color=GOOD)
    ax.add_patch(FancyArrowPatch((6.72, 2.6), (6.72, 2.28), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=GOOD))


def panel_b(ax):
    ax.set_xlim(0, 10.6); ax.set_ylim(-0.2, 7.2); ax.axis("off")
    ax.set_title("(b) the discarded contribution splits orthogonally",
                 fontsize=7.4, pad=3)

    ax.add_patch(Polygon([(0.5, 1.35), (7.7, 0.85), (9.9, 2.5), (2.7, 3.0)],
                         closed=True, fc=SPAN, ec="#4393c3", lw=1.0, alpha=0.9))
    ax.text(9.55, 1.25, r"$\mathrm{span}\,[U_S,\mathbf{1}]$", fontsize=6.8,
            color="#2166ac", ha="right")

    o, tip, foot = (2.6, 2.05), (5.5, 6.15), (6.3, 2.42)

    ax.add_patch(FancyArrowPatch(o, tip, arrowstyle="-|>", mutation_scale=10,
                                 lw=1.9, color=DROP))
    ax.text(2.35, 5.15, r"$D=U_{S^c}V_{S^c}^{\!\top}$", color=DROP,
            fontsize=7.2, ha="left")
    ax.text(2.35, 4.55, r"$\|D\|_F = R_{\mathrm{keep}}$", color=DROP,
            fontsize=6.8, ha="left")

    ax.add_patch(FancyArrowPatch(o, foot, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.6, color=GOOD))
    ax.text(4.3, 1.42, r"$P_S D$", color=GOOD, fontsize=7.2, ha="center")
    ax.text(4.3, 0.85, "what the refit recovers", color=GOOD, fontsize=6.4,
            ha="center")

    ax.add_patch(FancyArrowPatch(foot, tip, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.5, color="0.2"))
    ax.text(6.6, 4.55, r"$(I-P_S)D = R_{\mathrm{fit}}$", color="0.15",
            fontsize=7.2, ha="left")
    ax.text(6.6, 3.95, r"$\leq \|V\|_2\, \tau_M$", color="0.15",
            fontsize=7.0, ha="left")
    ax.text(6.6, 3.35, "irreducible", color="0.35", fontsize=6.4, ha="left")

    dx = 0.3
    ax.plot([foot[0] - dx + 0.05, foot[0] - dx + 0.12, foot[0] + 0.02],
            [foot[1] + 0.13, foot[1] + dx, foot[1] + dx - 0.05],
            color="0.35", lw=0.8)

    ax.text(5.3, -0.05,
            r"$R_{\mathrm{keep}}^{2} = R_{\mathrm{fit}}^{2} + \|P_S D\|_F^{2}$",
            fontsize=8.2, ha="center")


def panel_c(ax):
    d = json.loads((RES / "a8_theory.json").read_text())["by_kept_fraction"]
    ks = sorted(d, key=lambda s: float(s.split("-")[0]))
    x = [0.5 * (float(k.split("-")[0]) + float(k.split("-")[1])) for k in ks]
    rk = [d[k]["r_keep"] for k in ks]
    rf = [d[k]["r_fit"] for k in ks]
    ax.plot(x, rk, "o-", color=DROP, ms=3.0, lw=1.5, label=r"$R_{\mathrm{keep}}$")
    ax.plot(x, rf, "o-", color="0.2", ms=3.0, lw=1.5, label=r"$R_{\mathrm{fit}}$")
    ax.fill_between(x, rf, rk, color=GOOD, alpha=0.16)
    ax.text(0.50, 620, "recovered by\nthe correction", color=GOOD,
            fontsize=6.4, ha="center")
    ax.set_xlabel(r"fraction kept, $M/N$", labelpad=1.5)
    ax.set_ylabel(r"$\|\cdot\|_F$", labelpad=1.5)
    ax.set_title("(c) measured, 154 cells", fontsize=7.4, pad=3)
    ax.legend(fontsize=6.4, loc="upper right", handlelength=1.3)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6.3)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.08))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 0.68],
                          wspace=0.30, left=0.012, right=0.985,
                          top=0.905, bottom=0.135)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    panel_c(fig.add_subplot(gs[2]))
    fig.savefig(FIG / "fig1_mechanism.pdf")
    fig.savefig(FIG / "fig1_mechanism.png")
    print("  fig1_mechanism ok")
