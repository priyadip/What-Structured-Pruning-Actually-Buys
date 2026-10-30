#!/usr/bin/env python3
"""Figures for the report. Reads only `results/*.csv` and `results/*.json`."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figures"
plt.rcParams.update({"font.size": 8, "figure.dpi": 160,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def read(name):
    p = RESULTS / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def f(r, k):
    return float(r[k])


def fig_ladder():
    sweep = read("sweep.csv")
    g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
    dense = g0["dense_acc_validation_matched"]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7))

    for cloud, style in (("neuron", "-o"), ("filter_proj", "--s")):
        rs = sorted({(f(r, "alpha"), f(r, "acc"), f(r, "red_total_pct"))
                     for r in sweep if r["strategy"] == "omp_ls"
                     and r["cloud"] == cloud and r["target"] == "dense"
                     and r["seed"] == "0" and r["layers"] == "all"})
        if not rs:
            continue
        ax[0].plot([x[0] for x in rs], [x[1] for x in rs], style, ms=3, lw=1.2,
                   label=f"{cloud} cloud")
        ax[1].plot([x[2] for x in rs], [x[1] for x in rs], style, ms=3, lw=1.2,
                   label=f"{cloud} cloud")
    res = sorted({(f(r, "alpha"), f(r, "acc"), f(r, "red_total_pct"))
                  for r in sweep if r["strategy"] == "omp_ls"
                  and r["target"] == "residual" and r["seed"] == "0"})
    if res:
        ax[0].plot([x[0] for x in res], [x[1] for x in res], "-^", ms=3, lw=1.2,
                   color="tab:green", label="neuron, residual target")
        ax[1].plot([x[2] for x in res], [x[1] for x in res], "-^", ms=3, lw=1.2,
                   color="tab:green", label="neuron, residual target")

    # panel (c): the fine scan that actually settles the knee question
    kp = RESULTS / "knee_test.json"
    if kp.exists():
        kt = json.loads(kp.read_text())
        for sd, v in sorted(kt["seeds"].items()):
            xs = [p[0] for p in v["curve"]]
            ys = [p[1] for p in v["curve"]]
            ax[2].plot(xs, ys, "-o", ms=2.4, lw=1.0,
                       label=f"seed {sd}, slope ratio {v['slope_ratio']:.2f}")
            ax[2].axvline(v["breakpoint"], lw=0.7, ls="--", alpha=0.5,
                          color=ax[2].lines[-1].get_color())
        ax[2].axvline(1.0, color="tab:red", lw=1.0, ls=":")
        ax[2].set_xscale("log")
        ax[2].set_xlabel(r"multiplier $\alpha$, evenly sampled")
        ax[2].set_ylabel("MNLI matched accuracy (%)")
        ax[2].set_title("Fine scan: free breakpoints (dashed) land at\n"
                        r"0.65, 0.75, 0.85, never at $\alpha=1$ (dotted)",
                        fontsize=7.5)
        ax[2].legend(frameon=False, fontsize=6, loc="upper left")

    for a in ax[:2]:
        a.axhline(dense, color="k", lw=0.8)
        a.axhspan(dense - 2.5, dense, color="tab:blue", alpha=0.08)
        a.set_ylabel("MNLI matched accuracy (%)")
        a.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"multiplier $\alpha$   ($M_l=\mathrm{round}(\alpha d_l)$)")
    ax[0].axvline(1.0, color="tab:red", lw=0.9, ls=":")
    ax[0].annotate(r"$\alpha=1$", (1.0, 55), color="tab:red", fontsize=7,
                   rotation=90, va="bottom", ha="right")
    ax[1].set_xlabel("whole-model parameter reduction (%)")
    fig.suptitle("Accuracy is smooth in the budget, with no feature at the "
                 "intrinsic dimension", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_ladder.pdf")
    fig.savefig(FIGS / "fig1_ladder.png")
    plt.close(fig)


def fig_families():
    base = read("baselines.csv")
    if not base:
        return
    g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
    dense = g0["dense_acc_validation_matched"]
    refit = {"omp_ls", "fwdbwd_ls", "span_ls", "magnitude_ls",
             "magnitude_prod_ls", "random_ls", "ot_then_ls", "thinet_lsfull"}
    colour = {"omp_ls": "tab:blue", "fwdbwd_ls": "tab:cyan",
              "span_ls": "tab:purple", "magnitude_ls": "tab:orange",
              "magnitude_prod_ls": "tab:brown", "random_ls": "tab:gray",
              "ot_then_ls": "tab:green", "thinet_lsfull": "tab:olive",
              "omp": "tab:blue", "magnitude": "tab:orange",
              "random": "tab:gray", "thinet": "tab:red",
              "thinet_densetgt": "tab:pink", "ot_intrafusion": "tab:green",
              "kmeans_merge": "tab:olive"}
    by = defaultdict(list)
    for r in base:
        by[r["strategy"]].append((f(r, "alpha"), f(r, "acc")))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for stg, pts in sorted(by.items()):
        pts = sorted(set(pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "-o" if stg in refit else "--x", ms=2.6, lw=1.4 if stg == "omp_ls" else 0.9,
                color=colour.get(stg, "k"),
                alpha=1.0 if stg == "omp_ls" else 0.8, label=stg)
    ax.axhline(dense, color="k", lw=0.8)
    ax.axhspan(dense - 2.5, dense, color="tab:blue", alpha=0.08)
    ax.axhline(100 / 3, color="k", lw=0.6, ls=":")
    ax.annotate("chance", (2.2, 100 / 3 + 0.8), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel(r"multiplier $\alpha$ (identical widths for every operator)")
    ax.set_ylabel("MNLI matched accuracy (%)")
    ax.set_title("Fifteen operators at matched width\n"
                 "solid: consumer refitted; dashed: not", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_families.pdf")
    fig.savefig(FIGS / "fig2_families.png")
    plt.close(fig)


def fig_dims_and_perlayer():
    g = json.loads((RESULTS / "phase2_gride.json").read_text())
    pl = read("perlayer.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))

    L = list(range(6))
    for cloud, off, c in (("neuron", -0.22, "tab:blue"),
                          ("filter_proj", 0.0, "tab:orange"),
                          ("weight", 0.22, "tab:gray")):
        if cloud not in g["clouds"]:
            continue
        v = [g["clouds"][cloud][str(i)]["0"] for i in L]
        ax[0].bar([x + off for x in L], v, width=0.21, color=c, label=cloud)
    sp = g["seed_spread_neuron"]
    ax[0].errorbar(L, [sp[str(i)]["mean"] for i in L],
                   yerr=[sp[str(i)]["sd"] for i in L], fmt="none",
                   ecolor="k", elinewidth=0.8, capsize=2)
    ax[0].set_xticks(L)
    ax[0].set_xlabel("FFN block")
    ax[0].set_ylabel(r"GRIDE intrinsic dimension $d_l$")
    ax[0].set_title(r"$N=3072$ units per block; bars are seed 0, "
                    "whiskers seeds 0-2", fontsize=7.5)
    ax[0].legend(frameon=False, fontsize=6.5)

    if pl:
        by = defaultdict(list)
        for r in pl:
            by[int(r["tag"].split("_L")[1])].append((f(r, "alpha"), f(r, "acc")))
        g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
        dense = g0["dense_acc_validation_matched"]
        for l, pts in sorted(by.items()):
            pts = sorted(set(pts))
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2.6,
                       lw=1.0, label=f"block {l}  (d={g['dims'][l]:.0f})")
        ax[1].axhline(dense, color="k", lw=0.8)
        ax[1].axhspan(dense - 2.5, dense, color="tab:blue", alpha=0.08)
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"multiplier $\alpha$ for that block alone")
        ax[1].set_ylabel("MNLI matched accuracy (%)")
        ax[1].set_title("One block compressed, the other five left dense",
                        fontsize=7.5)
        ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_dims_perlayer.pdf")
    fig.savefig(FIGS / "fig3_dims_perlayer.png")
    plt.close(fig)


def fig_ablation():
    s = json.loads((RESULTS / "summary.json").read_text())
    ab = {k: v for k, v in s.get("ablation", {}).items() if v}
    if not ab:
        return
    order = sorted(ab, key=lambda k: ab[k]["mean"])
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    y = range(len(order))
    ax.barh(list(y), [ab[k]["mean"] for k in order],
            xerr=[2 * ab[k]["se"] for k in order],
            color=["tab:red" if ab[k]["mean"] < 0 else "tab:blue" for k in order],
            height=0.6, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([k.split(" (")[0].replace("_", " ") for k in order],
                       fontsize=6.5)
    ax.set_xlabel("paired accuracy difference (points), matched widths")
    ax.set_title("What each component is worth\n"
                 "whiskers are two standard errors over budgets", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_ablation.pdf")
    fig.savefig(FIGS / "fig4_ablation.png")
    plt.close(fig)


def fig_attention():
    at = [r for r in read("attention.csv") if r["tag"] == "attn_only"]
    if not at:
        return
    g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
    dense = g0["dense_acc_validation_matched"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    by = defaultdict(list)
    for r in at:
        by[r["strategy"]].append((f(r, "alpha"), f(r, "acc"), f(r, "red_total_pct")))
    refit = {"omp_ls", "span_ls", "magnitude_ls", "random_ls"}
    for stg, pts in sorted(by.items()):
        pts = sorted(set(pts))
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts],
                   "-o" if stg in refit else "--x", ms=3,
                   lw=1.6 if stg == "omp_ls" else 1.0, label=stg)
    # alpha spans only 1 to 8 here, so a log axis just crowds the tick labels
    ax[0].set_xlabel(r"multiplier $\alpha$ (heads $=\mathrm{round}(\mathrm{round}"
                     r"(\alpha d)/64)$)")
    ax[0].set_title("Attention heads, matched head counts", fontsize=8)

    # the same contrast on both unit types, to show it inverts
    s = json.loads((RESULTS / "summary.json").read_text())
    ffn_gap = s["ablation"]["omp_vs_span_given_refit (omp_ls - span_ls)"]["mean"]
    gaps = []
    for a, acc, _ in sorted(set(by.get("omp_ls", []))):
        m = [x for x in by.get("span_ls", []) if abs(x[0] - a) < 1e-9]
        if m:
            gaps.append((a, acc - m[0][1]))
    if gaps:
        ax[1].plot([g[0] for g in gaps], [g[1] for g in gaps], "-o", ms=3,
                   color="tab:red", label="attention heads (12 units per block)")
    ax[1].axhline(ffn_gap, color="tab:blue", lw=1.4,
                  label=f"FFN neurons (3072 per block), mean {ffn_gap:+.2f}")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xlabel(r"multiplier $\alpha$")
    ax[1].set_ylabel("OMP minus spanning (points)")
    ax[1].set_title("Targeted selection matters when there are few units\n"
                    "to choose among, and not when there are many", fontsize=8)

    for a in ax:
        a.axhline(dense, color="k", lw=0.8) if a is ax[0] else None
        a.legend(frameon=False, fontsize=6)
    ax[0].axhspan(dense - 2.5, dense, color="tab:blue", alpha=0.08)
    ax[0].set_ylabel("MNLI matched accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_attention.pdf")
    fig.savefig(FIGS / "fig5_attention.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    for fn in (fig_ladder, fig_families, fig_dims_and_perlayer, fig_ablation,
               fig_attention):
        try:
            fn()
            print(f"  {fn.__name__} ok")
        except Exception as exc:
            print(f"  {fn.__name__} skipped: {type(exc).__name__}: {exc}")
    print(f"figures in {FIGS}")
