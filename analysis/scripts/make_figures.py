#!/usr/bin/env python3
"""Figures. Every panel earns its place by carrying a claim the text makes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from load_all import load_all, summarise                           # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 7.5, "figure.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.22, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "legend.frameon": False})
CLR = {"mlp": "tab:blue", "cnn": "tab:orange", "transformer": "tab:green"}
MARK = {"mlp": "o", "cnn": "s", "transformer": "^"}


def gap_cells(runs, a, b):
    out = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == a}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == b}
        for k in set(ia) & set(ib):
            ra, rb = ia[k], ib[k]
            out.append({"arch": arch, "gap": ra.acc - rb.acc, "alpha": ra.mult,
                        "drop": ra.dense - max(ra.acc, rb.acc)})
    return out


def fig1():
    """The whole argument in three panels."""
    runs = load_all()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15))

    # (a) both contrasts against regime, per architecture
    sel = gap_cells(runs, "omp_ls", "span_ls")
    cor = gap_cells(runs, "ot_then_ls", "ot_intrafusion")
    bands = [(0, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 1e9)]
    xs = [1.25, 3.75, 7.5, 15, 30]
    for name, cs, c, ls in (("correction", cor, "tab:red", "-"),
                            ("selection", sel, "tab:blue", "-")):
        ys, es = [], []
        for lo, hi in bands:
            s = summarise([q["gap"] for q in cs if lo <= q["drop"] < hi])
            ys.append(s["mean"] if s else float("nan"))
            es.append(2 * s["se"] if s else 0)
        ax[0].errorbar(xs, ys, yerr=es, fmt="o" + ls, color=c, ms=3.4, lw=1.4,
                       capsize=2, label=name)
    ax[0].set_xscale("log")
    ax[0].set_yscale("symlog", linthresh=1)
    ax[0].axvspan(0.8, 2.5, color="tab:green", alpha=0.10)
    ax[0].annotate("deployable", (1.15, 26), fontsize=6, color="tab:green")
    ax[0].set_xlabel("points below dense accuracy")
    ax[0].set_ylabel("paired gap (points)")
    ax[0].set_title("(a) both effects shrink,\nselection far faster", fontsize=7.5)
    ax[0].legend(fontsize=6.2, loc="lower right")

    # (b) at each checkpoint's own frontier
    d = json.loads((RES / "a6_at_the_frontier.json").read_text())
    labs = ["correction (merge)", "correction",
            "select vs merge", "selection: OMP vs magnitude",
            "selection: target vs span", "backward elimination"]
    short = ["refit\n(vs merge)", "refit\n(vs magn.)", "select\nvs merge",
             "OMP vs\nmagnitude", "target vs\nspan", "backward\nelim."]
    means = [d["pooled"][k]["mean"] for k in labs]
    errs = [2 * d["pooled"][k]["se"] for k in labs]
    cols = ["tab:red", "tab:red", "tab:purple", "tab:blue", "tab:blue", "tab:gray"]
    ax[1].barh(range(len(labs)), means, xerr=errs, color=cols, height=0.62,
               error_kw={"elinewidth": 0.8, "capsize": 2})
    ax[1].set_xlim(-1.5, 140)
    for i, k in enumerate(labs):
        s = d["pooled"][k]
        ax[1].annotate(f"{s['pos']}/{s['n']}", (72, i), fontsize=5.8,
                       va="center", ha="left")
    ax[1].set_yticks(range(len(labs)))
    ax[1].set_yticklabels(short, fontsize=6)
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_xlabel("gap at the frontier (points)")
    ax[1].set_title("(b) read where you would ship:\n21 checkpoints, 3 families",
                    fontsize=7.5)

    # (c) the mechanism, no accuracy involved
    m = json.loads((RES / "a7_mechanism.json").read_text())
    bins = m["by_bin"]["by_kept_fraction"]
    ks = sorted(bins, key=lambda s: float(s.split(",")[0][1:]))
    xc = [0.5 * (float(k.split(",")[0][1:]) + float(k.split(",")[1][:-1])) for k in ks]
    ax[2].plot(xc, [bins[k]["correction"] for k in ks], "o-", color="tab:red",
               ms=3.4, lw=1.4, label="correction")
    ax[2].plot(xc, [bins[k]["selection"] for k in ks], "o-", color="tab:blue",
               ms=3.4, lw=1.4, label="selection")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("fraction of units kept, $M/N$")
    ax[2].set_ylabel(r"residual term / $\|Y\|_F$")
    ax[2].set_title("(c) same ordering with no\naccuracy: 154 layer-budget cells",
                    fontsize=7.5)
    ax[2].legend(fontsize=6.2, loc="lower left")
    a2 = ax[2].twinx()
    a2.plot(xc, [bins[k]["ratio"] for k in ks], ":", color="k", lw=1.0)
    a2.set_ylabel("ratio", fontsize=6.5)
    a2.grid(False)
    a2.tick_params(labelsize=6)

    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "fig1_main.pdf")
    fig.savefig(FIG / "fig1_main.png")
    plt.close(fig)


def fig2():
    """Where the received ranking comes from, and the counting result."""
    runs = load_all()
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.15))

    # (a) the refit family collapses onto one curve near the frontier
    cells = defaultdict(dict)
    for r in runs:
        if r.arch == "transformer" and r.phase not in ("matched", "ablation"):
            continue
        cells[(r.arch, r.model, r.mult)][r.strategy] = r
    fam = ["omp_ls", "span_ls", "magnitude_ls", "ot_then_ls", "fwdbwd_ls"]
    pts = defaultdict(list)
    for (arch, model, mult), ops in cells.items():
        got = [ops[s].acc for s in fam if s in ops]
        if len(got) < 3:
            continue
        best = max(ops.values(), key=lambda r: r.acc)
        pts[arch].append((best.dense - best.acc, max(got) - min(got)))
    for arch, v in pts.items():
        ax[0].scatter([p[0] for p in v], [max(p[1], 1e-3) for p in v], s=7,
                      alpha=0.6, color=CLR[arch], marker=MARK[arch], label=arch,
                      linewidths=0)
    ax[0].axvspan(0, 2.5, color="tab:green", alpha=0.10)
    ax[0].set_xscale("symlog", linthresh=2.5)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("points below dense accuracy")
    ax[0].set_ylabel("spread within refit family")
    ax[0].set_title("(a) five refitted operators become\n"
                    "interchangeable near dense", fontsize=7.5)
    ax[0].legend(fontsize=6, loc="upper left")

    # (b) who wins the cell
    d = json.loads((RES / "a4_frontier_winners.json").read_text())
    ins, outs = d["winners"]["inside"], d["winners"]["outside"]
    keys = sorted(set(ins) | set(outs), key=lambda k: -(ins.get(k, 0)))
    keys = [k for k in keys if ins.get(k, 0) + outs.get(k, 0) >= 3][:7]
    y = range(len(keys))
    ax[1].barh([i + 0.19 for i in y], [ins.get(k, 0) for k in keys], height=0.36,
               color="tab:green", label=f"inside window (n={d['winners']['n_inside']})")
    ax[1].barh([i - 0.19 for i in y], [outs.get(k, 0) for k in keys], height=0.36,
               color="tab:gray", label=f"outside (n={d['winners']['n_outside']})")
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([k.replace("_", " ") for k in keys], fontsize=6)
    ax[1].set_xlabel("cells won")
    ax[1].set_title("(b) the ranking is a property\nof the regime, not the operator",
                    fontsize=7.5)
    ax[1].legend(fontsize=5.8, loc="lower right")

    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "fig2_ranking.pdf")
    fig.savefig(FIG / "fig2_ranking.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for fn in (fig1, fig2):
        try:
            fn()
            print(f"  {fn.__name__} ok")
        except Exception as exc:
            import traceback
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"figures in {FIG}")
