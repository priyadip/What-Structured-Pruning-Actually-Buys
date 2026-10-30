#!/usr/bin/env python3
"""Three figures that carry claims the tables state only in numbers.

fig3  repair and selection against the budget, on accuracy
fig4  the scatter that validates Corollary 2 directly
fig5  the same two effects per architecture family, at the operating point
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from load_all import load_all, summarise, pearson, spearman        # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 7.0, "figure.dpi": 220, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "legend.frameon": False,
                     "mathtext.fontset": "cm", "axes.grid": True,
                     "grid.alpha": 0.22, "axes.spines.top": False,
                     "axes.spines.right": False})
REP = "#b2182b"     # repair
SEL = "#2166ac"     # selection
CLR = {"mlp": "#4d9221", "cnn": "#e08214", "transformer": "#8073ac"}


def paired(runs, a, b, keep=lambda r: True):
    out = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch and keep(r)]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == a}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == b}
        for k in set(ia) & set(ib):
            ra, rb = ia[k], ib[k]
            if ra.mult <= 0 or not ra.widths or not ra.orig_widths:
                continue
            frac = sum(ra.widths) / sum(ra.orig_widths)
            out.append({"arch": arch, "gap": ra.acc - rb.acc, "alpha": ra.mult,
                        "frac": frac})
    return out


# ---------------------------------------------------------------------------

def fig3_budget():
    """Both effects against how much of the network is retained."""
    runs = load_all()
    rep = paired(runs, "ot_then_ls", "ot_intrafusion")
    sel = paired(runs, "omp_ls", "span_ls")

    bins = [(0.0, 0.06), (0.06, 0.12), (0.12, 0.20), (0.20, 0.32),
            (0.32, 0.50), (0.50, 0.75), (0.75, 1.01)]
    fig, ax = plt.subplots(figsize=(3.35, 1.86))
    for name, cs, c, mk in (("repair", rep, REP, "o"), ("selection", sel, SEL, "s")):
        xs, ys, es = [], [], []
        for lo, hi in bins:
            v = [q["gap"] for q in cs if lo <= q["frac"] < hi]
            s = summarise(v)
            if not s or s["n"] < 3:
                continue
            xs.append(0.5 * (lo + hi))
            ys.append(s["mean"])
            es.append(2 * s["se"])
        ax.errorbar(xs, ys, yerr=es, fmt=mk + "-", color=c, ms=3.4, lw=1.6,
                    capsize=2, label=name)
    ax.axhline(0, color="k", lw=0.7)
    ax.axhspan(-0.9, 0.9, color="0.5", alpha=0.12)
    ax.annotate("within noise", (0.80, 1.4), fontsize=5.9, color="0.35",
                ha="center")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"fraction of units retained, $\sum M_\ell / \sum N_\ell$")
    ax.set_ylabel("paired accuracy gain (points)")
    ax.set_title("Repair holds its value; selection does not", fontsize=7.4,
                 pad=3)
    ax.legend(fontsize=6.4, loc="center right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig3_budget.pdf")
    fig.savefig(FIG / "fig3_budget.png")
    plt.close(fig)
    print("  fig3_budget ok")


def fig4_tail():
    """Corollary 2 as a scatter: the selection difference against the bound."""
    a7 = {(r["model"], r["layer"], r["M"]): r
          for r in json.loads((RES / "a7_mechanism.json").read_text())["rows"]}
    a8 = {(r["model"], r["layer"], r["M"]): r
          for r in json.loads((RES / "a8_theory.json").read_text())["rows"]}
    keys = sorted(set(a7) & set(a8), key=str)
    x, y, frac = [], [], []
    for k in keys:
        bound = a8[k]["V_op"] * a8[k]["tau_M"]
        diff = abs(a7[k]["selection_term"]) * a7[k]["normY"]
        if bound <= 0 or diff <= 0:
            continue
        x.append(bound)
        y.append(diff)
        frac.append(a7[k]["M"] / a7[k]["N"])

    lx = [math.log10(v) for v in x]
    ly = [math.log10(v) for v in y]
    r_p, r_s = pearson(lx, ly), spearman(lx, ly)

    fig, ax = plt.subplots(figsize=(3.35, 1.86))
    sc = ax.scatter(x, y, c=frac, cmap="viridis", s=11, alpha=0.85,
                    linewidths=0)
    lo, hi = min(x + y) * 0.5, max(x + y) * 2
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=0.9)
    ax.annotate(r"$y=x$ (the bound)", (hi * 0.5, hi * 0.62), fontsize=6.0,
                color="0.3", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Corollary 2 bound, $\|V\|_2\,\tau_M$")
    ax.set_ylabel("selection difference")
    ax.set_title(rf"Selection tracks the spectral tail "
                 rf"($r={r_p:+.2f}$, $\rho={r_s:+.2f}$)", fontsize=7.2, pad=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(r"$M/N$", fontsize=6.3)
    cb.ax.tick_params(labelsize=5.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig4_tail.pdf")
    fig.savefig(FIG / "fig4_tail.png")
    plt.close(fig)
    json.dump({"n": len(x), "pearson_log": round(r_p, 4),
               "spearman_log": round(r_s, 4),
               "below_bound": sum(1 for a, b in zip(x, y) if b <= a)},
              (RES / "fig4_tail.json").open("w"), indent=2)
    print(f"  fig4_tail ok  (n={len(x)}, r={r_p:+.3f}, rho={r_s:+.3f}, "
          f"under bound {sum(1 for a,b in zip(x,y) if b<=a)}/{len(x)})")


def fig5_arch():
    """The conclusion is architecture-independent, in one panel."""
    d = json.loads((RES / "a6_at_the_frontier.json").read_text())["per_checkpoint"]
    fam = {"mlp": "MLP", "cnn": "CNN", "transformer": "compression_transformer"}
    rep, sel = defaultdict(list), defaultdict(list)
    for k, e in d.items():
        a = k.split("/")[0]
        if "correction (merge)" in e:
            rep[a].append(e["correction (merge)"])
        elif "correction" in e:
            rep[a].append(e["correction"])
        if "selection: target vs span" in e:
            sel[a].append(e["selection: target vs span"])

    order = ["mlp", "cnn", "transformer"]
    fig, ax = plt.subplots(figsize=(3.35, 1.86))
    w = 0.34
    for i, a in enumerate(order):
        sr, ss = summarise(rep[a]), summarise(sel[a])
        ax.bar(i - w / 2, sr["mean"], w, yerr=2 * sr["se"], color=REP,
               error_kw={"elinewidth": 0.8, "capsize": 2},
               label="repair" if i == 0 else None)
        ax.bar(i + w / 2, ss["mean"], w, yerr=2 * ss["se"], color=SEL,
               error_kw={"elinewidth": 0.8, "capsize": 2},
               label="selection" if i == 0 else None)
        ax.annotate(f"{sr['mean']:+.1f}", (i - w / 2, sr["mean"] * 1.35),
                    ha="center", fontsize=6.0, color=REP)
        ax.annotate(f"{ss['mean']:+.2f}", (i + w / 2, max(ss["mean"], 0.02) * 2.6),
                    ha="center", fontsize=6.0, color=SEL)
        ax.annotate(f"$n{{=}}{sr['n']}$", (i, 0.016), ha="center", fontsize=5.7,
                    color="0.4", zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 400)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([fam[a] for a in order])
    ax.set_axisbelow(True)
    ax.set_ylabel("gain at the operating point (points)")
    ax.set_title("Same ordering in every architecture family", fontsize=7.4,
                 pad=3)
    ax.legend(fontsize=6.4, loc="upper right", ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig5_arch.pdf")
    fig.savefig(FIG / "fig5_arch.png")
    plt.close(fig)
    print("  fig5_arch ok")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for fn in (fig3_budget, fig4_tail, fig5_arch):
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
