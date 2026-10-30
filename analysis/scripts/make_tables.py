#!/usr/bin/env python3
"""Emit the paper's tables from the analysis JSONs. No number is typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
TAB = HERE.parent / "tables"


def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def t1_regime():
    d = json.loads((RES / "a3_component_regime.json").read_text())
    order = [("correction | merging", r"refit $\mid$ merging"),
             ("correction | magnitude selection", r"refit $\mid$ magnitude"),
             ("selection: OMP vs merging", r"select vs.\ merge"),
             ("selection: OMP vs magnitude", r"data-driven vs.\ norm"),
             ("selection: OMP vs span", r"target-aware vs.\ blind"),
             ("refinement: fwd-bwd vs OMP", r"backward elimination")]
    bands = ["0.0-2.5", "2.5-5.0", "5.0-10.0", "10.0-1000000000.0"]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{points below dense accuracy} & \\",
         r"\cmidrule(lr){2-5}",
         r"contrast & $<$2.5 & 2.5--5 & 5--10 & $>$10 & $n_{<2.5}$\\",
         r"\midrule"]
    for key, lab in order:
        row = d[key]["bands"]
        vals = []
        for b in bands:
            s = row.get(b)
            vals.append(f"${s['mean']:+.2f}$" if s else "--")
        n = row["0.0-2.5"]["n"] if row.get("0.0-2.5") else 0
        bold = key.startswith("correction")
        lab_s = rf"\textbf{{{lab}}}" if bold else lab
        L.append(f"{lab_s} & " + " & ".join(vals) + f" & {n}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t1_regime.tex").write_text("\n".join(L) + "\n")


def t2_frontier():
    d = json.loads((RES / "a6_at_the_frontier.json").read_text())["pooled"]
    order = [("correction (merge)", r"refit $\mid$ merging"),
             ("correction", r"refit $\mid$ magnitude"),
             ("select vs merge", r"select vs.\ merge"),
             ("selection: OMP vs magnitude", r"data-driven vs.\ norm"),
             ("selection: target vs span", r"target-aware vs.\ blind"),
             ("backward elimination", r"backward elimination")]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"contrast & $n$ & mean & s.d. & $t$ & wins\\", r"\midrule"]
    for k, lab in order:
        s = d[k]
        bold = k.startswith("correction")
        lab_s = rf"\textbf{{{lab}}}" if bold else lab
        m = rf"$\mathbf{{{s['mean']:+.2f}}}$" if bold else f"${s['mean']:+.2f}$"
        L.append(f"{lab_s} & {s['n']} & {m} & {s['sd']:.2f} & "
                 f"{s['t']:.2f} & {s['pos']}/{s['n']}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t2_frontier.tex").write_text("\n".join(L) + "\n")


def t3_mechanism():
    d = json.loads((RES / "a7_mechanism.json").read_text())["by_bin"]["by_kept_fraction"]
    ks = sorted(d, key=lambda s: float(s.split(",")[0][1:]))
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"kept fraction $M/N$ & correction & selection & ratio & $n$\\",
         r"\midrule"]
    for k in ks:
        lo = float(k.split(",")[0][1:])
        hi = float(k.split(",")[1][:-1])
        s = d[k]
        L.append(f"{lo:.3f}--{hi:.3f} & {s['correction']:.4f} & "
                 f"{s['selection']:.4f} & {s['ratio']:.1f} & {s['n']}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t3_mechanism.tex").write_text("\n".join(L) + "\n")


def t4_perckpt():
    d = json.loads((RES / "a6_at_the_frontier.json").read_text())["per_checkpoint"]
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"family & checkpoint & $\alpha^\star$ & refit & sel.\ (span) & "
         r"sel.\ (magn.)\\", r"\midrule"]
    for k in sorted(d):
        arch, model = k.split("/", 1)
        e = d[k]
        c = e.get("correction (merge)", e.get("correction"))
        s1 = e.get("selection: target vs span")
        s2 = e.get("selection: OMP vs magnitude")
        L.append(f"{arch} & {esc(model)} & {e['alpha_star']:.2f} & "
                 + (f"${c:+.2f}$" if c is not None else "--") + " & "
                 + (f"${s1:+.2f}$" if s1 is not None else "--") + " & "
                 + (f"${s2:+.2f}$" if s2 is not None else "--") + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t4_perckpt.tex").write_text("\n".join(L) + "\n")


def prose():
    """The in-text numbers, regenerated so the prose cannot drift."""
    a3 = json.loads((RES / "a3_component_regime.json").read_text())
    a4 = json.loads((RES / "a4_frontier_winners.json").read_text())
    a6 = json.loads((RES / "a6_at_the_frontier.json").read_text())
    a2 = json.loads((RES / "a2_regime.json").read_text())
    a7 = json.loads((RES / "a7_mechanism.json").read_text())
    m = a7["by_bin"]["by_kept_fraction"]
    ks = sorted(m, key=lambda s: float(s.split(",")[0][1:]))
    p = a6["pooled"]
    out = {
        "n_checkpoints": a6["n_checkpoints"],
        "frontier_refit_merge": p["correction (merge)"]["mean"],
        "frontier_refit_magn": p["correction"]["mean"],
        "frontier_sel_span": p["selection: target vs span"]["mean"],
        "ratio": round(p["correction (merge)"]["mean"]
                       / p["selection: target vs span"]["mean"], 0),
        "spread_in": a4["refit_family_spread"]["inside"],
        "spread_out": a4["refit_family_spread"]["outside"],
        "omp_not_winner": a4["frontier_summary"]["omp_not_winner"],
        "omp_gives_away": a4["frontier_summary"]["gap"]["mean"],
        "window_by_arch": a2["omp_ls_minus_span_ls_window"],
        "mech_lo": m[ks[0]], "mech_hi": m[ks[-1]],
        "mech_corr_tail": a7["corr_tail_selection"],
        "sel_bands": {k: a3["selection: OMP vs span"]["bands"][k]["mean"]
                      for k in a3["selection: OMP vs span"]["bands"]
                      if a3["selection: OMP vs span"]["bands"][k]},
    }
    (RES / "paper_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))




def t5_attention():
    """The counterexample the mechanism predicts, from the Transformer runs."""
    import csv
    src = HERE.parents[1] / "compression_transformer" / "results" / "attention.csv"
    if not src.exists():
        return
    rows = [r for r in csv.DictReader(src.open()) if r["tag"] == "attn_only"]
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], {})[float(r["alpha"])] = float(r["acc"])
    alphas = sorted(set(by.get("omp_ls", {})) & set(by.get("span_ls", {})))
    heads = {float(r["alpha"]): json.loads(r["widths"])
             for r in rows if r["strategy"] == "omp_ls"}
    L = [r"\begin{tabular}{rlrrr}", r"\toprule",
         r"$\alpha$ & heads kept & \textsc{omp} & \textsc{span} & difference\\",
         r"\midrule"]
    for a in alphas:
        h = heads.get(a, [])
        gap = by["omp_ls"][a] - by["span_ls"][a]
        bold = r"\textbf" if abs(gap) > 5 else ""
        g = rf"{bold}{{${gap:+.2f}$}}" if bold else f"${gap:+.2f}$"
        L.append(f"{a:g} & {sum(h)}/72 & {by['omp_ls'][a]:.2f} & "
                 f"{by['span_ls'][a]:.2f} & {g}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t5_attention.tex").write_text("\n".join(L) + "\n")
    print("  t5_attention ok")


def t6_frontiers():
    """What the compressed models actually achieve, so the ablation analysis is
    read against a method that works rather than one that does not."""
    from load_all import load_all
    runs = load_all()
    best = {}
    for r in runs:
        if not r.has_refit or r.params_cut != r.params_cut:
            continue
        if abs(r.drop) > 2.5:
            continue
        k = (r.arch, r.model)
        if k not in best or r.params_cut > best[k].params_cut:
            best[k] = r
    fam = {"mlp": "fully connected", "cnn": "convolutional",
           "transformer": "compression_transformer"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"family & $n$ & dense & compressed & $\Delta$ & param.\ cut\\",
         r"\midrule"]
    for arch in ("mlp", "cnn", "transformer"):
        v = [r for (a, _), r in best.items() if a == arch]
        if not v:
            continue
        n = len(v)
        de = sum(r.dense for r in v) / n
        ac = sum(r.acc for r in v) / n
        dr = sum(r.drop for r in v) / n
        cut = sum(r.params_cut for r in v) / n
        lo = min(r.params_cut for r in v)
        hi = max(r.params_cut for r in v)
        L.append(f"{fam[arch]} & {n} & {de:.1f}\\% & {ac:.1f}\\% & "
                 f"${dr:+.2f}$ & {cut:.1f}\\% ({lo:.0f}--{hi:.0f})\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t6_frontiers.tex").write_text("\n".join(L) + "\n")
    print("  t6_frontiers ok")


def t7_master():
    """One table with every model and every column the paper argues from."""
    from load_all import load_all
    from collections import defaultdict
    runs = load_all()

    cells = defaultdict(dict)
    for r in runs:
        if r.arch == "transformer" and r.phase not in ("matched", "ablation"):
            continue
        cells[(r.arch, r.model, r.mult)][r.strategy] = r

    frontier = {}
    for (arch, model, mult), ops in cells.items():
        if len(ops) < 4:
            continue
        best = max(ops.values(), key=lambda r: r.acc)
        if (best.dense - best.acc) <= 2.5:
            k = (arch, model)
            if k not in frontier or mult < frontier[k][0]:
                frontier[k] = (mult, ops, best)

    # largest cut still inside the window, over the repaired family
    bestcut = {}
    for r in runs:
        if not r.has_refit or abs(r.drop) > 2.5 or r.params_cut != r.params_cut:
            continue
        k = (r.arch, r.model)
        if k not in bestcut or r.params_cut > bestcut[k].params_cut:
            bestcut[k] = r

    SHORT = {"omp_ls": "omp", "span_ls": "span", "magnitude_ls": "magn",
             "ot_then_ls": "merge", "fwdbwd_ls": "fwd-bwd",
             "augdict_fwdbwd": "dict", "augdict_omp": "dict",
             "magnitude": "magn (raw)", "N05_omp_transport": "omp-t",
             "N09_fisher_omp_e2": "fisher", "thinet_lsfull": "thinet"}
    FAM = {"mlp": "MLP", "cnn": "CNN", "transformer": "Tf."}
    DS = {"mnist": "MNIST", "fashionmnist": "F-MNIST", "cifar10": "C10",
          "cifar100": "C100", "imagenet32": "IN32", "mnli": "MNLI"}

    L = [r"\begin{tabular}{lllrrrrrrrrrl}", r"\toprule",
         r"& & & \multicolumn{5}{c}{frontier configuration} & "
         r"\multicolumn{3}{c}{paired contrast at the frontier} & \\",
         r"\cmidrule(lr){4-8}\cmidrule(lr){9-11}",
         r"fam. & model & data & $\alpha^\star$ & acc. & $\Delta$ & par. & "
         r"MAC & repair & t.\ vs sp. & d.\ vs n. & best\\",
         r"\midrule"]
    order = {"mlp": 0, "cnn": 1, "transformer": 2}
    for (arch, model), (mult, ops, best) in sorted(
            frontier.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        # longest prefix first: "mnist" is a substring of "fashionmnist", and
        # stripping it from the middle of a name silently mangles it
        ds, name = "--", model
        for key in sorted(DS, key=len, reverse=True):
            if model.startswith(key + "_"):
                ds, name = DS[key], model[len(key) + 1:]
                break
        else:
            ds = DS.get(best.dataset.lower(), "--")
        cut = bestcut.get((arch, model))
        rep = ops["ot_then_ls"].acc - ops["ot_intrafusion"].acc \
            if "ot_then_ls" in ops and "ot_intrafusion" in ops else None
        s1 = ops["omp_ls"].acc - ops["span_ls"].acc \
            if "omp_ls" in ops and "span_ls" in ops else None
        s2 = ops["omp_ls"].acc - ops["magnitude_ls"].acc \
            if "omp_ls" in ops and "magnitude_ls" in ops else None
        w = SHORT.get(best.strategy, best.strategy.replace("_", " "))
        mac = cut.extra.get("macs_cut", float("nan")) if cut else float("nan")
        L.append(
            f"{FAM[arch]} & {esc(name)} & {ds} & {mult:.2f} & "
            f"{best.acc:.1f} & ${best.drop:+.2f}$ & "
            + (f"{cut.params_cut:.0f}\\%" if cut else "--") + " & "
            + (f"{mac:.0f}\\%" if mac == mac else "--") + " & "
            + (f"${rep:+.1f}$" if rep is not None else "--") + " & "
            + (f"${s1:+.2f}$" if s1 is not None else "--") + " & "
            + (f"${s2:+.2f}$" if s2 is not None else "--") + " & "
            + f"{w}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t7_master.tex").write_text("\n".join(L) + "\n")
    print("  t7_master ok")


def t8_spectra():
    """The spectra behind the reversal of Section 'Where selection must matter'."""
    f = RES / "a9_spectra.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())["mean_tail"]
    fracs = ["0.05", "0.10", "0.25", "0.50", "0.75"]
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"design matrix & cols & " + " & ".join(rf"$M/N\!=\!{x}$" for x in fracs)
         + r"\\", r"\midrule"]
    for k, lab, cols in (("ffn", "feed-forward", 3072), ("att", "attention", 768)):
        L.append(f"{lab} & {cols} & "
                 + " & ".join(f"{d[k][x]:.3f}" for x in fracs) + r"\\")
    r_ = [d["att"][x] / d["ffn"][x] for x in fracs]
    L.append(r"\midrule")
    L.append("ratio & & " + " & ".join(f"{v:.2f}" for v in r_) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t8_spectra.tex").write_text("\n".join(L) + "\n")
    print("  t8_spectra ok")


def t9_comparison():
    """Published methods against ours at matched budget, per architecture family.

    Every entry is the mean distance from dense accuracy, in points, evaluated at
    each checkpoint's own frontier multiplier so that all methods in a column
    receive identical per-layer widths. \textsc{ThiNet} was run on a four-model
    convolutional subset and on the Transformer; its column entries are means
    over exactly the cells where it ran, and the paired margin against ours on
    those same cells is given in the text.
    """
    import csv as _csv
    from load_all import load_all
    from collections import defaultdict
    import statistics as st
    runs = load_all()

    cells = defaultdict(dict)
    for r in runs:
        if r.arch == "transformer" and r.phase not in ("matched", "ablation"):
            continue
        cells[(r.arch, r.model, r.mult)][r.strategy] = r
    frontier = {}
    for (arch, model, mult), ops in cells.items():
        if len(ops) < 4:
            continue
        best = max(ops.values(), key=lambda r: r.acc)
        if (best.dense - best.acc) <= 2.5:
            k = (arch, model)
            if k not in frontier or mult < frontier[k][0]:
                frontier[k] = (mult, ops)

    # ThiNet: separate driver, four convolutional models at four budgets
    thi = defaultdict(list)
    f = HERE.parents[1] / "ablations" / "results" / "e3_thinet.csv"
    if f.exists():
        for r in _csv.DictReader(f.open()):
            thi[r["strategy"]].append(float(r["drop"]))

    ROWS = [
        ("magnitude",      r"Magnitude \citep{li2017pruning}",     "none"),
        ("ot_intrafusion", r"Intra-Fusion \citep{theus2024intrafusion}", "none"),
        ("kmeans_merge",   r"$k$-means merge",                      "none"),
        ("THINET",         r"\textsc{ThiNet} \citep{luo2017thinet}", "scalar"),
        ("ot_then_ls",     r"OT merge + refit",                     "full"),
        ("magnitude_ls",   r"Magnitude + refit",                    "full"),
        ("span_ls",        r"Span + refit",                         "full"),
        ("omp_ls",         r"OMP + refit",                          "full"),
        ("fwdbwd_ls",      r"OMP + backward + refit",               "full"),
    ]
    FAMS = [("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "Tf.")]

    def cellval(stg, arch):
        if stg == "THINET":
            if arch == "cnn" and thi.get("T1_thinet"):
                return st.mean(thi["T1_thinet"]), len(thi["T1_thinet"]), True
            if arch == "transformer":
                v = [(ops["thinet"].acc - ops["thinet"].dense)
                     for (a, _), (m, ops) in frontier.items()
                     if a == arch and "thinet" in ops]
                return (st.mean(v), len(v), False) if v else (None, 0, False)
            return None, 0, False
        v = [(ops[stg].acc - ops[stg].dense)
             for (a, _), (m, ops) in frontier.items() if a == arch and stg in ops]
        return (st.mean(v), len(v), False) if v else (None, 0, False)

    counts = {a: len([1 for (x, _) in frontier if x == a]) for a, _ in FAMS}
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"& & \multicolumn{3}{c}{$\Delta$ from dense (points)}\\",
         r"\cmidrule(lr){3-5}",
         r"method & repair & "
         + " & ".join(rf"{lab} ({counts[a]})" for a, lab in FAMS) + r"\\",
         r"\midrule"]
    for stg, lab, rep in ROWS:
        vals = []
        for a, _ in FAMS:
            m, n, sub = cellval(stg, a)
            if m is None:
                vals.append("--")
            else:
                s = f"${m:+.1f}$"
                if sub:
                    s += r"$^{\dagger}$"
                vals.append(rf"\textbf{{{s}}}" if stg == "omp_ls" else s)
        if stg == "ot_then_ls":
            L.append(r"\midrule")
        L.append(f"{lab} & {rep} & " + " & ".join(vals) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "t9_comparison.tex").write_text("\n".join(L) + "\n")

    # the paired ThiNet margin on its own cells
    if thi:
        marg = st.mean([a - b for a, b in zip(thi["S4_omp_ls"], thi["T1_thinet"])])
        marg_ls = st.mean([a - b for a, b in zip(thi["S4_omp_ls"],
                                                 thi["T2_thinet_lsfull"])])
        json.dump({"thinet_paired_margin_cnn": round(marg, 2),
                   "thinet_lsfull_paired_margin_cnn": round(marg_ls, 2),
                   "n_cells": len(thi["T1_thinet"]),
                   "thinet_mean_drop": round(st.mean(thi["T1_thinet"]), 2),
                   "thinet_lsfull_mean_drop": round(st.mean(thi["T2_thinet_lsfull"]), 2)},
                  (RES / "thinet_comparison.json").open("w"), indent=2)
    print("  t9_comparison ok")


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    for fn in (t1_regime, t2_frontier, t3_mechanism, t4_perckpt, t5_attention, t6_frontiers, t7_master, t8_spectra, t9_comparison):
        fn()
        print(f"  {fn.__name__} ok")
    prose()
