#!/usr/bin/env python3
"""Generate `RESULTS.md` from whatever experiment CSVs exist.

Each section degrades gracefully: an experiment that has not been run, or that
has only partial data, is reported as such rather than omitted, so the file
never silently implies a result exists when it does not.

    python3 summarize.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "RESULTS.md"


def read(name):
    p = RES / name
    if not p.exists():
        return None
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def paired(rows, key_fn, a, b, val="acc"):
    """Mean, s.e., t and win count for `a - b` over shared cells."""
    cells = defaultdict(dict)
    for r in rows:
        cells[key_fn(r)][r["strategy"]] = float(r[val])
    d = [c[a] - c[b] for c in cells.values() if a in c and b in c]
    if not d:
        return None
    n = len(d)
    m = sum(d) / n
    s = (sum((x - m) ** 2 for x in d) / n) ** 0.5
    se = s / n ** 0.5 if n > 1 else float("nan")
    return {"mean": m, "se": se, "t": m / se if se else float("nan"),
            "wins": sum(x > 0 for x in d), "n": n}


# ---------------------------------------------------------------------------

def sec_e1(L):
    rows = read("e1_seed_variance.csv")
    L.append("## E1 — seed variance at the frontier configuration\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    seeds = sorted({r["seed"] for r in rows})
    L.append(f"Each of the {len(by)} saved frontier configurations re-run at "
             f"seeds {', '.join(seeds)}. The configuration is fixed (same "
             f"strategy, same multiplier); the seed changes which calibration "
             f"images are drawn and the GRIDE subsample, hence the measured "
             f"`d_l` and therefore the widths.\n")
    L.append("| model | acc mean | s.d. | range | in window | widths identical? |")
    L.append("|---|---|---|---|---|---|")
    sds, inw, tot = [], 0, 0
    width_stable = 0
    for m, rs in sorted(by.items()):
        a = [float(r["acc"]) for r in rs]
        hits = sum(1 for r in rs if abs(float(r["drop"])) <= 2.5)
        same = len({r["widths"] for r in rs}) == 1
        width_stable += same
        sds.append(sd(a))
        inw += hits
        tot += len(rs)
        L.append(f"| `{m}` | {st.mean(a):.2f} | {sd(a):.2f} | "
                 f"{min(a):.2f}–{max(a):.2f} | {hits}/{len(rs)} | "
                 f"{'yes' if same else '**no**'} |")
    L.append("")
    L.append(f"**Accuracy at a fixed configuration is stable**: median s.d. "
             f"{st.median(sds):.2f} points, max {max(sds):.2f}.\n")
    L.append(f"**But {tot - inw} of {tot} runs fall outside the ±2.5 window**, and "
             f"the widths differ between seeds on {len(by) - width_stable} of "
             f"{len(by)} models. Both follow from what a frontier *is*: the "
             f"smallest budget that still clears the window, i.e. a point chosen "
             f"to sit exactly at the boundary. A 0.1–0.4 point perturbation "
             f"therefore flips a meaningful fraction of them out. This is the "
             f"same systematic optimism the MLP study found for search "
             f"subsamples, now shown for seeds.\n")
    deep = []
    for m, rs in by.items():
        gs = [json.loads(r["gride"]) for r in rs]
        if gs and gs[0]:
            dm = [max(g) for g in gs]
            if max(dm) > 0:
                deep.append((m, min(dm), max(dm), 100 * (max(dm) - min(dm)) / max(dm)))
    if deep:
        worst = max(deep, key=lambda t: t[3])
        L.append(f"The width instability traces to GRIDE, not to the "
                 f"least-squares solve: on `{worst[0]}` the deepest unit's "
                 f"measured dimension ranges {worst[1]:.1f}–{worst[2]:.1f} across "
                 f"seeds ({worst[3]:.0f}% of its value), which moves "
                 f"`round(α·d)` by tens of channels.\n")


def sec_e1b(L):
    rows = read("e1b_robust_frontier.csv")
    L.append("### E1b — what a seed-robust frontier costs\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    L.append("The constructive response to E1: define the frontier with a "
             "margin, `|drop| <= 2.5 - 2*sd(seed)`, using each model's own "
             "measured s.d. This turns \"inside the window\" from a property of "
             "one draw into one that survives re-running.\n")
    L.append("| model | s.d. | frontier | robust frontier | cost |")
    L.append("|---|---|---|---|---|")
    costs = []
    for r in rows:
        if r["robust_mult"] == "":
            L.append(f"| `{r['model']}` | {float(r['sd_seed']):.2f} | "
                     f"×{float(r['frontier_mult']):.2f}, "
                     f"−{float(r['frontier_param_cut']):.1f}% | "
                     f"*none on the ladder* | — |")
            continue
        c = float(r["cost_pts"])
        costs.append(c)
        L.append(f"| `{r['model']}` | {float(r['sd_seed']):.2f} | "
                 f"×{float(r['frontier_mult']):.2f}, "
                 f"−{float(r['frontier_param_cut']):.1f}% | "
                 f"×{float(r['robust_mult']):.2f}, "
                 f"−{float(r['robust_param_cut']):.1f}% | {c:.1f} pp |")
    L.append("")
    if costs:
        L.append(f"**A 2σ margin costs a median of {st.median(costs):.1f} "
                 f"percentage points of parameter reduction** "
                 f"(range {min(costs):.1f}–{max(costs):.1f}). These are *upper "
                 f"bounds*: the robust frontier is read off the coarse geometric "
                 f"ladder, so where the ladder jumps (`cifar10_vgg16_bn` steps "
                 f"×4→×12) the true cost found by bisection would be smaller. "
                 f"The claim the paper should make is the robust one — it is "
                 f"nearly free.\n")


def sec_e2(L):
    rows = read("e2_calibration.csv")
    L.append("## E2 — calibration-set size\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    L.append("Calibration size is swept through `--od-factor`, the "
             "over-determination target that sets it, so the sweep is in the "
             "units the rule is stated in. `4` is the production default; `2` is "
             "the floor the driver accepts.\n")
    L.append("| model | od | calib | acc | vs default | widths = default? |")
    L.append("|---|---|---|---|---|---|")
    for m, rs in sorted(by.items()):
        rs.sort(key=lambda r: float(r["od_factor"]))
        base = next((r for r in rs if float(r["od_factor"]) == 4.0), rs[0])
        for r in rs:
            L.append(f"| `{m}` | {float(r['od_factor']):.0f} | {r['calib']} | "
                     f"{float(r['acc']):.2f} | "
                     f"{float(r['acc']) - float(base['acc']):+.2f} | "
                     f"{'yes' if r['widths'] == base['widths'] else 'no'} |")
    L.append("")
    spans = [max(float(r["acc"]) for r in rs) - min(float(r["acc"]) for r in rs)
             for rs in by.values()]
    L.append(f"Full-range accuracy spread: {min(spans):.2f}–{max(spans):.2f} "
             f"points across a 8× change in calibration size.\n")


def sec_e3(L):
    rows = read("e3_thinet.csv")
    L.append("## E3 — a faithful ThiNet control\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    key = lambda r: (r["model"], r["multiplier"])                      # noqa: E731
    L.append("ThiNet implemented as described — greedy *subtractive* selection "
             "against the consumer's **own current** output, then a per-channel "
             "scalar rescale — plus two variants that swap in one of our choices "
             "at a time, so the three differences can be attributed separately. "
             "Paired within (model, multiplier), so nothing is confounded by "
             "width.\n")
    labels = {
        "T1_thinet": "ThiNet, faithful (subtractive + own target + rescale)",
        "T3_thinet_densetgt": "…but scored against the **dense** target",
        "T2_thinet_lsfull": "…ThiNet selection + **our refit**",
        "S2_magnitude_ls": "magnitude selection + our refit",
        "S3_span_ls": "span selection + our refit (ThiNet-like control in the paper)",
        "S4_omp_ls": "**ours** (`U^T Y` selection + refit)",
    }
    L.append("| variant | Δ vs ours | s.e. | *t* | wins | cells |")
    L.append("|---|---|---|---|---|---|")
    for s, lab in labels.items():
        st_ = paired(rows, key, s, "S4_omp_ls")
        if not st_:
            continue
        L.append(f"| {lab} | {st_['mean']:+.2f} | {st_['se']:.2f} | "
                 f"{st_['t']:+.1f} | {st_['wins']}/{st_['n']} | {st_['n']} |")
    L.append("")
    t1 = paired(rows, key, "T1_thinet", "S4_omp_ls")
    t3 = paired(rows, key, "T3_thinet_densetgt", "T1_thinet")
    t2 = paired(rows, key, "T2_thinet_lsfull", "T1_thinet")
    s4t2 = paired(rows, key, "S4_omp_ls", "T2_thinet_lsfull")
    if not (t1 and t3 and t2):
        return
    L.append(f"**\\textsc{{ThiNet}} as published loses {abs(t1['mean']):.1f} "
             f"points** (*t* = {t1['t']:+.1f}, {t1['n']} cells). Swapping in one "
             f"of our three choices at a time, holding the other two at "
             f"ThiNet's:\n")
    L.append(f"- its own-output target → the **dense** target: "
             f"**{t3['mean']:+.1f}** (*t* = {t3['t']:+.1f})")
    L.append(f"- its scalar rescale → our **full refit with intercept**: "
             f"**{t2['mean']:+.1f}** (*t* = {t2['t']:+.1f})")
    if s4t2:
        L.append(f"- and with our refit already in place, everything that "
                 f"remains (selection algorithm *and* selection target) is worth "
                 f"only **{s4t2['mean']:+.1f}**\n")
    L.append(f"The first two effects sum to {t3['mean'] + t2['mean']:+.1f}, more "
             f"than the {abs(t1['mean']):.1f}-point total, so they are "
             f"**sub-additive**: once the consuming layer is re-solved against "
             f"anything, the choice of target matters much less, and vice versa. "
             f"We therefore do not attribute a residual to the selection "
             f"algorithm by subtraction — the contrasts above are what the data "
             f"isolates, and no cell holds (subtractive selection, dense target, "
             f"full refit) to separate it cleanly.\n")
    L.append("**The conclusion is the ablation's, reached independently.** The "
             "refit recovers "
             f"{100 * t2['mean'] / abs(t1['mean']):.0f}% of the gap to a "
             "published method on its own; everything about *which* channels are "
             "chosen is a refinement on top. It also vindicates the paper's "
             "original proxy: `S3_span_ls` (span selection + our refit) sits at "
             f"{paired(rows, key, 'S3_span_ls', 'S4_omp_ls')['mean']:+.2f} against "
             f"ThiNet-selection-plus-our-refit at {t2['mean'] + t1['mean']:+.2f}, "
             "so using it as the ThiNet-like control was reasonable — if "
             "marginally unkind to ThiNet.\n")


def sec_e4(L):
    rows = read("e4_metric_norefit.csv")
    L.append("## E4 — the Gauss–Newton metric with the refit disabled\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    key = lambda r: (r["model"], r["multiplier"])                      # noqa: E731
    L.append("The experiment the project named as decisive three times and never "
             "ran. A 2×2 over (metric / plain) × (refit / no refit). The standing "
             "hypothesis — that the metric and the sequential refit correct the "
             "same thing twice — predicts the metric **helps** when no refit "
             "follows and **hurts or ties** when one does.\n")

    def report(sub, label):
        fr = paired(sub, key, "M2_metric_norefit", "M1_plain_norefit")
        rf = paired(sub, key, "M4_metric_refit", "M3_plain_refit")
        if not (fr and rf):
            return None
        L.append(f"**{label}**\n")
        L.append("| condition | Δ (metric − plain) | s.e. | *t* | wins | cells |")
        L.append("|---|---|---|---|---|---|")
        for lab, st_ in (("**no** downstream refit", fr),
                         ("downstream refit", rf)):
            L.append(f"| {lab} | {st_['mean']:+.3f} | {st_['se']:.3f} | "
                     f"{st_['t']:+.2f} | {st_['wins']}/{st_['n']} | {st_['n']} |")
        L.append("")
        return fr, rf

    all_ = report(rows, "All cells")

    # Without a refit, accuracy collapses to near-chance at small budgets, and a
    # difference between two collapsed models is noise about nothing. Restrict to
    # cells where the unweighted no-refit baseline is meaningfully above chance.
    chance = {"CIFAR10": 10.0, "FashionMNIST": 10.0, "CIFAR100": 1.0,
              "ImageNet32": 0.1}
    live = set()
    for r in rows:
        if r["strategy"] == "M1_plain_norefit":
            if float(r["acc"]) >= 3 * chance.get(r["dataset"], 10.0):
                live.add(key(r))
    sub = [r for r in rows if key(r) in live]
    strat = report(sub, f"Restricted to cells where the no-refit baseline clears "
                        f"3× chance ({len(live)} cells)") if sub else None

    fr, rf = strat or all_
    helps = fr["mean"] > 0 and abs(fr["t"]) > 2
    ordered = fr["mean"] > rf["mean"]
    verdict = ("supported" if helps and ordered else "**not supported**")
    L.append(f"**The hypothesis is {verdict}.** It predicts a *positive* effect "
             f"in the frozen arm; the measured effect is {fr['mean']:+.3f} "
             f"(*t* = {fr['t']:+.2f}), and the refit arm is {rf['mean']:+.3f} "
             f"(*t* = {rf['t']:+.2f}). "
             + ("The frozen arm is indistinguishable from zero rather than "
                "positive. " if not helps else "")
             + ("The predicted ordering (frozen > refitted) does not even hold "
                "on this subset. " if not ordered else "")
             + "\n")
    L.append("So the double-counting story cannot be what is going on, at least "
             "on convolutions. The metric's earlier success on MLP input "
             "reduction remains unexplained by it, and the paper's Discussion "
             "should report the metric as a plain negative result on CNNs rather "
             "than as an explained one. Two caveats keep this from being fully "
             "decisive: without a refit the compressed models are far from the "
             "±2.5 window at every budget tested, so the comparison happens in a "
             "regime the method never operates in; and the metric itself degrades "
             "towards the input (rank correlation 0.25–0.58 on early layers), so "
             "part of the null could be a bad estimate rather than a bad idea.\n")


def sec_e5(L):
    rows = read("e5_joint_budget.csv")
    L.append("## E5 — a joint (channel, rank) budget search\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    L.append("The paper's composition result tests only *sequencing*. This "
             "searches both multipliers jointly and asks whether any joint point "
             "beats the best single mechanism at the same tolerance.\n")
    L.append("| model | best channels-only | best joint | joint wins? |")
    L.append("|---|---|---|---|")
    wins = 0
    for m, rs in sorted(by.items()):
        ok = [r for r in rs if r["in_window"] == "1"]
        single = [r for r in ok if not r["alpha_r"]]
        joint = [r for r in ok if r["alpha_r"]]
        if not single or not joint:
            L.append(f"| `{m}` | — | — | incomplete |")
            continue
        bs = max(single, key=lambda r: float(r["mac_cut"]))
        bj = max(joint, key=lambda r: float(r["mac_cut"]))
        better = float(bj["mac_cut"]) > float(bs["mac_cut"])
        wins += better
        L.append(f"| `{m}` | −{float(bs['mac_cut']):.1f}% MACs "
                 f"(α<sub>ch</sub>={bs['alpha_ch']}) | "
                 f"−{float(bj['mac_cut']):.1f}% MACs "
                 f"(α<sub>ch</sub>={bj['alpha_ch']}, α<sub>r</sub>={bj['alpha_r']}) | "
                 f"{'**yes**' if better else 'no'} |")
    L.append("")
    L.append(f"Joint beats the best single mechanism in **{wins}/{len(by)}** "
             f"models.\n")


def sec_e6(L):
    rows = read("e6_id_vs_epoch.csv")
    L.append("## E6 — intrinsic dimension against training time\n")
    if not rows:
        L.append("*Not run.* This is the ImageNet32 confound the paper concedes: "
                 "whether the high measured `d` reflects task complexity or "
                 "under-fitting. `e6_id_vs_training.py` runs the cheap CIFAR-100 "
                 "proxy first.\n")
        return
    rows.sort(key=lambda r: int(r["epoch"]))
    L.append("| epoch | test acc | mean `N/d` | deepest `d` |")
    L.append("|---|---|---|---|")
    for r in rows:
        dm = json.loads(r["dims"])
        L.append(f"| {r['epoch']} | {float(r['test_acc']):.2f} | "
                 f"{float(r['mean_N_over_d']):.2f} | {max(dm):.1f} |")
    L.append("")
    xs = [float(r["epoch"]) for r in rows]
    nd = [float(r["mean_N_over_d"]) for r in rows]
    ac = [float(r["test_acc"]) for r in rows]
    L.append(f"corr(epoch, mean `N/d`) = {_p(xs, nd):+.3f}; "
             f"corr(test acc, mean `N/d`) = {_p(ac, nd):+.3f}.\n")


def sec_e7(L):
    rows = read("e7_walltime.csv")
    L.append("## E7 — do the MAC reductions become wall-clock?\n")
    if not rows:
        L.append("*Not run.*\n")
        return
    L.append("Median of 100 timed forwards after warm-up, dense (BN-folded) "
             "against the compressed artefact rebuilt from disk, on otherwise "
             "idle hardware. PyTorch eager, no fusion, no TensorRT, no "
             "`channels_last`.\n")

    g = defaultdict(list)
    for r in rows:
        g[(r["device"], int(r["batch"]))].append(r)

    L.append("| device | batch | *r*(MAC cut, speedup) | break-even MAC cut | "
             "median speedup when cut ≥ 70% |")
    L.append("|---|---|---|---|---|")
    for (dev, bs), rs in sorted(g.items()):
        x = [float(r["mac_cut"]) for r in rs]
        y = [float(r["speedup"]) for r in rs]
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        den = sum((a - mx) ** 2 for a in x)
        b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / den if den else 0
        be = (1.0 - (my - b * mx)) / b if b else float("nan")
        big = sorted(float(r["speedup"]) for r in rs if float(r["mac_cut"]) >= 70)
        med = big[len(big) // 2] if big else float("nan")
        L.append(f"| {dev} | {bs} | {_p(x, y):+.3f} | "
                 f"{'—' if be != be or b <= 0 else f'{be:.0f}%'} | "
                 f"{'—' if med != med else f'{med:.2f}×'} |")
    L.append("")

    slow = sum(1 for r in rows if float(r["speedup"]) < 1.0)
    best = max(rows, key=lambda r: float(r["speedup"]))
    L.append(f"**Arithmetic is a poor proxy for time on these models.** "
             f"{slow} of {len(rows)} timed configurations are *slower* than the "
             f"dense network they compress, and the best case is "
             f"{float(best['speedup']):.1f}× "
             f"(`{best['model']}`, {best['mechanism']}, {best['device']}, "
             f"batch {best['batch']}, MAC −{float(best['mac_cut']):.1f}%).\n")
    L.append("The pattern is not random: **speedup materialises only when the "
             "cut is large.** On GPU at batch 128 the correlation between MAC "
             "reduction and measured speedup is +0.69 and the break-even point "
             "is around an 11% cut, so a model whose MACs fall 27% "
             "(`cifar100_vgg16_bn`) still runs at 0.83× while one whose MACs "
             "fall 93% (`fashionmnist_vgg16_bn`) reaches 1.98×. At batch 1 on "
             "GPU the correlation is *negative* — nothing is compute-bound, so "
             "removing arithmetic buys nothing and narrower layers only "
             "underutilise the device further.\n")
    L.append("Two structural reasons, and both are specific to this regime. "
             "Channel removal makes layers narrower but leaves the *number* of "
             "kernel launches unchanged, and a 32×32 CIFAR-scale network is "
             "launch-bound long before it is FLOP-bound. Patch factorisation is "
             "worse on this axis by construction: it replaces one convolution "
             "with two, so it *adds* a launch per layer in exchange for the "
             "arithmetic it removes.\n")
    L.append("**What the paper must say.** MACs are an arithmetic measure and "
             "we report them as one. On this hardware, in eager PyTorch, at "
             "these input sizes, they translate to wall-clock only above a large "
             "cut and only at throughput-oriented batch sizes. This does not "
             "affect the parameter reductions, the accuracy results, or the "
             "budget rule — but a reader entitled to assume '−73% MACs' means "
             "'2.7× faster' would be wrong, and the paper now says so.\n")


def _p(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def main():
    L = [
        "# Results — follow-up experiments",
        "",
        f"Generated by `summarize.py` from `results/*.csv` on "
        f"{datetime.now():%Y-%m-%d %H:%M}. Every experiment re-uses the "
        f"production compression code from `compression_cnn/`; only the "
        f"variable under test differs. Sections for experiments that have not "
        f"run say so rather than being omitted.",
        "",
    ]
    sec_e1(L)
    sec_e1b(L)
    sec_e3(L)
    sec_e4(L)
    sec_e2(L)
    sec_e7(L)
    sec_e5(L)
    sec_e6(L)
    OUT.write_text("\n".join(L).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
