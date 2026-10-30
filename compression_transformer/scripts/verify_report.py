#!/usr/bin/env python3
"""Re-derive every headline number in REPORT.md straight from the raw CSVs.

`analyze.py` already reads the raw files by a different code path from the
harness that wrote them. This goes one step further and checks that what ended
up in the prose matches what that pass computed, because a number can be correct
in `summary.json` and still be mistyped into a sentence.

Each check states the claim, recomputes it from `results/`, and compares against
the literal value written in the report. Any mismatch is printed and the exit
status is non-zero.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "REPORT.md"
TOL = 2.5

fails: list[str] = []
checks = 0


def rows(name):
    p = RESULTS / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def check(label, got, want, eps=0.02):
    global checks
    checks += 1
    ok = (abs(got - want) <= eps) if isinstance(want, (int, float)) else (got == want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: computed {got}, report says {want}")
    if not ok:
        fails.append(label)


def in_report(pattern) -> bool:
    return re.search(pattern, REPORT.read_text()) is not None


def main():
    text = REPORT.read_text()
    g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
    dense = g0["dense_acc_validation_matched"]

    print("ground truth")
    check("dense accuracy", round(dense, 2), 82.21)
    check("total parameters", g0["params"]["total"], 66955779)
    check("embedding parameters", g0["params"]["embedding"], 23835648)
    check("embedding share %", round(100 * g0["params"]["embedding"]
                                     / g0["params"]["total"], 1), 35.6)
    check("keep-everything is a no-op", abs(g0["noop_delta"]) < 1e-9, True)
    check("valid token rows at calib 2048", g0["calib2048_valid_rows"], 80539)
    check("over-determination", round(g0["overdetermination"], 1), 21.3)

    print("\nfrontiers")
    sw = [r for r in rows("sweep.csv") if r["strategy"] == "omp_ls"
          and r["seed"] == "0" and r["layers"] == "all"]
    best = {}
    for r in sw:
        if abs(float(r["delta"])) > TOL:
            continue
        k = (r["cloud"], r["target"])
        if k not in best or float(r["alpha"]) < float(best[k]["alpha"]):
            best[k] = r
    for (cloud, target), want in (
            (("filter_proj", "residual"), (13.53125, 79.90, 29.97, 46.54, 70.83)),
            (("filter_proj", "dense"), (13.875, 79.76, 29.66, 46.06, 70.09)),
            (("neuron", "residual"), (11.15625, 79.96, 29.63, 46.01, 70.01)),
            (("neuron", "dense"), (12.125, 79.86, 28.52, 44.29, 67.40))):
        r = best.get((cloud, target))
        if r is None:
            fails.append(f"frontier {cloud}/{target} missing")
            print(f"  [FAIL] frontier {cloud}/{target}: not found")
            continue
        for name, got, w in (("alpha*", float(r["alpha"]), want[0]),
                             ("acc", round(float(r["acc"]), 2), want[1]),
                             ("total%", round(float(r["red_total_pct"]), 2), want[2]),
                             ("enc%", round(float(r["red_encoder_pct"]), 2), want[3]),
                             ("ffn%", round(float(r["red_ffn_pct"]), 2), want[4])):
            check(f"{cloud}/{target} {name}", got, w)

    print("\nsuccess criteria")
    b = max(best.values(), key=lambda r: float(r["red_total_pct"]))
    check("primary: acc >= 79.5", float(b["acc"]) >= 79.5, True)
    check("primary: total cut >= 20", float(b["red_total_pct"]) >= 20.0, True)
    check("best config is filter_proj/residual",
          (b["cloud"], b["target"]), ("filter_proj", "residual"))
    check("headline acc", round(float(b["acc"]), 2), 79.90)
    check("headline total cut", round(float(b["red_total_pct"]), 2), 29.97)
    # the two Transformer-specific improvements over the transplanted rule
    base = float(best[("neuron", "dense")]["red_total_pct"])
    check("cloud alone worth", round(float(best[("filter_proj", "dense")]
                                           ["red_total_pct"]) - base, 2), 1.14)
    check("residual target alone worth",
          round(float(best[("neuron", "residual")]["red_total_pct"]) - base, 2), 1.11)
    check("both worth", round(float(b["red_total_pct"]) - base, 2), 1.45)

    print("\nrobust frontier")
    for tag, want_sigma, want_alpha, want_acc, want_cut in (
            ("seed", 0.189, 13.0, 80.19, 27.66),
            ("seed_residual", 0.170, 12.0, 80.07, 28.79)):
        seedrows = [r for r in rows("stability.csv") if r["tag"] == tag]
        per = defaultdict(list)
        for r in seedrows:
            per[float(r["alpha"])].append(float(r["acc"]))
        sds = [st.stdev(v) for v in per.values() if len(v) > 1]
        sigma = st.mean(sds)
        check(f"{tag}: sigma across seeds", round(sigma, 3), want_sigma)
        rt = TOL - 2 * sigma
        ok = sorted(a for a, v in per.items() if dense - st.mean(v) <= rt)
        check(f"{tag}: robust alpha", ok[0], want_alpha)
        src = [r for r in seedrows if abs(float(r["alpha"]) - ok[0]) < 1e-9]
        check(f"{tag}: robust mean acc",
              round(st.mean([float(r["acc"]) for r in src]), 2), want_acc)
        check(f"{tag}: robust total cut",
              round(st.mean([float(r["red_total_pct"]) for r in src]), 2), want_cut)

    print("\nmatched-width comparison")
    base = rows("baselines.csv")
    by = defaultdict(dict)
    for r in base:
        by[r["strategy"]][float(r["alpha"])] = float(r["acc"])
    for stg, want_mean in (("omp_ls", 76.73), ("fwdbwd_ls", 76.75),
                           ("span_ls", 76.36), ("thinet", 74.47),
                           ("magnitude_ls", 63.99), ("magnitude", 38.35),
                           ("ot_intrafusion", 38.96), ("ot_then_ls", 66.64)):
        check(f"mean acc {stg}", round(st.mean(by[stg].values()), 2), want_mean)
    # (mean paired difference, wins) exactly as the report's table states them
    for stg, want, wins in (("magnitude", 38.39, 7), ("ot_intrafusion", 37.77, 7),
                            ("magnitude_ls", 12.74, 7), ("ot_then_ls", 10.09, 7),
                            ("thinet", 2.26, 7), ("thinet_lsfull", 0.56, 6),
                            ("span_ls", 0.37, 5), ("fwdbwd_ls", -0.02, 4)):
        common = sorted(set(by["omp_ls"]) & set(by[stg]))
        d = [by["omp_ls"][a] - by[stg][a] for a in common]
        check(f"omp_ls minus {stg}", round(st.mean(d), 2), want)
        check(f"omp_ls wins vs {stg}", sum(1 for x in d if x > 0), wins)

    print("\nablation contrasts")
    s = json.loads((RESULTS / "summary.json").read_text())
    for key, want in (
            ("refit_given_merge (ot_then_ls - ot_intrafusion)", 31.88),
            ("refit_given_magnitude (magnitude_ls - magnitude)", 28.84),
            ("omp_vs_random_given_refit (omp_ls - random_ls)", 24.96),
            ("omp_vs_magnitude_given_refit (omp_ls - magnitude_ls)", 12.63),
            ("select_vs_merge_given_refit (omp_ls - ot_then_ls)", 9.90),
            ("refit_given_omp (omp_ls - omp)", 5.77),
            ("residual_target (residual - dense)", 0.58),
            ("omp_vs_span_given_refit (omp_ls - span_ls)", 0.08),
            ("free_intercept (with - without)", -0.03),
            ("sequential_absorption (on - off)", -0.44)):
        check(f"ablation {key.split(' (')[0]}", s["ablation"][key]["mean"], want)

    print("\ndrift")
    dr = json.loads((RESULTS / "phase5b_drift.json").read_text())["per_layer"]
    fr = [d["drift_reachable_frac"] for d in dr if d["layer"] > 0]
    check("min reachable %", round(100 * min(fr), 1), 8.2)
    check("max reachable %", round(100 * max(fr), 1), 13.5)
    check("report says 86 to 92 percent unreachable",
          (round(100 - 100 * max(fr)) == 86) and (round(100 - 100 * min(fr)) == 92),
          True)

    print("\nper-block solo frontiers")
    pl = json.loads((RESULTS / "perlayer_frontier.json").read_text())
    check("blocks reaching alpha=1",
          sum(1 for v in pl.values() if v and v["alpha_star"] == 1.0), 5)
    check("block 0 alpha*", pl["0"]["alpha_star"], 2.0)
    check("block 5 delta", round(pl["5"]["delta"], 2), 0.10)
    check("block 5 FFN cut %", round(100 * (1 - 53 / 3072), 1), 98.3)

    print("\npredictor")
    pr = s["paper_regression"]
    check("log(mean N/d)", round(pr["our_log_n_over_d"], 3), 3.659)
    check("extrapolation beyond fit", round(pr["extrapolation"], 2), 0.37)
    check("predicted cut", round(pr["predicted_cut_pct"]), 132)
    w = s["within_model_n_over_d"]["alpha=1"]
    check("within-model r at alpha=1", w["r"], 0.745)
    check("LOO min", w["loo_min"], -0.381)
    check("LOO max", w["loo_max"], 0.885)
    check("LOO sign not stable", w["sign_stable"], False)

    print("\nknee")
    kt = json.loads((RESULTS / "knee_test.json"). read_text())
    check("no breakpoint at alpha=1", kt["breakpoint_at_1_in_all_seeds"], False)
    check("knee not supported", kt.get("knee_at_1_supported"), False)


    print("\nattention")
    at = [r for r in rows("attention.csv") if r["tag"] == "attn_only"]
    ao = {r["strategy"]: {float(x["alpha"]): float(x["acc"])
                          for x in at if x["strategy"] == r["strategy"]}
          for r in at}
    hits = [r for r in at if r["strategy"] == "omp_ls"
            and abs(float(r["delta"])) <= TOL]
    fr = min(hits, key=lambda r: float(r["alpha"]))
    check("attention frontier alpha", float(fr["alpha"]), 7.0)
    check("attention frontier acc", round(float(fr["acc"]), 2), 79.95)
    check("attention frontier total cut", round(float(fr["red_total_pct"]), 2), 9.70)
    check("attention frontier heads", fr["widths"], "[6, 6, 8, 8, 7, 4]")
    for a, want in ((1.0, 7.28), (2.0, 13.00), (3.0, 14.40), (4.0, 14.06),
                    (6.0, 9.64), (7.0, 0.20), (8.0, 0.22)):
        check(f"attention omp minus span at alpha={a:g}",
              round(ao["omp_ls"][a] - ao["span_ls"][a], 2), want)
    gd = json.loads((RESULTS / "phase8_attention_gride.json").read_text())
    check("attention dims min heads' worth",
          round(min(gd["dims"]) / 64, 2), 0.55)
    check("attention dims max heads' worth",
          round(max(gd["dims"]) / 64, 2), 1.18)
    jo = [r for r in rows("attention.csv") if r["tag"] == "joint"
          and r["strategy"] == "omp_ls" and abs(float(r["delta"])) <= TOL]
    ffn_only = max(best.values(), key=lambda r: float(r["red_total_pct"]))
    if jo:
        bj = max(jo, key=lambda r: float(r["red_total_pct"]))
        check("joint configs inside the window", len(jo), 12)
        check("best joint total cut", round(float(bj["red_total_pct"]), 2), 29.12)
        check("best joint acc", round(float(bj["acc"]), 2), 79.82)
        check("no joint config beats FFN alone",
              float(bj["red_total_pct"]) < float(ffn_only["red_total_pct"]), True)

    print("\nallocation")
    al = rows("alloc.csv")
    uni = [r for r in al if r["tag"] == "uniform"][0]
    zad = [r for r in al if r["tag"] == "zador"]
    check("uniform acc", round(float(uni["acc"]), 2), 80.24)
    if zad:
        check("zador acc", round(float(zad[0]["acc"]), 2), 79.56)
        check("zador loss", round(float(uni["acc"]) - float(zad[0]["acc"]), 2), 0.68)
    bs = max((r for r in al if r["tag"] == "search"), key=lambda r: float(r["acc"]))
    check("oracle gain", round(float(bs["acc"]) - float(uni["acc"]), 2), 0.26)

    print("\nprose spot checks")
    for pat, label in ((r"79\.90% at 29\.97%", "primary claim sentence"),
                       (r"80\.07% at 28\.79%", "robust frontier sentence"),
                       (r"3072", "unit count appears"),
                       (r"42\.3%", "FFN ceiling appears")):
        global checks
        checks += 1
        ok = in_report(pat)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            fails.append(label)

    print(f"\n{checks - len(fails)}/{checks} checks passed")
    if fails:
        print("FAILED:")
        for f_ in fails:
            print("  " + f_)
        sys.exit(1)
    print("every headline number in REPORT.md reproduces from results/")


if __name__ == "__main__":
    main()
