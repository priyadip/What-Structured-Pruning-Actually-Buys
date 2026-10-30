#!/usr/bin/env python3
"""Regenerate every reported number from the raw CSVs.

Deliberately a separate code path from the one that wrote them: this file
re-reads `results/*.csv` with the standard-library CSV reader and recomputes
frontiers, contrasts and correlations from scratch, so a bug in the harness that
happened to produce a plausible row cannot survive into the report unchallenged.
Writes `results/summary.json` and `results/tables.md`.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOL = 2.5


def read(name):
    p = RESULTS / name
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def frontier(rows, key=lambda r: True):
    """Smallest alpha whose accuracy is still inside the window."""
    hits = [r for r in rows if key(r) and abs(f(r, "delta")) <= TOL]
    return min(hits, key=lambda r: f(r, "alpha")) if hits else None


def paired(rows_a, rows_b, on=("alpha", "seed")):
    """Differences over cells present in both, matched on `on`."""
    ia = {tuple(r[k] for k in on): r for r in rows_a}
    ib = {tuple(r[k] for k in on): r for r in rows_b}
    keys = sorted(set(ia) & set(ib))
    return [(k, f(ia[k], "acc") - f(ib[k], "acc")) for k in keys]


def summarise(diffs):
    v = [d for _, d in diffs]
    if not v:
        return None
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    se = sd / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"n": len(v), "mean": round(m, 3), "sd": round(sd, 3),
            "se": round(se, 3), "t": round(m / se, 2) if se > 0 else None,
            "min": round(min(v), 3), "max": round(max(v), 3)}


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    out = {}
    gride = json.loads((RESULTS / "phase2_gride.json").read_text())
    g0 = json.loads((RESULTS / "phase0_ground_truth.json").read_text())
    dense = g0["dense_acc_validation_matched"]
    out["dense_acc"] = dense
    out["params"] = g0["params"]
    out["dims"] = gride["dims"]
    out["n_over_d"] = gride["n_over_d"]
    out["mean_n_over_d"] = gride["mean_n_over_d"]
    out["log_mean_n_over_d"] = gride["log_mean_n_over_d"]
    out["gride_seed_spread"] = gride["seed_spread_neuron"]

    sweep = read("sweep.csv")
    method = [r for r in sweep if r["strategy"] == "omp_ls"
              and r["cloud"] == "neuron" and r["target"] == "dense"
              and r["seed"] == "0" and r["layers"] == "all"]

    # ---- headline frontier ------------------------------------------------
    fr = frontier(method)
    if fr:
        out["frontier"] = {
            "alpha_star": f(fr, "alpha"),
            "acc": f(fr, "acc"), "delta": f(fr, "delta"),
            "widths": json.loads(fr["widths"]),
            "red_total_pct": f(fr, "red_total_pct"),
            "red_encoder_pct": f(fr, "red_encoder_pct"),
            "red_ffn_pct": f(fr, "red_ffn_pct"),
            "params_total": int(fr["params_total"]),
        }
        out["primary_criterion_met"] = bool(
            f(fr, "acc") >= dense - TOL and f(fr, "red_total_pct") >= 20.0)

    # ---- ladder shape: is there a knee at alpha = 1? ----------------------
    lad = sorted({(f(r, "alpha"), f(r, "acc")) for r in method})
    out["ladder"] = [{"alpha": a, "acc": v} for a, v in lad]
    out["ladder_monotone"] = all(lad[i][1] <= lad[i + 1][1] + 1e-9
                                 for i in range(len(lad) - 1))
    # The curvature scan must run on the geometric ladder alone. Bisection adds
    # points at 12.125, 12.25, 12.5 whose log-spacing is a hundredth of the
    # ladder's, and a second difference over spacings that uneven reports the
    # sampling pattern rather than the curve: on the full set the three largest
    # curvatures are all bisection artefacts.
    lad = sorted({(f(r, "alpha"), f(r, "acc")) for r in method
                  if r["tag"] == "ladder"})
    out["knee_scan_grid"] = [a for a, _ in lad]
    knee = []
    for i in range(1, len(lad) - 1):
        a0, a1, a2 = lad[i - 1][0], lad[i][0], lad[i + 1][0]
        s1 = (lad[i][1] - lad[i - 1][1]) / (math.log(a1) - math.log(a0))
        s2 = (lad[i + 1][1] - lad[i][1]) / (math.log(a2) - math.log(a1))
        knee.append({"alpha": a1, "slope_left": round(s1, 3),
                     "slope_right": round(s2, 3), "curv": round(s2 - s1, 3)})
    out["knee_scan"] = knee
    at1 = [k for k in knee if abs(k["alpha"] - 1.0) < 1e-9]
    others = [abs(k["curv"]) for k in knee if abs(k["alpha"] - 1.0) >= 1e-9]
    if at1 and others:
        out["knee_at_alpha1"] = {
            "curv_at_1": at1[0]["curv"],
            "median_abs_curv_elsewhere": round(st.median(others), 3),
            "rank_of_alpha1_by_abs_curv":
                1 + sum(1 for c in others if c > abs(at1[0]["curv"])),
            "n_interior_points": len(knee)}

    # ---- every (cloud, target) frontier ------------------------------------
    # The cloud changes `d_l` and therefore the widths, so the clouds cannot be
    # compared at fixed alpha; they are compared as whole frontiers, accuracy
    # against parameters. The target does not change the widths but is crossed
    # with the cloud here because the two turn out to compound.
    out["cloud_frontiers"] = {}
    for cloud in ("neuron", "filter_proj", "weight"):
        for target in ("dense", "residual"):
            rs = [r for r in sweep if r["strategy"] == "omp_ls"
                  and r["cloud"] == cloud and r["target"] == target
                  and r["seed"] == "0" and r["layers"] == "all"]
            fc = frontier(rs)
            if fc:
                out["cloud_frontiers"][f"{cloud}/{target}"] = {
                    "alpha_star": f(fc, "alpha"), "acc": f(fc, "acc"),
                    "delta": f(fc, "delta"),
                    "red_total_pct": f(fc, "red_total_pct"),
                    "red_encoder_pct": f(fc, "red_encoder_pct"),
                    "red_ffn_pct": f(fc, "red_ffn_pct"),
                    "widths": json.loads(fc["widths"]),
                    "dims": json.loads(fc["dims"])}
    if out["cloud_frontiers"]:
        bk = max(out["cloud_frontiers"],
                 key=lambda k: out["cloud_frontiers"][k]["red_total_pct"])
        out["best_frontier"] = dict(out["cloud_frontiers"][bk], config=bk)

    # ---- matched-width baselines ------------------------------------------
    base = read("baselines.csv")
    by = defaultdict(list)
    for r in base:
        by[r["strategy"]].append(r)
    out["baselines"] = {}
    alphas = sorted({f(r, "alpha") for r in base})
    for stg, rs in sorted(by.items()):
        per = {a: f(x, "acc") for a in alphas
               for x in rs if abs(f(x, "alpha") - a) < 1e-9}
        out["baselines"][stg] = {"by_alpha": {str(k): v for k, v in per.items()},
                                 "mean_acc": round(st.mean(per.values()), 3)
                                 if per else None}
    if "omp_ls" in by:
        wins = {}
        for stg, rs in by.items():
            if stg == "omp_ls":
                continue
            d = paired(by["omp_ls"], rs, on=("alpha", "seed"))
            wins[stg] = summarise(d)
            if wins[stg]:
                wins[stg]["ours_wins"] = sum(1 for _, x in d if x > 0)
        out["omp_ls_vs_each"] = wins

    # ---- ablation contrasts ------------------------------------------------
    abl = read("ablation.csv")
    tag = lambda t: [r for r in abl if r["tag"] == t]
    stg = lambda s, t="AB": [r for r in abl if r["tag"] == t and r["strategy"] == s]
    contrasts = {
        "refit_given_magnitude (magnitude_ls - magnitude)":
            paired(stg("magnitude_ls"), stg("magnitude")),
        "refit_given_omp (omp_ls - omp)":
            paired(stg("omp_ls"), stg("omp")),
        "refit_given_random (random_ls - random)":
            paired(stg("random_ls"), stg("random")),
        "refit_given_merge (ot_then_ls - ot_intrafusion)":
            paired(stg("ot_then_ls"), stg("ot_intrafusion")),
        "select_vs_merge_given_refit (omp_ls - ot_then_ls)":
            paired(stg("omp_ls"), stg("ot_then_ls")),
        "omp_vs_magnitude_given_refit (omp_ls - magnitude_ls)":
            paired(stg("omp_ls"), stg("magnitude_ls")),
        "omp_vs_span_given_refit (omp_ls - span_ls)":
            paired(stg("omp_ls"), stg("span_ls")),
        "omp_vs_random_given_refit (omp_ls - random_ls)":
            paired(stg("omp_ls"), stg("random_ls")),
        "sequential_absorption (on - off)":
            paired(stg("omp_ls"), tag("C_noabsorb")),
        "free_intercept (with - without)":
            paired(stg("omp_ls"), tag("D_nointercept")),
        "residual_target (residual - dense)":
            paired(tag("E_residual"), stg("omp_ls")),
    }
    out["ablation"] = {k: summarise(v) for k, v in contrasts.items()}

    # ---- seeds and the robust frontier -------------------------------------
    stab = read("stability.csv")
    seedrows = [r for r in stab if r["tag"] == "seed"]
    per_alpha = defaultdict(list)
    for r in seedrows:
        per_alpha[f(r, "alpha")].append(f(r, "acc"))
    spread = {a: {"n": len(v), "mean": round(st.mean(v), 3),
                  "sd": round(st.stdev(v), 3) if len(v) > 1 else 0.0}
              for a, v in sorted(per_alpha.items())}
    out["seed_spread"] = {str(k): v for k, v in spread.items()}
    sds = [v["sd"] for v in spread.values() if v["n"] > 1]
    if sds:
        sigma = st.mean(sds)
        out["sigma_across_seeds"] = round(sigma, 3)
        robust_tol = TOL - 2 * sigma
        out["robust_tol"] = round(robust_tol, 3)
        ok = [a for a, v in spread.items() if dense - v["mean"] <= robust_tol]
        if ok:
            a_r = min(ok)
            src = [r for r in seedrows if abs(f(r, "alpha") - a_r) < 1e-9]
            out["robust_frontier"] = {
                "alpha": a_r, "mean_acc": spread[a_r]["mean"],
                "sd": spread[a_r]["sd"],
                "red_total_pct": round(st.mean([f(r, "red_total_pct") for r in src]), 3),
                "red_ffn_pct": round(st.mean([f(r, "red_ffn_pct") for r in src]), 3)}

    calibrows = [r for r in stab if r["tag"] == "calib"]
    out["calibration_curve"] = [
        {"calib": int(r["calib"]), "rows": int(r["rows"]), "alpha": f(r, "alpha"),
         "acc": f(r, "acc")} for r in sorted(calibrows, key=lambda r: (f(r, "alpha"),
                                                                      int(r["calib"])))]

    # ---- per-layer -----------------------------------------------------------
    pl = read("perlayer.csv")
    if pl:
        solo = defaultdict(list)
        for r in pl:
            solo[r["tag"]].append(r)
        out["per_layer"] = {}
        for t, rs in sorted(solo.items()):
            l = int(t.split("_L")[1])
            fr_l = frontier(rs)
            out["per_layer"][str(l)] = {
                "d": gride["dims"][l],
                "alpha_star": f(fr_l, "alpha") if fr_l else None,
                "m": json.loads(fr_l["widths"])[l] if fr_l else None,
                "acc": f(fr_l, "acc") if fr_l else None,
                "curve": sorted([(f(r, "alpha"), f(r, "acc")) for r in rs])}

    # ---- placement against the paper's regression ---------------------------
    # cut = 61.0109 * log(mean N/d) - 91.0705, r = +0.898, fit on 16 CNNs over
    # log(N/d) in [1.803, 3.289]. Recomputed in this session from
    # original_research/scripts/load_data.py; see FINDINGS F7.
    x = out["log_mean_n_over_d"]
    pred = 61.0109 * x - 91.0705
    out["paper_regression"] = {
        "slope": 61.0109, "intercept": -91.0705, "r": 0.898,
        "fit_range_log_n_over_d": [1.803, 3.289],
        "our_log_n_over_d": round(x, 4),
        "extrapolation": round(x - 3.289, 4),
        "predicted_cut_pct": round(pred, 2),
        "observed_ffn_cut_pct": out.get("frontier", {}).get("red_ffn_pct"),
        "observed_total_cut_pct": out.get("frontier", {}).get("red_total_pct")}

    # Within-model version of the same question. The six blocks differ in N/d by
    # a factor of 1.9, so the predictor can be tested without leaving the model,
    # using each block's solo accuracy at a fixed alpha as its compressibility.
    if "per_layer" in out:
        for a_fix in (1.0, 2.0, 4.0):
            pts = []
            for l in range(6):
                cur = dict(out["per_layer"][str(l)]["curve"])
                if a_fix in cur:
                    pts.append((math.log(3072.0 / out["dims"][l]), cur[a_fix]))
            if len(pts) == 6:
                # With six points one block can carry the whole correlation, so
                # the leave-one-out range is reported beside it and is what the
                # conclusion has to rest on.
                loo = []
                for k in range(6):
                    rest = pts[:k] + pts[k + 1:]
                    loo.append(round(pearson([p[0] for p in rest],
                                             [p[1] for p in rest]), 4))
                out.setdefault("within_model_n_over_d", {})[f"alpha={a_fix:g}"] = {
                    "n": 6,
                    "r": round(pearson([p[0] for p in pts], [p[1] for p in pts]), 4),
                    "loo_r": loo,
                    "loo_min": min(loo), "loo_max": max(loo),
                    "sign_stable": bool(min(loo) > 0 or max(loo) < 0),
                    "points": [(round(x, 3), y) for x, y in pts]}

    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("dense_acc", "frontier", "primary_criterion_met",
                       "mean_n_over_d", "paper_regression")
                      if k in out}, indent=2))
    print(f"\nwrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
