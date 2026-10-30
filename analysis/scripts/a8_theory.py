#!/usr/bin/env python3
"""A8. Verify the exact decomposition and the tail bound it implies.

The consumer's target is by construction `Y = U V^T + 1 b^T`, so removing the
units outside `S` and keeping the consumer's own weights leaves exactly the
discarded contribution

    D  =  U_{S^c} V_{S^c}^T ,          R_keep(S) = ||D||_F .

Re-solving the consumer projects `Y` onto `span([U_S, 1])`. Since
`(I - P_S) U_S = 0`, the residual after re-solving is exactly the part of `D`
that the survivors cannot express:

    R_fit(S) = ||(I - P_S) D||_F ,

and because `P_S D` and `(I - P_S) D` are orthogonal,

    R_keep^2 - R_fit^2 = ||P_S D||_F^2 .                             (Theorem 1)

Two consequences are checked numerically here.

(i)  The identity itself, to floating-point accuracy, on every layer and budget.
(ii) `R_fit(S) <= ||V||_2 * tau_M` with `tau_M^2 = sum_{i>M} sigma_i(U)^2`,
     which is what makes the *selection* term small: two selections of the same
     size differ by at most twice that quantity, whereas the *correction* term
     is `R_keep - R_fit` and is governed by `||D||`, which does not vanish with
     the spectral tail.

The predicted ratio is therefore `R_keep / (2 ||V||_2 tau_M) - 1/2`, and the
last block below checks that this tracks the ratio measured in A7.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "compression_mlp"), str(ROOT / "compression_cnn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import linalg as LA                                     # noqa: E402
from a7_mechanism import build_bench, load_ref                      # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MODELS = ["mnist_mlp2_bn", "mnist_mlp3_bn", "fashionmnist_mlp2_bn",
          "fashionmnist_mlp3_bn", "cifar10_mlp3_quad_bn", "cifar10_mlp3_oct_bn",
          "cifar100_mlp3_quad_bn", "cifar100_mlp3_oct_bn"]


@torch.no_grad()
def one_cell(U, Y, W2, b2, keep, sig):
    """Every quantity in Theorem 1, in float64, with no ridge so the identity is
    exact rather than approximately exact."""
    n, N = U.shape
    U64, W64, b64 = U.double(), W2.double(), b2.double()
    k = torch.as_tensor(sorted(keep), device=U.device, dtype=torch.long)
    mask = torch.ones(N, dtype=torch.bool, device=U.device)
    mask[k] = False
    drop = mask.nonzero().flatten()

    D = U64[:, drop] @ W64[:, drop].t()               # discarded contribution
    r_keep = float(D.norm())

    Phi = torch.cat([U64[:, k], torch.ones(n, 1, dtype=U64.dtype,
                                           device=U64.device)], 1)
    # projector applied to D, via least squares (no ridge)
    sol = torch.linalg.lstsq(Phi, D).solution
    PD = Phi @ sol
    r_fit = float((D - PD).norm())
    in_span = float(PD.norm())

    # the same residual computed the other way, as ||Y - P_S Y||: must agree
    solY = torch.linalg.lstsq(Phi, Y.double()).solution
    r_fit_Y = float((Y.double() - Phi @ solY).norm())

    M = len(keep)
    tau = float(torch.sqrt(torch.clamp((sig[M:] ** 2).sum(), min=0.0)))
    return {"M": M, "N": N, "r_keep": r_keep, "r_fit": r_fit,
            "r_fit_via_Y": r_fit_Y, "in_span": in_span, "tau_M": tau,
            "identity_gap": abs(r_keep ** 2 - r_fit ** 2 - in_span ** 2)
            / max(r_keep ** 2, 1e-30),
            "consistency_gap": abs(r_fit - r_fit_Y) / max(r_fit, 1e-30),
            "V_op": float(torch.linalg.matrix_norm(W64, 2))}


def main():
    ref = load_ref()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        try:
            bench = build_bench(ref, name, dev)
        except Exception as exc:
            print(f"  {name}: skipped ({type(exc).__name__})")
            continue
        for l in range(len(bench.A_orig)):
            U, Y = bench.A_orig[l], bench.Y_orig[l]
            W2, b2 = bench.Ws[l + 1], bench.bs[l + 1]
            N = U.shape[1]
            sig = torch.linalg.svdvals(U.double())
            G, P, us, ys, n = LA.gram_cross(U, Y)
            for frac in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
                M = max(4, int(round(frac * N)))
                if M >= N:
                    continue
                keep = LA.group_select(G, P, 1, M)
                r = one_cell(U, Y, W2, b2, keep, sig)
                r.update(model=name, layer=l)
                rows.append(r)
        print(f"  {name}: {len([r for r in rows if r['model'] == name])} cells")
        del bench
        if dev == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("Theorem 1, checked numerically")
    print("=" * 72)
    idg = [r["identity_gap"] for r in rows]
    csg = [r["consistency_gap"] for r in rows]
    print(f"  cells                                  {len(rows)}")
    print(f"  max relative error in R_keep^2 = R_fit^2 + ||P_S D||^2   "
          f"{max(idg):.3e}")
    print(f"  max relative disagreement of the two ways of computing R_fit  "
          f"{max(csg):.3e}")

    print()
    print("=" * 72)
    print("The tail bound:  R_fit <= ||V||_2 * tau_M")
    print("=" * 72)
    viol = [r for r in rows if r["r_fit"] > r["V_op"] * r["tau_M"] * 1.000001]
    ratio = [r["r_fit"] / max(r["V_op"] * r["tau_M"], 1e-30) for r in rows]
    print(f"  violations: {len(viol)} of {len(rows)}")
    print(f"  R_fit / (||V||_2 tau_M):  median {sorted(ratio)[len(ratio)//2]:.4f}, "
          f"max {max(ratio):.4f}")

    print()
    print("=" * 72)
    print("Predicted vs measured separation, by kept fraction")
    print("=" * 72)
    print(f"{'M/N':>12s} {'R_keep/||Y||':>13s} {'R_fit/||Y||':>12s} "
          f"{'predicted':>10s} {'measured':>9s} {'n':>4s}")
    bins = [(0.0, 0.075), (0.075, 0.15), (0.15, 0.275), (0.275, 0.425),
            (0.425, 0.6), (0.6, 0.8), (0.8, 1.01)]
    tab = {}
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["M"] / r["N"] < hi]
        if not sel:
            continue
        # predicted separation: correction / selection >= R_keep/(2||V|| tau) - 1/2
        pred = sum(r["r_keep"] / max(2 * r["V_op"] * r["tau_M"], 1e-30) - 0.5
                   for r in sel) / len(sel)
        meas = sum((r["r_keep"] - r["r_fit"]) / max(r["r_fit"], 1e-30)
                   for r in sel) / len(sel)
        rk = sum(r["r_keep"] for r in sel) / len(sel)
        rf = sum(r["r_fit"] for r in sel) / len(sel)
        print(f"{lo:.3f}-{hi:.3f} {rk:13.1f} {rf:12.1f} {pred:10.2f} "
              f"{meas:9.2f} {len(sel):4d}")
        tab[f"{lo:.3f}-{hi:.3f}"] = {"r_keep": round(rk, 3), "r_fit": round(rf, 3),
                                     "predicted": round(pred, 3),
                                     "measured": round(meas, 3), "n": len(sel)}

    xs = [math.log(max(r["r_keep"] / max(2 * r["V_op"] * r["tau_M"], 1e-30), 1e-9))
          for r in rows]
    ys = [math.log(max((r["r_keep"] - r["r_fit"]) / max(r["r_fit"], 1e-30), 1e-9))
          for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    r_pm = sxy / math.sqrt(sxx * syy)
    print()
    print(f"  correlation between the predicted and the measured separation "
          f"(log-log): r = {r_pm:+.4f}")

    res = {"n_cells": len(rows), "max_identity_rel_error": max(idg),
           "max_consistency_rel_error": max(csg),
           "tail_bound_violations": len(viol),
           "tail_ratio_median": sorted(ratio)[len(ratio) // 2],
           "tail_ratio_max": max(ratio),
           "by_kept_fraction": tab,
           "pred_vs_measured_logr": round(r_pm, 4),
           "rows": rows}
    (OUT / "a8_theory.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a8_theory.json'}")


if __name__ == "__main__":
    main()
