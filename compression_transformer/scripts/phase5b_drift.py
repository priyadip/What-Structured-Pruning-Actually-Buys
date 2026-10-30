#!/usr/bin/env python3
"""Why sequential absorption does nothing here.

The ablation says absorbing upstream error costs 0.44 points rather than
earning anything, which reverses the paper. The candidate explanation is
structural: a residual block's output is dominated by the stream, the FFN only
writes a small correction onto it, and the correction each block writes is
largely independent of the ones before. If that is right then the drift a
downstream unit inherits is small in relative terms, there is little to absorb,
and the cost of fitting on a design matrix that has itself drifted is not repaid.

This measures the three quantities that claim rests on, per block:

  write ratio   ||ffn_out|| / ||a||, how much the FFN moves the stream
  drift         ||a_comp - a_dense|| / ||a_dense||, what arrives from upstream
  reachable     the fraction of that drift lying in the span of the surviving
                units, which is the most a refit could remove even in principle
"""

from __future__ import annotations

import argparse
import json

import torch

import core as C
import harness as H
from common import linalg as LA


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    s = H.Session()
    rows, cache = s.cache(a.seed)
    widths = s.widths(a.alpha)
    H.log(f"alpha {a.alpha} widths {widths}")

    import copy
    comp = copy.deepcopy(s.model)
    comp.eval()
    out = []
    for i in range(C.N_LAYERS):
        a_dense, Y = cache[i]
        X = C.comp_input(comp, rows, i, 64)
        layer = comp.distilbert.transformer.layer[i]
        u = C.Unit(layer.ffn.lin1, layer.ffn.lin2, layer.ffn.activation, X, Y,
                   seed=a.seed)
        G, P, us, ys, n = u.stats
        m = int(widths[i])
        keep = LA.group_select(G, P, 1, m)

        # how much of the inherited drift a refit on these units could remove
        D = (a_dense - X)
        cols = u.cols(keep)
        Us = u.U[:, cols]
        Gs = LA.f64(Us.t() @ Us) + 1e-8 * torch.eye(m, dtype=torch.float64,
                                                    device=Us.device)
        Ps = LA.f64(Us.t() @ D)
        proj_sq = float((Ps * torch.linalg.solve(Gs, Ps)).sum())
        d_sq = float(LA.f64(D).pow(2).sum())
        reachable = proj_sq / d_sq if d_sq > 0 else float("nan")

        ffn_out = Y
        rec = {"layer": i,
               "write_ratio": float(ffn_out.norm() / a_dense.norm()),
               "drift_rel": rel(X, a_dense),
               "drift_reachable_frac": reachable}
        out.append(rec)
        H.log(f"  layer {i}: FFN writes {rec['write_ratio']:.3f} of the stream, "
              f"inherited drift {rec['drift_rel']:.4f}, "
              f"{100 * reachable:.1f}% of it reachable by the refit")

        W1, b1, W2, b2 = C.STRATEGIES["omp_ls"](u, m)
        C._replace_ffn(layer, W1, b1, W2, b2)
        u.free()
        del X, Us, D
        torch.cuda.empty_cache()

    (C.RESULTS / "phase5b_drift.json").write_text(
        json.dumps({"alpha": a.alpha, "seed": a.seed, "per_layer": out}, indent=2))
    H.log("wrote phase5b_drift.json")


if __name__ == "__main__":
    main()
