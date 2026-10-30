#!/usr/bin/env python3
"""Phase 9: is a uniform multiplier the right way to spend the budget?

A single global `alpha` assumes every block converts units into accuracy at the
same rate. Zador's bound says it does not, and the paper derives a per-layer
allocation from the same inequality that sets the widths. It then finds, on both
CIFAR datasets, that freeing the multipliers and running coordinate descent
returns the uniform allocation anyway. Six FFN blocks with intrinsic dimensions
spanning 52.7 to 97.8 are a fair second test of that.

Two allocations are tried, and both stay inside the rule because every width is
still `round(alpha_l d_l)`; only `alpha_l` is allowed to differ.

  zador   `m_l = d_l (K w_l / (d_l^2 c_l))^(d_l / (2 + d_l))`, one global `K`
          swept to move along the budget. Every FFN unit costs the same 1537
          parameters, so `c_l` is constant and the allocation is driven purely
          by the dimensions.

  search  coordinate descent over pairs, moving multiplier from one block to
          another at fixed parameter count. Held-out accuracy is the objective,
          so this is an oracle and is reported as an upper bound on what any
          allocation rule could buy, not as a method.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import core as C
import harness as H
from common.gride_conv import zador_widths

OUT = C.RESULTS / "alloc.csv"
PER_UNIT = 2 * C.DIM + 1                      # lin1 row + lin1 bias + lin2 column


def widths_of(alphas, dims):
    return [int(np.clip(round(a * d), 4, C.HIDDEN_DIM)) for a, d in zip(alphas, dims)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=12.125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="1.5,0.75")
    ap.add_argument("--what", default="zador,search")
    a = ap.parse_args()
    what = set(a.what.split(","))

    s = H.Session()
    dims = s.dims()
    base_alphas = [a.alpha] * C.N_LAYERS
    base_w = widths_of(base_alphas, dims)
    budget = sum(base_w)
    H.log(f"uniform alpha={a.alpha} widths {base_w} total {budget} units")

    def evaluate(alphas, tag, note=""):
        w = widths_of(alphas, dims)
        rows, cache = s.cache(a.seed)
        comp, _ = C.compress(s.model, rows, w, "omp_ls", cache, seed=a.seed,
                             target="residual")
        acc = C.accuracy(comp, s.eval_ids, s.eval_mask, s.eval_lab)
        pc = C.param_counts(comp)
        row = {
            "timestamp": H.datetime.now().isoformat(timespec="seconds"),
            "tag": tag, "strategy": "omp_ls", "alpha": round(float(np.mean(alphas)), 4),
            "seed": a.seed, "cloud": s.cloud, "target": "residual", "sequential": 1,
            "ridge": 1e-4, "calib": s.calib, "rows": rows.n, "layers": "all",
            "acc": round(acc, 4), "dense_acc": round(s.dense_acc, 4),
            "delta": round(acc - s.dense_acc, 4),
            "hit": int(abs(acc - s.dense_acc) <= s.TOL),
            "params_total": pc["total"], "params_encoder": pc["encoder"],
            "params_ffn": pc["ffn"],
            "red_total_pct": round(100 * (1 - pc["total"] / s.base["total"]), 4),
            "red_encoder_pct": round(100 * (1 - pc["encoder"] / s.base["encoder"]), 4),
            "red_ffn_pct": round(100 * (1 - pc["ffn"] / s.base["ffn"]), 4),
            "widths": json.dumps(w),
            "dims": json.dumps([round(d, 3) for d in dims]),
            "seconds": 0, "note": note or json.dumps([round(x, 3) for x in alphas]),
        }
        H.csv_append(OUT, row)
        H.log(f"  {tag:12s} units={sum(w):5d} w={w} acc={acc:6.2f}")
        del comp
        C.torch.cuda.empty_cache()
        return acc, w

    uni_acc, _ = evaluate(base_alphas, "uniform")

    if "zador" in what:
        cost = [float(PER_UNIT)] * C.N_LAYERS
        # K enters as (K / (d^2 c))^(d/(2+d)) with c = 1537 parameters per unit
        # and d ~ 85, so the constant sits near 1e7 and reaching 1000 units per
        # block needs K ~ 1e8. A bracket that stops at 1e6 silently returns the
        # smallest allocation the rule can express, which is what the first run
        # did.
        lo, hi = 1e-6, 1e14
        for _ in range(80):                       # bisect K to the same unit count
            K = (lo * hi) ** 0.5
            w, _ = zador_widths(dims, K, [C.HIDDEN_DIM] * C.N_LAYERS, cost)
            if sum(w) > budget:
                hi = K
            else:
                lo = K
        K = (lo * hi) ** 0.5
        w, mults = zador_widths(dims, K, [C.HIDDEN_DIM] * C.N_LAYERS, cost)
        H.log(f"zador K={K:.4g} multipliers {[round(m, 3) for m in mults]}")
        evaluate(mults, "zador", note=f"K={K:.6g}")

    if "search" in what:
        cur = list(base_alphas)
        best = uni_acc
        for step in [float(x) for x in a.steps.split(",")]:
            improved = True
            while improved:
                improved = False
                for i, j in itertools.permutations(range(C.N_LAYERS), 2):
                    trial = list(cur)
                    trial[i] = max(0.5, trial[i] - step)
                    # parameter-neutral: every unit costs the same, so the
                    # multiplier moved out of block i buys step*d_i/d_j in block j
                    trial[j] = trial[j] + step * dims[i] / dims[j]
                    if abs(sum(widths_of(trial, dims)) - budget) > 2:
                        continue
                    acc, _ = evaluate(trial, "search",
                                      note=f"move {step} from L{i} to L{j}")
                    if acc > best + 1e-9:
                        best, cur, improved = acc, trial, True
                        H.log(f"    accepted, now {best:.2f} at "
                              f"{[round(x, 3) for x in cur]}")
                        break
        H.log(f"search finished: {best:.2f} vs uniform {uni_acc:.2f} "
              f"({best - uni_acc:+.2f}) at {[round(x, 3) for x in cur]}")
        (C.RESULTS / "alloc_search.json").write_text(json.dumps(
            {"uniform_acc": uni_acc, "best_acc": best, "alphas": cur,
             "widths": widths_of(cur, dims), "gain": best - uni_acc}, indent=2))


if __name__ == "__main__":
    main()
