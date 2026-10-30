#!/usr/bin/env python3
"""Phase 0: measure what everything else is compared against.

Nothing here is taken on trust. The dense accuracy is measured rather than
quoted from the task brief, the label mapping is checked against every
permutation so a 33% result cannot be mistaken for a broken model, and the
harness is required to reproduce dense accuracy exactly when the budget keeps
every unit, because a compression pipeline that silently perturbs a no-op will
silently perturb everything else.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import torch

import core as C


def main():
    t0 = time.time()
    dev = C.device()
    print(f"device: {dev}")
    model = C.load_model(dev)

    pc = C.param_counts(model)
    print("\nparameter accounting")
    for k, v in pc.items():
        print(f"  {k:10s} {v:>12,}  ({100 * v / pc['total']:5.2f}% of total)")
    max_total_cut = 100.0 * (pc["ffn"] - 6 * (768 + 4 * 768 + 4)) / pc["total"]
    print(f"  ceiling on total-model reduction from FFN alone: ~{max_total_cut:.1f}%")

    print("\nevaluating dense on MNLI validation_matched ...")
    ids, mask, lab = C.eval_set()
    accs = {}
    for perm in itertools.permutations(range(3)):
        pass
    base = C.accuracy(model, ids, mask, lab)
    accs["identity"] = base
    print(f"  identity label mapping: {base:.2f}%")
    if base < 50.0:
        print("  below chance-plus; checking all label permutations")
        for perm in itertools.permutations(range(3)):
            if perm == (0, 1, 2):
                continue
            a = C.accuracy(model, ids, mask, lab, perm=perm)
            accs[str(perm)] = a
            print(f"  perm {perm}: {a:.2f}%")

    dense_acc = max(accs.values())
    print(f"\ndense accuracy taken as {dense_acc:.2f}%")

    # ---- no-op check: budget that keeps everything must change nothing -----
    print("\nno-op check (widths = 3072 everywhere, omp_ls) ...")
    rows = C.Rows(*C.calibration_set(256, seed=0), budget=16384, seed=0)
    cache = C.dense_targets(model, rows)
    comp, info = C.compress(model, rows, [C.HIDDEN_DIM] * 6, "omp_ls", cache, seed=0)
    noop = C.accuracy(comp, ids, mask, lab)
    print(f"  no-op accuracy {noop:.4f}% vs dense {base:.4f}%  "
          f"(delta {noop - base:+.4f})")
    assert abs(noop - base) < 1e-9, "keep-everything is not a no-op; harness is wrong"

    # ---- overdetermination at the real calibration size --------------------
    rows_big = C.Rows(*C.calibration_set(2048, seed=0), budget=65536, seed=0)
    od = rows_big.n / (C.HIDDEN_DIM + 1)
    print(f"\ncalibration 2048 examples: {rows_big.n_valid:,} valid token rows, "
          f"{rows_big.n:,} sampled, {od:.1f}x over-determined")

    out = {
        "model": C.MODEL_ID,
        "dense_acc_validation_matched": base,
        "dense_acc_best_permutation": dense_acc,
        "label_mapping": "identity" if base >= 50 else "see permutations",
        "permutation_accs": accs,
        "params": pc,
        "ffn_share_pct": 100 * pc["ffn"] / pc["total"],
        "embedding_share_pct": 100 * pc["embedding"] / pc["total"],
        "max_total_reduction_from_ffn_pct": max_total_cut,
        "noop_acc": noop,
        "noop_delta": noop - base,
        "calib2048_valid_rows": rows_big.n_valid,
        "calib2048_sampled_rows": rows_big.n,
        "overdetermination": od,
        "seconds": time.time() - t0,
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "phase0_ground_truth.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {C.RESULTS / 'phase0_ground_truth.json'}  "
          f"({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
