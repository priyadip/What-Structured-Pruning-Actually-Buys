#!/usr/bin/env python3
"""Checks that must pass before any number in the report is believed.

1. `group_select` at `gs = 1` reproduces the MLP study's `omp_select` and
   `pivoted_cholesky_select` index for index, so the Transformer port is
   demonstrably running the same selection rule as the paper.
2. The masked backward elimination agrees with the production kernel.
3. The consumer refit really is the least-squares minimiser, checked against a
   dense `lstsq` on the raw rows rather than the cached Gram.
4. A keep-everything budget leaves the FFN numerically unchanged.
5. The design matrix carries no padding rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C                                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compression_mlp"))
from common import linalg as LA                                     # noqa: E402


def t1_selection_matches_mlp():
    import importlib.util
    p = (Path(__file__).resolve().parents[2] / "compression_mlp"
         / "run_autonomous_compression.py")
    spec = importlib.util.spec_from_file_location("mlp_ref", p)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["mlp_ref"] = ref        # dataclasses needs the module registered
    spec.loader.exec_module(ref)
    g = torch.Generator().manual_seed(0)
    for N, out, m in ((60, 9, 17), (128, 3, 40), (200, 7, 5)):
        A = torch.randn(900, N, generator=g)
        Y = torch.randn(900, out, generator=g)
        G, P, us, ys, n = LA.gram_cross(A, Y)
        assert LA.group_select(G, P, 1, m) == ref.omp_select(G, P, m), \
            f"OMP mismatch at N={N}, m={m}"
        assert LA.group_select(G, None, 1, m) == ref.pivoted_cholesky_select(G, m), \
            f"span mismatch at N={N}, m={m}"
        print(f"  [PASS] gs=1 selection == MLP study  (N={N}, m={m})")


def t2_backward_matches_production():
    g = torch.Generator().manual_seed(1)
    for N, out, m in ((40, 7, 11), (55, 4, 23)):
        A = torch.randn(700, N, generator=g)
        Y = torch.randn(700, out, generator=g)
        G, P, us, ys, n = LA.gram_cross(A, Y)
        ours = C.backward_eliminate(G, P, us, ys, n, m, list(range(N)), ridge=1e-8)
        theirs = LA.forward_backward_select(G, P, us, ys, n, 1, m,
                                            ridge=1e-8, over=float(N))
        assert ours == theirs, f"N={N} m={m}: {ours} != {theirs}"
        print(f"  [PASS] masked backward == production  (N={N}, m={m})")


def t3_refit_is_the_minimiser():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(500, 30, generator=g).double()
    Y = torch.randn(500, 6, generator=g).double()
    keep = sorted(torch.randperm(30, generator=g)[:11].tolist())
    G, P, us, ys, n = LA.gram_cross(A.float(), Y.float())
    V, c = LA.ls_consuming(G, P, us, ys, n, torch.as_tensor(keep), ridge=0.0)
    Phi = torch.cat([A[:, keep], torch.ones(500, 1, dtype=A.dtype)], 1)
    Th = torch.linalg.lstsq(Phi, Y).solution
    err = max(float((V.double() - Th[:-1].t()).abs().max()),
              float((c.double() - Th[-1]).abs().max()))
    assert err < 1e-4, f"refit differs from lstsq by {err}"
    print(f"  [PASS] refit == lstsq with intercept  (max |diff| = {err:.2e})")


def t4_noop_and_padding():
    dev = C.device()
    model = C.load_model(dev)
    ids, mask = C.calibration_set(512, seed=0)
    rows = C.Rows(ids, mask, budget=16384, seed=0)

    # every sampled row must be a real token
    assert bool((rows.keep & ~mask.bool()).sum() == 0), "padding leaked into rows"
    print(f"  [PASS] no padding rows  ({rows.n} of {rows.n_valid} valid)")

    # the under-determination guard must actually fire
    thin = C.Rows(*C.calibration_set(64, seed=0), budget=4096, seed=0)
    thin_cache = C.dense_targets(model, thin)
    try:
        C.compress(model, thin, [C.HIDDEN_DIM] * 6, "omp_ls", thin_cache, seed=0)
        raise AssertionError("under-determined fit was accepted")
    except RuntimeError as exc:
        assert "unknowns" in str(exc)
        print(f"  [PASS] under-determined fit refused  ({thin.n} rows)")
    del thin_cache, thin

    cache = C.dense_targets(model, rows)
    before = [l.ffn.lin1.weight.clone() for l in model.distilbert.transformer.layer]
    comp, _ = C.compress(model, rows, [C.HIDDEN_DIM] * 6, "omp_ls", cache, seed=0)
    for i, l in enumerate(comp.distilbert.transformer.layer):
        assert torch.equal(l.ffn.lin1.weight, before[i]), f"layer {i} changed"
    print("  [PASS] keep-everything is a no-op on all six blocks")

    # the unit's consumer really is lin2 and nothing else reads the 3072 dims
    for i, l in enumerate(model.distilbert.transformer.layer):
        readers = [n for n, mod in l.named_modules()
                   if isinstance(mod, torch.nn.Linear)
                   and mod.in_features == C.HIDDEN_DIM]
        assert readers == ["ffn.lin2"], f"layer {i}: {readers} read the intermediate"
    print("  [PASS] lin2 is the only consumer of the FFN intermediate")


if __name__ == "__main__":
    for fn in (t1_selection_matches_mlp, t2_backward_matches_production,
               t3_refit_is_the_minimiser, t4_noop_and_padding):
        print(fn.__name__)
        fn()
    print("\nall self-tests passed")
