#!/usr/bin/env python3
"""A7. The mechanism, measured in the linear algebra rather than in accuracy.

A2 to A6 are phenomenology: they show *that* the correction dominates and *that*
the selection rule stops mattering as the budget loosens. This measures the two
quantities the explanation is stated in, directly, on the layers themselves.

Fix a unit with design matrix `U` (post-activation of the producer, `n x N`) and
target `Y` (the dense consumer's pre-activation, `n x C`). For a chosen index
set `S` of size `M`, write `P_S` for the orthogonal projector onto the column
space of `[U_S, 1]`. Two residuals:

    refit      `R_fit(S) = || Y - P_S Y ||_F`      the best any consumer can do
                                                    given those units
    no refit   `R_keep(S) = || Y - U_S V_S^T - b ||_F`
                                                    what the layer's own weights
                                                    give on the survivors

The claim under test has two halves.

(1)  `R_keep - R_fit` does not shrink as `M` grows towards `N`, because keeping
     the original columns is not a projection at any `M`. This is the correction
     term and the prediction is that it stays large.

(2)  `R_fit(S_span) - R_fit(S_omp)` shrinks as `M` grows past the effective rank
     of `U`, because once the retained columns span the directions carrying the
     signal, which columns were retained stops mattering. This is the selection
     term and the prediction is that it decays.

Both are computed exactly, with no accuracy evaluation and no search, so the
result cannot be an artefact of a ladder or a window. Everything runs on the
eight MLP checkpoints, which are small enough that this is seconds of compute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "compression_mlp"), str(ROOT / "compression_cnn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import linalg as LA                                    # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MODELS = ["mnist_mlp2_bn", "mnist_mlp3_bn", "fashionmnist_mlp2_bn",
          "fashionmnist_mlp3_bn", "cifar10_mlp3_quad_bn", "cifar10_mlp3_oct_bn",
          "cifar100_mlp3_quad_bn", "cifar100_mlp3_oct_bn"]


def load_ref():
    """The MLP driver, imported for its checkpoint loading and BN folding, so
    this analysis sees exactly the network the original sweep saw."""
    import importlib.util
    p = ROOT / "compression_mlp" / "run_autonomous_compression.py"
    spec = importlib.util.spec_from_file_location("mlp_ref", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mlp_ref"] = m
    spec.loader.exec_module(m)
    return m


def build_bench(ref, name: str, dev: str, calib: int = 8192, seed: int = 0):
    """Rebuild exactly what the original sweep built: BN folded into the linear
    stack, and the same unlabelled calibration draw. Nothing is trained."""
    ckpt = ROOT / "trainer" / "checkpoint" / f"{name}.safetensors"
    _, model_part = ref._split_ckpt_stem(ckpt.stem)
    disc = ref.load_discovered(ckpt, model_part, dev)
    train_loader, _ = ref.get_dataloaders(disc.dataset, batch_size=256,
                                          num_workers=2, augment=False)
    Ws, bs = ref.fold_bn(disc.model)
    Ws = [w.to(dev) for w in Ws]
    bs = [b.to(dev) for b in bs]

    torch.manual_seed(seed)
    xs, got = [], 0
    for xb, _ in train_loader:
        xs.append(xb.reshape(xb.shape[0], -1))
        got += xb.shape[0]
        if got >= calib:
            break
    X = torch.cat(xs, 0)[:calib].to(dev)
    return ref.Bench(Ws, bs, X).prepare()


@torch.no_grad()
def residuals(U, Y, keep, W2, b2, ridge=1e-4):
    """`R_fit` and `R_keep` for one index set, both in float64."""
    U64, Y64 = U.double(), Y.double()
    cols = torch.as_tensor(sorted(keep), device=U.device, dtype=torch.long)
    Us = U64[:, cols]
    Phi = torch.cat([Us, torch.ones(Us.shape[0], 1, dtype=Us.dtype,
                                    device=Us.device)], 1)
    G = Phi.t() @ Phi
    lam = ridge * (G.diagonal()[:-1].mean() + 1e-30)
    reg = torch.zeros(G.shape[0], dtype=G.dtype, device=G.device)
    reg[:-1] = lam
    Th = torch.linalg.solve(G + torch.diag(reg), Phi.t() @ Y64)
    r_fit = float((Y64 - Phi @ Th).norm())

    # what the layer's own weights give on the same survivors
    keep_pred = Us @ W2.double()[:, cols].t() + b2.double()
    r_keep = float((Y64 - keep_pred).norm())
    return r_fit, r_keep


def main():
    ref = load_ref()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for name in MODELS:
        try:
            bench = build_bench(ref, name, dev)
        except Exception as exc:
            print(f"  {name}: cannot build bench ({type(exc).__name__}: {exc})")
            continue
        A_list = bench.A_orig
        for l in range(len(A_list)):
            U = A_list[l]
            N = U.shape[1]
            Y = bench.Y_orig[l]
            # the consumer of layer l is stack entry l+1
            W2, b2 = bench.Ws[l + 1], bench.bs[l + 1]

            # effective rank of U, for the x-axis the theory is stated in
            s = torch.linalg.svdvals(U.double())
            energy = (s ** 2).cumsum(0) / (s ** 2).sum()
            d_eff = int((energy < 0.99).sum()) + 1

            G, P, us, ys, n = LA.gram_cross(U, Y)
            for frac in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
                M = max(4, int(round(frac * N)))
                if M >= N:
                    continue
                s_omp = LA.group_select(G, P, 1, M)
                s_span = LA.group_select(G, None, 1, M)
                f_omp, k_omp = residuals(U, Y, s_omp, W2, b2)
                f_span, _ = residuals(U, Y, s_span, W2, b2)
                tail = float(1.0 - energy[min(M, len(energy)) - 1])
                rows.append({
                    "model": name, "layer": l, "N": N, "M": M,
                    "M_over_deff": M / d_eff, "d_eff": d_eff,
                    "tail_energy_beyond_M": tail,
                    "R_fit_omp": f_omp, "R_fit_span": f_span,
                    "R_keep_omp": k_omp, "normY": float(Y.double().norm()),
                    # the two terms, normalised by ||Y|| so models are comparable
                    "correction_term": (k_omp - f_omp) / float(Y.double().norm()),
                    "selection_term": (f_span - f_omp) / float(Y.double().norm()),
                })
            print(f"  {name} layer {l}: N={N}, d_eff(99%)={d_eff}, "
                  f"{len([r for r in rows if r['model'] == name and r['layer'] == l])} budgets")
        del bench
        if dev == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("no rows produced")
        return

    # ---- the two predictions ------------------------------------------------
    import math

    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

    x = [math.log(r["M_over_deff"]) for r in rows]
    print()
    print("=" * 74)
    print("Both terms against the budget, in units of the effective rank")
    print("=" * 74)
    print(f"{'kept fraction M/N':18s} {'correction':>12s} {'selection':>12s} "
          f"{'ratio':>8s} {'n':>5s}")
    summary = {}
    for lo, hi in [(0.0, 0.075), (0.075, 0.15), (0.15, 0.275), (0.275, 0.425),
                   (0.425, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        sel = [r for r in rows if lo <= r["M"] / r["N"] < hi]
        if not sel:
            continue
        c = sum(r["correction_term"] for r in sel) / len(sel)
        s_ = sum(r["selection_term"] for r in sel) / len(sel)
        ratio = c / s_ if s_ > 1e-12 else float("inf")
        lab = f"[{lo:5.3f},{hi:5.3f})"
        print(f"{lab:18s} {c:12.4f} {s_:12.4f} {ratio:8.1f} {len(sel):5d}")
        summary[lab] = {"correction": round(c, 5), "selection": round(s_, 5),
                        "ratio": round(ratio, 2), "n": len(sel)}
    print()
    print(f"{'M/d_eff bin':18s} {'correction':>12s} {'selection':>12s} "
          f"{'ratio':>8s} {'n':>5s}")
    summary_deff = {}
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]:
        sel = [r for r in rows if lo <= r["M_over_deff"] < hi]
        if not sel:
            continue
        c = sum(r["correction_term"] for r in sel) / len(sel)
        s_ = sum(r["selection_term"] for r in sel) / len(sel)
        ratio = c / s_ if s_ > 1e-12 else float("inf")
        lab = f"[{lo:5.2f},{hi:6.2f})" if hi < 1e9 else f"[{lo:5.2f},   inf)"
        print(f"{lab:18s} {c:12.4f} {s_:12.4f} {ratio:8.1f} {len(sel):5d}")
        summary_deff[lab.strip()] = {"correction": round(c, 5),
                                     "selection": round(s_, 5),
                                     "ratio": round(ratio, 2), "n": len(sel)}
    summary = {"by_kept_fraction": summary, "by_M_over_deff": summary_deff}

    print()
    print(f"corr(log M/d_eff, correction term) = "
          f"{corr(x, [r['correction_term'] for r in rows]):+.4f}")
    print(f"corr(log M/d_eff, selection  term) = "
          f"{corr(x, [r['selection_term'] for r in rows]):+.4f}")
    t = [math.log(max(r["tail_energy_beyond_M"], 1e-12)) for r in rows]
    print(f"corr(log tail energy, selection term) = "
          f"{corr(t, [r['selection_term'] for r in rows]):+.4f}")
    print(f"corr(log tail energy, correction term) = "
          f"{corr(t, [r['correction_term'] for r in rows]):+.4f}")

    res = {"rows": rows, "by_bin": summary,
           "corr_logM_correction": round(corr(x, [r["correction_term"] for r in rows]), 4),
           "corr_logM_selection": round(corr(x, [r["selection_term"] for r in rows]), 4),
           "corr_tail_selection": round(corr(t, [r["selection_term"] for r in rows]), 4),
           "corr_tail_correction": round(corr(t, [r["correction_term"] for r in rows]), 4)}
    (OUT / "a7_mechanism.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a7_mechanism.json'}  ({len(rows)} unit-budget cells)")


if __name__ == "__main__":
    main()
