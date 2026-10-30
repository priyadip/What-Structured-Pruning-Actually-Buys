#!/usr/bin/env python3
"""E3 -- a faithful ThiNet control.

The paper differentiates its selection rule from ThiNet (Luo, Wu & Lin, ICCV
2017) by argument plus one proxy (`S3_span_ls`, span selection + our refit).
That is the weakest link in the novelty claim, so this implements ThiNet as
described and runs it in the same cells.

**What ThiNet actually does**, and where each piece differs from ours:

  1. *Greedy subtractive selection.* Start from all channels and repeatedly
     remove the one whose removal least increases the reconstruction error of
     the next layer's output. We use greedy *additive* selection (group OMP)
     against `U^T Y`.
  2. *Target = the next layer's own output.* ThiNet reconstructs what the
     consumer currently produces. We target the **dense** network's
     pre-activation, so each solve corrects error inherited from upstream
     instead of reproducing an already-degraded value.
  3. *Scalar per-channel rescale.* After selection ThiNet fits one scale per
     surviving channel by least squares. We re-solve the whole consumer with a
     free intercept.

Three variants are run so the three differences can be attributed separately:

  T1_thinet          all three ThiNet choices (the faithful control)
  T2_thinet_lsfull   ThiNet selection, but our full consumer refit
                     -> isolates (3), the rescale-vs-refit difference
  T3_thinet_densetgt ThiNet selection scored against the *dense* target
                     -> isolates (2), the target difference

    python3 e3_thinet.py [--models ...] [--mults ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import torch
import torch.nn as nn

import harness as H
from common import linalg as LA          # noqa: E402
import strategies as ST                  # noqa: E402

OUT = H.RESULTS / "e3_thinet.csv"
FIELDS = ["model", "dataset", "multiplier", "strategy", "acc", "dense_acc", "drop",
          "params", "macs", "param_cut", "widths", "seconds"]


# ---------------------------------------------------------------------------
# ThiNet's selection: greedy subtractive on the consumer's reconstruction
# ---------------------------------------------------------------------------

def batched_backward_select(G, P, us, ys, n, gs: int, m: int,
                            ridge: float = 1e-8) -> list[int]:
    """Greedy backward elimination of channel groups, starting from all channels.

    Algorithmically identical to `common.linalg.forward_backward_select` with the
    candidate set set to everything -- same removal cost
    `dSSE_g = tr(Theta_g^T (C_gg)^-1 Theta_g)`, same block downdate of both `C`
    and `Theta`, so no system is refactorised. The only change is that the scan
    over candidate groups is **batched**: the production version issues one
    `linalg.solve` per surviving group per step, which for a 512-channel unit is
    ~235k tiny GPU kernels and is entirely launch-bound. Extracting all diagonal
    blocks as `[k, gs, gs]` and solving once per step makes the same computation
    ~50x faster.

    `_assert_matches_production` checks the two agree index-for-index; run this
    file with `--selftest` to execute it.
    """
    C_all = G.shape[0] // gs
    if m >= C_all:
        return list(range(C_all))
    dev = G.device
    cand = list(range(C_all))
    cols = LA.group_cols(cand, gs, dev)
    Gs, Ps, uss = LA.f64(G)[cols][:, cols], LA.f64(P)[cols], LA.f64(us)[cols]
    F = Gs.shape[0]

    Gaug = torch.zeros(F + 1, F + 1, dtype=Gs.dtype, device=dev)
    Gaug[:F, :F], Gaug[:F, F], Gaug[F, :F], Gaug[F, F] = Gs, uss, uss, float(n)
    lam = ridge * (Gs.diagonal().mean() + 1e-12)
    reg = torch.zeros(F + 1, dtype=Gs.dtype, device=dev)
    reg[:F] = lam
    Cinv = torch.linalg.inv(Gaug + torch.diag(reg))
    Theta = Cinv @ torch.cat([Ps, LA.f64(ys).unsqueeze(0)], 0)

    alive = list(range(len(cand)))
    eps = 1e-30
    while len(alive) > m:
        k = len(alive)
        idx = torch.arange(k * gs, device=dev).view(k, gs)
        # all k diagonal blocks of Cinv at once -> [k, gs, gs]
        Cgg = Cinv[:k * gs, :k * gs].view(k, gs, k, gs)
        Cgg = torch.stack([Cgg[i, :, i, :] for i in range(k)])
        Th = Theta[:k * gs].view(k, gs, -1)                     # [k, gs, out]
        eye = torch.eye(gs, dtype=Cinv.dtype, device=dev).expand(k, gs, gs)
        cost = (Th * torch.linalg.solve(Cgg + eps * eye, Th)).sum((1, 2))
        bj = int(torch.argmin(cost))

        sl = slice(bj * gs, (bj + 1) * gs)
        Cb = Cinv[:, sl]
        Cd = Cinv[sl, sl] + eps * torch.eye(gs, dtype=Cinv.dtype, device=dev)
        Theta = Theta - Cb @ torch.linalg.solve(Cd, Theta[sl])
        Cinv = Cinv - Cb @ torch.linalg.solve(Cd, Cinv[sl])
        keep = torch.ones(Cinv.shape[0], dtype=torch.bool, device=dev)
        keep[sl] = False
        Cinv = Cinv[keep][:, keep]
        Theta = Theta[keep]
        alive.pop(bj)
    return sorted(cand[i] for i in alive)


def thinet_select(G, P, us, ys, n, gs: int, m: int, ridge: float = 1e-8) -> list[int]:
    """ThiNet's selection: greedy subtractive removal from the full channel set,
    each step deleting the channel whose removal least increases the
    reconstruction error of the layer's target."""
    return batched_backward_select(G, P, us, ys, n, gs, m, ridge)


def _assert_matches_production(seed: int = 0):
    """The batched scan must return exactly what the production kernel does."""
    g = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for gs, C, out, m in ((1, 40, 7, 11), (9, 24, 5, 6), (4, 30, 3, 9)):
        A = torch.randn(600, C * gs, generator=g).to(dev)
        Y = torch.randn(600, out, generator=g).to(dev)
        G, P, us, ys, n = LA.gram_cross(A, Y)
        ours = batched_backward_select(G, P, us, ys, n, gs, m, ridge=1e-8)
        theirs = LA.forward_backward_select(G, P, us, ys, n, gs, m,
                                            ridge=1e-8, over=float(C))
        assert ours == theirs, f"gs={gs} C={C} m={m}: {ours} != {theirs}"
        print(f"  [PASS] batched == production  (gs={gs}, C={C}, m={m})")


def _scalar_rescale(ctx, keep):
    """ThiNet's per-channel scalar fit: one scale per surviving input channel,
    applied to the consumer's existing weights. No intercept, no full re-solve.

    Solves `min_s || U_S diag(s (x) 1_gs) v - Y ||` in the least-squares sense
    over the `|S|` scales jointly, which is a small normal-equation system in
    `|S|` unknowns rather than `|S|*gs*C_out`."""
    G, P, us, ys, n = ctx.stats
    cols = ctx.cols(keep)
    V = ctx.consumer.weight.detach()
    Vf = V.reshape(V.shape[0], ctx.C, -1)[:, keep].reshape(V.shape[0], -1)
    Vd = LA.f64(Vf)                                        # [out, |S|*gs]
    Gs = LA.f64(G)[cols][:, cols]                          # [|S|gs, |S|gs]
    Ps = LA.f64(P)[cols]                                   # [|S|gs, out]
    k, gs = len(keep), ctx.gs

    # A[i,j] = sum_out (V_i^T G_ij V_j) with V_i the i-th channel's block
    Vb = Vd.reshape(Vd.shape[0], k, gs)                    # [out, k, gs]
    Gb = Gs.reshape(k, gs, k, gs)
    A = torch.einsum("oip,ipjq,ojq->ij", Vb, Gb, Vb)
    b = torch.einsum("oip,ipo->i", Vb, Ps.reshape(k, gs, -1))
    lam = 1e-8 * (A.diagonal().mean().abs() + 1e-30)
    s = torch.linalg.solve(A + lam * torch.eye(k, dtype=A.dtype, device=A.device), b)
    scaled = (Vb * s.reshape(1, k, 1)).reshape(Vd.shape[0], k * gs).float()
    return scaled, ctx.consumer.bias.detach().clone()


class ThiNet(ST.Strategy):
    """Faithful control: subtractive selection on the *current* consumer output,
    then a per-channel scalar rescale."""
    name = "T1_thinet"
    note = "ThiNet: greedy subtractive selection + per-channel scalar rescale"
    family = "select"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        Pc = self._target(ctx, P)
        keep = thinet_select(G, Pc, us, ys, n, ctx.gs, m)
        W, b = ST._rows(ctx, keep)
        Vf, c = _scalar_rescale(ctx, keep)
        return W, b, Vf, c

    @staticmethod
    def _target(ctx, P):
        """ThiNet reconstructs the consumer's *own current* output. `ctx.Y` is
        the dense target, so we rebuild the cross-product against what the
        consumer actually produces from the compressed input it is being fed."""
        V = ctx.consumer.weight.detach()
        Vf = V.reshape(V.shape[0], -1)
        Ycur = ctx.U @ Vf.t()                     # current pre-activation, no bias
        return LA.f64(ctx.U).t() @ LA.f64(Ycur)


class ThiNetFullRefit(ThiNet):
    """ThiNet selection, our consumer refit. Isolates difference (3)."""
    name = "T2_thinet_lsfull"
    note = "ThiNet selection + our closed-form consumer refit with intercept"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = thinet_select(G, self._target(ctx, P), us, ys, n, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*ST._rows(ctx, keep), V, c)


class ThiNetDenseTarget(ThiNet):
    """ThiNet selection scored against the DENSE target, then its own rescale.
    Isolates difference (2)."""
    name = "T3_thinet_densetgt"
    note = "ThiNet selection against the dense target + scalar rescale"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = thinet_select(G, P, us, ys, n, ctx.gs, m)   # P is already U^T Y_dense
        W, b = ST._rows(ctx, keep)
        Vf, c = _scalar_rescale(ctx, keep)
        return W, b, Vf, c


VARIANTS = [ThiNet(), ThiNetFullRefit(), ThiNetDenseTarget()]
COMPARE = ["S4_omp_ls", "S3_span_ls", "S2_magnitude_ls"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["cifar10_vgg11_bn", "cifar10_resnet18",
                             "fashionmnist_vgg16_bn", "cifar100_vgg11_bn"])
    ap.add_argument("--mults", type=float, nargs="*", default=[2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("checking the batched elimination against the production kernel")
        _assert_matches_production()
        return

    for v in VARIANTS:
        H.register(v)
    H.redirect_csv(H.RESULTS / "e3_driver_log.csv")

    rows = []
    for tag in a.models:
        args = H.base_args(seed=a.seed)
        sess = None
        try:
            sess = H.make_session(tag, args)
            fr = H.frontiers().get(tag)
            print(f"\n=== {tag}  dense {sess.dense_acc:.2f}%"
                  + (f"  (paper frontier x{fr.multiplier})" if fr else ""))
            for mlt in a.mults:
                for name in [v.name for v in VARIANTS] + COMPARE:
                    t0 = time.time()
                    r = H.run_config(sess, name, mlt, stage="e3_thinet")
                    if r is None:
                        continue
                    rows.append({
                        "model": tag, "dataset": sess.ck.dataset, "multiplier": mlt,
                        "strategy": name, "acc": round(r.acc, 4),
                        "dense_acc": round(sess.dense_acc, 4), "drop": round(r.drop, 4),
                        "params": r.params, "macs": r.macs,
                        "param_cut": round(100 * (1 - r.params / sess.dense_params), 3),
                        "widths": json.dumps(r.widths),
                        "seconds": round(time.time() - t0, 1)})
                _flush(rows)
        finally:
            del sess
            H.free()
    report(rows)


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["multiplier"]), {})[r["strategy"]] = r["acc"]

    print("\n" + "=" * 74)
    print("E3 -- faithful ThiNet control, paired within (model, multiplier)")
    print("=" * 74)
    order = ["S4_omp_ls", "T2_thinet_lsfull", "S3_span_ls", "S2_magnitude_ls",
             "T3_thinet_densetgt", "T1_thinet"]
    gaps = {s: [] for s in order}
    for cell in cells.values():
        if "S4_omp_ls" not in cell:
            continue
        for s in order:
            if s in cell:
                gaps[s].append(cell[s] - cell["S4_omp_ls"])
    print(f"{'strategy':22s} {'mean vs ours':>13s} {'se':>6s} {'wins':>7s} {'cells':>6s}")
    for s in order:
        g = gaps[s]
        if not g:
            continue
        st = _stats(g)
        print(f"{s:22s} {st['mean']:+13.2f} {H.fmt(st['se']):>6s} "
              f"{sum(x > 0 for x in g):>3d}/{len(g):<3d} {len(g):>6d}")
    print(f"\nwrote {OUT}")


def _stats(v):
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return {"mean": m, "se": float("nan")}
    var = sum((x - m) ** 2 for x in v) / n
    return {"mean": m, "se": (var / n) ** 0.5}


if __name__ == "__main__":
    main()
