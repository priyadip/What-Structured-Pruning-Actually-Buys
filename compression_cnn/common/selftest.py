#!/usr/bin/env python3
"""Correctness checks for the CNN port. Run this before trusting any number.

Each check is designed to fail loudly if a *silent* mistake is present -- the
kind that still produces plausible accuracies. In the MLP study three of the four
real bugs were of exactly this kind, so the rule adopted there applies here too:
anything that gets quoted has to be reproducible by a second code path.

    1. BN folding is exact                 (folded net == original net, logits)
    2. unfold/target alignment is exact    (LS with ALL channels recovers the
                                            consumer's own weights)
    3. group selection generalises the MLP (gs=1 reproduces compression_mlp's omp_select)
    4. the walk is faithful                (keep-everything compression is a no-op)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT.parent), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import linalg as LA           # noqa: E402
from common import patches as PT          # noqa: E402
from common import surgery as SG          # noqa: E402
from common.discovery import (            # noqa: E402
    calibration_batch, discover, evaluate, load_checkpoint, loaders_for)
from common.foldbn import fold_bn         # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_fails = 0


def check(name: str, ok: bool, detail: str = ""):
    global _fails
    print(f"  [{OK if ok else BAD}] {name}{('  ' + detail) if detail else ''}")
    _fails += 0 if ok else 1


# ---------------------------------------------------------------------------
def t_group_select_matches_mlp(device):
    """`gs=1` must reproduce the MLP study's scalar OMP exactly."""
    sys.path.insert(0, str(ROOT / "compression_mlp"))
    from run_autonomous_compression import omp_select, pivoted_cholesky_select

    g = torch.Generator().manual_seed(7)
    A = torch.randn(400, 60, generator=g).to(device)
    Y = torch.randn(400, 12, generator=g).to(device)
    G, P = A.t().double() @ A.double(), A.t().double() @ Y.double()
    for m in (5, 17, 40):
        check(f"group OMP == scalar OMP (m={m})",
              LA.group_select(G, P, 1, m) == omp_select(G, P, m))
        check(f"group span == pivoted Cholesky (m={m})",
              LA.group_select(G, None, 1, m) == pivoted_cholesky_select(G, m))


def t_ls_recovers_weights(device):
    """With every channel kept, the closed-form refit must return the consumer's
    own weights -- proof that `design_matrix` and `target_matrix` are aligned."""
    torch.manual_seed(0)
    for consumer, A in ((nn.Conv2d(6, 5, 3, padding=1, bias=True).to(device),
                         torch.randn(8, 6, 9, 9, device=device)),
                        (nn.Conv2d(6, 5, 3, stride=2, padding=1, bias=True).to(device),
                         torch.randn(8, 6, 9, 9, device=device)),
                        # a Linear consumer contributes ONE row per image, so the
                        # calibration set has to exceed C*H*W or the system is
                        # rank-deficient -- the same constraint the driver asserts.
                        (nn.Linear(6 * 4 * 4, 5).to(device),
                         torch.randn(600, 6, 4, 4, device=device))):
        per = PT.n_positions(consumer, A)
        gs = PT.group_size(consumer, A)
        Y = (consumer(A) if isinstance(consumer, nn.Conv2d)
             else consumer(A.reshape(A.shape[0], -1)))
        rows = torch.arange(A.shape[0] * per, device=device)
        U = PT.design_matrix(A, consumer, rows, per)
        Yr = PT.target_matrix(Y, rows, per)
        G, P, us, ys, n = LA.gram_cross(U, Yr)
        cols = LA.group_cols(list(range(A.shape[1])), gs, device)
        V, c = LA.ls_consuming(G, P, us, ys, n, cols, ridge=1e-12)
        W0 = (consumer.weight.reshape(consumer.weight.shape[0], -1)
              if isinstance(consumer, nn.Conv2d) else consumer.weight)
        eW = float((V - W0).abs().max())
        eb = float((c - consumer.bias).abs().max())
        check(f"LS recovers {type(consumer).__name__} weights "
              f"(stride={getattr(consumer,'stride','-')})",
              eW < 1e-4 and eb < 1e-4, f"maxerr W={eW:.2e} b={eb:.2e}")


def t_fold_and_walk(device, tag_filter, n_calib):
    found = discover()
    if tag_filter:
        found = [(p, m) for p, m in found if tag_filter in p.stem]
    if not found:
        print("  (no checkpoints matched; skipping model-level checks)")
        return
    for path, mname in found:
        ck = load_checkpoint(path, mname, device)
        tr, te = loaders_for(ck, 256, 4)
        folded = fold_bn(ck.model, ck.family).to(device)
        probe = calibration_batch(tr, 8, device)
        need, info = SG.plan_calibration(folded, ck.family, probe, factor=4.0)
        n_use = max(n_calib, need)
        tight = max(info, key=lambda i: i["need"])
        print(f"  {ck.tag}: {len(info)} units, tightest {tight['unit']} "
              f"({tight['unknowns']} unknowns, {tight['per_image']} rows/image) "
              f"-> needs {need} calib images; using {n_use}")
        X = calibration_batch(tr, n_use, device)

        with torch.no_grad():
            d = SG._chunked(ck.model, X[:128], 64)
            f = SG._chunked(folded, X[:128], 64)
        err = float((d - f).abs().max() / d.abs().max())
        check(f"{ck.tag}: BN folding exact", err < 5e-3, f"rel logit err={err:.2e}")

        comp = SG.clone_eval(folded)
        # Use the ridge the drivers actually use. At ridge=1e-8 this check also
        # measures the conditioning of `gram_cross`'s float32 accumulation, which
        # on the deepest units (cond(G) ~ 7e11) swamps the alignment signal it is
        # meant to test: the same fit with a float64 Gram drops the residual from
        # 2.8e-3 to 6e-9. At 1e-4 the two agree to within a factor of 1.4, so any
        # residual left here is a bookkeeping error and not float noise.
        walker = SG.Walker(folded, comp, ck.family, X, patch_budget=80000,
                           ridge=1e-4, seed=0)
        worst = []

        def decide(ctx):
            ctx.require_overdetermined(2.0)
            groups = list(range(ctx.C))
            V, c = ctx.ls(groups)
            # Recovering the consumer's literal weights is ill-posed wherever the
            # ReLU activations are rank-deficient (dead channels give exactly-zero
            # columns), and that is common in deep layers. The refit is only ever
            # asked to reproduce the consumer's *output*, so that is what is
            # checked: relative residual of the fit against the dense target.
            Yr = PT.target_matrix(ctx.Y, ctx.rows, ctx.per_image)
            res = (ctx.U @ V.t() + c - Yr).pow(2).sum() / Yr.pow(2).sum().clamp(min=1e-30)
            worst.append((ctx.name, float(res), ctx.rows.numel(), ctx.C * ctx.gs))
            SG.set_module(comp, _path_of(comp, ctx.consumer),
                          SG.new_consumer(ctx.consumer, V, c, ctx.C))
            return ctx.A

        walker.walk(decide)
        with torch.no_grad():
            a = SG._chunked(folded, X[:256], 64)
            b = SG._chunked(comp, X[:256], 64)
        rel = float((a - b).abs().max() / a.abs().max())
        bad = [w for w in worst if w[1] > 1e-5]
        check(f"{ck.tag}: keep-all walk is a no-op ({len(worst)} units)",
              rel < 5e-2 and not bad, f"rel logit err={rel:.2e}, "
              f"max unit residual={max(w[1] for w in worst):.2e}" +
              (f"  offenders: {[(w[0], f'{w[1]:.1e}') for w in bad[:3]]}" if bad else ""))

        acc_o = evaluate(ck.model, te, device, max_batches=8)
        acc_f = evaluate(folded, te, device, max_batches=8)
        check(f"{ck.tag}: folded accuracy matches", abs(acc_o - acc_f) <= 0.15,
              f"{acc_o:.2f} vs {acc_f:.2f}")
        del comp, folded, X
        torch.cuda.empty_cache()


def _path_of(root: nn.Module, mod: nn.Module) -> str:
    for name, m in root.named_modules():
        if m is mod:
            return name
    raise KeyError("module not found in tree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None, help="restrict model checks to this substring")
    ap.add_argument("--calib", type=int, default=256)
    a = ap.parse_args()

    print("\n1. group selection vs the MLP implementation")
    t_group_select_matches_mlp(a.device)
    print("\n2. unfold / target alignment")
    t_ls_recovers_weights(a.device)
    print("\n3. BN folding + lock-step walk")
    t_fold_and_walk(a.device, a.tag, a.calib)

    print(f"\n{'all checks passed' if not _fails else f'{_fails} CHECK(S) FAILED'}")
    raise SystemExit(1 if _fails else 0)


if __name__ == "__main__":
    main()
