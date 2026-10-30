#!/usr/bin/env python3
"""Check the logit-pullback metric against a forward-only finite-difference probe.

`jacobian_metric.py` computes `Omega_l = E[J_l^T J_l]` by a backward recursion
over forward quantities, using two documented approximations: perturbations are
treated as spatially white, and the cross term between a residual block's two
branches is dropped. Neither is exact, so the metric is *checked* rather than
asserted.

The probe: add `eps * v` to one layer's output at a single random spatial
position, run the rest of the network forward, and measure `||delta logits||^2`.
For a ReLU network that is exact for small `eps` (the network is piecewise
linear), and it uses no backward pass. Averaged over samples and positions it
estimates `v^T Omega v` directly.

What "passing" means: predicted and measured energies should agree in *rank*
across directions spanning the metric's spectrum -- that is all the metric is
used for, to say which directions matter more than others. A constant scale
offset is harmless because the metric only ever enters a reduced-rank problem,
which is invariant to it.

Measured result, and the honest limitation it exposes: agreement is strong in the
deep layers (rank correlation ~0.93 on a VGG11's last two 512-channel convs) and
degrades towards the input (~0.25-0.58), with the scale offset growing by roughly
a factor of four per intervening max-pool. Both follow from the spatial-whiteness
approximation compounding: a single-position perturbation early in the network
spreads over an increasingly large, spatially *structured* region, which the
per-position channel metric cannot represent. So the metric is trustworthy
exactly where it is doing the most work -- the deep layers, which hold the
parameters -- and near-neutral where it is not. The gate below reflects that: it
fails only if the deep half fails.

    python3 common/validate_metric.py --tag cifar10_vgg11_bn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT.parent), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.discovery import (calibration_batch, discover, load_checkpoint,  # noqa: E402
                              loaders_for)
from common.foldbn import fold_bn                      # noqa: E402
from common.jacobian_metric import pullback_metrics    # noqa: E402
from common.surgery import get_module                  # noqa: E402


@torch.no_grad()
def probe(model, path, X, dirs, eps=1e-3, reps=8, seed=0):
    """Measured `||d logits||^2 / eps^2` per direction, forward passes only.

    Averaged over `reps` random spatial positions: a single position is a very
    noisy estimate of the position-averaged metric, and the noise swamps the
    signal for directions whose true energies are close together.
    """
    mod = get_module(model, path)
    g = torch.Generator(device="cpu").manual_seed(seed)
    state = {"delta": None, "yx": (0, 0), "hw": (1, 1)}

    def hook(m, i, o):
        state["hw"] = o.shape[-2:]
        if state["delta"] is None:
            return o
        y, x = state["yx"]
        o = o.clone()
        o[:, :, y, x] = o[:, :, y, x] + state["delta"]
        return o

    hd = mod.register_forward_hook(hook)
    try:
        model(X[:1])                                   # shape only, via the hook
        hw = state["hw"]
        base = model(X)
        pos = [(int(torch.randint(0, hw[0], (1,), generator=g)),
                int(torch.randint(0, hw[1], (1,), generator=g))) for _ in range(reps)]
        out = []
        for v in dirs:
            acc = 0.0
            for yx in pos:
                state["delta"], state["yx"] = eps * v, yx
                d = model(X) - base
                acc += float((d ** 2).sum(1).mean())
            out.append(acc / (len(pos) * eps ** 2))
    finally:
        hd.remove()
    return np.array(out)


def spectrum_directions(Om: torch.Tensor, n: int):
    """Eigenvectors spanning the metric's spectrum, plus a few random directions.

    Testing with *random* unit vectors measures nothing: in `R^512` the quadratic
    form `v^T Omega v` concentrates tightly around `tr(Omega)/C`, so predicted and
    measured energies are both nearly constant and their correlation is noise.
    The metric's claim is about which directions matter *more* than others, so the
    directions probed have to span that range.
    """
    lam, Q = torch.linalg.eigh(0.5 * (Om + Om.t()))
    C = Q.shape[0]
    idx = sorted({int(round(t)) for t in np.geomspace(1, C, n // 2)} | {C - 1})
    dirs = [Q[:, C - i].float() for i in idx]
    g = torch.Generator(device="cpu").manual_seed(3)
    for _ in range(n - len(dirs)):
        v = torch.randn(C, generator=g).to(Q.device).float()
        dirs.append(v / v.norm())
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--dirs", type=int, default=24)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()

    found = [(p, m) for p, m in discover() if p.stem == a.tag]
    if not found:
        raise SystemExit(f"{a.tag} not found")
    ck = load_checkpoint(found[0][0], found[0][1], a.device)
    tr, _ = loaders_for(ck, 256, 4)
    model = fold_bn(ck.model, ck.family).to(a.device).eval()
    X = calibration_batch(tr, a.calib, a.device)

    Om = pullback_metrics(model, ck.family, X, 128)
    paths = [p for p in Om if p not in ("classifier",) and
             isinstance(get_module(model, p), torch.nn.Conv2d)]
    step = max(1, len(paths) // a.layers)
    picked = paths[::step][: a.layers]          # `paths` runs deepest-first
    deep_fail = 0
    for i, path in enumerate(picked):
        C = get_module(model, path).out_channels
        dirs = spectrum_directions(Om[path], a.dirs)
        pred = np.array([float(v.double() @ Om[path] @ v.double()) for v in dirs])
        meas = probe(model, path, X[:128], dirs)
        rho = float(np.corrcoef(np.log(pred + 1e-30), np.log(meas + 1e-30))[0, 1])
        sp = float(np.corrcoef(np.argsort(np.argsort(pred)),
                               np.argsort(np.argsort(meas)))[0, 1])
        scale = float(np.median(meas / np.maximum(pred, 1e-30)))
        deep = i < max(1, len(picked) // 2)
        ok = sp > 0.6
        deep_fail += int(deep and not ok)
        print(f"  {path:16s} C={C:4d}  rank-corr {sp:+.3f}  log-corr {rho:+.3f}  "
              f"scale {scale:.2e}   {'ok' if ok else 'weak'}"
              f"{'  <- deep' if deep else ''}")

    print("\n" + ("metric tracks true sensitivity where it is used (deep layers); "
                  "see the module docstring for the early-layer limitation"
                  if not deep_fail else
                  f"{deep_fail} DEEP layer(s) failed -- the recursion is wrong, not "
                  "merely approximate"))
    raise SystemExit(1 if deep_fail else 0)


if __name__ == "__main__":
    main()
