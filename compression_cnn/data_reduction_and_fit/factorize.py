"""Reducing what each layer *reads*: rank-`r` factorisation of its patch space.

The MLP study reduced the input by factoring the first layer as `W_1 ~ A P` with
`P` an `r x D` projection, `r = round(multiplier * GRIDE d_input)`. A conv layer
never reads the whole image, it reads a `C_in*k*k` patch, so the object with a
reducible dimension is the **patch space** -- and every conv layer has one.

All five strategies below produce exactly `r` intermediate channels and are
closed form. They differ only in *which* `r`-dimensional subspace of patch space
they keep, and that is the whole experiment:

    F5_random        an arbitrary subspace                    (floor)
    F4_weight_svd    the subspace the weights are largest in  (ignores the data)
    F3_patch_pca     the subspace the patches vary most in    (ignores the layer)
    F1_rrr           the subspace that best predicts the dense layer's own output
    F2_rrr_jacobian  the same, measured in the metric the rest of the frozen
                     network induces on that output

`F1` is reduced-rank regression -- Eckart-Young in the data metric -- and is
therefore *provably optimal* among rank-`r` factorisations for the objective
`E||V u - B P u||^2`. Reporting `F3`/`F4` without it would be uninterpretable,
since no amount of subspace cleverness can beat a closed-form optimum.

`F2` is the one that can beat it, and only because that objective is a surrogate:
not every coordinate of the layer's output matters equally to the logits. This
is `N1_jacobian` from the MLP input-reduction study, carried over unchanged in
spirit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from common import linalg as LA          # noqa: E402


# ---------------------------------------------------------------------------
# centred second moments and their square roots
# ---------------------------------------------------------------------------

@torch.no_grad()
def _moments(G, P, us, ys, n):
    """Centred `(Sigma, Gamma, u_bar, y_bar)` from the raw sufficient statistics.

    Centring is what lets the bias stay free: only the linear part is
    rank-constrained, so the factorisation never spends a direction representing
    a constant offset.
    """
    G, P, us, ys = LA.f64(G), LA.f64(P), LA.f64(us), LA.f64(ys)
    ub, yb = us / n, ys / n
    return G / n - torch.outer(ub, ub), P / n - torch.outer(ub, yb), ub, yb


@torch.no_grad()
def _sqrt_pair(S: torch.Tensor, rel: float = 1e-8):
    lam, Q = torch.linalg.eigh(0.5 * (S + S.t()))
    lam = lam.clamp(min=float(lam.max().clamp(min=1e-30)) * rel)
    r = torch.sqrt(lam)
    return (Q * r) @ Q.t(), (Q / r) @ Q.t()


@torch.no_grad()
def _pad(U: torch.Tensor, r: int) -> torch.Tensor:
    """Give `U` exactly `r` orthonormal columns.

    A rank-`r` factorisation of a `[out, F]` map cannot have rank above
    `min(F, out)`, so for large `r` the SVD returns fewer directions. Silently
    returning a narrower `P` would make every reported `r` a lie; padding with an
    orthonormal complement is the honest statement that the extra coordinates
    carry no signal.
    """
    k = U.shape[1]
    if k >= r:
        return U[:, :r]
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.randn(U.shape[0], r - k, generator=g).to(U.device, U.dtype)
    R = R - U @ (U.t() @ R)
    Q, _ = torch.linalg.qr(R)
    return torch.cat([U, Q[:, : r - k]], 1)


@torch.no_grad()
def _finish(Pmat, Bmat, ub, yb):
    """`c = y_bar - B P u_bar` -- the intercept the centring left implicit."""
    return Pmat.float(), Bmat.float(), (yb - Bmat @ (Pmat @ ub)).float()


# ---------------------------------------------------------------------------
# the strategies
# ---------------------------------------------------------------------------

@torch.no_grad()
def rrr(ctx, r: int, ridge: float = 1e-4, metric_sqrt=None, metric_inv=None):
    """Reduced-rank regression of the dense target on the compressed patches.

    `Theta = Sigma^-1 Gamma` is the unconstrained fit; the best rank-`r`
    approximation *in the data metric* is `Sigma^-1/2 [Sigma^1/2 Theta]_r`, which
    is Eckart-Young after whitening. With an output metric `L` the same holds
    for `Theta L^T`, so substituting the target turns it back into the ordinary
    problem -- exactly as in the MLP study's `weighted_optimal_linear`.
    """
    G, Pc, us, ys, n = ctx.stats
    S, Gam, ub, yb = _moments(G, Pc, us, ys, n)
    lam = ridge * S.diagonal().mean().clamp(min=1e-30)
    Sr = S + lam * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    Th = torch.linalg.solve(Sr, Gam)                       # [F, out]
    Ssq, Sinv = _sqrt_pair(Sr)

    T = Th @ metric_sqrt.t() if metric_sqrt is not None else Th
    Um, Sm, Vm = torch.linalg.svd(Ssq @ T, full_matrices=False)
    k = min(r, int((Sm > float(Sm.max().clamp(min=1e-30)) * 1e-10).sum()))
    k = max(k, 1)
    Ur = _pad(Um[:, :k], r)
    Pmat = (Sinv @ Ur).t()                                 # [r, F]
    Bt = Vm[:k].t() * Sm[:k]                               # [out, k]
    if r > k:
        Bt = torch.cat([Bt, torch.zeros(Bt.shape[0], r - k, dtype=Bt.dtype,
                                        device=Bt.device)], 1)
    if metric_inv is not None:
        Bt = metric_inv @ Bt
    return _finish(Pmat, Bt, ub, yb)


@torch.no_grad()
def _fit_B(ctx, Pmat, ridge: float = 1e-4):
    """Least-squares mixing matrix for a *given* projection -- the "and fit" half.

    Every strategy gets its `B` this way, so the comparison isolates the choice
    of subspace and never the fitting procedure.
    """
    G, Pc, us, ys, n = ctx.stats
    S, Gam, ub, yb = _moments(G, Pc, us, ys, n)
    Sz = Pmat @ S @ Pmat.t()
    lam = ridge * Sz.diagonal().mean().clamp(min=1e-30)
    Bt = torch.linalg.solve(
        Sz + lam * torch.eye(Sz.shape[0], dtype=Sz.dtype, device=Sz.device),
        Pmat @ Gam).t()
    return _finish(Pmat, Bt, ub, yb)


@torch.no_grad()
def patch_pca(ctx, r: int, ridge: float = 1e-4):
    G, Pc, us, ys, n = ctx.stats
    S, _, _, _ = _moments(G, Pc, us, ys, n)
    lam, Q = torch.linalg.eigh(0.5 * (S + S.t()))
    return _fit_B(ctx, _pad(Q.flip(1), r).t(), ridge)


@torch.no_grad()
def weight_svd(ctx, r: int, ridge: float = 1e-4):
    W = ctx.consumer.weight.detach()
    Wf = LA.f64(W.reshape(W.shape[0], -1))
    _, _, Vh = torch.linalg.svd(Wf, full_matrices=False)
    return _fit_B(ctx, _pad(Vh.t(), r).t(), ridge)


@torch.no_grad()
def random_proj(ctx, r: int, ridge: float = 1e-4):
    G, _, _, _, _ = ctx.stats
    g = torch.Generator(device="cpu").manual_seed(ctx.seed)
    R = torch.randn(G.shape[0], r, generator=g).to(G.device, G.dtype)
    Q, _ = torch.linalg.qr(R)
    return _fit_B(ctx, Q.t(), ridge)


class Factorizer:
    name = "base"
    note = ""
    needs_metric = False

    def apply(self, ctx, r, metric=None):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class F1_RRR(Factorizer):
    name = "F1_rrr"
    note = "reduced-rank regression on the dense target (optimal for the plain objective)"

    def apply(self, ctx, r, metric=None):
        return rrr(ctx, r, ctx.ridge)


class F2_RRRJacobian(Factorizer):
    name = "F2_rrr_jacobian"
    note = "reduced-rank regression in the logit-pullback metric of the frozen tail"
    needs_metric = True

    def apply(self, ctx, r, metric=None):
        if metric is None:
            return rrr(ctx, r, ctx.ridge)
        return rrr(ctx, r, ctx.ridge, metric[0], metric[1])


class F3_PatchPCA(Factorizer):
    name = "F3_patch_pca"
    note = "top-r principal directions of the patch cloud + LS mixing"

    def apply(self, ctx, r, metric=None):
        return patch_pca(ctx, r, ctx.ridge)


class F4_WeightSVD(Factorizer):
    name = "F4_weight_svd"
    note = "top-r right singular vectors of the weight matrix + LS mixing (data-agnostic)"

    def apply(self, ctx, r, metric=None):
        return weight_svd(ctx, r, ctx.ridge)


class F5_Random(Factorizer):
    name = "F5_random"
    note = "random orthonormal subspace + LS mixing (floor)"

    def apply(self, ctx, r, metric=None):
        return random_proj(ctx, r, ctx.ridge)


ALL: list[Factorizer] = [F1_RRR(), F2_RRRJacobian(), F3_PatchPCA(),
                         F4_WeightSVD(), F5_Random()]
BY_NAME = {f.name: f for f in ALL}
