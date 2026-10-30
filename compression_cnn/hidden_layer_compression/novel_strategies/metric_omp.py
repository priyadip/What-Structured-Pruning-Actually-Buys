"""Selecting channels in the metric the rest of the network actually uses.

The open question this tests
----------------------------
`S4_omp_ls` scores a candidate channel by how much it reduces
`||Y - U_S U_S^+ Y||_F^2` -- error in the consumer's pre-activation, measured in
the plain Frobenius norm. That norm treats every coordinate of `Y` as equally
important, and the frozen remainder of the network does not: some directions of
`Y` are amplified all the way to the logits and others are annihilated.

Replacing the norm with the one pulled back from the logits,

        Omega = E_x [ J^T J ],   J = d logits / d Y,

is the same idea as `N1_jacobian` in the MLP input-reduction study, where it beat
the closed-form optimum. In the MLP *hidden-layer* study the analogous strategies
(`N09_fisher_omp`, `N10_fisher_span_ls`) were a **negative** result, and the
explanation offered there was that each layer's least-squares refit already
corrects downstream error, so the metric double-counts.

That explanation makes a prediction, and it is worth checking rather than
assuming: the metric should help where nothing downstream can compensate and be
neutral-to-harmful where something can. Study 2 confirmed the first half (frozen
layers, metric helps). This module tests the second half on CNNs.

Both strategies still return exactly `target_m` channels and are closed form.
The metric is a backward recursion over forward quantities -- no autograd, no
backward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import linalg as LA          # noqa: E402

import strategies as ST                  # noqa: E402


def _sqrt(Om: torch.Tensor, damp: float = 1e-3) -> torch.Tensor:
    lam, Q = torch.linalg.eigh(0.5 * (Om + Om.t()))
    lam = lam.clamp(min=float(lam.max().clamp(min=1e-30)) * damp)
    return (Q * torch.sqrt(lam)) @ Q.t()


class N1_MetricOMP(ST.Strategy):
    """Selection scored in the logit metric; the refit stays in the plain norm.

    Only the *choice* of channels changes. The consuming convolution is still
    solved by ordinary least squares against the dense target, so any difference
    from `S4_omp_ls` is attributable to the selection criterion alone.
    """
    name = "N1_metric_omp"
    note = "group-OMP scored against U^T Y L^T (logit-pullback metric) + plain refit"
    needs_metric = True

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        if ctx.metric is not None:
            L = _sqrt(LA.f64(ctx.metric))
            keep = LA.group_select(G, LA.f64(P) @ L.t(), ctx.gs, m)
        else:
            keep = LA.group_select(G, P, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*ST._rows(ctx, keep), V, c)


class N2_MetricOMPMetricLS(ST.Strategy):
    """Both the selection *and* the refit in the logit metric.

    `min ||L (Y - [U_S,1] Theta)||` is the ordinary problem after substituting
    `Y <- Y L^T`, and the solution maps back as `Theta <- Theta L^-T`. If the
    "the refit already corrects downstream error" explanation is right, this
    should be no better than `N1` and possibly worse, because it also distorts
    the fit and not only the choice.
    """
    name = "N2_metric_omp_ls"
    note = "selection and consumer refit both in the logit-pullback metric"
    needs_metric = True

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        if ctx.metric is None:
            return ST.OMPLS().apply(ctx, m)
        L = _sqrt(LA.f64(ctx.metric))
        Linv = torch.linalg.inv(L)
        Pm, ysm = LA.f64(P) @ L.t(), LA.f64(ys) @ L.t()
        keep = LA.group_select(G, Pm, ctx.gs, m)
        cols = LA.group_cols(keep, ctx.gs, G.device)
        V, c = LA.ls_consuming(G, Pm, us, ysm, n, cols, ctx.ridge)
        V = (LA.f64(V).t() @ Linv.t()).t().float()       # undo the substitution
        c = (Linv @ LA.f64(c)).float()
        return (*ST._rows(ctx, keep), V, c)


STRATEGIES = [N1_MetricOMP(), N2_MetricOMPMetricLS()]
