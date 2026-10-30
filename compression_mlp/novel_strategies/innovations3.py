"""Third wave: fix the *norm*, not the selection rule.

Rounds 1-2 showed the selection rule is not the bottleneck -- an augmented
dictionary and exact forward-backward selection both tie plain OMP+LS. So the
error is not in *which* neurons are kept, it is in *what we are asking them to
reproduce*.

Every layer-wise compression method in the literature (ThiNet, GPTQ, SparseGPT,
Intra-Fusion, DOTResize, and `N02_omp_ls`) minimises a plain Frobenius residual
on the layer's own output:

    min || Yhat_l - Y_l ||_F .

That treats every direction of Y_l as equally valuable. It is not. The rest of
the network amplifies some directions of Y_l and annihilates others; an error
lying in a direction the classifier ignores costs nothing, and an error of the
same size along a discriminative direction can flip the prediction. The correct
norm is the one pulled back from the logits,

    min E_x || (Yhat_l - Y_l) J_l(x)^T ||^2 ,   J_l = d logits / d Y_l ,

whose sample-averaged form is the Gauss-Newton (Fisher) metric

    M_l = E_x[ J_l^T J_l ] .

For a ReLU MLP that metric has a closed-form backward recursion built purely
from forward quantities -- weight matrices and activation masks -- so it costs
no gradients (see `Bench._jacobian_metrics`). With S = M^{1/2} the weighted
problem becomes an ordinary least-squares problem on the rotated target Y S,
so both the OMP selection scores and the closed-form solve inherit it for free;
the solution is mapped back by V = S^{-1} V~.

`eps` floors the metric's eigenvalues at `eps * lambda_max`, interpolating
between the pure Fisher metric (eps -> 0) and plain least squares (eps -> 1);
the floor is needed because M is genuinely rank-deficient -- which is the point:
those are the directions worth spending no budget on.
"""

from __future__ import annotations

import torch

from run_autonomous_compression import Strategy, _f64, ls_consuming, omp_select


def metric_sqrt(M: torch.Tensor | None, eps: float, n: int, device, dtype
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """(S, S^-1) with S = M^{1/2}, eigenvalues floored at eps*lambda_max and the
    spectrum normalised to unit mean so the ridge keeps its meaning."""
    if M is None:
        I = torch.eye(n, dtype=dtype, device=device)
        return I, I
    Md = M.to(torch.float64)
    Md = 0.5 * (Md + Md.t())
    lam, Q = torch.linalg.eigh(Md)
    lam = torch.clamp(lam, min=float(eps) * float(lam.max().clamp(min=1e-30)))
    lam = lam / lam.mean()
    s = torch.sqrt(lam)
    S = (Q * s) @ Q.t()
    Sinv = (Q * (1.0 / s)) @ Q.t()
    return S.to(dtype), Sinv.to(dtype)


class N09_FisherOMP(Strategy):
    """OMP + least squares in the output-Jacobian metric. The headline of round 3."""
    name = "N09_fisher_omp"
    note = "OMP + LS in the logit-pullback (Gauss-Newton/Fisher) metric, not Frobenius"

    def __init__(self, eps: float = 1e-3, tag: str = ""):
        self.eps = eps
        if tag:
            self.name = f"N09_fisher_omp_{tag}"
            self.note += f" (eps={eps})"

    def apply(self, ctx, m):
        Y = ctx.Y
        S, Sinv = metric_sqrt(ctx.metric, self.eps, Y.shape[1], Y.device, Y.dtype)
        Yt = Y @ S                                  # rotate into the metric
        keep = omp_select(ctx.gram(), _f64(ctx.A).t() @ _f64(Yt), m)
        Vt, ct = ls_consuming(ctx.A[:, keep], Yt, ctx.ridge)
        V = Sinv @ Vt                               # rotate the solution back
        c = Sinv @ ct
        return ctx.W[keep], ctx.b[keep], V, c


class N10_FisherSpanLS(Strategy):
    """Ablation: keep the Fisher metric for the *fit* but select by activation
    span only. Isolates how much of N09's gain comes from the metric-aware
    selection versus the metric-aware solve."""
    name = "N10_fisher_span_ls"
    note = "span selection + Fisher-metric LS (ablation of N09's selection term)"

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def apply(self, ctx, m):
        from run_autonomous_compression import pivoted_cholesky_select
        Y = ctx.Y
        S, Sinv = metric_sqrt(ctx.metric, self.eps, Y.shape[1], Y.device, Y.dtype)
        keep = pivoted_cholesky_select(ctx.gram(), m)
        Vt, ct = ls_consuming(ctx.A[:, keep], Y @ S, ctx.ridge)
        return ctx.W[keep], ctx.b[keep], Sinv @ Vt, Sinv @ ct


STRATEGIES = [
    N09_FisherOMP(eps=1e-2, tag="e2"),
    N09_FisherOMP(eps=1e-3),
    N09_FisherOMP(eps=1e-4, tag="e4"),
    N10_FisherSpanLS(eps=1e-3),
]
