"""Encoders derived from what stages A-D actually showed.

The diagnostic that motivates all of these: on every model tested, the closed-form
`opt_linear` encoder beat srGW-MDS, and within srGW-MDS the *more faithfully
manifold-preserving* the out-of-sample map was, the *worse* the accuracy
(`relaxed` < `nystrom` < `gower` < `opt_linear`). Preserving input geometry is
not the objective; preserving what the network reads off the input is.

`opt_linear` is provably optimal among linear encoders — but only for the
objective `E||W_1 x - A P x||^2`. That objective is a **surrogate**. Two ways
past it, one per strategy family below:

1. **Fix the metric.** Not all coordinates of `W_1 x` matter equally: the frozen
   remainder of the network amplifies some and annihilates others. Minimising the
   error in the metric pulled back from the logits is strictly better aligned
   with accuracy than the plain Frobenius norm. (`N1`, `N3`.)

2. **Fix the capacity.** A linear encoder is a rank-`r` bottleneck. A non-linear
   feature map has more capacity at the same output width `r`. (`N2`.)

Both keep the golden rule (`r = round(multiplier x GRIDE d_input)`) and both are
closed-form — the frozen network never receives a gradient.
"""

from __future__ import annotations

import numpy as np
import torch

from core.baselines import MatrixEncoder, _data_eig, pad_basis
from core.oos import Encoder


# ---------------------------------------------------------------------------
# the metric the rest of the (frozen) network induces on layer-1 pre-activations
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_pullback_metric(sess, damp: float = 1e-3) -> torch.Tensor:
    """`Omega = E_x[ J^T J ]` with `J = d logits / d (layer-1 pre-activation)`.

    For a ReLU MLP the Jacobian is a product of weight matrices and diagonal
    activation masks, so this is a closed-form backward recursion over forward
    quantities only:

        Omega_L   = I                                   (at the logits)
        Omega_l   = (W_{l+1}^T Omega_{l+1} W_{l+1}) (*) E[m m^T]

    with `(*)` the Hadamard product and `m = 1[pre-activation > 0]`. No autograd,
    no gradient on any weight.

    Note this is the same construction that *failed* in the hidden-layer study.
    It failed there because each layer's least-squares refit already corrected
    downstream error, so the metric double-counted. Here the hidden layers are
    **frozen** — nothing downstream can compensate — so the metric is the only
    thing that can express "this direction does not matter", and it should pay.
    """
    model, X = sess.cm.model, sess.Xtr[sess.fit_idx]
    Ws = [l.weight.double() for l in model.layers] + [model.classifier.weight.double()]
    bs = [l.bias.double() for l in model.layers] + [model.classifier.bias.double()]

    pre, h = [], X.double()
    for W, b in zip(Ws[:-1], bs[:-1]):
        z = h @ W.t() + b
        pre.append(z)
        h = torch.relu(z)

    L = len(pre)
    Om = torch.eye(Ws[-1].shape[0], dtype=torch.float64, device=X.device)
    for l in range(L - 1, -1, -1):
        Wn = Ws[l + 1]                                   # consumes layer l
        Om = Wn.t() @ Om @ Wn                            # [n_l, n_l]
        if l > 0 or True:
            m = (pre[l] > 0).double()
            Om = Om * (m.t() @ m / m.shape[0])
        if l == 0:
            break
    Om = 0.5 * (Om + Om.t())
    lam, Q = torch.linalg.eigh(Om)
    lam = torch.clamp(lam, min=float(lam.max().clamp(min=1e-30)) * damp)
    return (Q * lam) @ Q.t()


@torch.no_grad()
def _metric_sqrt(Om: torch.Tensor):
    lam, Q = torch.linalg.eigh(0.5 * (Om + Om.t()))
    lam = torch.clamp(lam, min=float(lam.max().clamp(min=1e-30)) * 1e-8)
    return (Q * torch.sqrt(lam)) @ Q.t()


@torch.no_grad()
def weighted_optimal_linear(X: torch.Tensor, W1: torch.Tensor, r: int,
                            Lsqrt: torch.Tensor | None) -> MatrixEncoder:
    """Reduced-rank regression in an arbitrary output metric.

    Minimises `E_x || L (W_1 x - A P x) ||^2`. Substituting `W_1 <- L W_1` turns
    it back into the ordinary problem, so the same Eckart-Young-in-the-data-metric
    solution applies: with `X = U S V^T`, the optimum is the top-`r` left singular
    subspace of `C = S V^T (L W_1)^T`, giving `P = (V S^-1 U_c[:, :r])^T`.
    `L = I` recovers `baselines.optimal_linear_encoder` exactly.
    """
    V, S = _data_eig(X)
    Sinv = 1.0 / S.clamp(min=float(S.max()) * 1e-7)
    Wt = W1.double().t()                                  # [D, M1]
    if Lsqrt is not None:
        Wt = Wt @ Lsqrt.t()
    C = S.unsqueeze(1) * (V.t() @ Wt)
    Uc, _, _ = torch.linalg.svd(C, full_matrices=False)
    P = (V * Sinv.unsqueeze(0)) @ pad_basis(Uc, r)
    return MatrixEncoder(P.t().float(), name="weighted_opt_linear")


# ---------------------------------------------------------------------------
# N1 — the headline: optimal linear encoder in the logit-pullback metric
# ---------------------------------------------------------------------------

def n1_jacobian_linear(sess, r: int):
    if not hasattr(sess, "_Lsqrt"):
        sess._Lsqrt = _metric_sqrt(logit_pullback_metric(sess))
    enc = weighted_optimal_linear(sess.Xtr[sess.fit_idx], sess.cm.W1, r, sess._Lsqrt)
    return enc, {"strategy": "jacobian_metric", "oos": "linear", "K": 0}


# ---------------------------------------------------------------------------
# N3 — cheap ablation: weight units by how often they actually fire
# ---------------------------------------------------------------------------

def n3_actfreq_linear(sess, r: int):
    """Diagonal metric only: a unit that is almost always dead contributes almost
    nothing downstream, so reconstructing it accurately is wasted budget. This is
    the rank-1 shadow of N1 and isolates how much of N1's gain is just that."""
    with torch.no_grad():
        X = sess.Xtr[sess.fit_idx]
        z = X @ sess.cm.W1.t() + sess.cm.b1
        freq = (z > 0).double().mean(0).clamp(min=1e-3)
        Ls = torch.diag(torch.sqrt(freq))
    enc = weighted_optimal_linear(X, sess.cm.W1, r, Ls)
    return enc, {"strategy": "act_frequency", "oos": "linear", "K": 0}


# ---------------------------------------------------------------------------
# N2 — non-linear capacity at the same output width
# ---------------------------------------------------------------------------

class RFFEncoder(Encoder):
    """`z = top-r linear readout of random Fourier features of x`.

    A linear encoder is a rank-`r` bottleneck on `W_1 x`; a non-linear feature map
    is not. Random Fourier features approximate an RBF kernel, and the readout is
    chosen by the same weighted reduced-rank regression, so this is `N1` with the
    linear dictionary `x` swapped for `[cos(Wx+b), sin(Wx+b)]`. Still closed-form.

    It does not fold into the first layer, so its cost is reported as
    `d_feat x D` — the honest price of the extra capacity.
    """
    name = "rff"
    is_linear = False

    def __init__(self, Omega, bias, P, q, D):
        self.Omega, self.bias, self.P, self.q, self.D = Omega, bias, P, q, D

    @torch.no_grad()
    def _feat(self, X):
        proj = X.float() @ self.Omega.t() + self.bias
        return torch.cat([torch.cos(proj), torch.sin(proj)], 1)

    @torch.no_grad()
    def encode(self, X):
        return self._feat(X) @ self.P.t() + self.q

    def cost(self, D):
        d_feat = self.Omega.shape[0]
        r = self.P.shape[0]
        return {"params": d_feat * D + 2 * d_feat * r,
                "flops_per_sample": d_feat * D + 2 * d_feat * r,
                "folds_into_first_layer": False}


def n2_rff(sess, r: int, d_feat: int = 1024, gamma_scale: float = 1.0):
    with torch.no_grad():
        X = sess.Xtr[sess.fit_idx]
        sub = X[torch.randperm(X.shape[0], device=X.device)[:2000]].float()
        med = torch.cdist(sub, sub).median().clamp(min=1e-6)
        gamma = gamma_scale / (med ** 2)
        g = torch.Generator(device="cpu").manual_seed(sess.args.seed)
        Om = (torch.randn(d_feat, X.shape[1], generator=g).to(X.device)
              * torch.sqrt(2 * gamma).cpu().item())
        bias = (torch.rand(d_feat, generator=g).to(X.device) * 2 * np.pi)
        enc0 = RFFEncoder(Om, bias, torch.eye(2 * d_feat, device=X.device),
                          torch.zeros(2 * d_feat, device=X.device), X.shape[1])
        F = enc0._feat(X)                                  # [n, 2*d_feat]
        if not hasattr(sess, "_Lsqrt"):
            sess._Lsqrt = _metric_sqrt(logit_pullback_metric(sess))
        # reduced-rank regression from features F to the (metric-weighted) target
        Y = (X.double() @ sess.cm.W1.double().t()) @ sess._Lsqrt.t()
        Fd = F.double()
        G = Fd.t() @ Fd
        G += 1e-4 * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        Beta = torch.linalg.solve(G, Fd.t() @ Y)           # [2d, M1] full-rank map
        Uc, Sc, _ = torch.linalg.svd(torch.linalg.cholesky(G).t() @ Beta,
                                     full_matrices=False)
        Rinv = torch.linalg.solve_triangular(
            torch.linalg.cholesky(G).t(), torch.eye(G.shape[0], dtype=G.dtype, device=G.device),
            upper=True)
        P = (Rinv @ pad_basis(Uc, r)).t().float()          # [r, 2d]
    return RFFEncoder(Om, bias, P, torch.zeros(r, device=X.device), X.shape[1]), \
        {"strategy": f"rff{d_feat}", "oos": "kernel", "K": 0}


STRATEGIES = [
    ("N1_jacobian_linear", n1_jacobian_linear),
    ("N3_actfreq_linear", n3_actfreq_linear),
    ("N2_rff1024", n2_rff),
]
