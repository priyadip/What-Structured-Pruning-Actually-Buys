"""Out-of-sample extension — turning a landmark embedding into an *encoder*.

srGW-MDS gives coordinates for K landmarks. Inference needs a map for arbitrary
`x`, and that map is what actually gets deployed, so its cost matters as much as
its accuracy.

Four extensions, and the distinction between them is not cosmetic:

* `gower`, `nystrom`, `relaxed` are **non-linear** and distance-based: they need
  `K` distances to landmarks, i.e. `K x D` multiply-adds per sample. For K=500,
  D=3072 that is 1.5M flops -- comparable to the first layer they replace. They
  can beat any linear map, but they do not shrink the model, they *reshape* it.
* `linear` distils the non-linear encoder into a matrix `P in R^{r x D}` by ridge
  regression. Then `A (P x) = (A P) x`, so the encoder **folds into the first
  layer** and the result is a genuine rank-`r` factorisation with real parameter
  savings. This is the deployable one -- and it is also why
  `baselines.optimal_linear_encoder` matters: it upper-bounds this whole family.

Every encoder reports `params` and `flops_per_sample` so the summary can quote
honest numbers instead of counting only `A`.
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import cdist_chunked


class Encoder:
    name = "base"
    is_linear = False

    def encode(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def cost(self, D: int) -> dict:
        raise NotImplementedError


class GowerEncoder(Encoder):
    """Classical Gower / Landmark-MDS interpolation — closed form, one shot.

    Solving `||z - z_i||^2 = delta_i^2` for all landmarks and subtracting the
    mean over `i` linearises to `Z_c z = b`, so `z = Z_c^+ b`. Exact when the
    landmark configuration is exactly Euclidean, a good projection otherwise.
    """
    name = "gower"

    def __init__(self, Xl: torch.Tensor, Z: torch.Tensor):
        self.Xl = Xl
        self.mu = Z.mean(0, keepdim=True)
        self.Zc = Z - self.mu                              # centring is assumed below
        self.pinv = torch.linalg.pinv(self.Zc)             # [r, K]
        self.zn2 = (self.Zc ** 2).sum(1)                   # [K]
        self.zn2_mean = self.zn2.mean()

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        d2 = cdist_chunked(X.float(), self.Xl.float()) ** 2            # [n, K]
        b = 0.5 * (self.zn2.unsqueeze(0) - self.zn2_mean
                   - d2 + d2.mean(1, keepdim=True))                    # [n, K]
        return b @ self.pinv.t() + self.mu

    def cost(self, D: int) -> dict:
        K, r = self.Xl.shape[0], self.Zc.shape[1]
        return {"params": K * D + K * r, "flops_per_sample": K * D + K * r,
                "folds_into_first_layer": False}


class NystromEncoder(GowerEncoder):
    """Gower init + SMACOF majorisation steps on the *un-squared* distances.

    Gower matches squared distances, which over-weights far landmarks. The
    SMACOF update `z <- mean_i [ z_i + delta_i (z - z_i)/||z - z_i|| ]` decreases
    the true stress monotonically, so a handful of steps strictly improves the
    fit at negligible cost.
    """
    name = "nystrom"

    def __init__(self, Xl, Z, steps: int = 12):
        super().__init__(Xl, Z)
        self.steps = steps
        self.Zl = Z

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        z = super().encode(X)
        d = cdist_chunked(X.float(), self.Xl.float())                  # [n, K]
        for _ in range(self.steps):
            diff = z.unsqueeze(1) - self.Zl.unsqueeze(0)               # [n, K, r]
            nrm = torch.linalg.vector_norm(diff, dim=2).clamp(min=1e-8)
            z = (self.Zl.unsqueeze(0) + d.unsqueeze(2) * diff / nrm.unsqueeze(2)).mean(1)
        return z

    def cost(self, D: int) -> dict:
        c = super().cost(D)
        c["flops_per_sample"] += self.steps * self.Xl.shape[0] * self.Zc.shape[1] * 3
        return c


class RelaxedEncoder(GowerEncoder):
    """Relaxed srGW-MDS: landmarks frozen, new points optimised by Adam on the
    same stress the landmarks were fitted with. The most faithful extension and
    the slowest; included to check whether the cheaper closed forms give anything
    away."""
    name = "relaxed"

    def __init__(self, Xl, Z, steps: int = 60, lr: float = 0.05, batch: int = 8192):
        super().__init__(Xl, Z)
        self.Zl, self.steps, self.lr, self.batch = Z, steps, lr, batch

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, X.shape[0], self.batch):
            xb = X[i:i + self.batch].float()
            with torch.no_grad():
                z0 = GowerEncoder.encode(self, xb)
                d = cdist_chunked(xb, self.Xl.float())
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=self.lr)
            for _ in range(self.steps):
                opt.zero_grad()
                dz = torch.cdist(z, self.Zl)
                ((dz - d) ** 2).mean().backward()
                opt.step()
            outs.append(z.detach())
        return torch.cat(outs)

    def cost(self, D: int) -> dict:
        c = super().cost(D)
        c["flops_per_sample"] += self.steps * self.Xl.shape[0] * self.Zc.shape[1] * 4
        return c


class LinearProbeEncoder(Encoder):
    """Distil a non-linear encoder into `P in R^{r x D}` by ridge regression.

    Fitted on the landmarks *plus* a larger subsample coded by the teacher --
    fitting on K<=500 points alone in D=3072 dimensions is hopeless. Because the
    result is linear it folds: `A (P x + q) = (A P) x + (A q + c)`, so the
    deployed model is exactly a rank-`r` factorised first layer.
    """
    name = "linear"
    is_linear = True

    def __init__(self, teacher: Encoder, X_fit: torch.Tensor, ridge: float = 1e-3):
        with torch.no_grad():
            Zt = teacher.encode(X_fit).double()
            Xd = X_fit.double()
            n = Xd.shape[0]
            Phi = torch.cat([Xd, torch.ones(n, 1, dtype=Xd.dtype, device=Xd.device)], 1)
            G = Phi.t() @ Phi
            lam = ridge * (G.diagonal()[:-1].mean() + 1e-12)
            reg = torch.zeros(G.shape[0], dtype=Xd.dtype, device=Xd.device)
            reg[:-1] = lam
            Th = torch.linalg.solve(G + torch.diag(reg), Phi.t() @ Zt)   # [D+1, r]
            self.P = Th[:-1].t().float()                                 # [r, D]
            self.q = Th[-1].float()                                      # [r]

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return X.float() @ self.P.t() + self.q

    def cost(self, D: int) -> dict:
        r = self.P.shape[0]
        return {"params": r * D + r, "flops_per_sample": r * D,
                "folds_into_first_layer": True}


OOS_METHODS = ("gower", "nystrom", "relaxed", "linear")


def build_encoder(method: str, Xl: torch.Tensor, Z: torch.Tensor,
                  X_fit: torch.Tensor | None = None) -> Encoder:
    if method == "gower":
        return GowerEncoder(Xl, Z)
    if method == "nystrom":
        return NystromEncoder(Xl, Z)
    if method == "relaxed":
        return RelaxedEncoder(Xl, Z)
    if method == "linear":
        return LinearProbeEncoder(NystromEncoder(Xl, Z), X_fit if X_fit is not None else Xl)
    raise ValueError(f"unknown OOS method {method!r}; options {OOS_METHODS}")
