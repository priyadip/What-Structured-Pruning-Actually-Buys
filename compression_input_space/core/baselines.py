"""Linear-encoder baselines, including the provable ceiling for the whole family.

Why this file exists: if the encoder is linear, `A (P x) = (A P) x`, so input
reduction is *exactly* a rank-`r` factorisation of the compressed first layer.
That means there is a best-possible linear encoder, it is computable in closed
form, and no amount of srGW-MDS machinery can beat it while staying linear.
Reporting srGW-MDS without that number would be uninterpretable.

`optimal_linear_encoder` is that ceiling: the reduced-rank-regression solution,
i.e. Eckart-Young in the data metric. `pca`, `w1_svd` and `random` are the
conventional choices it dominates.
"""

from __future__ import annotations

import numpy as np
import torch

from .oos import Encoder


class MatrixEncoder(Encoder):
    """z = P x + q. Linear, so it folds into the first layer."""
    is_linear = True

    def __init__(self, P: torch.Tensor, q: torch.Tensor | None = None, name: str = "matrix"):
        self.P = P
        self.q = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype) if q is None else q
        self.name = name

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return X.float() @ self.P.t() + self.q

    def cost(self, D: int) -> dict:
        r = self.P.shape[0]
        return {"params": r * D + r, "flops_per_sample": r * D,
                "folds_into_first_layer": True}


@torch.no_grad()
def pad_basis(U: torch.Tensor, r: int) -> torch.Tensor:
    """Ensure `U` has exactly `r` orthonormal columns.

    The reduced-rank solution has rank at most `M1` (the first layer's width), so
    for `r > M1` the SVD returns fewer than `r` directions. Silently returning a
    narrower `P` would make every reported `r` a lie. Pad with an orthonormal
    complement instead: the extra coordinates carry no signal, which is the
    honest statement that `r > M1` cannot help.
    """
    k = U.shape[1]
    if k >= r:
        return U[:, :r]
    n = U.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.randn(n, r - k, generator=g).to(U.device, U.dtype)
    R = R - U @ (U.t() @ R)                        # project out the used subspace
    Q, _ = torch.linalg.qr(R)
    return torch.cat([U, Q[:, : r - k]], 1)


_EIG_CACHE: dict = {}


@torch.no_grad()
def _data_eig(X: torch.Tensor, eps: float = 1e-8):
    """Eigen-decomposition of the (uncentred) second-moment matrix X^T X.

    Cheaper than an SVD of X and gives the same V, S: `X^T X = V S^2 V^T`.
    Memoised: identical for every r and every encoder family on a given fit set,
    and at D=3072 the float64 eigh is seconds, not milliseconds.
    """
    key = (X.data_ptr(), tuple(X.shape), str(X.device))
    if key in _EIG_CACHE:
        return _EIG_CACHE[key]
    Xd = X.double()
    C = Xd.t() @ Xd
    lam, V = torch.linalg.eigh(0.5 * (C + C.t()))
    lam = lam.flip(0).clamp(min=0.0)
    V = V.flip(1)
    S = torch.sqrt(lam + eps)
    if len(_EIG_CACHE) > 8:
        _EIG_CACHE.clear()
    _EIG_CACHE[key] = (V, S)
    return V, S


@torch.no_grad()
def optimal_linear_encoder(X: torch.Tensor, W1: torch.Tensor, r: int) -> MatrixEncoder:
    """The best possible linear encoder — reduced-rank regression.

    Minimises `E_x || W_1 x - A P x ||^2` exactly. With `X = U S V^T`, the
    objective equals `|| S V^T W_1^T - M ||_F` over rank-`r` `M`, so Eckart-Young
    gives `M* = Uc_r Sc_r Vc_r^T` and hence `P = (V S^-1 Uc_r)^T`. Any linear
    encoder — PCA, random projection, or a distilled srGW-MDS probe — is bounded
    by this.
    """
    V, S = _data_eig(X)
    Sinv = 1.0 / S.clamp(min=float(S.max()) * 1e-7)
    C = (S.unsqueeze(1) * (V.t() @ W1.double().t()))        # [D, M1] = S V^T W1^T
    Uc, _, _ = torch.linalg.svd(C, full_matrices=False)
    P = (V * Sinv.unsqueeze(0)) @ pad_basis(Uc, r)          # [D, r]
    return MatrixEncoder(P.t().float(), name="opt_linear")


@torch.no_grad()
def pca_encoder(X: torch.Tensor, r: int) -> MatrixEncoder:
    mu = X.double().mean(0)
    Xc = X.double() - mu
    _, V = torch.linalg.eigh(Xc.t() @ Xc)
    P = pad_basis(V.flip(1), r).t().float()
    return MatrixEncoder(P, q=-(P @ mu.float()), name="pca")


@torch.no_grad()
def w1_svd_encoder(W1: torch.Tensor, r: int) -> MatrixEncoder:
    """Top-`r` right singular vectors of `W_1` — the data-agnostic choice."""
    _, _, Vt = torch.linalg.svd(W1.double(), full_matrices=False)
    return MatrixEncoder(pad_basis(Vt.t(), r).t().float(), name="w1_svd")


@torch.no_grad()
def random_encoder(D: int, r: int, device: str, seed: int = 0) -> MatrixEncoder:
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(r, D, generator=g).to(device) / np.sqrt(D)
    return MatrixEncoder(P, name="random")


BASELINES = ("opt_linear", "pca", "w1_svd", "random")


def build_baseline(name: str, X: torch.Tensor, W1: torch.Tensor, r: int,
                   device: str, seed: int = 0) -> MatrixEncoder:
    if name == "opt_linear":
        return optimal_linear_encoder(X, W1, r)
    if name == "pca":
        return pca_encoder(X, r)
    if name == "w1_svd":
        return w1_svd_encoder(W1, r)
    if name == "random":
        return random_encoder(X.shape[1], r, device, seed)
    raise ValueError(f"unknown baseline {name!r}; options {BASELINES}")
