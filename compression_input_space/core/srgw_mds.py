"""Regularized srGW-MDS: joint manifold embedding + first-layer reconstruction.

Solve, over landmark coordinates `Z in R^{K x r}`,

    L(Z) = stress(Z ; D_geo)  +  lambda * min_{A,c} || Y - Z A^T - 1 c^T ||_F^2

with `Y = X_L W_1^T in R^{K x M1}` the *compressed* first layer's pre-activations
on the landmarks. The first term keeps the embedding faithful to the data
manifold; the second forces the embedding to retain whatever the first layer
actually reads off the input.

`A` is eliminated analytically -- for fixed `Z` the inner problem is ordinary
least squares, so its optimum is a closed-form function of `Z` and gradients flow
straight through `torch.linalg.solve`. That turns a bilevel problem into plain
first-order optimisation over `Z` alone, and it is why an intercept `c` is free
to include (same argument as the hidden-layer work: a bias is the only thing that
can absorb a mean offset).

Both terms are normalised to be dimensionless, so `lambda` means the same thing
across datasets whose pixel scales and pre-activation magnitudes differ by orders
of magnitude. Without that, a lambda sweep is uninterpretable.

**Only `Z` (and the derived `A`) are optimised. No network weight ever receives a
gradient** -- the compressed model is frozen throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EmbeddingFit:
    Z: torch.Tensor          # [K, r] landmark coordinates
    A: torch.Tensor          # [M1, r] new first-layer weights
    c: torch.Tensor          # [M1]    new first-layer bias correction
    stress: float            # normalised MDS stress (0 = perfect)
    recon: float             # normalised reconstruction error (0 = perfect)
    alpha: float             # fraction of first-layer energy retained


def classical_mds_init(D: np.ndarray, r: int) -> torch.Tensor:
    """Torgerson scaling — the exact optimum of the stress term alone, so it is
    the right place to start the joint problem from (random init is far worse)."""
    D2 = torch.as_tensor(D, dtype=torch.float64) ** 2
    K = D2.shape[0]
    J = torch.eye(K, dtype=torch.float64) - 1.0 / K
    B = -0.5 * J @ D2 @ J
    lam, V = torch.linalg.eigh(0.5 * (B + B.T))
    lam = torch.clamp(lam.flip(0)[:r], min=0.0)
    V = V.flip(1)[:, :r]
    return (V * torch.sqrt(lam)).float()


def _solve_A(Z: torch.Tensor, Y: torch.Tensor, ridge: float):
    """Closed-form inner LS with intercept: returns (A [M1,r], c [M1], Yhat)."""
    K = Z.shape[0]
    Phi = torch.cat([Z, torch.ones(K, 1, dtype=Z.dtype, device=Z.device)], 1)
    G = Phi.t() @ Phi
    lam = ridge * (G.diagonal()[:-1].mean().detach() + 1e-12)
    reg = torch.zeros(G.shape[0], dtype=Z.dtype, device=Z.device)
    reg[:-1] = lam                                   # never shrink the intercept
    Theta = torch.linalg.solve(G + torch.diag(reg), Phi.t() @ Y)   # [r+1, M1]
    return Theta[:-1].t(), Theta[-1], Phi @ Theta


def fit_embedding(D_geo: np.ndarray, Y: torch.Tensor, r: int, lam: float, *,
                  epochs: int = 400, lr: float = 0.05, ridge: float = 1e-6,
                  device: str = "cuda", seed: int = 0,
                  init: torch.Tensor | None = None) -> EmbeddingFit:
    """Optimise landmark coordinates for the joint objective. `Y` is [K, M1]."""
    torch.manual_seed(seed)
    D = torch.as_tensor(D_geo, dtype=torch.float32, device=device)
    Y = Y.to(device=device, dtype=torch.float32)
    K = D.shape[0]
    iu = torch.triu_indices(K, K, offset=1, device=device)
    d_target = D[iu[0], iu[1]]
    d_scale = (d_target ** 2).sum() + 1e-12
    y_scale = (Y ** 2).sum() + 1e-12

    Z = (classical_mds_init(D_geo, r) if init is None else init.clone())
    Z = Z.to(device=device, dtype=torch.float32)
    Z = Z / (Z.std() + 1e-12)                        # unit scale keeps lr meaningful
    Z.requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        dz = torch.linalg.vector_norm(Z[iu[0]] - Z[iu[1]], dim=1)
        stress = ((dz - d_target) ** 2).sum() / d_scale
        _, _, Yhat = _solve_A(Z, Y, ridge)
        recon = ((Yhat - Y) ** 2).sum() / y_scale
        (stress + lam * recon).backward()
        opt.step()

    with torch.no_grad():
        Zf = Z.detach()
        A, c, Yhat = _solve_A(Zf, Y, ridge)
        dz = torch.linalg.vector_norm(Zf[iu[0]] - Zf[iu[1]], dim=1)
        stress = float(((dz - d_target) ** 2).sum() / d_scale)
        recon = float(((Yhat - Y) ** 2).sum() / y_scale)
    return EmbeddingFit(Zf, A, c, stress, recon, 1.0 - recon)
