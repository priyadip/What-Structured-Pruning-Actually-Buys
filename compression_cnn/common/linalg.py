"""Closed-form kernels. No autograd, no optimiser, no backward pass anywhere.

Everything here is the MLP study's linear algebra with one generalisation: the
design matrix's columns come in contiguous **groups** of `gs = k*k` (all the taps
of one input channel), and a channel is kept or dropped as a whole. Setting
`gs = 1` recovers the scalar routines in `../compression_mlp/run_autonomous_compression.py`
exactly, which is what `selftest.py` checks.
"""

from __future__ import annotations

import torch


def f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


# ---------------------------------------------------------------------------
# sufficient statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def gram_cross(U: torch.Tensor, Y: torch.Tensor, chunk: int = 8192):
    """`G = U^T U`, `P = U^T Y`, plus the column sums the intercept needs.

    Accumulated in float64 across float32 chunk products: a single float32
    reduction over ~1e5 rows loses about five digits, which a ridge of 1e-4
    cannot rescue, while a full float64 product costs 60x more on this hardware.
    Chunked, the per-chunk error is ~sqrt(8192)*eps and the sum across chunks is
    exact to float64.

    **This is safe only because the ridge is relative.** The deepest conv units
    have `cond(G) ~ 7e11`, and at `ridge=1e-8` a float32-accumulated Gram gives a
    least-squares residual of 2.8e-3 where a float64 Gram gives 6e-9 -- six orders
    of magnitude. At the `ridge=1e-4` the drivers use, the two agree to within a
    factor of 1.4 (residuals ~1e-8), because the ridge caps the effective
    condition number far below where float32 accumulation matters. Lower the
    ridge and this function must go to float64.
    """
    F, C = U.shape[1], Y.shape[1]
    dev = U.device
    G = torch.zeros(F, F, dtype=torch.float64, device=dev)
    P = torch.zeros(F, C, dtype=torch.float64, device=dev)
    us = torch.zeros(F, dtype=torch.float64, device=dev)
    ys = torch.zeros(C, dtype=torch.float64, device=dev)
    for i in range(0, U.shape[0], chunk):
        u, y = U[i:i + chunk], Y[i:i + chunk]
        G += (u.t() @ u).double()
        P += (u.t() @ y).double()
        us += u.sum(0).double()
        ys += y.sum(0).double()
    return G, P, us, ys, int(U.shape[0])


def group_cols(groups: list[int] | torch.Tensor, gs: int, device) -> torch.Tensor:
    g = torch.as_tensor(groups, device=device, dtype=torch.long)
    return (g.unsqueeze(1) * gs + torch.arange(gs, device=device)).reshape(-1)


# ---------------------------------------------------------------------------
# group selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def group_select(G: torch.Tensor, P: torch.Tensor | None, gs: int, m: int,
                 jitter: float = 1e-10) -> list[int]:
    """Greedy group-wise selection by block pivoted Cholesky.

    `P is not None`  -- **output-targeted** (group OMP): at each step take the
    channel whose remaining block most reduces `||Y - U_S U_S^+ Y||_F^2`, i.e.
    `argmax_g tr(Pres_g^T D_g^-1 Pres_g)` with `D_g` the current Schur
    complement block. This is the CNN form of the MLP study's `omp_select`, and
    it is the departure from all OT-style pruning: the cost references the
    *downstream target*, not a geometric distance between filters.

    `P is None`      -- **span-only** (block pivoted Cholesky): `argmax_g tr(D_g)`,
    the channels that best span the patch space regardless of what the next
    layer reads. This is the CNN form of `pivoted_cholesky_select`.

    The Schur complement is carried, never recomputed, so the whole sweep is
    `O(m^2 * gs * F)` rather than `m` separate least-squares fits.
    """
    G = f64(G)
    F = G.shape[0]
    C = F // gs
    if m >= C:
        return list(range(C))
    dev = G.device

    idx = torch.arange(F, device=dev).view(C, gs)
    D = G[idx.reshape(-1)][:, idx.reshape(-1)].view(C, gs, C, gs)
    D = torch.stack([D[g, :, g, :] for g in range(C)])           # [C, gs, gs]
    scale = float(D.diagonal(dim1=1, dim2=2).mean().clamp(min=1e-30))
    eye = torch.eye(gs, dtype=G.dtype, device=dev)

    Pres = f64(P).clone() if P is not None else None
    L = torch.zeros(m * gs, F, dtype=G.dtype, device=dev)
    taken = torch.zeros(C, dtype=torch.bool, device=dev)
    picked: list[int] = []
    t = 0
    for _ in range(m):
        Dj = D + (jitter * scale) * eye
        if Pres is not None:
            Pg = Pres.view(C, gs, -1)
            sol = torch.linalg.solve(Dj, Pg)                     # [C, gs, out]
            score = (Pg * sol).sum((1, 2))
        else:
            score = D.diagonal(dim1=1, dim2=2).sum(1)
        score = score.masked_fill(taken, -float("inf"))
        j = int(torch.argmax(score))
        if not torch.isfinite(score[j]):
            j = int(torch.nonzero(~taken).flatten()[0])
        picked.append(j)
        taken[j] = True

        cols = idx[j]
        B = G[cols] - L[:t, cols].t() @ L[:t]                    # [gs, F]
        Djj = B[:, cols] + (jitter * scale) * eye
        try:
            R = torch.linalg.cholesky(0.5 * (Djj + Djj.t()))
        except Exception:                                        # numerically dead block
            D[j] = 0.0
            continue
        Ln = torch.linalg.solve_triangular(R, B, upper=False)    # [gs, F]
        if Pres is not None:
            Z = torch.linalg.solve_triangular(R, Pres[cols], upper=False)
            Pres = Pres - Ln.t() @ Z
        Lg = Ln.view(gs, C, gs)                              # [tap_of_new, chan, tap]
        D = D - torch.einsum("acp,acq->cpq", Lg, Lg)         # per-channel Schur update
        D[j] = 0.0
        L[t:t + gs] = Ln
        t += gs
    return sorted(picked)


@torch.no_grad()
def forward_backward_select(G: torch.Tensor, P: torch.Tensor, us: torch.Tensor,
                            ys: torch.Tensor, n: int, gs: int, m: int,
                            ridge: float = 1e-4, over: float = 3.0) -> list[int]:
    """Overshoot with group OMP, then *exact* backward elimination of channels.

    Forward greedy never reconsiders: a channel taken early to explain a
    direction can become redundant once later ones arrive. For a ridged
    least-squares system the increase in error from deleting a group is available
    in closed form -- the block generalisation of the MLP study's
    `||Theta_j||^2 / C_jj`,

            dSSE_g = tr( Theta_g^T (C_gg)^-1 Theta_g ),   C = (Phi^T Phi + reg)^-1

    and both `C` and `Theta` are downdated by the matching block formula, so no
    system is ever refactorised. The intercept row is present throughout and is
    never a candidate for removal.
    """
    C_all = G.shape[0] // gs
    K = int(min(C_all, max(m, round(over * m))))
    cand = group_select(G, P, gs, K) if K < C_all else list(range(C_all))
    if len(cand) <= m:
        return sorted(cand)

    dev = G.device
    cols = group_cols(cand, gs, dev)
    Gs, Ps, uss = f64(G)[cols][:, cols], f64(P)[cols], f64(us)[cols]
    F = Gs.shape[0]
    Gaug = torch.zeros(F + 1, F + 1, dtype=Gs.dtype, device=dev)
    Gaug[:F, :F], Gaug[:F, F], Gaug[F, :F], Gaug[F, F] = Gs, uss, uss, float(n)
    lam = ridge * (Gs.diagonal().mean() + 1e-12)
    reg = torch.zeros(F + 1, dtype=Gs.dtype, device=dev)
    reg[:F] = lam
    Cinv = torch.linalg.inv(Gaug + torch.diag(reg))
    Theta = Cinv @ torch.cat([Ps, f64(ys).unsqueeze(0)], 0)

    alive = list(range(len(cand)))
    while len(alive) > m:
        best, bj = None, 0
        for pos in range(len(alive)):
            sl = slice(pos * gs, (pos + 1) * gs)
            Cgg = Cinv[sl, sl]
            th = Theta[sl]
            cost = float((th * torch.linalg.solve(
                Cgg + 1e-30 * torch.eye(gs, dtype=Cgg.dtype, device=dev), th)).sum())
            if best is None or cost < best:
                best, bj = cost, pos
        sl = slice(bj * gs, (bj + 1) * gs)
        Cb = Cinv[:, sl]
        Cgg = Cinv[sl, sl] + 1e-30 * torch.eye(gs, dtype=Cinv.dtype, device=dev)
        Theta = Theta - Cb @ torch.linalg.solve(Cgg, Theta[sl])
        Cinv = Cinv - Cb @ torch.linalg.solve(Cgg, Cinv[sl])
        keep = torch.ones(Cinv.shape[0], dtype=torch.bool, device=dev)
        keep[sl] = False
        Cinv = Cinv[keep][:, keep]
        Theta = Theta[keep]
        alive.pop(bj)
    return sorted(cand[i] for i in alive)


# ---------------------------------------------------------------------------
# the consuming layer's closed-form refit
# ---------------------------------------------------------------------------

@torch.no_grad()
def ls_consuming(G: torch.Tensor, P: torch.Tensor, us: torch.Tensor, ys: torch.Tensor,
                 n: int, cols: torch.Tensor, ridge: float = 1e-4):
    """`min ||[U_S , 1] [V';c']^T - Y||_F` from the cached sufficient statistics.

    `Y` is the **dense** network's pre-activation at the consumer, so the fit
    both compensates for the channels removed here and absorbs error inherited
    from layers already compressed upstream -- the GPTQ/SparseGPT idea applied to
    structured channel removal, unchanged from the MLP study.

    The intercept is not optional. Dropping a ReLU channel removes a
    non-negative contribution with a non-zero *mean*; no reweighting of the
    survivors can represent a constant, only a bias can. It is never shrunk by
    the ridge, and the ridge itself is relative to the mean Gram diagonal so it
    behaves consistently across very different budgets.
    """
    Gs, Ps, uss = f64(G)[cols][:, cols], f64(P)[cols], f64(us)[cols]
    F = Gs.shape[0]
    dev = Gs.device
    A = torch.zeros(F + 1, F + 1, dtype=Gs.dtype, device=dev)
    A[:F, :F], A[:F, F], A[F, :F], A[F, F] = Gs, uss, uss, float(n)
    lam = ridge * (Gs.diagonal().mean() + 1e-12)
    reg = torch.zeros(F + 1, dtype=Gs.dtype, device=dev)
    reg[:F] = lam
    rhs = torch.cat([Ps, f64(ys).unsqueeze(0)], 0)
    Th = torch.linalg.solve(A + torch.diag(reg), rhs)
    return Th[:F].t().float(), Th[F].float()          # V_flat [out, F], c [out]


@torch.no_grad()
def residual_sse(G, P, us, ys, n, cols, ridge, yy: torch.Tensor) -> float:
    """`||Y - [U_S,1] Theta||_F^2` without touching the raw rows again."""
    V, c = ls_consuming(G, P, us, ys, n, cols, ridge)
    Th = torch.cat([f64(V).t(), f64(c).unsqueeze(0)], 0)
    Gs = f64(G)[cols][:, cols]
    uss = f64(us)[cols]
    F = Gs.shape[0]
    A = torch.zeros(F + 1, F + 1, dtype=Gs.dtype, device=Gs.device)
    A[:F, :F], A[:F, F], A[F, :F], A[F, F] = Gs, uss, uss, float(n)
    rhs = torch.cat([f64(P)[cols], f64(ys).unsqueeze(0)], 0)
    return float(yy - 2 * (Th * rhs).sum() + (Th * (A @ Th)).sum())
