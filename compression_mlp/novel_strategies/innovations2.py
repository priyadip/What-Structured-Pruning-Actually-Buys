"""Second wave of novel strategies: enlarge the *dictionary*, and make the
selection exact rather than greedy.

Both still obey the golden rule -- exactly `target_m = round(multiplier *
GRIDE_d_l)` neurons come out -- and both are closed-form.

Motivation
----------
`N02_omp_ls` is limited in two independent ways:

1. **The dictionary.** It may only choose among the N neurons that happen to be
   in the layer. But a compressed layer is allowed to contain *any* neuron. OT
   merging exploits this (a barycentre is a new neuron) but then has no way to
   tell whether the merge helped. So: build an explicit **augmented dictionary**
   of candidate neurons -- the originals, barycentric merges at several
   granularities, and threshold-shifted copies -- and let the selection decide.
   A merged atom is kept only if it earns its slot on the calibration data,
   which is exactly the check OT merging lacks.

   The threshold-shifted copies matter more than they look. `relu(w.x + b')` for
   several b' along one direction is a spline basis in that direction: it lets M
   neurons express a *piecewise-linear* response where selection alone can only
   express a single kink. Rescaling is pointless (the consuming layer absorbs
   any positive scale) -- the threshold is the one thing that is genuinely free.

2. **The selection rule.** Forward greedy (OMP) never reconsiders. A neuron
   picked early to explain a direction may become redundant once later ones
   arrive. Forward-backward fixes this exactly: overshoot to K = over*m atoms,
   then remove atoms one at a time, each time deleting the one whose removal
   least increases the least-squares error. For an LS system that increase is
   available in closed form,

       dSSE_j = ||Theta_j||^2 / C_jj ,     C = (Phi^T Phi + ridge)^{-1},

   and the inverse is downdated by the same rank-1 formula, so the whole sweep
   costs one K x K inverse plus O(K^3) of downdates -- no refitting.
"""

from __future__ import annotations

import numpy as np
import torch

from run_autonomous_compression import (
    Strategy, _f64, kmeans_labels, ls_consuming, omp_select, relabel_dense,
)


# ---------------------------------------------------------------------------
# dictionary construction
# ---------------------------------------------------------------------------

def build_dictionary(ctx, *, shifts=(0.25, 0.75), merge_levels=(2, 4),
                     max_atoms: int = 8192):
    """Candidate neurons for this layer: (W_atoms [K,din], b_atoms [K]).

    Atom families
      * the N original neurons (always first, so pure selection stays reachable)
      * `merge_levels` barycentric merges: neurons k-means'd on their (normalised)
        activation profiles into N/level groups, each group replaced by its
        importance-weighted mean -- i.e. exactly an OT/k-means merge, offered as
        a candidate rather than imposed
      * `shifts` threshold-shifted copies: same direction, bias retuned so the
        neuron fires on a fraction p of the calibration set
    """
    W, b, H, A = ctx.W, ctx.b, ctx.H, ctx.A
    N = W.shape[0]
    Ws, bs = [W], [b]

    imp = A.mean(0) + 1e-9
    An = A / (A.norm(dim=0, keepdim=True) + 1e-12)
    rng = np.random.default_rng(ctx.seed)
    rows = rng.choice(An.shape[0], min(512, An.shape[0]), replace=False)
    feats = An[rows].t().float().cpu().numpy()          # [N, 512] activation profile

    for lv in merge_levels:
        k = max(2, N // lv)
        if k >= N:
            continue
        labels = relabel_dense(kmeans_labels(feats, k, ctx.seed), k)
        lab = torch.as_tensor(labels, device=W.device, dtype=torch.long)
        Wm = torch.zeros(k, W.shape[1], dtype=W.dtype, device=W.device)
        bm = torch.zeros(k, dtype=b.dtype, device=b.device)
        for c in range(k):
            idx = torch.nonzero(lab == c).flatten()
            if idx.numel() == 0:
                continue
            w = imp[idx] / imp[idx].sum()
            Wm[c] = (w.unsqueeze(1) * W[idx]).sum(0)
            bm[c] = (w * b[idx]).sum()
        Ws.append(Wm)
        bs.append(bm)

    if shifts:
        zb = H @ W.t() + b                              # [T, N] pre-activations
        sub = zb[rng.choice(zb.shape[0], min(4096, zb.shape[0]), replace=False)]
        for p in shifts:
            thr = torch.quantile(sub.float(), 1.0 - p, dim=0)   # fire on a fraction p
            Ws.append(W)
            bs.append(b - thr.to(b.dtype))

    Wa, ba = torch.cat(Ws, 0), torch.cat(bs, 0)
    if Wa.shape[0] > max_atoms:                          # keep originals, trim the rest
        keep = torch.arange(N, device=Wa.device)
        extra = torch.arange(N, Wa.shape[0], device=Wa.device)
        sel = extra[torch.randperm(extra.numel(), device=Wa.device)[: max_atoms - N]]
        idx = torch.cat([keep, sel.sort().values])
        Wa, ba = Wa[idx], ba[idx]
    return Wa, ba


# ---------------------------------------------------------------------------
# exact forward-backward selection
# ---------------------------------------------------------------------------

def forward_backward_select(A: torch.Tensor, Y: torch.Tensor, m: int,
                            ridge: float = 1e-4, over: float = 3.0) -> list[int]:
    """Overshoot with OMP to K = over*m atoms, then exact backward elimination.

    Removal cost for column j of the ridged LS system is ||Theta_j||^2 / C_jj
    with C = (Phi^T Phi + lam I)^-1; deleting j downdates C and Theta by the
    standard rank-1 formulas, so no system is ever refactorised. The intercept
    column is present throughout and never eliminated.
    """
    Ad, Yd = _f64(A), _f64(Y)
    K_all = Ad.shape[1]
    K = int(min(K_all, max(m, round(over * m))))
    cand = omp_select(Ad.t() @ Ad, Ad.t() @ Yd, K) if K < K_all else list(range(K_all))
    if len(cand) <= m:
        return sorted(cand)

    T = Ad.shape[0]
    Phi = torch.cat([Ad[:, cand], torch.ones(T, 1, dtype=Ad.dtype, device=Ad.device)], 1)
    G = Phi.t() @ Phi
    lam = ridge * (G.diagonal()[:-1].mean() + 1e-12)
    reg = torch.full((G.shape[0],), float(lam), dtype=G.dtype, device=G.device)
    reg[-1] = 0.0                                     # never shrink the intercept
    C = torch.linalg.inv(G + torch.diag(reg))
    Theta = C @ (Phi.t() @ Yd)                        # [K+1, out]

    alive = list(range(len(cand)))                    # positions into `cand`
    n_int = len(alive)                                # index of the intercept row in C
    while len(alive) > m:
        d = C.diagonal()[:-1].clamp(min=1e-30)
        cost = (Theta[:-1] ** 2).sum(1) / d
        j = int(torch.argmin(cost))
        cj = C[:, j].clone()
        Theta = Theta - torch.outer(cj, Theta[j]) / C[j, j]
        C = C - torch.outer(cj, cj) / C[j, j]
        keep = [i for i in range(C.shape[0]) if i != j]
        C = C[keep][:, keep]
        Theta = Theta[keep]
        alive.pop(j)
        n_int -= 1
    return sorted(cand[i] for i in alive)


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

class N06_AugmentedDictOMP(Strategy):
    name = "N06_augdict_omp"
    note = "OMP over an augmented dictionary (originals + OT merges + threshold shifts)"

    def __init__(self, shifts=(0.25, 0.75), merge_levels=(2, 4), tag=""):
        self.shifts, self.merge_levels = shifts, merge_levels
        if tag:
            self.name = f"N06_augdict_omp_{tag}"

    def apply(self, ctx, m):
        Wa, ba = build_dictionary(ctx, shifts=self.shifts, merge_levels=self.merge_levels)
        Aa = torch.relu(ctx.H @ Wa.t() + ba)
        Ad = _f64(Aa)
        keep = omp_select(Ad.t() @ Ad, Ad.t() @ _f64(ctx.Y), m)
        V, c = ls_consuming(Aa[:, keep], ctx.Y, ctx.ridge)
        return Wa[keep], ba[keep], V, c


class N07_ForwardBackwardOMP(Strategy):
    name = "N07_fwdbwd_omp"
    note = "OMP overshoot + exact closed-form backward elimination, original neurons only"

    def __init__(self, over: float = 3.0):
        self.over = over

    def apply(self, ctx, m):
        keep = forward_backward_select(ctx.A, ctx.Y, m, ctx.ridge, self.over)
        V, c = ls_consuming(ctx.A[:, keep], ctx.Y, ctx.ridge)
        return ctx.W[keep], ctx.b[keep], V, c


class N08_AugDictForwardBackward(Strategy):
    """Both ideas at once -- the strongest configuration in this file."""
    name = "N08_augdict_fwdbwd"
    note = "augmented dictionary + OMP overshoot + exact backward elimination"

    def __init__(self, over: float = 3.0, shifts=(0.25, 0.75), merge_levels=(2, 4)):
        self.over, self.shifts, self.merge_levels = over, shifts, merge_levels

    def apply(self, ctx, m):
        Wa, ba = build_dictionary(ctx, shifts=self.shifts, merge_levels=self.merge_levels)
        Aa = torch.relu(ctx.H @ Wa.t() + ba)
        keep = forward_backward_select(Aa, ctx.Y, m, ctx.ridge, self.over)
        V, c = ls_consuming(Aa[:, keep], ctx.Y, ctx.ridge)
        return Wa[keep], ba[keep], V, c


STRATEGIES = [
    N06_AugmentedDictOMP(),
    N06_AugmentedDictOMP(shifts=(0.1, 0.25, 0.5, 0.75), merge_levels=(2, 4, 8), tag="wide"),
    N07_ForwardBackwardOMP(over=3.0),
    N08_AugDictForwardBackward(over=3.0),
]
