"""Channel-compression strategies for conv layers.

Every strategy compresses ONE unit to EXACTLY `target_m` output channels, where
`target_m = round(multiplier * d_l)` and `d_l` is the layer's GRIDE intrinsic
dimension. `target_m` is handed down, never chosen here, and the driver asserts
the returned tensor has exactly that many rows.

Everything is closed form. No strategy takes a gradient, calls `.backward()`, or
touches an optimiser.

The families, and what separates them
-------------------------------------
**merge** (`ot_intrafusion`, `kmeans_merge`) replaces a group of filters by their
barycentre. This is what OT-based pruning does. It is only output-preserving when
every filter in the group co-activates on every patch -- the ReLU exactness
condition -- and on a trained network that fails badly. Worse, the error is
invisible downstream: nothing is refitted, so nothing can undo it.

**select-and-correct** (`span_ls`, `omp_ls`, `magnitude_ls`, `fwdbwd_ls`,
`augdict_*`) keeps a subset of filters *exactly*, so their ReLU responses are
exact by construction, and then re-solves the consuming convolution in closed
form against the **dense** network's own pre-activation. Two things make that
correction stronger than a plain reconstruction map: the target is the dense
value at the consumer (so error inherited from already-compressed layers is
absorbed rather than compounded), and the fit carries a free intercept (deleting
a ReLU channel removes a contribution with non-zero mean, which no reweighting
of the survivors can represent).

`omp_ls` is the headline: selection scored against `U^T Y`, i.e. against the
subspace the next layer actually reads, rather than against a geometric distance
between filters. That is the point where this departs from OT pruning, and in
the MLP study it was worth ~5 points at heavy compression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from common import linalg as LA          # noqa: E402
from common import surgery as SG         # noqa: E402

try:
    import ot as pot                     # POT: exact EMD for the Intra-Fusion baseline
except Exception:                        # pragma: no cover
    pot = None


class Strategy:
    """`apply(ctx, m) -> (W_new [m,Cin,kh,kw], b_new [m], V_flat [out, m*gs], c_new)`."""
    name = "base"
    note = ""
    family = "select"
    needs_metric = False        # set True to receive ctx.metric (logit-pullback)

    def apply(self, ctx: SG.UnitCtx, m: int):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


def _rows(ctx, keep):
    W = ctx.prod.weight.detach()
    return W[keep].clone(), ctx.prod.bias.detach()[keep].clone()


# ---------------------------------------------------------------------------
# select-and-correct
# ---------------------------------------------------------------------------

class MagnitudeOnly(Strategy):
    name = "S1_magnitude"
    note = "classic structured pruning: keep the m filters of largest L2 norm, no refit"
    family = "prune"

    def apply(self, ctx, m):
        imp = ctx.filters_flat().norm(dim=1)
        keep = torch.topk(imp, m).indices.sort().values.tolist()
        W, b = _rows(ctx, keep)
        V = ctx.consumer.weight.detach()
        Vf = (V.reshape(V.shape[0], ctx.C, -1)[:, keep].reshape(V.shape[0], -1)
              if isinstance(ctx.consumer, nn.Conv2d)
              else V.reshape(V.shape[0], ctx.C, -1)[:, keep].reshape(V.shape[0], -1))
        return W, b, Vf.clone(), ctx.consumer.bias.detach().clone()


class MagnitudeLS(Strategy):
    name = "S2_magnitude_ls"
    note = "magnitude selection + closed-form consumer refit (isolates what the refit is worth)"

    def apply(self, ctx, m):
        imp = ctx.filters_flat().norm(dim=1)
        keep = torch.topk(imp, m).indices.sort().values.tolist()
        V, c = ctx.ls(keep)
        return (*_rows(ctx, keep), V, c)


class SpanLS(Strategy):
    name = "S3_span_ls"
    note = "block pivoted-Cholesky selection (spans the patch space) + consumer refit"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = LA.group_select(G, None, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*_rows(ctx, keep), V, c)


class OMPLS(Strategy):
    """The headline method: output-targeted selection + intercept refit."""
    name = "S4_omp_ls"
    note = "group-OMP selection against U^T Y (what the next layer reads) + consumer refit"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = LA.group_select(G, P, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*_rows(ctx, keep), V, c)


class ForwardBackwardLS(Strategy):
    name = "S5_fwdbwd_ls"
    note = "group-OMP overshoot + exact closed-form backward elimination + consumer refit"

    def __init__(self, over: float = 3.0):
        self.over = over

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = LA.forward_backward_select(G, P, us, ys, n, ctx.gs, m, ctx.ridge, self.over)
        V, c = ctx.ls(keep)
        return (*_rows(ctx, keep), V, c)


# ---------------------------------------------------------------------------
# merge (the OT / clustering family)
# ---------------------------------------------------------------------------

def _mix(ctx, T: torch.Tensor):
    """Apply a `[C, m]` mixing matrix to producer rows and consumer input channels."""
    W = ctx.prod.weight.detach()
    b = ctx.prod.bias.detach()
    Wn = torch.einsum("cm,cikl->mikl", T, W)
    bn = T.t() @ b
    V = ctx.consumer.weight.detach()
    Vg = V.reshape(V.shape[0], ctx.C, -1)                    # [out, C, gs]
    Vn = torch.einsum("ocg,cm->omg", Vg, T).reshape(V.shape[0], -1)
    return Wn, bn, Vn, ctx.consumer.bias.detach().clone()


def _ot_plan(ctx, m):
    """Intra-Fusion's plan: keep the top-m filters by L1 mass, transport the rest
    onto them under a cosine cost, column-normalise."""
    S = ctx.filters_flat()
    imp = S.abs().sum(1)
    keep = torch.topk(imp, m).indices.sort().values
    mask = torch.zeros(ctx.C, dtype=torch.bool, device=S.device)
    mask[keep] = True
    drop = torch.nonzero(~mask).flatten()

    Sn = S / (S.norm(dim=1, keepdim=True) + 1e-12)
    T = torch.zeros(ctx.C, m, dtype=torch.float64, device=S.device)
    T[keep, torch.arange(m, device=S.device)] = imp[keep].double()
    if drop.numel():
        cost = (1.0 - Sn[drop] @ Sn[keep].t()).double().cpu().numpy()
        a = (imp[drop] / imp[drop].sum()).double().cpu().numpy()
        bb = (imp[keep] / imp[keep].sum()).double().cpu().numpy()
        if pot is not None:
            plan = pot.emd(a, bb, np.ascontiguousarray(cost), numItermax=200000)
        else:                                                # greedy nearest-kept
            plan = np.zeros_like(cost)
            plan[np.arange(len(a)), cost.argmin(1)] = a
        T[drop] = torch.as_tensor(plan, device=S.device) * imp[drop].sum().double()
    T = T / (T.sum(0, keepdim=True) + 1e-12)
    return T.to(S.dtype)


class OTIntraFusion(Strategy):
    name = "S6_ot_intrafusion"
    note = "Intra-Fusion (ICLR'24) baseline: exact-EMD filter merging, no refit"
    family = "merge"

    def apply(self, ctx, m):
        return _mix(ctx, _ot_plan(ctx, m))


class OTThenLS(Strategy):
    name = "S7_ot_then_ls"
    note = "OT merge for the producer, then re-solve the consumer in closed form"
    family = "merge"

    def apply(self, ctx, m):
        Wn, bn, _, _ = _mix(ctx, _ot_plan(ctx, m))
        A = ctx.acts_for(Wn, bn)
        G, P, us, ys, n = ctx.stats_for(A)
        cols = LA.group_cols(list(range(m)), ctx.gs, A.device)
        V, c = LA.ls_consuming(G, P, us, ys, n, cols, ctx.ridge)
        return Wn, bn, V, c


class KMeansMerge(Strategy):
    name = "S8_kmeans_merge"
    note = "k-means the filters in patch space, replace each cluster by its barycentre"
    family = "merge"

    def apply(self, ctx, m):
        from sklearn.cluster import KMeans
        S = ctx.filters_flat().float().cpu().numpy()
        lab = KMeans(n_clusters=m, n_init=1 if m > 128 else 3,
                     random_state=ctx.seed).fit(S).labels_
        T = torch.zeros(ctx.C, m, device=ctx.A.device, dtype=ctx.prod.weight.dtype)
        w = ctx.A.mean((0, 2, 3)) + 1e-9 if ctx.A.dim() == 4 else ctx.A.mean(0) + 1e-9
        for k in range(m):
            idx = np.nonzero(lab == k)[0]
            if idx.size == 0:
                continue
            ii = torch.as_tensor(idx, device=T.device)
            T[ii, k] = w[ii] / w[ii].sum()
        empty = (T.sum(0) == 0).nonzero().flatten()
        for k in empty.tolist():                             # keep the exact-m contract
            T[int(torch.argmax(w)), k] = 1.0
        return _mix(ctx, T)


# ---------------------------------------------------------------------------
# augmented dictionary -- the MLP study's overall winner
# ---------------------------------------------------------------------------

def build_dictionary(ctx, shifts=(0.25, 0.75), merge_levels=(2, 4), max_atoms=2048):
    """Candidate filters for this unit: originals, barycentric merges, shifted copies.

    A compressed layer is allowed to contain *any* filter, not only one of the C
    already present. Merging exploits that but cannot tell whether a merge helped;
    offering merges as *candidates* and letting the output-targeted selection
    decide restores the check that OT merging lacks.

    The threshold-shifted copies matter more than they look: `relu(w.x + b')` for
    several `b'` along one direction is a spline basis in that direction, so `m`
    channels can express a piecewise-linear response where selection alone gets a
    single kink. Rescaling would be pointless -- the consumer absorbs any positive
    scale -- so the bias is the one genuinely free parameter.
    """
    W = ctx.prod.weight.detach()
    b = ctx.prod.bias.detach()
    C = W.shape[0]
    Ws, bs = [W], [b]
    S = ctx.filters_flat()

    act = ctx.A.mean((0, 2, 3)) if ctx.A.dim() == 4 else ctx.A.mean(0)
    imp = act + 1e-9
    if merge_levels:
        from sklearn.cluster import KMeans
        feats = (S / (S.norm(dim=1, keepdim=True) + 1e-12)).float().cpu().numpy()
        for lv in merge_levels:
            k = max(2, C // lv)
            if k >= C:
                continue
            lab = KMeans(n_clusters=k, n_init=1, random_state=ctx.seed).fit(feats).labels_
            T = torch.zeros(C, k, device=W.device, dtype=W.dtype)
            for j in range(k):
                idx = torch.as_tensor(np.nonzero(lab == j)[0], device=W.device)
                if idx.numel():
                    T[idx, j] = imp[idx] / imp[idx].sum()
            Ws.append(torch.einsum("cm,cikl->mikl", T, W))
            bs.append(T.t() @ b)

    if shifts:
        z = ctx.acts_for(W, b)                               # post-ReLU, full width
        zf = z.permute(1, 0, 2, 3).reshape(C, -1) if z.dim() == 4 else z.t()
        sub = zf[:, torch.randperm(zf.shape[1], device=zf.device)[:8192]].float()
        for p in shifts:
            thr = torch.quantile(sub, 1.0 - p, dim=1)
            Ws.append(W.clone())
            bs.append(b - thr.to(b.dtype))
        del z, zf, sub

    Wa, ba = torch.cat(Ws, 0), torch.cat(bs, 0)
    if Wa.shape[0] > max_atoms:                              # keep originals, trim the rest
        g = torch.Generator(device="cpu").manual_seed(ctx.seed)
        extra = torch.randperm(Wa.shape[0] - C, generator=g)[: max_atoms - C] + C
        idx = torch.cat([torch.arange(C), extra.sort().values]).to(Wa.device)
        Wa, ba = Wa[idx], ba[idx]
    return Wa, ba


class AugDictOMP(Strategy):
    name = "S9_augdict_omp"
    note = "group-OMP over an augmented dictionary (originals + merges + threshold shifts)"
    family = "dict"

    def __init__(self, shifts=(0.25, 0.75), merge_levels=(2, 4)):
        self.shifts, self.merge_levels = shifts, merge_levels

    def _select(self, ctx, m, G, P, us, ys, n):
        return LA.group_select(G, P, ctx.gs, m)

    def apply(self, ctx, m):
        Wa, ba = build_dictionary(ctx, self.shifts, self.merge_levels)
        A = ctx.acts_for(Wa, ba)
        G, P, us, ys, n = ctx.stats_for(A)
        keep = self._select(ctx, m, G, P, us, ys, n)
        cols = LA.group_cols(keep, ctx.gs, A.device)
        V, c = LA.ls_consuming(G, P, us, ys, n, cols, ctx.ridge)
        del A
        return Wa[keep].clone(), ba[keep].clone(), V, c


class AugDictForwardBackward(AugDictOMP):
    """Both ideas at once -- the strongest configuration in the MLP study."""
    name = "S10_augdict_fwdbwd"
    note = "augmented dictionary + OMP overshoot + exact backward elimination"
    family = "dict"

    def __init__(self, over: float = 3.0, **kw):
        super().__init__(**kw)
        self.over = over

    def _select(self, ctx, m, G, P, us, ys, n):
        return LA.forward_backward_select(G, P, us, ys, n, ctx.gs, m, ctx.ridge, self.over)


ALL: list[Strategy] = [
    MagnitudeOnly(), MagnitudeLS(), SpanLS(), OMPLS(), ForwardBackwardLS(),
    OTIntraFusion(), OTThenLS(), KMeansMerge(), AugDictOMP(), AugDictForwardBackward(),
]
BY_NAME = {s.name: s for s in ALL}
