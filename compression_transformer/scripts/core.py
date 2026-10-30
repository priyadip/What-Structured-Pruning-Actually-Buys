#!/usr/bin/env python3
"""Select-and-correct for DistilBERT FFN units.

The unit is `(ffn.lin1, ffn.lin2)`. `lin1` emits 3072 neurons and exactly one
module reads them, so this is the identical producer/consumer pair the MLP study
compressed, with a different carrier around it. Everything mathematical is
imported from `compression_cnn/common/linalg.py` at `gs = 1`, which is the
scalar case that package's self-test already checks against the MLP code, so no
linear algebra is reimplemented here.

What is genuinely new is the plumbing: rows of the design matrix are
(example, token) pairs with padding masked out, the walk through the network is
a lock-step advance of a dense and a compressed copy so each fit sees the error
inherited from upstream, and the target admits a residual-absorbing variant that
a ResNet's internal refit was never asked for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TF_ROOT = HERE.parent
PROJECT_ROOT = TF_ROOT.parent
EXT_ROOT = PROJECT_ROOT / "compression_cnn"
for _p in (str(PROJECT_ROOT), str(EXT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import linalg as LA                                    # noqa: E402
from intrinsic_rank.id_estimator import gride as _gride            # noqa: E402

MODEL_ID = "typeform/distilbert-base-uncased-mnli"
RESULTS = TF_ROOT / "results"
CACHE = TF_ROOT / "cache"
N_LAYERS = 6
HIDDEN_DIM = 3072
DIM = 768


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# model and data
# ---------------------------------------------------------------------------

def load_model(dev=None):
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    m.eval().to(dev or device())
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _encode(tok, prem, hyp, max_len):
    # datasets >= 4 returns a lazy Column, which the tokenizer rejects
    enc = tok(list(prem), list(hyp), truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def calibration_set(n: int, seed: int, max_len: int = 128):
    """`n` unlabelled MNLI *training* inputs. Labels are never read here."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    sub = ds.select(sorted(int(i) for i in idx))
    tok = load_tokenizer()
    return _encode(tok, sub["premise"], sub["hypothesis"], max_len)


def eval_set(max_len: int = 128, split: str = "validation_matched"):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli", split=split)
    tok = load_tokenizer()
    ids, mask = _encode(tok, ds["premise"], ds["hypothesis"], max_len)
    return ids, mask, torch.tensor(list(ds["label"]), dtype=torch.long)


@torch.inference_mode()
def accuracy(model, ids, mask, labels, batch: int = 256, perm=None) -> float:
    dev = next(model.parameters()).device
    ok = 0
    for i in range(0, ids.shape[0], batch):
        lo = model(input_ids=ids[i:i + batch].to(dev),
                   attention_mask=mask[i:i + batch].to(dev)).logits
        pred = lo.argmax(-1).cpu()
        if perm is not None:
            pred = torch.as_tensor(perm)[pred]
        ok += int((pred == labels[i:i + batch]).sum())
    return 100.0 * ok / ids.shape[0]


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------

def param_counts(model) -> dict:
    """Three denominators, because a Transformer has no single honest one.

    The word-embedding table is 23.4M of 66.96M parameters and is not touched by
    any method here, so quoting a reduction against the total understates the
    operator and quoting it against the FFN alone overstates the model. Both are
    reported, plus the encoder without embeddings.
    """
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for p in model.distilbert.embeddings.parameters())
    ffn = sum(p.numel() for l in model.distilbert.transformer.layer
              for p in l.ffn.parameters())
    return {"total": total, "embedding": emb, "encoder": total - emb, "ffn": ffn}


# ---------------------------------------------------------------------------
# row capture
# ---------------------------------------------------------------------------

class Rows:
    """A fixed subsample of valid (example, token) positions.

    Shared by the dense and the compressed pass so the two are row-aligned;
    padding positions are excluded because their activations are an artefact of
    the batch shape and would enter the Gram with the same weight as real
    tokens.
    """

    def __init__(self, ids, mask, budget: int, seed: int):
        self.ids, self.mask = ids, mask
        valid = mask.bool()
        flat = valid.flatten().nonzero().flatten()
        rng = np.random.default_rng(seed + 9973)
        if flat.numel() > budget:
            sel = rng.choice(flat.numel(), size=budget, replace=False)
            flat = flat[torch.as_tensor(np.sort(sel))]
        keep = torch.zeros(valid.numel(), dtype=torch.bool)
        keep[flat] = True
        self.keep = keep.view(valid.shape)
        self.n = int(self.keep.sum())
        self.n_valid = int(valid.sum())

    def take(self, h: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return h[self.keep[lo:hi].to(h.device)]


@torch.inference_mode()
def sweep(model, rows: Rows, want: dict, batch: int = 64) -> dict:
    """One forward pass, gathering whatever `want` asks for.

    `want` maps a name to `(module, "in"|"out")`. Returns name -> `[R, D]` on the
    model's device, in row order.
    """
    dev = next(model.parameters()).device
    buf = {k: [] for k in want}
    cur = {}
    handles = []

    def mk(name, where):
        if where == "in":
            def hook(_m, inp):
                cur[name] = inp[0]
            return hook

        def hook(_m, _i, out):
            cur[name] = out
        return hook

    for name, (mod, where) in want.items():
        h = (mod.register_forward_pre_hook(mk(name, "in")) if where == "in"
             else mod.register_forward_hook(mk(name, "out")))
        handles.append(h)
    try:
        for lo in range(0, rows.ids.shape[0], batch):
            hi = min(lo + batch, rows.ids.shape[0])
            model(input_ids=rows.ids[lo:hi].to(dev),
                  attention_mask=rows.mask[lo:hi].to(dev))
            for name in want:
                buf[name].append(rows.take(cur[name], lo, hi).float())
            cur.clear()
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, 0) for k, v in buf.items()}


def dense_targets(model, rows: Rows, batch: int = 64) -> dict:
    """`a_dense[l]` (lin1's input) and `Y_dense[l]` (lin2's output) for all
    layers, from a single pass. Cached across the whole sweep since the dense
    network never changes."""
    layers = model.distilbert.transformer.layer
    want = {}
    for i, l in enumerate(layers):
        want[f"a{i}"] = (l.ffn.lin1, "in")
        want[f"Y{i}"] = (l.ffn.lin2, "out")
    got = sweep(model, rows, want, batch)
    return {i: (got[f"a{i}"], got[f"Y{i}"]) for i in range(len(layers))}


def comp_input(model, rows: Rows, layer: int, batch: int = 64) -> torch.Tensor:
    """`lin1`'s input at `layer` on the *compressed* path."""
    l = model.distilbert.transformer.layer[layer]
    return sweep(model, rows, {"a": (l.ffn.lin1, "in")}, batch)["a"]


# ---------------------------------------------------------------------------
# GRIDE
# ---------------------------------------------------------------------------

def gride_id(X: np.ndarray, *, max_points: int = 2200, max_dims: int = 2048,
             n_neighbors: int = 16, n_iters: int = 8, seed: int = 0) -> float:
    """Median over the estimator's neighbour-order scales, as everywhere else in
    this project. Subsampling coordinates is a Johnson-Lindenstrauss projection
    and leaves the estimate alone."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    X = X[~np.isnan(X).any(1)]
    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], max_points, replace=False)]
    if X.shape[1] > max_dims:
        X = X[:, rng.choice(X.shape[1], max_dims, replace=False)]
    X = np.ascontiguousarray(X)
    if X.shape[0] < 2 * n_neighbors + 2:
        n_neighbors = max(2, (X.shape[0] - 2) // 2)
    if X.shape[0] < 6:
        raise RuntimeError(f"cloud too small for GRIDE: {X.shape}")
    est = np.asarray(_gride(X, n_neighbors=n_neighbors, n_iters=n_iters), dtype=float)
    est = est[np.isfinite(est) & (est > 0)]
    if est.size == 0:
        raise RuntimeError("GRIDE returned no finite estimate; widths cannot be set")
    return float(np.median(est))


@torch.no_grad()
def row_space(G: torch.Tensor, rel_tol: float = 1e-6):
    lam, Q = torch.linalg.eigh(G.double())
    keep = lam > float(lam.max()) * rel_tol
    return Q[:, keep], lam[keep].clamp(min=0)


@torch.no_grad()
def unit_clouds(W1: torch.Tensor, X: torch.Tensor, U: torch.Tensor,
                max_coords: int = 2048, seed: int = 0) -> dict[str, np.ndarray]:
    """The three defensible clouds for one FFN layer, all returned.

    `neuron`      post-activation matrix transposed. Each of the 3072 units is a
                  point whose coordinates are its activations over calibration
                  rows. This is the measure Zador's bound is stated for and is
                  what the MLP study used, so it is the default.
    `filter_proj` rows of `W1` projected onto the row space of the layer input,
                  the part of each unit the data can actually see. The CNN
                  study's default.
    `weight`      rows of `W1` unprojected, ignoring what the data excites.
    """
    rng = np.random.default_rng(seed)
    R = U.shape[0]
    sel = rng.choice(R, size=min(R, max_coords), replace=False)
    sel = torch.as_tensor(np.sort(sel), device=U.device)
    neuron = U.index_select(0, sel).t().contiguous()

    Gx = (X.t() @ X).double()
    V, _ = row_space(Gx)
    fp = W1.double() @ V

    return {"neuron": neuron.cpu().numpy(),
            "filter_proj": fp.cpu().numpy(),
            "weight": W1.double().cpu().numpy()}


def widths_from_gride(dims, multiplier: float, orig, m_min: int = 4) -> list[int]:
    """THE RULE. `M_l = round(multiplier * d_l)`, clipped to `[m_min, N_l]`."""
    return [int(np.clip(round(multiplier * float(d)), m_min, int(n)))
            for d, n in zip(dims, orig)]


def multiplier_ladder(dims, orig) -> list[float]:
    """Geometric ladder up to the point the *widest* layer saturates.

    With `N = 3072` and `d` in the tens, the ceiling here is two orders of
    magnitude above the CNN study's, so the CNN ladder is extended rather than
    reused.
    """
    ratio = max(float(n) / max(float(d), 1e-9) for n, d in zip(orig, dims))
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0,
            32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0]
    out = [m for m in grid if m <= ratio]
    return out or [round(max(0.25, ratio), 3)]


# ---------------------------------------------------------------------------
# one compressible unit
# ---------------------------------------------------------------------------

class Unit:
    """Producer `lin1`, consumer `lin2`, and the cached sufficient statistics.

    `X` is the producer's input on the compressed path and `Y` the target on the
    dense path. `U` is rebuilt from `X` rather than stored by the caller, so a
    merging strategy that changes the producer can ask for the design matrix its
    own merged filters would have produced.
    """

    def __init__(self, lin1: nn.Linear, lin2: nn.Linear, act, X, Y,
                 ridge: float = 1e-4, seed: int = 0, chunk: int = 16384):
        self.lin1, self.lin2, self.act = lin1, lin2, act
        self.X, self.Y = X, Y
        self.ridge, self.seed, self.chunk = ridge, seed, chunk
        self._U = None
        self._stats = None
        self.gs = 1

    @property
    def C(self) -> int:
        return int(self.lin1.weight.shape[0])

    @property
    def n_unknowns(self) -> int:
        return self.C + 1

    @property
    def overdetermination(self) -> float:
        return self.X.shape[0] / self.n_unknowns

    def require_overdetermined(self, factor: float = 2.0):
        if self.overdetermination < factor:
            raise RuntimeError(
                f"{self.X.shape[0]} rows for {self.n_unknowns} unknowns "
                f"({self.overdetermination:.2f}x); raise --calib or --rows")

    @torch.no_grad()
    def acts_for(self, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty(self.X.shape[0], W.shape[0], device=self.X.device,
                          dtype=torch.float32)
        for i in range(0, self.X.shape[0], self.chunk):
            out[i:i + self.chunk] = self.act(
                torch.nn.functional.linear(self.X[i:i + self.chunk], W, b))
        return out

    @property
    def U(self) -> torch.Tensor:
        if self._U is None:
            self._U = self.acts_for(self.lin1.weight.detach(), self.lin1.bias.detach())
        return self._U

    def stats_for(self, U):
        return LA.gram_cross(U, self.Y, chunk=self.chunk)

    @property
    def stats(self):
        if self._stats is None:
            self._stats = self.stats_for(self.U)
        return self._stats

    def cols(self, keep) -> torch.Tensor:
        return torch.as_tensor(sorted(keep), device=self.X.device, dtype=torch.long)

    def ls(self, keep):
        G, P, us, ys, n = self.stats
        return LA.ls_consuming(G, P, us, ys, n, self.cols(keep), self.ridge)

    def rows_of(self, keep):
        W = self.lin1.weight.detach()
        b = self.lin1.bias.detach()
        k = self.cols(keep)
        return W[k].clone(), b[k].clone()

    def free(self):
        self._U = None
        self._stats = None


# ---------------------------------------------------------------------------
# strategies: (unit, m) -> (W1_new, b1_new, W2_new, b2_new)
# ---------------------------------------------------------------------------

def _consumer_cols(u: Unit, keep):
    V = u.lin2.weight.detach()
    return V[:, u.cols(keep)].clone(), u.lin2.bias.detach().clone()


def s_magnitude(u, m):
    keep = torch.topk(u.lin1.weight.detach().norm(dim=1), m).indices.tolist()
    return (*u.rows_of(keep), *_consumer_cols(u, keep))


def s_magnitude_ls(u, m):
    keep = torch.topk(u.lin1.weight.detach().norm(dim=1), m).indices.tolist()
    return (*u.rows_of(keep), *u.ls(keep))


def s_magnitude_prod(u, m):
    """Row norm of `W1` times column norm of `W2`, the saliency structured
    Transformer pruning usually uses. Not the paper's baseline; reported beside
    it so the paper's weaker magnitude rule is not a straw man."""
    imp = u.lin1.weight.detach().norm(dim=1) * u.lin2.weight.detach().norm(dim=0)
    keep = torch.topk(imp, m).indices.tolist()
    return (*u.rows_of(keep), *u.ls(keep))


def s_random(u, m):
    g = torch.Generator().manual_seed(u.seed)
    keep = torch.randperm(u.C, generator=g)[:m].tolist()
    return (*u.rows_of(keep), *_consumer_cols(u, keep))


def s_random_ls(u, m):
    g = torch.Generator().manual_seed(u.seed)
    keep = torch.randperm(u.C, generator=g)[:m].tolist()
    return (*u.rows_of(keep), *u.ls(keep))


def s_span_ls(u, m):
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, None, 1, m)
    return (*u.rows_of(keep), *u.ls(keep))


def s_omp_ls(u, m):
    """The method: output-targeted selection, then rebuild the consumer."""
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, P, 1, m)
    return (*u.rows_of(keep), *u.ls(keep))


@torch.no_grad()
def s_omp_ls_noint(u, m):
    """Same selection, same ridge, but the consumer's bias is frozen at its
    original value instead of being re-solved.

    The paper argues the free intercept is load-bearing because dropping a ReLU
    unit removes a non-negative contribution with non-zero mean. GELU is not
    non-negative, so the argument does not transfer and the term has to be
    re-measured rather than inherited. See RESEARCH R3.
    """
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, P, 1, m)
    cols = u.cols(keep)
    Gs = LA.f64(G)[cols][:, cols]
    b0 = LA.f64(u.lin2.bias.detach())
    rhs = LA.f64(P)[cols] - torch.outer(LA.f64(us)[cols], b0)
    lam = u.ridge * (Gs.diagonal().mean() + 1e-12)
    V = torch.linalg.solve(Gs + lam * torch.eye(Gs.shape[0], dtype=Gs.dtype,
                                                device=Gs.device), rhs)
    return (*u.rows_of(keep), V.t().float(), u.lin2.bias.detach().clone())


def s_omp_only(u, m):
    G, P, us, ys, n = u.stats
    keep = LA.group_select(G, P, 1, m)
    return (*u.rows_of(keep), *_consumer_cols(u, keep))


@torch.no_grad()
def backward_eliminate(G, P, us, ys, n, m: int, cand: list[int],
                       ridge: float = 1e-4) -> list[int]:
    """Exact backward elimination at `gs = 1`, masked instead of resized.

    Same cost as `common.linalg.forward_backward_select`, namely
    `dSSE_j = ||Theta_j||^2 / C_jj` with `C = (Phi^T Phi + reg)^-1`, and the same
    rank-one downdate of both `C` and `Theta`. The production kernel scans
    candidates with a Python loop and one `linalg.solve` per surviving unit per
    step, which at 3072 units is millions of microscopic GPU launches and does
    not finish. At `gs = 1` the whole scan is one vectorised division, and after
    a downdate the eliminated row and column of `C` are exactly zero, so the
    matrix can be masked rather than reallocated.

    `selftest.py` checks this returns the same indices as the production kernel.
    """
    dev = G.device
    cols = torch.as_tensor(sorted(cand), device=dev, dtype=torch.long)
    Gs, Ps, uss = LA.f64(G)[cols][:, cols], LA.f64(P)[cols], LA.f64(us)[cols]
    F = Gs.shape[0]
    if m >= F:
        return sorted(cand)
    A = torch.zeros(F + 1, F + 1, dtype=Gs.dtype, device=dev)
    A[:F, :F], A[:F, F], A[F, :F], A[F, F] = Gs, uss, uss, float(n)
    lam = ridge * (Gs.diagonal().mean() + 1e-12)
    reg = torch.zeros(F + 1, dtype=Gs.dtype, device=dev)
    reg[:F] = lam
    Cinv = torch.linalg.inv(A + torch.diag(reg))
    Theta = Cinv @ torch.cat([Ps, LA.f64(ys).unsqueeze(0)], 0)

    alive = torch.ones(F, dtype=torch.bool, device=dev)
    tiny = torch.tensor(1e-300, dtype=Cinv.dtype, device=dev)
    for _ in range(F - m):
        d = Cinv.diagonal()[:F].clamp_min(tiny)
        cost = (Theta[:F] ** 2).sum(1) / d
        cost = cost.masked_fill(~alive, float("inf"))
        j = int(torch.argmin(cost))
        piv = Cinv[j, j].clamp_min(tiny)
        col = Cinv[:, j].clone()
        Theta = Theta - torch.outer(col, Theta[j] / piv)
        Cinv = Cinv - torch.outer(col, Cinv[j] / piv)
        alive[j] = False
    return sorted(int(cols[i]) for i in alive.nonzero().flatten().tolist())


def s_fwdbwd_ls(u, m, over: float = 3.0):
    """OMP overshoot, then exact backward elimination. Forward greedy never
    reconsiders, so a unit taken early to explain a direction can be redundant
    once later ones arrive; this is what removes it."""
    G, P, us, ys, n = u.stats
    K = int(min(u.C, max(m, round(over * m))))
    cand = LA.group_select(G, P, 1, K) if K < u.C else list(range(u.C))
    keep = backward_eliminate(G, P, us, ys, n, m, cand, u.ridge)
    return (*u.rows_of(keep), *u.ls(keep))


@torch.no_grad()
def _ot_plan(u: Unit, m: int) -> torch.Tensor:
    """Intra-Fusion's plan: keep the top-m units by L1 mass, transport the rest
    onto them under a cosine cost, column-normalise."""
    try:
        import ot as pot
    except Exception:
        pot = None
    S = u.lin1.weight.detach()
    imp = S.abs().sum(1)
    keep = torch.topk(imp, m).indices.sort().values
    mask = torch.zeros(u.C, dtype=torch.bool, device=S.device)
    mask[keep] = True
    drop = torch.nonzero(~mask).flatten()

    Sn = S / (S.norm(dim=1, keepdim=True) + 1e-12)
    T = torch.zeros(u.C, m, dtype=torch.float64, device=S.device)
    T[keep, torch.arange(m, device=S.device)] = imp[keep].double()
    if drop.numel():
        cost = (1.0 - Sn[drop] @ Sn[keep].t()).double().cpu().numpy()
        a = (imp[drop] / imp[drop].sum()).double().cpu().numpy()
        bb = (imp[keep] / imp[keep].sum()).double().cpu().numpy()
        if pot is not None:
            plan = pot.emd(a, bb, np.ascontiguousarray(cost), numItermax=500000)
        else:
            plan = np.zeros_like(cost)
            plan[np.arange(len(a)), cost.argmin(1)] = a
        T[drop] = torch.as_tensor(plan, device=S.device) * imp[drop].sum().double()
    T = T / (T.sum(0, keepdim=True) + 1e-12)
    return T.to(S.dtype)


def _mix(u: Unit, T: torch.Tensor):
    W = T.t() @ u.lin1.weight.detach()
    b = T.t() @ u.lin1.bias.detach()
    V = u.lin2.weight.detach() @ T
    return W, b, V, u.lin2.bias.detach().clone()


def s_ot_intrafusion(u, m):
    return _mix(u, _ot_plan(u, m))


def s_ot_then_ls(u, m):
    W, b, _, _ = _mix(u, _ot_plan(u, m))
    Um = u.acts_for(W, b)
    G, P, us, ys, n = u.stats_for(Um)
    cols = torch.arange(m, device=Um.device)
    V, c = LA.ls_consuming(G, P, us, ys, n, cols, u.ridge)
    del Um
    return W, b, V, c


def s_kmeans_merge(u, m):
    from sklearn.cluster import KMeans
    S = u.lin1.weight.detach().float().cpu().numpy()
    lab = KMeans(n_clusters=m, n_init=1 if m > 128 else 3,
                 random_state=u.seed).fit(S).labels_
    T = torch.zeros(u.C, m, device=u.X.device, dtype=u.lin1.weight.dtype)
    w = u.U.abs().mean(0) + 1e-9
    for k in range(m):
        idx = np.nonzero(lab == k)[0]
        if idx.size == 0:
            continue
        ii = torch.as_tensor(idx, device=T.device)
        T[ii, k] = w[ii] / w[ii].sum()
    for k in (T.sum(0) == 0).nonzero().flatten().tolist():
        T[int(torch.argmax(w)), k] = 1.0
    return _mix(u, T)


@torch.no_grad()
def _contribution_gram(u: Unit):
    """`A_ij = <u_i v_i^T, u_j v_j^T> = (U^T U)_ij (V^T V)_ij`.

    Unit `i`'s entire contribution to the consumer's output is the rank-one
    matrix `u_i v_i^T`, so every quantity \\textsc{ThiNet} needs, both its greedy
    score and its per-channel rescale, is a bilinear form in this one `C x C`
    matrix. Building it once turns the whole method into `O(C^2)` bookkeeping.
    """
    U, V = u.U, u.lin2.weight.detach()
    G = torch.zeros(u.C, u.C, dtype=torch.float64, device=U.device)
    for i in range(0, U.shape[0], u.chunk):
        Uc = U[i:i + u.chunk]
        G += (Uc.t() @ Uc).double()
    return G * (V.t().double() @ V.double())


@torch.no_grad()
def _thinet_select(A: torch.Tensor, m: int) -> list[int]:
    """\\textsc{ThiNet}'s Algorithm 1, verbatim: grow the *removed* set greedily,
    at each step deleting the unit that keeps the norm of the discarded sum
    smallest. Not forward selection of what to keep, and not backward
    elimination against a target; those are different algorithms."""
    C = A.shape[0]
    removed, acc = [], torch.zeros(C, dtype=A.dtype, device=A.device)
    gone = torch.zeros(C, dtype=torch.bool, device=A.device)
    for _ in range(C - m):
        score = (A.diagonal() + 2.0 * acc).masked_fill(gone, float("inf"))
        j = int(torch.argmin(score))
        removed.append(j)
        gone[j] = True
        acc = acc + A[j]
    return sorted(int(i) for i in (~gone).nonzero().flatten().tolist())


@torch.no_grad()
def _thinet_rescale(u: Unit, A: torch.Tensor, b: torch.Tensor, keep):
    """One scale per surviving unit, applied to the consumer's existing columns:
    `min_s || sum_i s_i u_i v_i^T - Y ||_F^2`. No intercept, no re-solve of the
    consumer's full matrix. This is what separates \\textsc{ThiNet}'s correction
    from ours."""
    k = u.cols(keep)
    As, bs = A[k][:, k], b[k]
    lam = 1e-8 * (As.diagonal().mean().abs() + 1e-30)
    s = torch.linalg.solve(As + lam * torch.eye(len(k), dtype=As.dtype,
                                                device=As.device), bs)
    V = u.lin2.weight.detach()[:, k]
    return (V * s.to(V.dtype).unsqueeze(0)).contiguous(), u.lin2.bias.detach().clone()


def _thinet_b(u: Unit, A: torch.Tensor, dense_target: bool) -> torch.Tensor:
    """`b_i = <u_i v_i^T, Y>`. \\textsc{ThiNet}'s own target is the consumer's
    current output `Z = sum_j u_j v_j^T`, giving `b = A 1`; the dense target
    gives `b_i = v_i . (U^T Y)_i`."""
    if not dense_target:
        return A.sum(1)
    G, P, us, ys, n = u.stats
    V = u.lin2.weight.detach()
    return (LA.f64(V).t() * LA.f64(P)).sum(1)


def s_thinet(u, m):
    """The faithful control: all three of \\textsc{ThiNet}'s choices at once."""
    A = _contribution_gram(u)
    b = _thinet_b(u, A, dense_target=False)
    keep = _thinet_select(A, m)
    return (*u.rows_of(keep), *_thinet_rescale(u, A, b, keep))


@torch.no_grad()
def s_thinet_densetgt(u, m):
    """\\textsc{ThiNet}'s subtractive greedy and its rescale, but scored against
    the dense target. Isolates what the target choice alone is worth.

    The objective is still stated on the removed set, and that is where the
    algebra has to be done carefully. Minimising `||Z_S - Y||^2` is not
    minimising `||Z_T - Y||^2`; with `R = Z_full - Y` the part of the current
    output that overshoots the target,

        ||Z_S - Y||^2 = ||R - Z_T||^2 ,

    so adding `j` to `T` costs `A_jj + 2<u_j v_j, Z_T> - 2<u_j v_j, R>`. At
    `Y = Z_full` the last term vanishes and this reduces to plain
    \\textsc{ThiNet}, which is the check that the generalisation is the right
    one.
    """
    A = _contribution_gram(u)
    b_dense = _thinet_b(u, A, dense_target=True)
    r = A.sum(1) - b_dense                       # <u_j v_j, Z_full - Y>
    C = u.C
    acc = torch.zeros(C, dtype=A.dtype, device=A.device)
    gone = torch.zeros(C, dtype=torch.bool, device=A.device)
    for _ in range(C - m):
        score = (A.diagonal() + 2.0 * acc - 2.0 * r).masked_fill(gone, float("inf"))
        j = int(torch.argmin(score))
        gone[j] = True
        acc = acc + A[j]
    keep = sorted(int(i) for i in (~gone).nonzero().flatten().tolist())
    return (*u.rows_of(keep), *_thinet_rescale(u, A, b_dense, keep))


def s_thinet_lsfull(u, m):
    """\\textsc{ThiNet}'s selection, our consumer refit. Isolates what the
    correction alone is worth, holding selection fixed."""
    A = _contribution_gram(u)
    keep = _thinet_select(A, m)
    return (*u.rows_of(keep), *u.ls(keep))


STRATEGIES = {
    "magnitude": s_magnitude,
    "magnitude_ls": s_magnitude_ls,
    "magnitude_prod_ls": s_magnitude_prod,
    "random": s_random,
    "random_ls": s_random_ls,
    "span_ls": s_span_ls,
    "omp": s_omp_only,
    "omp_ls": s_omp_ls,
    "omp_ls_noint": s_omp_ls_noint,
    "fwdbwd_ls": s_fwdbwd_ls,
    "ot_intrafusion": s_ot_intrafusion,
    "ot_then_ls": s_ot_then_ls,
    "kmeans_merge": s_kmeans_merge,
    "thinet": s_thinet,
    "thinet_densetgt": s_thinet_densetgt,
    "thinet_lsfull": s_thinet_lsfull,
}
MERGE_FAMILY = {"ot_intrafusion", "ot_then_ls", "kmeans_merge"}


# ---------------------------------------------------------------------------
# the lock-step walk
# ---------------------------------------------------------------------------

def _replace_ffn(layer, W1, b1, W2, b2):
    dev, dt = W1.device, layer.ffn.lin1.weight.dtype
    m = W1.shape[0]
    l1 = nn.Linear(W1.shape[1], m).to(dev, dt)
    l2 = nn.Linear(m, W2.shape[0]).to(dev, dt)
    with torch.no_grad():
        l1.weight.copy_(W1)
        l1.bias.copy_(b1)
        l2.weight.copy_(W2)
        l2.bias.copy_(b2)
    for p in list(l1.parameters()) + list(l2.parameters()):
        p.requires_grad_(False)
    layer.ffn.lin1, layer.ffn.lin2 = l1, l2


def compress(dense, rows: Rows, widths: list[int], strategy: str,
             dense_cache: dict, *, ridge: float = 1e-4, seed: int = 0,
             target: str = "dense", sequential: bool = True,
             layers=None, batch: int = 64, verbose=lambda s: None):
    """Walk the six blocks in order, compressing each against the dense target.

    The design matrix comes from the compressed path so every fit sees the error
    inherited from the blocks already done; the target comes from the dense path
    so that error is absorbed rather than compounded. `sequential=False` feeds
    the dense path's input instead, which is the control that isolates what the
    absorption is worth.

    `target="residual"` adds `a_dense - a_comp` to the target, asking the refit
    to also correct the drift that arrived on the residual stream. See RESEARCH
    R2; this is not in the paper because a ResNet's internal refit was never
    given the option.
    """
    import copy
    comp = copy.deepcopy(dense)
    comp.eval()
    fn = STRATEGIES[strategy]
    todo = list(range(N_LAYERS)) if layers is None else sorted(layers)
    info = []
    for i in todo:
        m = int(widths[i])
        a_dense, Y = dense_cache[i]
        X = a_dense if not sequential else comp_input(comp, rows, i, batch)
        Yt = Y
        if target == "residual" and sequential:
            Yt = Y + (a_dense - X)
        layer = comp.distilbert.transformer.layer[i]
        u = Unit(layer.ffn.lin1, layer.ffn.lin2, layer.ffn.activation,
                 X, Yt, ridge=ridge, seed=seed)
        u.require_overdetermined(2.0)
        if m >= u.C:
            info.append({"layer": i, "m": u.C, "kept_all": True})
            u.free()
            continue
        W1, b1, W2, b2 = fn(u, m)
        assert W1.shape[0] == m and W2.shape[1] == m, \
            f"strategy {strategy} returned {W1.shape[0]}/{W2.shape[1]}, expected {m}"
        _replace_ffn(layer, W1, b1, W2, b2)
        info.append({"layer": i, "m": m, "kept_all": False})
        verbose(f"    layer {i}: {u.C} -> {m}")
        u.free()
        del X, Yt
        torch.cuda.empty_cache()
    return comp, info


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------

def save_artifact(model, path: Path, meta: dict):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path), metadata={"meta": json.dumps(meta)})
