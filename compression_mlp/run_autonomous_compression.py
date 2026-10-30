#!/usr/bin/env python3
"""Autonomous zero-shot neuron-compression research driver.

AIM
---
Take a *converged* MLP checkpoint from `trainer/checkpoint/` and shrink its
hidden layers with **no fine-tuning at all** (zero backward passes, zero
`optimizer.step()`), landing within +/- `TOL` percentage points of the dense
model's test accuracy.

GOLDEN CONSTRAINT (non-negotiable)
----------------------------------
Every compressed layer width comes **only** from that layer's GRIDE intrinsic
dimension:

        M_l = round(multiplier * d_l),  d_l = GRIDE ID of layer l

`multiplier` is swept over a grid whose range is derived at runtime from the
discovered model's widths. No hand-set widths, no PCA-99% rank, no arbitrary
sparsity ratios. `Strategy.apply()` receives `target_m` and must return exactly
`target_m` neurons.

This mirrors the paper's thesis: Zador's quantization bound E_m ~ m^{-1/d} makes
the achievable width a function of the *intrinsic* dimension d, so d is what
sets the budget.

EVERYTHING is discovered from the filesystem: which checkpoints exist, which is
the largest MLP, which dataset it belongs to, its dense accuracy, its per-layer
GRIDE dimensions, and the multiplier range.

Phases
------
1. Sweep the 10 seed strategies x the GRIDE multiplier grid; log every run.
2. If the aim is unmet, write a diagnostic report and dynamically load novel
   strategies from `novel_strategies/*.py` (up to 3 rounds).
3. Emit `final_report.md` with a ranked table and the GRIDE foundation.

Usage
-----
    python3 run_autonomous_compression.py                 # largest MLP found
    python3 run_autonomous_compression.py --checkpoint cifar10_mlp3_oct_bn
    python3 run_autonomous_compression.py --all-mlps
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from safetensors.torch import load_file, save_file  # noqa: E402

from models.mlp import MLP  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from utils.config import CHECKPOINT_DIR, DATASETS  # noqa: E402
from utils.data_loader import get_dataloaders  # noqa: E402

try:
    from intrinsic_rank.id_estimator import gride as _gride_estimator
except Exception as exc:  # pragma: no cover - the golden constraint depends on this
    raise RuntimeError(
        "GRIDE is mandatory: could not import intrinsic_rank.id_estimator.gride. "
        "Layer widths may not be set by any other means."
    ) from exc

try:
    import ot as pot  # POT: exact EMD for the Intra-Fusion baseline
except Exception:  # pragma: no cover
    pot = None

RESULTS_DIR = HERE / "results"
NOVEL_DIR = HERE / "novel_strategies"
CSV_PATH = RESULTS_DIR / "compression_results.csv"
PROGRESS_PATH = HERE / "PROGRESS.md"
REPORT_PATH = HERE / "final_report.md"
REQUEST_PATH = HERE / "adaptive_research_request.txt"

CSV_FIELDS = [
    "timestamp", "model_name", "strategy_name", "multiplier",
    "M1", "M2", "M3", "compressed_acc", "acc_drop", "target_hit",
    # extras (beyond the required schema) that make the log self-describing
    "dataset", "phase", "id_space", "gride_dims", "orig_widths", "widths",
    "params", "param_reduction_pct", "original_acc", "calib", "seconds",
]

log = logging.getLogger("autoc")


# ===========================================================================
# 1. DISCOVERY -- nothing about the model or dataset is hardcoded
# ===========================================================================

@dataclass
class Discovered:
    """A checkpoint that was successfully matched to an architecture + dataset."""
    path: Path
    dataset: str
    model_name: str
    model: nn.Module
    n_params: int


def _split_ckpt_stem(stem: str) -> tuple[str | None, str | None]:
    """`cifar10_mlp3_quad_bn` -> ('CIFAR10', 'mlp3_quad_bn') using the project's
    own dataset registry as the vocabulary (so nothing is hardcoded here)."""
    for ds in DATASETS:
        prefix = ds.lower() + "_"
        if stem.startswith(prefix):
            return ds, stem[len(prefix):]
    return None, None


def discover_mlp_checkpoints(ckpt_dir: Path) -> list[tuple[Path, str]]:
    """Every `.safetensors` whose model part names an MLP, largest file first."""
    found = []
    for p in sorted(ckpt_dir.glob("*.safetensors")):
        _, model_name = _split_ckpt_stem(p.stem)
        if model_name and model_name.startswith("mlp"):
            found.append((p, model_name))
    found.sort(key=lambda t: t[0].stat().st_size, reverse=True)
    return found


def load_discovered(path: Path, model_name: str, device: str) -> Discovered:
    """Try every dataset in the registry until the state dict loads *strictly*.

    A wrong dataset gives a shape mismatch (different in_dim / num_classes), so
    a strict load is a genuine identification test, not a filename lookup. The
    filename's prefix only *orders* the candidates.
    """
    state = load_file(str(path), device="cpu")
    hinted, _ = _split_ckpt_stem(path.stem)
    candidates = ([hinted] if hinted else []) + [d for d in DATASETS if d != hinted]

    errors = {}
    for ds in candidates:
        try:
            model = build_model(model_name, ds)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            errors[ds] = str(exc).splitlines()[0][:120]
            continue
        try:  # the dataset must also actually be loadable on this machine
            get_dataloaders(ds, batch_size=8, num_workers=0, augment=False)
        except Exception as exc:
            errors[ds] = f"weights fit but data unavailable: {exc}"
            continue
        n_params = sum(p.numel() for p in model.parameters())
        return Discovered(path, ds, model_name, model.to(device).eval(), n_params)

    raise RuntimeError(f"Could not match {path.name} to any dataset. Tried: {errors}")


# ===========================================================================
# 2. BATCHNORM FOLDING + A PURE (W, b) STACK
# ===========================================================================

@torch.no_grad()
def fold_bn(model: MLP) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Fold BatchNorm1d into the preceding Linear so a neuron is exactly (w, b).

        w <- w * gamma/sqrt(var+eps),   b <- (b-mu)*gamma/sqrt(var+eps) + beta

    Returns L+1 (weight, bias) pairs: L hidden layers then the classifier.
    """
    Ws, bs = [], []
    for lin, bn in zip(model.layers, model.bns):
        w, b = lin.weight.detach().clone(), lin.bias.detach().clone()
        if isinstance(bn, nn.BatchNorm1d):
            s = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            b = (b - bn.running_mean.detach()) * s + bn.bias.detach()
            w = w * s.unsqueeze(1)
        Ws.append(w)
        bs.append(b)
    Ws.append(model.classifier.weight.detach().clone())
    bs.append(model.classifier.bias.detach().clone())
    return Ws, bs


def build_from_stack(Ws, bs, in_dim: int, num_classes: int, activation: str) -> MLP:
    """Rebuild a BN-free MLP from a folded (and possibly narrowed) weight stack."""
    hidden = [int(w.shape[0]) for w in Ws[:-1]]
    m = MLP(in_dim=in_dim, hidden_dims=hidden, num_classes=num_classes,
            activation=activation, batch_norm=False)
    with torch.no_grad():
        for i, (w, b) in enumerate(zip(Ws[:-1], bs[:-1])):
            m.layers[i].weight.copy_(w)
            m.layers[i].bias.copy_(b)
        m.classifier.weight.copy_(Ws[-1])
        m.classifier.bias.copy_(bs[-1])
    return m.eval()


def act_name(model: MLP) -> str:
    return {"ReLU": "relu", "Tanh": "tanh", "GELU": "gelu"}[type(model.act).__name__]


@torch.no_grad()
def stack_forward(X: torch.Tensor, Ws, bs, upto: int) -> torch.Tensor:
    """Post-activation output of hidden layer `upto-1` (X itself when upto==0)."""
    h = X
    for j in range(upto):
        h = torch.relu(h @ Ws[j].t() + bs[j])
    return h


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model = model.to(device).eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


# ===========================================================================
# 3. GRIDE -- the ONLY legal source of layer widths
# ===========================================================================

def _subsample(X: np.ndarray, max_points: int, max_dims: int, rng) -> np.ndarray:
    n, d = X.shape
    if n > max_points:
        X = X[rng.choice(n, max_points, replace=False)]
    if d > max_dims:  # random coordinate projection preserves distances (JL)
        X = X[:, rng.choice(d, max_dims, replace=False)]
    return np.ascontiguousarray(X, dtype=np.float64)


def gride_id(X: np.ndarray, *, max_points: int = 2200, max_dims: int = 2048,
             n_neighbors: int = 16, n_iters: int = 8, seed: int = 0) -> float:
    """Scalar GRIDE intrinsic dimension = median over the estimator's scales.

    GRIDE (Denti et al. 2022) returns one estimate per neighbour-order scale;
    the median is the robust plateau value and is what we feed to the Zador
    width rule.
    """
    rng = np.random.default_rng(seed)
    Xs = _subsample(X, max_points, max_dims, rng)
    if Xs.shape[0] < 2 * n_neighbors + 2:
        n_neighbors = max(2, (Xs.shape[0] - 2) // 2)
    est = _gride_estimator(Xs, n_neighbors=n_neighbors, n_iters=n_iters)
    est = np.asarray(est, dtype=float)
    est = est[np.isfinite(est) & (est > 0)]
    if est.size == 0:
        raise RuntimeError("GRIDE returned no finite estimate -- cannot set widths.")
    return float(np.median(est))


def layer_gride_dims(A_list: list[torch.Tensor], id_space: str, seed: int) -> list[float]:
    """GRIDE ID of every hidden layer.

    id_space='neuron'      -> the *neuron cloud*: each neuron is a point whose
                              coordinates are its activations over calibration
                              samples. This is the measure mu_l that Zador's
                              bound is stated for, so it is the default.
    id_space='activation'  -> the *activation manifold*: each sample is a point
                              in R^{h_l}. The paper's r_l.
    """
    dims = []
    for A in A_list:
        M = A.detach().float().cpu().numpy()
        cloud = M.T if id_space == "neuron" else M
        dims.append(gride_id(cloud, seed=seed))
    return dims


def multiplier_grid(dims: list[float], widths: list[int]) -> list[float]:
    """Derived at runtime: geometric ladder from 1.5x up to the multiplier at
    which the *last* layer saturates (M_l >= N_l for every l).

    The ceiling is `max(N_l / d_l)`, not `min`. Using the minimum stops the sweep
    as soon as the single tightest layer fills up, which on a model with an
    uneven GRIDE profile (e.g. a CIFAR-100 MLP whose last layer has d=39 in 512
    units) truncates the grid long before the *other* layers have stopped being
    compressed. Widths stay legal past that point because
    `widths_from_gride` clips each layer at its own N_l -- a saturated layer is
    simply left uncompressed while the rest keep growing.
    """
    ratio_to_full = max(n / d for n, d in zip(widths, dims))
    grid = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]
    out = [m for m in grid if m <= ratio_to_full]
    if not out:
        out = [round(max(1.0, ratio_to_full), 2)]
    return out


def widths_from_gride(dims: list[float], multiplier: float, orig: list[int],
                      m_min: int = 4) -> list[int]:
    """THE GOLDEN RULE. M_l = round(multiplier * d_l), clipped to [m_min, N_l]."""
    return [int(np.clip(round(multiplier * d), m_min, n)) for d, n in zip(dims, orig)]


# ===========================================================================
# 4. LINEAR-ALGEBRA CORE (all closed form -- no gradients anywhere)
# ===========================================================================

def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


def ls_consuming(A_keep: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-4
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form least squares for the CONSUMING layer, **with an intercept**.

        min_{V',c'}  || [A_keep , 1] [V' ; c']^T  -  Y ||_F

    `Y` is the *original* network's pre-activation at the next layer, so the fit
    simultaneously (a) compensates for the neurons removed here and (b) corrects
    error inherited from already-compressed upstream layers.

    The intercept is essential and is what the plain reconstruction map misses:
    a removed ReLU neuron contributes a non-zero *mean* downstream, and only a
    free bias can absorb it. The intercept column is never shrunk by the ridge.
    The ridge is *relative* (scaled by the mean Gram diagonal) so it regularises
    consistently across very different M.
    """
    A, Y = _f64(A_keep), _f64(Y)
    T, M = A.shape
    Phi = torch.cat([A, torch.ones(T, 1, dtype=A.dtype, device=A.device)], 1)
    G = Phi.t() @ Phi
    lam = ridge * (G.diagonal()[:M].mean() + 1e-12)
    reg = torch.zeros(M + 1, dtype=A.dtype, device=A.device)
    reg[:M] = lam
    Theta = torch.linalg.solve(G + torch.diag(reg), Phi.t() @ Y)  # [M+1, out]
    return Theta[:M].t().float(), Theta[M].float()


def pivoted_cholesky_select(G: torch.Tensor, m: int) -> list[int]:
    """Greedy column-subset selection on the Gram matrix G = A^T A.

    Pivoted Cholesky of A^T A visits columns in exactly the order that
    column-pivoted QR of A does, but costs O(N*m) instead of O(T*N^2) -- the
    difference between seconds and minutes at N=2048, T=8192.
    """
    G = _f64(G)
    N = G.shape[0]
    if m >= N:
        return list(range(N))
    d = G.diagonal().clone()
    d0 = d.clone()
    L = torch.zeros(m, N, dtype=G.dtype, device=G.device)
    piv: list[int] = []
    chosen = torch.zeros(N, dtype=torch.bool, device=G.device)
    tol = 1e-12 * float(d0.max())
    for k in range(m):
        dm = d.masked_fill(chosen, -float("inf"))
        j = int(torch.argmax(dm))
        if dm[j] <= tol:  # numerically rank-deficient: fall back to raw energy
            rem = torch.nonzero(~chosen).flatten()
            j = int(rem[torch.argmax(d0[rem])])
            piv.append(j)
            chosen[j] = True
            continue
        piv.append(j)
        chosen[j] = True
        v = G[j] - L[:k, j] @ L[:k]
        L[k] = v / torch.sqrt(d[j])
        d = torch.clamp(d - L[k] ** 2, min=0.0)
    return sorted(piv)


def omp_select(G: torch.Tensor, P: torch.Tensor, m: int) -> list[int]:
    """Output-targeted greedy neuron selection (matrix OMP).

    G = A^T A [N,N], P = A^T Y [N,out]. At each step pick the neuron that most
    reduces || Y - A_S (A_S)^+ Y ||_F^2, i.e. argmax_j ||P_res[j]||^2 / d[j].

    Unlike pivoted QR/Cholesky (which spans the *activation* space), this spans
    the space the next layer actually reads. Same pivoted-Cholesky recursion,
    run on P as well, so it stays O(N*m*out).
    """
    G, P = _f64(G), _f64(P)
    N = G.shape[0]
    if m >= N:
        return list(range(N))
    d = G.diagonal().clone()
    d0 = d.clone()
    L = torch.zeros(m, N, dtype=G.dtype, device=G.device)
    Pres = P.clone()
    piv: list[int] = []
    chosen = torch.zeros(N, dtype=torch.bool, device=G.device)
    tol = 1e-12 * float(d0.max())
    for k in range(m):
        safe = torch.clamp(d, min=tol)
        score = (Pres ** 2).sum(1) / safe
        score = score.masked_fill(chosen | (d <= tol), -float("inf"))
        j = int(torch.argmax(score))
        if not torch.isfinite(score[j]):
            rem = torch.nonzero(~chosen).flatten()
            j = int(rem[torch.argmax(d0[rem])])
            piv.append(j)
            chosen[j] = True
            continue
        piv.append(j)
        chosen[j] = True
        v = G[j] - L[:k, j] @ L[:k]
        Lk = v / torch.sqrt(d[j])
        # incremental residual update: project Y onto the newly added direction
        zk = Pres[j] / torch.sqrt(d[j])
        Pres = Pres - torch.outer(Lk, zk)
        L[k] = Lk
        d = torch.clamp(d - Lk ** 2, min=0.0)
    return sorted(piv)


def merge_by_labels(labels: np.ndarray, W, b, V, weights: torch.Tensor, m: int):
    """Collapse neurons sharing a cluster label into one neuron.

    Incoming = importance-weighted mean (the cluster's barycentre); outgoing =
    **sum** (so that if the cluster's members co-activate identically the layer
    output is exactly preserved: sum_i v_i a_i = (sum_i v_i) a).
    """
    Wn = torch.zeros(m, W.shape[1], dtype=W.dtype, device=W.device)
    bn = torch.zeros(m, dtype=b.dtype, device=b.device)
    Vn = torch.zeros(V.shape[0], m, dtype=V.dtype, device=V.device)
    lab = torch.as_tensor(labels, device=W.device, dtype=torch.long)
    for k in range(m):
        idx = torch.nonzero(lab == k).flatten()
        if idx.numel() == 0:  # keep the contract: exactly m neurons out
            j = int(torch.argmax(weights))
            Wn[k], bn[k], Vn[:, k] = W[j], b[j], 0.0
            continue
        w = weights[idx]
        w = w / w.sum() if float(w.sum()) > 0 else torch.full_like(w, 1.0 / idx.numel())
        Wn[k] = (w.unsqueeze(1) * W[idx]).sum(0)
        bn[k] = (w * b[idx]).sum()
        Vn[:, k] = V[:, idx].sum(1)
    return Wn, bn, Vn


def kmeans_labels(X: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    m = min(m, X.shape[0])
    km = KMeans(n_clusters=m, n_init=1 if m > 256 else 3, random_state=seed).fit(X)
    return km.labels_


def relabel_dense(labels: np.ndarray, m: int) -> np.ndarray:
    """Force labels into exactly {0..m-1} with every id used at most once."""
    uniq = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
    return np.array([uniq[v] for v in labels], dtype=int)


# ===========================================================================
# 5. STRATEGY FRAMEWORK
# ===========================================================================

@dataclass
class LayerContext:
    """Everything a strategy may look at when compressing hidden layer `idx`."""
    idx: int
    W: torch.Tensor       # [N, din]   producing weights (BN already folded)
    b: torch.Tensor       # [N]
    V: torch.Tensor       # [out, N]   consuming weights
    c: torch.Tensor       # [out]
    H: torch.Tensor       # [T, din]   layer input, through ALREADY-compressed layers
    A: torch.Tensor       # [T, N]     post-ReLU activations of this layer
    Y: torch.Tensor       # [T, out]   ORIGINAL net's pre-activation at layer idx+1
    ridge: float = 1e-4
    seed: int = 0
    metric: torch.Tensor | None = None   # [out, out] output-Jacobian (Fisher) metric on Y

    @property
    def N(self) -> int:
        return int(self.W.shape[0])

    def acts_for(self, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.H @ W.t() + b)

    def importance(self, p: int = 1) -> torch.Tensor:
        return torch.linalg.vector_norm(torch.cat([self.W, self.b.unsqueeze(1)], 1), ord=p, dim=1)

    def gram(self) -> torch.Tensor:
        A = _f64(self.A)
        return A.t() @ A

    def cross(self) -> torch.Tensor:
        return _f64(self.A).t() @ _f64(self.Y)


class Strategy:
    """Compress ONE hidden layer to EXACTLY `target_m` neurons.

    Returns (W_new [M,din], b_new [M], V_new [out,M], c_new [out]).
    `target_m` is handed down from the GRIDE rule and must be obeyed exactly.
    """
    name = "base"
    note = ""

    def apply(self, ctx: LayerContext, target_m: int):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"


# --------------------------------------------------------------------------
# The 10 seed strategies
# --------------------------------------------------------------------------

class S01_DirectLeastSquares(Strategy):
    name = "01_direct_ls_svd"
    note = "SVD/leverage neuron selection + closed-form LS on the consuming layer"

    def apply(self, ctx, m):
        # right singular vectors of A == eigenvectors of A^T A, at O(N^3) not O(T*N^2)
        lam, Q = torch.linalg.eigh(ctx.gram())
        k = min(m, Q.shape[1])
        lev = (Q[:, -k:] ** 2).sum(1)            # rank-m leverage score per neuron
        keep = torch.topk(lev, m).indices.sort().values.tolist()
        Vn, cn = ls_consuming(ctx.A[:, keep], ctx.Y, ctx.ridge)
        return ctx.W[keep], ctx.b[keep], Vn, cn


class S02_TopMRowSampling(Strategy):
    name = "02_topm_l2_norm"
    note = "classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows"

    def apply(self, ctx, m):
        keep = torch.topk(ctx.importance(2), m).indices.sort().values.tolist()
        return ctx.W[keep], ctx.b[keep], ctx.V[:, keep], ctx.c.clone()


class S03_ActivationPatternClustering(Strategy):
    name = "03_activation_pattern_kmeans"
    note = "binarise ReLU masks, k-means the neurons, merge each cluster"

    def apply(self, ctx, m):
        rng = np.random.default_rng(ctx.seed)
        A = ctx.A
        cols = rng.choice(A.shape[0], min(512, A.shape[0]), replace=False)
        B = (A[cols] > 0).float().t().cpu().numpy()          # [N, 512] binary masks
        labels = relabel_dense(kmeans_labels(B, m, ctx.seed), m)
        w = ctx.A.mean(0) + 1e-9
        Wn, bn, Vn = merge_by_labels(labels, ctx.W, ctx.b, ctx.V, w, m)
        return Wn, bn, Vn, ctx.c.clone()


class S04_CosineOTMerging(Strategy):
    name = "04_cosine_ot_intrafusion"
    note = "Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised"

    def apply(self, ctx, m):
        imp = ctx.importance(1)
        keep = torch.topk(imp, m).indices.sort().values
        mask = torch.zeros(ctx.N, dtype=torch.bool, device=imp.device)
        mask[keep] = True
        drop = torch.nonzero(~mask).flatten()

        S = torch.cat([ctx.W, ctx.b.unsqueeze(1)], 1)
        Sn = S / (torch.linalg.vector_norm(S, dim=1, keepdim=True) + 1e-12)
        cost = (1.0 - Sn[drop] @ Sn[keep].t()).double().cpu().numpy()

        Tmap = torch.zeros(ctx.N, m, dtype=torch.float64, device=imp.device)
        Tmap[keep, torch.arange(m, device=imp.device)] = imp[keep].double()
        if drop.numel():
            a = (imp[drop] / imp[drop].sum()).double().cpu().numpy()
            bb = (imp[keep] / imp[keep].sum()).double().cpu().numpy()
            if pot is not None:
                plan = pot.emd(a, bb, np.ascontiguousarray(cost), numItermax=200000)
            else:                                    # greedy nearest-kept fallback
                plan = np.zeros_like(cost)
                plan[np.arange(len(a)), cost.argmin(1)] = a
            plan = torch.as_tensor(plan, device=imp.device) * imp[drop].sum().double()
            Tmap[drop] = plan
        Tmap = Tmap / (Tmap.sum(0, keepdim=True) + 1e-12)    # column-normalised
        Tm = Tmap.to(ctx.W.dtype)
        return Tm.t() @ ctx.W, Tm.t() @ ctx.b, ctx.V @ Tm, ctx.c.clone()


class S05_BalancedSignature(Strategy):
    name = "05_balanced_signature"
    note = "k-means on [incoming | bias | outgoing] with both blocks scale-normalised"

    def apply(self, ctx, m):
        inc = torch.cat([ctx.W, ctx.b.unsqueeze(1)], 1)
        out = ctx.V.t()
        inc = inc / (inc.norm(dim=1).mean() + 1e-12)
        out = out / (out.norm(dim=1).mean() + 1e-12)
        sig = torch.cat([inc, out], 1).float().cpu().numpy()
        labels = relabel_dense(kmeans_labels(sig, m, ctx.seed), m)
        Wn, bn, Vn = merge_by_labels(labels, ctx.W, ctx.b, ctx.V, ctx.A.mean(0) + 1e-9, m)
        return Wn, bn, Vn, ctx.c.clone()


class S06_GreedyW1Merging(Strategy):
    name = "06_greedy_wasserstein1"
    note = "exact 1-D W1 between neuron activation laws + average-linkage agglomeration"

    def apply(self, ctx, m):
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform
        A = ctx.A
        q = min(256, A.shape[0])
        # W1 between two empirical 1-D laws = mean |sorted_i - sorted_j|; quantile
        # sketch keeps that exact up to the quantile grid.
        idx = torch.linspace(0, A.shape[0] - 1, q, device=A.device).long()
        Q = torch.sort(A, dim=0).values[idx].t().contiguous()      # [N, q]
        D = torch.cdist(Q.float(), Q.float(), p=1) / q
        Dn = D.cpu().numpy().astype(np.float64)
        np.fill_diagonal(Dn, 0.0)
        Z = sch.linkage(squareform(Dn, checks=False), method="average")
        labels = relabel_dense(sch.fcluster(Z, t=m, criterion="maxclust") - 1, m)
        if labels.max() + 1 < m:  # split the biggest cluster until we hit exactly m
            labels = _pad_clusters(labels, m, ctx)
        Wn, bn, Vn = merge_by_labels(labels, ctx.W, ctx.b, ctx.V, ctx.A.mean(0) + 1e-9, m)
        return Wn, bn, Vn, ctx.c.clone()


def _pad_clusters(labels: np.ndarray, m: int, ctx: LayerContext) -> np.ndarray:
    """maxclust can return < m clusters; peel off the highest-energy members of
    the largest cluster into fresh singleton clusters until exactly m exist."""
    labels = labels.copy()
    energy = (ctx.A.mean(0)).float().cpu().numpy()
    nxt = labels.max() + 1
    while nxt < m:
        big = np.bincount(labels).argmax()
        members = np.nonzero(labels == big)[0]
        if members.size <= 1:
            break
        labels[members[np.argmax(energy[members])]] = nxt
        nxt += 1
    return labels


class S07_MixedImportance(Strategy):
    name = "07_mixed_importance"
    note = "protect the top-20% highest-variance neurons verbatim, merge the rest, then LS"

    def apply(self, ctx, m):
        var = ctx.A.var(0)
        k = max(1, int(round(0.2 * m)))
        prot = torch.topk(var, k).indices
        mask = torch.zeros(ctx.N, dtype=torch.bool, device=var.device)
        mask[prot] = True
        rest = torch.nonzero(~mask).flatten()
        m_rest = m - k
        if m_rest <= 0 or rest.numel() == 0:
            keep = torch.topk(var, m).indices.sort().values.tolist()
            Vn, cn = ls_consuming(ctx.A[:, keep], ctx.Y, ctx.ridge)
            return ctx.W[keep], ctx.b[keep], Vn, cn

        inc = torch.cat([ctx.W[rest], ctx.b[rest].unsqueeze(1)], 1)
        out = ctx.V[:, rest].t()
        inc = inc / (inc.norm(dim=1).mean() + 1e-12)
        out = out / (out.norm(dim=1).mean() + 1e-12)
        sig = torch.cat([inc, out], 1).float().cpu().numpy()
        labels = relabel_dense(kmeans_labels(sig, m_rest, ctx.seed), m_rest)
        Wm, bm, _ = merge_by_labels(labels, ctx.W[rest], ctx.b[rest], ctx.V[:, rest],
                                    ctx.A[:, rest].mean(0) + 1e-9, m_rest)
        Wn = torch.cat([ctx.W[prot], Wm], 0)
        bn = torch.cat([ctx.b[prot], bm], 0)
        An = ctx.acts_for(Wn, bn)
        Vn, cn = ls_consuming(An, ctx.Y, ctx.ridge)
        return Wn, bn, Vn, cn


class S08_TwoStageCorrection(Strategy):
    name = "08_ot_then_ls"
    note = "Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS"

    def apply(self, ctx, m):
        Wn, bn, _, _ = S04_CosineOTMerging().apply(ctx, m)
        An = ctx.acts_for(Wn, bn)
        Vn, cn = ls_consuming(An, ctx.Y, ctx.ridge)
        return Wn, bn, Vn, cn


class S09_KernelKMeans(Strategy):
    name = "09_kernel_kmeans_rbf"
    note = "RBF kernel k-means on balanced signatures (non-linear cluster shapes)"

    def apply(self, ctx, m):
        inc = torch.cat([ctx.W, ctx.b.unsqueeze(1)], 1)
        out = ctx.V.t()
        inc = inc / (inc.norm(dim=1).mean() + 1e-12)
        out = out / (out.norm(dim=1).mean() + 1e-12)
        S = torch.cat([inc, out], 1).float()
        D2 = torch.cdist(S, S) ** 2
        gamma = 1.0 / (float(torch.median(D2[D2 > 0])) + 1e-12)
        K = torch.exp(-gamma * D2).double()

        labels = torch.as_tensor(relabel_dense(kmeans_labels(S.cpu().numpy(), m, ctx.seed), m),
                                 device=K.device)
        for _ in range(15):
            oh = torch.zeros(ctx.N, m, dtype=K.dtype, device=K.device)
            oh[torch.arange(ctx.N, device=K.device), labels] = 1.0
            cnt = oh.sum(0).clamp(min=1)
            KO = K @ oh                                   # [N, m]
            third = (oh.t() @ KO).diagonal() / cnt ** 2   # [m]
            dist = K.diagonal().unsqueeze(1) - 2 * KO / cnt + third.unsqueeze(0)
            new = dist.argmin(1)
            if torch.equal(new, labels):
                break
            labels = new
        lab = relabel_dense(labels.cpu().numpy(), m)
        Wn, bn, Vn = merge_by_labels(lab, ctx.W, ctx.b, ctx.V, ctx.A.mean(0) + 1e-9, m)
        return Wn, bn, Vn, ctx.c.clone()


class S10_IterativeRefinement(Strategy):
    name = "10_iterative_refinement"
    note = "two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage"

    def apply(self, ctx, m):
        mid = int(np.clip(round(np.sqrt(ctx.N * m)), m + 1, ctx.N - 1)) if m < ctx.N - 1 else m
        W, b = ctx.W, ctx.b
        for target in ([mid, m] if mid > m else [m]):
            A = torch.relu(ctx.H @ W.t() + b)
            Ad = _f64(A)
            keep = pivoted_cholesky_select(Ad.t() @ Ad, target)
            W, b = W[keep], b[keep]
        An = torch.relu(ctx.H @ W.t() + b)
        Vn, cn = ls_consuming(An, ctx.Y, ctx.ridge)
        return W, b, Vn, cn


SEED_STRATEGIES: list[Strategy] = [
    S01_DirectLeastSquares(), S02_TopMRowSampling(), S03_ActivationPatternClustering(),
    S04_CosineOTMerging(), S05_BalancedSignature(), S06_GreedyW1Merging(),
    S07_MixedImportance(), S08_TwoStageCorrection(), S09_KernelKMeans(),
    S10_IterativeRefinement(),
]


# ===========================================================================
# 6. COMPRESSION DRIVER
# ===========================================================================

@dataclass
class Bench:
    """The immutable reference: the dense, BN-folded network + calibration data."""
    Ws: list[torch.Tensor]
    bs: list[torch.Tensor]
    X: torch.Tensor
    A_orig: list[torch.Tensor] = field(default_factory=list)
    Y_orig: list[torch.Tensor] = field(default_factory=list)
    metrics: list[torch.Tensor | None] = field(default_factory=list)

    @torch.no_grad()
    def prepare(self):
        h = self.X
        self.A_orig = []
        for l in range(len(self.Ws) - 1):
            h = torch.relu(h @ self.Ws[l].t() + self.bs[l])
            self.A_orig.append(h)
        # LS target for layer l = the DENSE net's pre-activation at layer l+1
        self.Y_orig = [self.A_orig[l] @ self.Ws[l + 1].t() + self.bs[l + 1]
                       for l in range(len(self.A_orig))]
        self._jacobian_metrics()
        return self

    @torch.no_grad()
    def _jacobian_metrics(self):
        """Metric M_l on Y_l induced by the rest of the DENSE network.

        A layer-wise fit minimises ||Yhat_l - Y_l||_F, which treats every
        direction of Y_l as equally important. It is not: the remaining layers
        amplify some directions and annihilate others. The right norm is the one
        pulled back from the logits by the network's Jacobian,

            M_l = E_x[ J_l(x)^T J_l(x) ],   J_l = d logits / d Y_l .

        For a ReLU MLP the Jacobian is a product of weight matrices and diagonal
        activation masks, so M obeys a closed-form backward recursion needing
        only forward quantities (no autograd, no backward pass):

            M_{L-1} = I                                     (Y_{L-1} = logits)
            M_l     = (W_{l+2}^T M_{l+1} W_{l+2}) (*) E[m m^T],   m = 1[Y_l > 0]

        where (*) is the Hadamard product. This is the Gauss-Newton / Fisher
        metric of the layer-wise subproblem.
        """
        L = len(self.A_orig)
        T = self.X.shape[0]
        M: list[torch.Tensor | None] = [None] * L
        out_dim = self.Ws[-1].shape[0]
        M[L - 1] = torch.eye(out_dim, dtype=torch.float64, device=self.X.device)
        for l in range(L - 2, -1, -1):
            Wn = _f64(self.Ws[l + 2])                      # [n_{l+2}, n_{l+1}]
            core = Wn.t() @ M[l + 1] @ Wn                  # [n_{l+1}, n_{l+1}]
            m = (self.A_orig[l + 1] > 0).to(torch.float64)  # [T, n_{l+1}]
            M[l] = core * (m.t() @ m / T)
        self.metrics = M


@torch.no_grad()
def compress_stack(bench: Bench, strategy: Strategy, targets: list[int],
                   ridge: float, seed: int):
    """Sequential left-to-right compression with full error correction.

    Layer l's activations are recomputed through the layers already compressed,
    and its least-squares target is the DENSE network's value at layer l+1 --
    so each layer's fit absorbs the upstream error rather than compounding it
    (the GPTQ / SparseGPT idea, applied to structured neuron removal).
    """
    Ws = [w.clone() for w in bench.Ws]
    bs = [b.clone() for b in bench.bs]
    L = len(Ws) - 1
    for l in range(L):
        N = int(Ws[l].shape[0])
        m = int(targets[l])
        if m >= N:
            continue
        H = stack_forward(bench.X, Ws, bs, l)
        A = torch.relu(H @ Ws[l].t() + bs[l])
        ctx = LayerContext(idx=l, W=Ws[l], b=bs[l], V=Ws[l + 1], c=bs[l + 1],
                           H=H, A=A, Y=bench.Y_orig[l], ridge=ridge, seed=seed,
                           metric=bench.metrics[l] if bench.metrics else None)
        Wn, bn, Vn, cn = strategy.apply(ctx, m)
        assert Wn.shape[0] == m, f"{strategy.name} returned {Wn.shape[0]} != target_m {m}"
        Ws[l], bs[l] = Wn.to(Ws[l].dtype), bn.to(bs[l].dtype)
        Ws[l + 1], bs[l + 1] = Vn.to(Ws[l + 1].dtype), cn.to(bs[l + 1].dtype)
    return Ws, bs


# ===========================================================================
# 7. LOGGING
# ===========================================================================

def csv_append(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def progress_append(text: str):
    with PROGRESS_PATH.open("a") as f:
        f.write(text + "\n")


# ===========================================================================
# 8. THE AUTONOMOUS RUN
# ===========================================================================

@dataclass
class RunResult:
    strategy: str
    multiplier: float
    widths: list[int]
    acc: float
    drop: float
    params: int
    phase: int
    hit: bool
    seconds: float


class Session:
    def __init__(self, disc: Discovered, args):
        self.d = disc
        self.args = args
        self.device = args.device
        self.tol = args.tol
        self.results: list[RunResult] = []
        self._novel: list[Strategy] = []
        self.frontier: float | None = None
        self.tightest: RunResult | None = None
        self.best: RunResult | None = None
        self.errors: dict[str, str] = {}

    # -- setup ------------------------------------------------------------
    def setup(self):
        a = self.args
        self.train_loader, self.test_loader = get_dataloaders(
            self.d.dataset, batch_size=a.batch_size, num_workers=a.workers, augment=False)

        self.orig_acc = evaluate(self.d.model, self.test_loader, self.device)
        self.lo, self.hi = self.orig_acc - self.tol, self.orig_acc + self.tol
        print(f"\n\U0001F3AF AIM: Compress {self.d.model_name} from {self.orig_acc:.2f}% "
              f"to within [{self.lo:.2f}%, {self.hi:.2f}%] without fine-tuning.")
        print(f"   dataset={self.d.dataset}  params={self.d.n_params:,}  device={self.device}")

        Ws, bs = fold_bn(self.d.model)
        Ws = [w.to(self.device) for w in Ws]
        bs = [b.to(self.device) for b in bs]
        self.in_dim = int(Ws[0].shape[1])
        self.num_classes = int(Ws[-1].shape[0])
        self.activation = act_name(self.d.model)
        self.orig_widths = [int(w.shape[0]) for w in Ws[:-1]]

        X = self._calibration()
        self.bench = Bench(Ws, bs, X).prepare()

        folded = build_from_stack(Ws, bs, self.in_dim, self.num_classes, self.activation)
        self.folded_acc = evaluate(folded, self.test_loader, self.device)
        print(f"   BN-folded dense sanity check: {self.folded_acc:.2f}% "
              f"(delta {self.folded_acc - self.orig_acc:+.2f} pts)")

        self._gride()

    @torch.no_grad()
    def _calibration(self) -> torch.Tensor:
        xs, n = [], 0
        for x, _ in self.train_loader:
            xs.append(x.view(x.size(0), -1).to(self.device))
            n += x.size(0)
            if n >= self.args.calib:
                break
        X = torch.cat(xs)[: self.args.calib]
        print(f"   calibration set: {tuple(X.shape)} (train split, no augmentation)")
        return X

    def _gride(self):
        t0 = time.time()
        self.dims = layer_gride_dims(self.bench.A_orig, self.args.id_space, self.args.seed)
        alt = "activation" if self.args.id_space == "neuron" else "neuron"
        self.dims_alt = layer_gride_dims(self.bench.A_orig, alt, self.args.seed)
        pretty = ", ".join(f"Layer {i+1}: {d:.2f}" for i, d in enumerate(self.dims))
        print(f"\n\U0001F4D0 GRIDE dimensions: {pretty}   [{self.args.id_space} cloud, "
              f"{time.time()-t0:.1f}s]")
        print("   (cross-check, {} cloud: {})".format(
            alt, ", ".join(f"{d:.2f}" for d in self.dims_alt)))
        print(f"   original widths: {self.orig_widths}")
        self.mults = multiplier_grid(self.dims, self.orig_widths)
        if self.args.multipliers:
            self.mults = [float(x) for x in self.args.multipliers.split(",")]
        print(f"   multiplier sweep (runtime-derived): {self.mults}")
        print("   -> every M_l = round(multiplier * d_l). No other width source is used.\n")

    # -- one experiment ---------------------------------------------------
    def run_one(self, strategy: Strategy, mult: float, phase: int,
                per_layer: list[float] | None = None) -> RunResult | None:
        dims = self.dims
        mults = per_layer or [mult] * len(dims)
        targets = [int(np.clip(round(mu * d), 4, n))
                   for mu, d, n in zip(mults, dims, self.orig_widths)]
        t0 = time.time()
        try:
            Ws, bs = compress_stack(self.bench, strategy, targets, self.args.ridge, self.args.seed)
            model = build_from_stack(Ws, bs, self.in_dim, self.num_classes, self.activation)
            acc = evaluate(model, self.test_loader, self.device)
        except Exception as exc:
            self.errors[f"{strategy.name}@{mult}"] = f"{type(exc).__name__}: {exc}"
            log.warning("%s @ x%.2f failed: %s", strategy.name, mult, exc)
            return None
        secs = time.time() - t0

        params = sum(p.numel() for p in model.parameters())
        widths = [int(w.shape[0]) for w in Ws[:-1]]
        drop = self.orig_acc - acc
        hit = self.lo <= acc <= self.hi
        r = RunResult(strategy.name, mult, widths, acc, drop, params, phase, hit, secs)
        self.results.append(r)
        if self.best is None or acc > self.best.acc:
            self.best = r
            self._save_model(model, r, "compressed")
        # the frontier model -- smallest one still inside the window -- is the
        # artifact worth quoting; the most accurate one is the least compressed.
        if hit and (self.tightest is None or params < self.tightest.params):
            self.tightest = r
            self._save_model(model, r, "frontier")

        red = 100.0 * (1 - params / self.d.n_params)
        flag = "  ✅ HIT" if hit else ""
        print(f"   [{strategy.name:<28}] x{mult:<5.2f} M={str(widths):<20} "
              f"acc={acc:6.2f}%  drop={drop:6.2f}  params-{red:4.1f}%  {secs:4.1f}s{flag}")

        csv_append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.d.model_name, "strategy_name": strategy.name,
            "multiplier": mult if per_layer is None else json.dumps(mults),
            "M1": widths[0] if len(widths) > 0 else "",
            "M2": widths[1] if len(widths) > 1 else "",
            "M3": widths[2] if len(widths) > 2 else "",
            "compressed_acc": round(acc, 4), "acc_drop": round(drop, 4),
            "target_hit": int(hit), "dataset": self.d.dataset, "phase": phase,
            "id_space": self.args.id_space,
            "gride_dims": json.dumps([round(x, 3) for x in self.dims]),
            "orig_widths": json.dumps(self.orig_widths), "widths": json.dumps(widths),
            "params": params, "param_reduction_pct": round(red, 3),
            "original_acc": round(self.orig_acc, 4), "calib": int(self.bench.X.shape[0]),
            "seconds": round(secs, 2),
        })
        progress_append(
            f"- `{datetime.now():%Y-%m-%d %H:%M}` **{self.d.dataset}/{self.d.model_name}** "
            f"phase {phase} | `{strategy.name}` | x{mult} | GRIDE d={[round(x,1) for x in self.dims]} "
            f"| M={widths} | acc **{acc:.2f}%** (dense {self.orig_acc:.2f}%, drop {drop:+.2f}) "
            f"| params {params:,} (-{red:.1f}%) | {'HIT' if hit else 'miss'}")
        return r

    def _save_model(self, model: nn.Module, r: RunResult, kind: str = "compressed"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{self.d.dataset.lower()}_{self.d.model_name}_{kind}"
        save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()},
                  str(RESULTS_DIR / f"{tag}.safetensors"))
        (RESULTS_DIR / f"{tag}.arch.json").write_text(json.dumps({
            "in_dim": self.in_dim, "hidden_dims": r.widths, "num_classes": self.num_classes,
            "activation": self.activation, "batch_norm": False,
            "source_checkpoint": self.d.path.name, "strategy": r.strategy,
            "multiplier": r.multiplier, "gride_dims": self.dims,
            "id_space": self.args.id_space, "original_widths": self.orig_widths,
            "original_acc": self.orig_acc, "compressed_acc": r.acc,
            "params": r.params, "target_hit": r.hit,
        }, indent=2))

    # -- phase 1 ----------------------------------------------------------
    def phase1(self) -> bool:
        print("=" * 100)
        print("PHASE 1 -- seed strategies x GRIDE multiplier sweep")
        print("=" * 100)
        pool = SEED_STRATEGIES
        if self.args.strategies:
            pats = [p.strip() for p in self.args.strategies.split(",")]
            pool = [s for s in pool if any(p in s.name for p in pats)]
        for mult in self.mults:
            targets = widths_from_gride(self.dims, mult, self.orig_widths)
            print(f"\n multiplier x{mult}  ->  M = {targets}  (from GRIDE d="
                  f"{[round(x,2) for x in self.dims]})")
            for s in pool:
                r = self.run_one(s, mult, phase=1)
                if r and r.hit and not self.args.exhaustive:
                    print(f"\n\U0001F389 AIM ACHIEVED! {s.name} @ x{mult} -> {r.acc:.2f}% "
                          f"(dense {self.orig_acc:.2f}%, target [{self.lo:.2f}, {self.hi:.2f}])")
                    return True
        return any(r.hit for r in self.results)

    # -- phase 2 ----------------------------------------------------------
    def diagnostics(self) -> str:
        ranked = sorted(self.results, key=lambda r: -r.acc)[:12]
        by_strategy: dict[str, RunResult] = {}
        for r in self.results:
            if r.strategy not in by_strategy or r.acc > by_strategy[r.strategy].acc:
                by_strategy[r.strategy] = r
        lines = [
            "ADAPTIVE RESEARCH REQUEST",
            "=" * 78,
            f"generated      : {datetime.now().isoformat(timespec='seconds')}",
            f"model          : {self.d.dataset}/{self.d.model_name}  ({self.d.n_params:,} params)",
            f"dense accuracy : {self.orig_acc:.2f}%",
            f"target window  : [{self.lo:.2f}%, {self.hi:.2f}%]  (+/- {self.tol})",
            f"GRIDE dims     : {[round(x,3) for x in self.dims]}  ({self.args.id_space} cloud)",
            f"cross-check    : {[round(x,3) for x in self.dims_alt]}",
            f"original widths: {self.orig_widths}",
            f"multipliers    : {self.mults}",
            f"runs completed : {len(self.results)}",
            "",
            "BEST SO FAR",
            "-" * 78,
        ]
        if self.best:
            lines.append(f"{self.best.strategy} @ x{self.best.multiplier} M={self.best.widths} "
                         f"-> {self.best.acc:.2f}% (gap to window: "
                         f"{self.lo - self.best.acc:+.2f} pts)")
        lines += ["", "TOP 12 RUNS", "-" * 78]
        lines += [f"{r.acc:6.2f}%  {r.strategy:<30} x{r.multiplier:<6} M={r.widths}"
                  for r in ranked]
        lines += ["", "BEST PER STRATEGY", "-" * 78]
        lines += [f"{v.acc:6.2f}%  {k:<30} x{v.multiplier:<6} M={v.widths}"
                  for k, v in sorted(by_strategy.items(), key=lambda kv: -kv[1].acc)]

        # which multiplier band looks most promising?
        band: dict[float, float] = {}
        for r in self.results:
            band[r.multiplier] = max(band.get(r.multiplier, -1), r.acc)
        lines += ["", "BEST ACCURACY PER MULTIPLIER (is the budget or the method the bottleneck?)",
                  "-" * 78]
        lines += [f"  x{m:<6} -> {a:6.2f}%" for m, a in sorted(band.items())]

        # per-layer reconstruction error of the best configuration
        lines += ["", "PER-LAYER LS RESIDUAL AT THE BEST BUDGET", "-" * 78]
        for l, res in self._layer_residuals().items():
            lines.append(f"  layer {l+1}: relative residual ||Y-Yhat||/||Y|| = {res:.4f}")
        if self.errors:
            lines += ["", "FAILURES", "-" * 78] + [f"  {k}: {v}" for k, v in self.errors.items()]
        lines += [
            "", "WHAT IS NEEDED", "-" * 78,
            "Drop new Strategy subclasses into novel_strategies/*.py, each module exposing",
            "`STRATEGIES = [...]` or `get_strategies()`. They are imported and swept",
            "automatically on the next round.",
            "",
            "REMINDER -- GOLDEN RULE: a strategy receives `target_m` and MUST return exactly",
            "that many neurons. target_m is round(multiplier * GRIDE_d_l). A strategy may not",
            "choose its own width, use PCA rank, or use a fixed sparsity ratio.",
            "No fine-tuning: no backward passes, no optimizer.step().",
        ]
        return "\n".join(lines)

    @torch.no_grad()
    def _layer_residuals(self) -> dict[int, float]:
        """How much of each layer's downstream target survives the best budget?
        Uses pivoted-Cholesky selection + LS as the reference reconstructor."""
        if not self.best:
            return {}
        out = {}
        Ws = [w.clone() for w in self.bench.Ws]
        bs = [b.clone() for b in self.bench.bs]
        for l, m in enumerate(self.best.widths):
            H = stack_forward(self.bench.X, Ws, bs, l)
            A = torch.relu(H @ Ws[l].t() + bs[l])
            Ad = _f64(A)
            keep = pivoted_cholesky_select(Ad.t() @ Ad, min(m, A.shape[1]))
            Y = self.bench.Y_orig[l]
            V, c = ls_consuming(A[:, keep], Y, self.args.ridge)
            Yh = A[:, keep] @ V.t() + c
            out[l] = float(torch.linalg.norm(Yh - Y) / (torch.linalg.norm(Y) + 1e-12))
            Ws[l], bs[l] = Ws[l][keep], bs[l][keep]
            Ws[l + 1], bs[l + 1] = V, c
        return out

    def load_novel(self, already: set[str]) -> list[Strategy]:
        NOVEL_DIR.mkdir(parents=True, exist_ok=True)
        new: list[Strategy] = []
        for f in sorted(NOVEL_DIR.glob("*.py")):
            if f.name.startswith("_") or f.name in already:
                continue
            already.add(f.name)
            try:
                spec = importlib.util.spec_from_file_location(f"novel_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                got = getattr(mod, "STRATEGIES", None)
                if got is None and hasattr(mod, "get_strategies"):
                    got = mod.get_strategies()
                for s in (got or []):
                    new.append(s)
                self._novel.extend(got or [])
                print(f"   loaded {len(got or [])} strategies from novel_strategies/{f.name}")
            except Exception as exc:
                print(f"   !! failed to load {f.name}: {type(exc).__name__}: {exc}")
        return new

    def phase2(self) -> bool:
        already: set[str] = set()
        for rnd in range(1, self.args.rounds + 1):
            print("\n" + "=" * 100)
            print(f"PHASE 2 -- ADAPTIVE RESEARCH, round {rnd}/{self.args.rounds}")
            print("=" * 100)
            diag = self.diagnostics()
            REQUEST_PATH.write_text(diag)
            (RESULTS_DIR / f"adaptive_request_{self.d.dataset.lower()}_"
             f"{self.d.model_name}.txt").write_text(diag)
            print(f"   diagnostics written to {REQUEST_PATH.relative_to(PROJECT_ROOT)}")
            if self.args.wait_for_operator:
                input(f"   add strategies to {NOVEL_DIR}/ then press Enter... ")
            novel = self.load_novel(already)
            if not novel:
                print("   no new strategies available -- ending adaptive phase.")
                return False
            for s in novel:
                for mult in self.mults:
                    r = self.run_one(s, mult, phase=1 + rnd)
                    if r and r.hit and not self.args.exhaustive:
                        print(f"\n\U0001F389 AIM ACHIEVED! {s.name} @ x{mult} -> {r.acc:.2f}%")
                        return True
            if any(r.hit for r in self.results):
                return True
        return False

    # -- phase 3: how the GRIDE budget is spread across layers -------------
    def _params_for(self, widths: list[int]) -> int:
        dims = [self.in_dim] + list(widths) + [self.num_classes]
        return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

    def phase3_allocation(self) -> bool:
        """A uniform multiplier is not the only GRIDE-legal budget.

        Every width is still `round(mult_l * d_l)` with `mult_l` drawn from the
        same runtime-derived grid -- only the *per-layer* choice is freed. Held
        at a fixed parameter budget this asks whether the Zador rule's *shape*
        (M_l proportional to d_l) is the right allocation, or only its scale.
        Coordinate descent over the grid, two passes.
        """
        if not self.best:
            return False
        strat = {s.name: s for s in SEED_STRATEGIES + self._novel}.get(self.best.strategy)
        if strat is None:
            return False
        base = self.best.multiplier if isinstance(self.best.multiplier, float) else self.mults[-1]
        budget = self._params_for(widths_from_gride(self.dims, base, self.orig_widths))

        print("\n" + "=" * 100)
        print(f"PHASE 3 -- per-layer GRIDE allocation search ({strat.name}, "
              f"budget = {budget:,} params = uniform x{base})")
        print("=" * 100)

        cur = [base] * len(self.dims)
        best_acc = self.best.acc
        seen: set[tuple[float, ...]] = {tuple(cur)}
        for sweep in range(2):
            for l in range(len(cur)):
                for cand in self.mults:
                    trial = list(cur)
                    trial[l] = cand
                    if tuple(trial) in seen:
                        continue
                    seen.add(tuple(trial))
                    w = [int(np.clip(round(mu * d), 4, n))
                         for mu, d, n in zip(trial, self.dims, self.orig_widths)]
                    if self._params_for(w) > budget:
                        continue
                    r = self.run_one(strat, base, phase=4, per_layer=trial)
                    if r and r.acc > best_acc:
                        best_acc, cur = r.acc, trial
                        if r.hit and not self.args.exhaustive:
                            print(f"\n\U0001F389 AIM ACHIEVED via allocation {trial} -> {r.acc:.2f}%")
                            return True
            print(f"   sweep {sweep+1}: best allocation {cur} -> {best_acc:.2f}%")
        return any(r.hit for r in self.results)

    # -- phase 4: find the TIGHTEST GRIDE budget that still lands in window -
    def phase4_refine(self) -> bool:
        """Bisect the multiplier between the largest miss and the smallest hit.

        The ladder in `multiplier_grid` is geometric and therefore coarse -- it
        steps 12 -> 16, so the first hit it reports can be well inside the
        window and much larger than necessary. The compression *frontier* is the
        smallest multiplier that still lands in the window, and that is the
        number worth quoting. Bisection finds it in ~5 runs. Widths remain
        `round(mult * d_l)` throughout, so this is still purely GRIDE-driven --
        only the grid resolution changes, not the rule.
        """
        hits = [r for r in self.results if r.hit and isinstance(r.multiplier, float)]
        if not hits:
            return False
        lo_hit = min(r.multiplier for r in hits)
        # bisect with the top strategies *at that budget*, not only those that
        # happened to clear the window there -- a method that just missed at
        # lo_hit can still be the best one below it.
        best_at: dict[str, RunResult] = {}
        for r in self.results:
            if r.multiplier == lo_hit and (
                    r.strategy not in best_at or r.acc > best_at[r.strategy].acc):
                best_at[r.strategy] = r
        winners = sorted(best_at.values(), key=lambda r: -r.acc)[:3]
        pool = {s.name: s for s in SEED_STRATEGIES + self._novel}
        strats = [pool[r.strategy] for r in winners if r.strategy in pool]
        if not strats:
            return False
        misses = [r.multiplier for r in self.results
                  if not r.hit and isinstance(r.multiplier, float) and r.multiplier < lo_hit]
        lo_miss = max(misses) if misses else 1.0

        print("\n" + "=" * 100)
        print(f"PHASE 4 -- budget refinement: bisecting x{lo_miss} (miss) .. x{lo_hit} (hit) "
              f"with {[s.name for s in strats]}")
        print("=" * 100)

        a, b = lo_miss, lo_hit
        seen = {lo_miss, lo_hit}
        for _ in range(5):
            mid = round((a + b) / 2, 2)
            if mid in seen or widths_from_gride(self.dims, mid, self.orig_widths) == \
                    widths_from_gride(self.dims, b, self.orig_widths):
                break
            seen.add(mid)
            got = [self.run_one(s, mid, phase=5) for s in strats]
            if any(r and r.hit for r in got):
                b = mid
            else:
                a = mid
        w = widths_from_gride(self.dims, b, self.orig_widths)
        print(f"   frontier: smallest GRIDE budget inside the window is x{b} -> M={w} "
              f"({self._params_for(w):,} params, "
              f"-{100*(1-self._params_for(w)/self.d.n_params):.1f}%)")
        self.frontier = b
        return True

    # -- report -----------------------------------------------------------
    def report(self, achieved: bool):
        uniq: dict[tuple, RunResult] = {}
        for r in self.results:
            key = (r.strategy, tuple(r.widths))
            if key not in uniq or r.acc > uniq[key].acc:
                uniq[key] = r
        ranked = sorted(uniq.values(), key=lambda r: -r.acc)
        by_strategy: dict[str, RunResult] = {}
        for r in self.results:
            if r.strategy not in by_strategy or r.acc > by_strategy[r.strategy].acc:
                by_strategy[r.strategy] = r

        L = [
            f"# Autonomous zero-shot compression report -- {self.d.dataset}/{self.d.model_name}",
            "",
            f"*generated {datetime.now():%Y-%m-%d %H:%M}*",
            "",
            "## Aim",
            "",
            f"Compress **{self.d.model_name}** ({self.d.n_params:,} params, dense test accuracy "
            f"**{self.orig_acc:.2f}%**) to within **+/-{self.tol}** points -- i.e. into "
            f"**[{self.lo:.2f}%, {self.hi:.2f}%]** -- with **no fine-tuning** "
            "(zero backward passes).",
            "",
            f"**Outcome: {'AIM ACHIEVED' if achieved else 'AIM NOT MET'}.** "
            + (f"Best: **{self.best.acc:.2f}%** via `{self.best.strategy}` at multiplier "
               f"x{self.best.multiplier}, widths {self.best.widths} "
               f"({self.best.params:,} params, "
               f"-{100*(1-self.best.params/self.d.n_params):.1f}%), "
               f"drop {self.best.drop:+.2f} pts." if self.best else ""),
            "",
            "## The GRIDE foundation (all widths derive from this)",
            "",
            f"Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the "
            f"GRIDE intrinsic dimension of layer *l*'s **{self.args.id_space} cloud**, measured "
            f"at runtime on {self.bench.X.shape[0]} calibration samples.",
            "",
            "| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |",
            "|---|---|---|---|",
        ]
        nd = self.dims if self.args.id_space == "neuron" else self.dims_alt
        ad = self.dims_alt if self.args.id_space == "neuron" else self.dims
        for i, n in enumerate(self.orig_widths):
            L.append(f"| h{i+1} | {n} | {nd[i]:.2f} | {ad[i]:.2f} |")
        L += ["", f"Multiplier grid swept (derived from the discovered widths): "
                  f"`{self.mults}`", ""]

        L += ["## Ranked results (top 25 of "
              f"{len(self.results)} runs)", "",
              "| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |",
              "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(ranked[:25], 1):
            L.append(f"| {i} | **{r.acc:.2f}%** | {r.drop:+.2f} | `{r.strategy}` | "
                     f"x{r.multiplier} | {r.widths} | {r.params:,} | "
                     f"{100*(1-r.params/self.d.n_params):.1f}% |")

        L += ["", "## Best per strategy", "",
              "| strategy | best accuracy | multiplier | widths | what it does |",
              "|---|---|---|---|---|"]
        notes = {s.name: s.note for s in SEED_STRATEGIES}
        for k, v in sorted(by_strategy.items(), key=lambda kv: -kv[1].acc):
            L.append(f"| `{k}` | {v.acc:.2f}% | x{v.multiplier} | {v.widths} | "
                     f"{notes.get(k, 'novel (phase 2)')} |")

        band: dict[float, RunResult] = {}
        for r in self.results:
            if r.multiplier not in band or r.acc > band[r.multiplier].acc:
                band[r.multiplier] = r
        L += ["", "## Accuracy vs. GRIDE budget", "",
              "Is the bottleneck the *budget* (M too small) or the *method*?", "",
              "| multiplier | widths | best accuracy | drop |", "|---|---|---|---|"]
        for m, r in sorted(band.items()):
            L.append(f"| x{m} | {r.widths} | {r.acc:.2f}% | {r.drop:+.2f} |")

        L += ["", "## Why the winner works", ""]
        L += self._explain(by_strategy)
        L += ["", "## Reproduce", "", "```bash",
              f"cd {PROJECT_ROOT}",
              f"python3 compression_mlp/run_autonomous_compression.py --checkpoint {self.d.path.stem} "
              f"--id-space {self.args.id_space} --calib {self.args.calib}",
              "```", "",
              f"Raw log: `compression_mlp/results/compression_results.csv` ({len(self.results)} runs). "
              f"Best model: `compression_mlp/results/{self.d.dataset.lower()}_{self.d.model_name}"
              "_compressed.safetensors`."]
        text = "\n".join(L) + "\n"
        REPORT_PATH.write_text(text)
        per_model = RESULTS_DIR / f"final_report_{self.d.dataset.lower()}_{self.d.model_name}.md"
        per_model.write_text(text)
        print(f"\n\U0001F4C4 report -> {REPORT_PATH.relative_to(PROJECT_ROOT)} "
              f"and {per_model.relative_to(PROJECT_ROOT)}")

    def _explain(self, by_strategy: dict[str, RunResult]) -> list[str]:
        if not self.best:
            return ["No successful run to explain."]
        w = self.best.strategy
        out = [f"The winning method is **`{w}`**.", ""]
        ls_based = any(k in w for k in ("ls", "reconstruct", "omp", "correction", "iterative"))
        if ls_based:
            out += [
                "It combines two things that pure OT/k-means merging does not have:",
                "",
                "1. **Selection instead of averaging.** The kept neurons retain their *exact* "
                "weights, so their ReLU activations are exact. Merging two neurons is only "
                "output-preserving when they co-activate on every input (Section 6.5's "
                "exactness condition); when they do not, the merge injects error that no "
                "downstream layer can see or undo.",
                "2. **A closed-form correction of the consuming layer against the DENSE "
                "network's own target,** solved with an intercept. Removing a ReLU neuron "
                "removes a non-negative contribution with a non-zero *mean*; only a free bias "
                "can absorb that mean. Because the target is the dense net's pre-activation at "
                "layer l+1 -- not the layer's own activations -- the fit also absorbs the error "
                "inherited from already-compressed upstream layers instead of compounding it.",
                "",
                "In the OT language of the paper, the least-squares map plays the role of the "
                "transport plan, but it is chosen to minimise the *functional* error the next "
                "layer actually sees, rather than a geometric cost in parameter space. That is "
                "exactly the gap the counterexample in Section 4.5 predicts: distance between "
                "neuron signatures is not the quantity that governs output error.",
            ]
        else:
            out += [
                "It is a merging method: neurons are grouped and replaced by cluster "
                "barycentres, with outgoing weights summed. This preserves the layer output "
                "exactly when the cluster's members co-activate, which is the exactness "
                "condition for ReLU merging.",
            ]
        gap = self.lo - self.best.acc
        if gap > 0:
            out += ["", f"**Remaining gap: {gap:.2f} points** below the target window. "
                        "The per-multiplier table above separates the two possible causes: if "
                        "accuracy is still climbing with the multiplier, the GRIDE budget is "
                        "binding; if it has plateaued below the window, the method is binding."]
        return out


# ===========================================================================
# 9. ENTRY POINT
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint stem, e.g. cifar10_mlp3_oct_bn (default: largest MLP found)")
    p.add_argument("--all-mlps", action="store_true", help="run every discovered MLP in turn")
    p.add_argument("--tol", type=float, default=2.5, help="target window, +/- points")
    p.add_argument("--calib", type=int, default=8192, help="calibration samples")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ridge", type=float, default=1e-4, help="relative ridge for the LS solves")
    p.add_argument("--id-space", choices=["neuron", "activation"], default="neuron",
                   help="which cloud GRIDE measures (both are always reported)")
    p.add_argument("--rounds", type=int, default=3, help="adaptive research rounds")
    p.add_argument("--wait-for-operator", action="store_true",
                   help="pause each adaptive round so code can be pasted into novel_strategies/")
    p.add_argument("--exhaustive", action="store_true",
                   help="do not stop on the first hit; sweep everything")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--only", default=None,
                   help="run exactly one configuration, 'STRATEGY@MULTIPLIER', and save it")
    p.add_argument("--no-refine", action="store_true",
                   help="skip phase 4 (bisecting for the tightest GRIDE budget in window)")
    p.add_argument("--alloc-search", action="store_true",
                   help="phase 3: coordinate-descent over PER-LAYER GRIDE multipliers at a "
                        "fixed parameter budget (every width still round(mult_l * d_l))")
    p.add_argument("--multipliers", default=None,
                   help="comma-separated override of the runtime-derived multiplier grid")
    p.add_argument("--strategies", default=None,
                   help="comma-separated substrings; only matching seed strategies run")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    return a


def run_for(path: Path, model_name: str, args) -> bool:
    disc = load_discovered(path, model_name, args.device)
    print("\n" + "#" * 100)
    print(f"# {disc.path.name}  ->  dataset={disc.dataset}  arch={disc.model_name}")
    print("#" * 100)
    sess = Session(disc, args)
    sess.setup()
    if args.only:
        name, _, mult = args.only.partition("@")
        sess.load_novel(set())
        pool = {s.name: s for s in SEED_STRATEGIES + sess._novel}
        if name not in pool:
            raise SystemExit(f"--only: unknown strategy {name!r}")
        r = sess.run_one(pool[name], float(mult), phase=6)
        # deliberately no report() here -- a one-config run must not overwrite
        # the full sweep's final_report_*.md
        return bool(r and r.hit)

    achieved = False if args.skip_phase1 else sess.phase1()
    # --exhaustive means "sweep everything", so the adaptive phase must still run
    # even when phase 1 already cleared the window -- otherwise a full sweep
    # silently omits every novel strategy.
    if not achieved or args.exhaustive:
        achieved = sess.phase2() or achieved
    if (not achieved or args.exhaustive) and args.alloc_search:
        achieved = sess.phase3_allocation() or achieved
    if achieved and not args.no_refine:
        sess.phase4_refine()
    if achieved:
        print(f"\n\U0001F389 AIM ACHIEVED for {disc.model_name}.")
    else:
        b = sess.best
        print(f"\n⚠️  Aim not met for {disc.model_name}. Best found: "
              f"{b.acc:.2f}% via {b.strategy} @ x{b.multiplier} (dense {sess.orig_acc:.2f}%, "
              f"need >= {sess.lo:.2f}%)" if b else "\n⚠️  No successful run.")
    sess.report(achieved)
    return achieved


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text(
            "# Explore / autonomous zero-shot compression -- run log\n\n"
            "Every line is one (strategy x GRIDE-multiplier) experiment. Widths always\n"
            "come from `M_l = round(multiplier * GRIDE_d_l)`.\n\n")

    found = discover_mlp_checkpoints(CHECKPOINT_DIR)
    if not found:
        raise SystemExit(f"No MLP checkpoints found in {CHECKPOINT_DIR}")
    print(f"discovered {len(found)} MLP checkpoints in {CHECKPOINT_DIR}:")
    for p, mn in found:
        print(f"   {p.stem:<32} {p.stat().st_size/1e6:8.1f} MB")

    if args.checkpoint:
        sel = [(p, mn) for p, mn in found if p.stem == args.checkpoint]
        if not sel:
            raise SystemExit(f"--checkpoint {args.checkpoint} not among the discovered MLPs")
    elif args.all_mlps:
        sel = found
    else:
        sel = [found[0]]  # largest

    ok = [run_for(p, mn, args) for p, mn in sel]
    raise SystemExit(0 if any(ok) else 1)


if __name__ == "__main__":
    main()
