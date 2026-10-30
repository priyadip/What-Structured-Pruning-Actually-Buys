"""Landmark selection — which K points the embedding is actually solved on.

Five strategies. `centroid_boundary` is the one specified for this study; the
rest are controls. Note that `centroid_boundary` and `class_balanced` read
**labels**, while `kmeans`, `greedy_cover` and `random` do not — worth keeping
straight, because the hidden-layer compression this builds on was entirely
label-free, so a label-using landmark rule is a (small) extra assumption.
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import cdist_chunked


def _kmeans_idx(X: torch.Tensor, K: int, seed: int) -> np.ndarray:
    """K-means centroids, snapped to their nearest real data points."""
    from sklearn.cluster import KMeans
    Xn = X.float().cpu().numpy()
    km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(Xn)
    C = torch.as_tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)
    return cdist_chunked(C, X).argmin(1).cpu().numpy()


def _greedy_cover(X: torch.Tensor, K: int, seed: int) -> np.ndarray:
    """Farthest-first traversal: maximally spread points (2-approx k-center)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    first = int(rng.integers(n))
    chosen = [first]
    d = torch.cdist(X[first:first + 1].float(), X.float()).squeeze(0)
    for _ in range(K - 1):
        j = int(torch.argmax(d))
        chosen.append(j)
        d = torch.minimum(d, torch.cdist(X[j:j + 1].float(), X.float()).squeeze(0))
    return np.array(chosen)


def _class_balanced(y: np.ndarray, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, K // len(classes))
    out = []
    for c in classes:
        idx = np.nonzero(y == c)[0]
        out.append(rng.choice(idx, min(per, len(idx)), replace=False))
    out = np.concatenate(out)
    if len(out) < K:                                   # top up at random
        rest = np.setdiff1d(np.arange(len(y)), out)
        out = np.concatenate([out, rng.choice(rest, K - len(out), replace=False)])
    return out[:K]


def _centroid_boundary(X: torch.Tensor, y: np.ndarray, K: int, seed: int) -> np.ndarray:
    """C class centroids + boundary representatives for the closest class pairs.

    For a pair (i, j) the two representatives are the sample of class i nearest
    to any sample of class j, and vice versa -- the points that sit on the
    decision boundary, which is exactly where a coarse embedding does damage.

    The specification gives C + 2*C(C,2) landmarks (100 for C=10). That is
    10 000 for CIFAR-100, so pairs are ranked by centroid distance and taken
    closest-first until the budget K is filled: the confusable pairs are the ones
    whose boundary is worth spending landmarks on.
    """
    classes = np.unique(y)
    C = len(classes)
    Xf = X.float()
    cents, cent_idx = [], []
    for c in classes:
        idx = np.nonzero(y == c)[0]
        mu = Xf[idx].mean(0, keepdim=True)
        cents.append(mu)
        cent_idx.append(int(idx[int(torch.cdist(mu, Xf[idx]).argmin())]))
    cents = torch.cat(cents)

    chosen = list(dict.fromkeys(cent_idx))
    budget = K - len(chosen)
    if budget <= 0:
        return np.array(chosen[:K])

    cd = torch.cdist(cents, cents)
    cd.fill_diagonal_(float("inf"))
    pairs = [(float(cd[i, j]), i, j) for i in range(C) for j in range(i + 1, C)]
    pairs.sort()

    # cap the per-class candidate pool so CIFAR-100 stays tractable
    rng = np.random.default_rng(seed)
    pool = {}
    for ci, c in enumerate(classes):
        idx = np.nonzero(y == c)[0]
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        pool[ci] = idx

    seen = set(chosen)
    for _, i, j in pairs:
        if budget <= 0:
            break
        Ai, Aj = pool[i], pool[j]
        d = cdist_chunked(Xf[Ai], Xf[Aj])
        a = int(Ai[int(d.min(1).values.argmin())])      # class i point closest to class j
        b = int(Aj[int(d.min(0).values.argmin())])      # class j point closest to class i
        for v in (a, b):
            if v not in seen and budget > 0:
                seen.add(v)
                chosen.append(v)
                budget -= 1
    if budget > 0:                                      # top up
        rest = np.setdiff1d(np.arange(len(y)), np.array(chosen))
        chosen += list(rng.choice(rest, budget, replace=False))
    return np.array(chosen[:K])


STRATEGIES = ("centroid_boundary", "class_balanced", "kmeans", "greedy_cover", "random")
USES_LABELS = {"centroid_boundary", "class_balanced"}


def select_landmarks(X: torch.Tensor, y: np.ndarray, K: int, strategy: str,
                     seed: int = 0) -> np.ndarray:
    K = min(K, X.shape[0])
    if strategy == "centroid_boundary":
        return _centroid_boundary(X, y, K, seed)
    if strategy == "class_balanced":
        return _class_balanced(y, K, seed)
    if strategy == "kmeans":
        return _kmeans_idx(X, K, seed)
    if strategy == "greedy_cover":
        return _greedy_cover(X, K, seed)
    if strategy == "random":
        return np.random.default_rng(seed).choice(X.shape[0], K, replace=False)
    raise ValueError(f"unknown landmark strategy {strategy!r}; options {STRATEGIES}")
