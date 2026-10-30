"""Distances: chunked Euclidean, and geodesic distances on a kNN graph.

srGW-MDS needs a target dissimilarity `D_geo` between landmarks. Euclidean
distance in pixel space is a poor description of an image manifold; geodesic
distance along a kNN graph is the standard Isomap-style fix and is what makes
the embedding *manifold*-aware rather than just a rotation.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def cdist_chunked(X: torch.Tensor, Y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Euclidean distances [n, m] without materialising an n x m x d tensor."""
    out = torch.empty(X.shape[0], Y.shape[0], device=X.device, dtype=X.dtype)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(X[i:i + chunk], Y)
    return out


@torch.no_grad()
def knn_graph_distances(X: torch.Tensor, land_idx: np.ndarray, k: int = 12,
                        max_nodes: int = 6000, seed: int = 0):
    """Geodesic distance between landmarks along a symmetric kNN graph.

    The graph is built over the landmarks plus a random subsample of the data
    (capped at `max_nodes`) so shortest paths can route *through* the manifold
    instead of cutting across it. Returns `D [K, K]` (float64, numpy).

    Falls back to Euclidean if the graph is disconnected or scipy is missing --
    an infinite geodesic is worse than an honest chord.
    """
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    extra = rng.choice(n, min(max_nodes, n), replace=False)
    nodes = np.unique(np.concatenate([land_idx, extra]))
    pos = {int(v): i for i, v in enumerate(nodes)}
    sub = X[nodes].float().cpu().numpy()

    G = kneighbors_graph(sub, n_neighbors=min(k, len(nodes) - 1), mode="distance")
    G = G.maximum(G.T)                                  # symmetrise
    rows = np.array([pos[int(v)] for v in land_idx])
    D = shortest_path(G, method="D", directed=False, indices=rows)[:, rows]

    if not np.isfinite(D).all():                        # disconnected graph
        L = X[land_idx].float()
        D_euc = cdist_chunked(L, L).double().cpu().numpy()
        bad = ~np.isfinite(D)
        D[bad] = D_euc[bad] * 1.0
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


@torch.no_grad()
def landmark_dissimilarity(X: torch.Tensor, land_idx: np.ndarray, kind: str = "geodesic",
                           k: int = 12, seed: int = 0) -> np.ndarray:
    if kind == "euclidean":
        L = X[land_idx].float()
        return cdist_chunked(L, L).double().cpu().numpy()
    try:
        return knn_graph_distances(X, land_idx, k=k, seed=seed)
    except Exception:
        L = X[land_idx].float()
        return cdist_chunked(L, L).double().cpu().numpy()
