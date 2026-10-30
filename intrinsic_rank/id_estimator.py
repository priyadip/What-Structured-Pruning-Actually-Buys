"""Core intrinsic-dimension (ID) estimators: TwoNN, GRIDE, and a simple MLE
baseline. All operate on a point cloud X of shape (n_samples, n_features) and
return a scalar (or, for GRIDE, an array of estimates over scales)."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _pairwise_ratios(X: np.ndarray, k1: int, k2: int, n_neighbors_cap: int):
    nn = NearestNeighbors(n_neighbors=n_neighbors_cap + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] is 0 (self); shift by one.
    r1 = distances[:, k1]
    r2 = distances[:, k2]
    valid = r1 > 0
    return r2[valid] / r1[valid]


def two_nn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Facco et al. (2017) TwoNN estimator: uses only the 1st and 2nd nearest
    neighbor distances. Robust to curvature/non-uniform density."""
    mu = _pairwise_ratios(X, k1=1, k2=2, n_neighbors_cap=2)
    mu = np.sort(mu)
    n = len(mu)
    n_keep = int(np.floor(n * (1 - discard_fraction)))
    mu = mu[:n_keep] if n_keep > 0 else mu

    # Linear fit of -log(1 - F(mu)) = d * log(mu), F = empirical CDF.
    F_emp = (np.arange(1, len(mu) + 1)) / n
    x = np.log(mu)
    y = -np.log(1 - F_emp + 1e-12)
    d = np.sum(x * y) / np.sum(x * x)
    return float(d)


def gride(X: np.ndarray, n_neighbors: int = 16, n_iters: int = 10) -> np.ndarray:
    """Generalized Ratios Intrinsic Dimension Estimator (Denti et al., 2022).
    Extends TwoNN to neighbor-order ratios (2k, k) across increasing k, giving
    an ID estimate per scale — useful for spotting multi-scale structure."""
    ks = np.unique(np.geomspace(1, n_neighbors, num=n_iters).astype(int))
    n_neighbors_cap = int(2 * ks[-1])
    estimates = []
    for k in ks:
        mu = _pairwise_ratios(X, k1=k, k2=2 * k, n_neighbors_cap=n_neighbors_cap)
        mu = mu[mu > 1]
        if len(mu) == 0:
            estimates.append(np.nan)
            continue
        # Closed-form MLE for generalized (2k, k) neighbor ratios (Denti et al., 2022).
        d_hat = 1.0 / np.mean(np.log(mu))
        estimates.append(d_hat)
    return np.array(estimates, dtype=float)


def mle_id(X: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel MLE estimator, averaged over the k-NN neighborhood size."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # drop self

    log_ratios = np.log(distances[:, -1:] / distances[:, :-1] + 1e-12)
    inv_d = log_ratios.mean(axis=1)
    per_point_d = 1.0 / (inv_d + 1e-12)
    return float(np.mean(per_point_d))
