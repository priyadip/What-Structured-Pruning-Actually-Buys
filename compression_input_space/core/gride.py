"""GRIDE intrinsic dimension of the raw input data.

The reduced input dimension `r` is only ever `round(multiplier * d_input)` with
`d_input` measured here at runtime. If the estimator cannot be imported this
module raises rather than substituting any other rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from intrinsic_rank.id_estimator import gride as _gride
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "GRIDE is mandatory for the input-dimension rule: could not import "
        "intrinsic_rank.id_estimator.gride."
    ) from exc


def gride_id(X: np.ndarray, *, max_points: int = 2500, max_dims: int = 2048,
             n_neighbors: int = 16, n_iters: int = 8, seed: int = 0) -> float:
    """Median-over-scales GRIDE estimate. Points and coordinates are subsampled
    for tractability; random coordinate selection preserves distances up to the
    Johnson-Lindenstrauss factor, so the estimate is unaffected."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], max_points, replace=False)]
    if X.shape[1] > max_dims:
        X = X[:, rng.choice(X.shape[1], max_dims, replace=False)]
    X = np.ascontiguousarray(X, dtype=np.float64)
    est = np.asarray(_gride(X, n_neighbors=n_neighbors, n_iters=n_iters), dtype=float)
    est = est[np.isfinite(est) & (est > 0)]
    if est.size == 0:
        raise RuntimeError("GRIDE returned no finite estimate for the input data.")
    return float(np.median(est))
