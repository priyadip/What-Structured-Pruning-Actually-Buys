"""GRIDE intrinsic dimension for convolutional layers -- the only legal source
of a compressed layer's width.

        M_l = round(multiplier * d_l),   d_l = GRIDE ID of layer l's cloud

No width in this project comes from a PCA variance threshold, a sparsity ratio,
or a hand-set number. If GRIDE cannot be computed the run stops.

Which point cloud?
------------------
For an MLP the answer was unambiguous: a neuron is a row of `W`, and its
"coordinates" are its activations over the calibration set. A conv layer offers
three readings, and they are *not* the same estimate, so all three are measured
and logged:

`filter_proj`   each filter flattened to `R^(C_in*k*k)` and projected onto the
                column space of the unfolded input, i.e. the part of the filter
                the data can actually see. `C_out` points. This is the direct
                translation of "effective neuron cloud" and is the default.

`neuron`        the same projection, but with each direction scaled by its
                singular value -- which makes the coordinates literally the
                channel's pre-activation vector (up to a rotation). This is what
                `../compression_mlp` used for MLPs, so it is the continuity check.

`patch`         the *input* manifold: one point per `C_in*k*k` patch. This is not
                a neuron cloud at all; it is the intrinsic dimension of what the
                layer reads, and it is what the patch-space stage budgets
                against. For the first conv layer this cloud is the RGB patch
                manifold -- see `rgb_patch_id`.

Why they differ: `filter_proj` measures how the *filters* are distributed and
therefore how few of them can stand in for all of them (Zador's `E_m ~ m^-1/d`
is a statement about quantising that measure). `patch` measures how few
*directions of the input* the layer needs. Compressing output channels is
governed by the first; compressing what each layer reads is governed by the
second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from intrinsic_rank.id_estimator import gride as _gride
except Exception as exc:  # pragma: no cover -- the golden rule depends on this
    raise RuntimeError(
        "GRIDE is mandatory: could not import intrinsic_rank.id_estimator.gride. "
        "Layer widths may not be set by any other means."
    ) from exc


def gride_id(X: np.ndarray, *, max_points: int = 2500, max_dims: int = 2048,
             n_neighbors: int = 16, n_iters: int = 8, seed: int = 0) -> float:
    """Median-over-scales GRIDE estimate of a point cloud's intrinsic dimension.

    GRIDE (Denti et al. 2022) returns one estimate per neighbour-order scale; the
    median is the robust plateau. Points and coordinates are subsampled -- random
    coordinate selection preserves distances up to the Johnson-Lindenstrauss
    factor, so the estimate is unaffected. `n_neighbors` shrinks automatically
    for small clouds, which matters here: a 64-channel layer offers only 64
    points.
    """
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
        raise RuntimeError("GRIDE returned no finite estimate -- cannot set widths.")
    return float(np.median(est))


@torch.no_grad()
def _row_space(G: torch.Tensor, rel_tol: float = 1e-6):
    """Numerical row space of the unfolded input, from its Gram `G = U^T U`.

    This is a *projection*, not a width: it only discards directions the
    calibration data does not excite at all (they carry no information about the
    filters either). No width is ever read off this rank -- that would be the
    PCA rule the golden constraint forbids.
    """
    lam, Q = torch.linalg.eigh(G.double())
    keep = lam > float(lam.max()) * rel_tol
    return Q[:, keep], lam[keep].clamp(min=0)


@torch.no_grad()
def layer_clouds(Wf: torch.Tensor, G: torch.Tensor, U: torch.Tensor | None = None,
                 max_patch_points: int = 2500, seed: int = 0) -> dict[str, np.ndarray]:
    """The three clouds for one conv layer.

    `Wf` -- filters flattened to `[C_out, C_in*k*k]`.
    `G`  -- Gram of the sampled unfolded input, `[C_in*k*k, C_in*k*k]`.
    `U`  -- the sampled unfolded input itself (only needed for the patch cloud).
    """
    V, lam = _row_space(G)
    coords = Wf.double() @ V                                   # [C_out, k]
    out = {"filter_proj": coords.cpu().numpy(),
           "neuron": (coords * torch.sqrt(lam)).cpu().numpy()}
    if U is not None:
        rng = np.random.default_rng(seed)
        n = U.shape[0]
        sel = rng.choice(n, min(n, max_patch_points), replace=False)
        out["patch"] = U[torch.as_tensor(sel, device=U.device)].double().cpu().numpy()
    return out


@torch.no_grad()
def rgb_patch_id(X: torch.Tensor, k: int = 3, stride: int = 2, n_points: int = 4000,
                 seed: int = 0) -> dict[str, float]:
    """Intrinsic dimension of an RGB image dataset, measured two ways.

    This is the question a CNN forces that an MLP never did. An MLP's first layer
    reads the *whole flattened image*, so `d_input` is the intrinsic dimension of
    the image manifold in `R^(3*H*W)` -- that is what
    `../compression_input_space` measured. A conv's first layer reads a
    `3*k*k` **patch**, and never anything larger. The two are different manifolds
    with different dimensions, and it is the patch one that bounds how few
    directions the first layer can be given.

    Colour is not a nuisance here, it is part of the measurement: the three
    channels of a patch are strongly correlated (natural images are close to
    grey plus two low-variance opponent directions), so the RGB patch manifold
    sits at a dimension well below the nominal `3*k*k`, and that gap is exactly
    the compressible part. `image` and `patch_gray` are reported alongside so the
    colour contribution can be read off directly.
    """
    Xc = X.detach()
    N, C, H, W = Xc.shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    flat = Xc.reshape(N, -1)
    sel = torch.randperm(N, generator=g)[:min(N, 3000)]
    out = {"image": gride_id(flat[sel].float().cpu().numpy(), seed=seed)}

    patches = torch.nn.functional.unfold(Xc, k, stride=stride)      # [N, C*k*k, L]
    patches = patches.transpose(1, 2).reshape(-1, C * k * k)
    sel = torch.randperm(patches.shape[0], generator=g)[:n_points].to(patches.device)
    Pt = patches.index_select(0, sel).float().cpu().numpy()
    out["patch"] = gride_id(Pt, seed=seed)
    out["patch_dim"] = float(C * k * k)

    if C == 3:                                                       # colour ablation
        gray = Pt.reshape(-1, 3, k * k).mean(1)
        out["patch_gray"] = gride_id(gray, seed=seed)
    return out


def widths_from_gride(dims: list[float], multiplier: float, orig: list[int],
                      m_min: int = 4) -> list[int]:
    """THE GOLDEN RULE. `M_l = round(multiplier * d_l)`, clipped to `[m_min, N_l]`."""
    return [int(np.clip(round(multiplier * d), m_min, n)) for d, n in zip(dims, orig)]


def zador_widths(dims: list[float], K: float, orig: list[int], cost: list[float],
                 weight: list[float] | None = None,
                 m_min: int = 4) -> tuple[list[int], list[float]]:
    """Budget allocation *derived from the same bound that sets the widths*.

    A single global multiplier spends the budget as if every layer converted
    channels into accuracy at the same rate. Zador says it does not. The
    quantisation distortion of a layer whose neuron measure has intrinsic
    dimension `d_l` falls as

            D_l(m_l) ~ (m_l / d_l)^(-2/d_l)

    so a layer with small `d_l` improves quickly with extra channels and one with
    large `d_l` barely improves at all. Minimising `sum_l D_l(m_l)` subject to a
    linear cost `sum_l c_l m_l = B` gives, from `dD_l/dm_l = -lambda c_l`,

            m_l = d_l * ( K / (d_l^2 c_l) ) ^ ( d_l / (2 + d_l) )

    with one global `K` that is swept to move along the budget. The widths are
    still `multiplier_l * d_l` -- GRIDE remains the only source of a width -- but
    the multiplier is now allowed to differ per layer, in the specific way the
    bound prescribes.

    On a CNN this is not a marginal refinement. Parameters concentrate in the
    deep, low-resolution layers while multiply-accumulates concentrate in the
    early, high-resolution ones, so a uniform multiplier that looks efficient in
    parameters can leave the compute almost untouched. Passing `cost` as MACs
    per channel is what lets the allocation target that instead.

    `weight` is `w_l`, how much distortion at layer `l` actually costs at the
    logits. Leaving it at 1 assumes every layer's error reaches the output
    equally, which is false and measurably so: the unweighted allocation starves
    the early, high-MAC layers, and error there is amplified by everything
    downstream with nothing able to correct it. `w_l = tr(Omega_l Sigma_l)` --
    the logit-pullback metric contracted with the layer's own output covariance
    -- is the expected squared logit error from replacing that layer's output
    with noise of matching scale, and is what the weighted variant uses.

    Returns `(widths, per_layer_multipliers)`.
    """
    if weight is None:
        weight = [1.0] * len(dims)
    mults, out = [], []
    for d, n, c, w in zip(dims, orig, cost, weight):
        d = max(float(d), 1e-6)
        c = max(float(c), 1e-9)
        w = max(float(w), 1e-30)
        mult = (K * w / (d * d * c)) ** (d / (2.0 + d))
        mults.append(mult)
        out.append(int(np.clip(round(mult * d), m_min, n)))
    return out, mults


def multiplier_grid(dims: list[float], widths: list[int]) -> list[float]:
    """Ladder from 1.5x up to the point where the *last* layer saturates.

    The ceiling is `max(N_l/d_l)`, not `min`: stopping when the single tightest
    layer fills up truncates the sweep long before the other layers have stopped
    being compressed, and widths stay legal past that point because
    `widths_from_gride` clips each layer at its own `N_l`. This is the same fix
    the MLP study needed before CIFAR-100 could reach its frontier.
    """
    ratio = max(n / max(d, 1e-9) for n, d in zip(widths, dims))
    grid = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0]
    out = [m for m in grid if m <= ratio]
    return out or [round(max(1.0, ratio), 2)]
