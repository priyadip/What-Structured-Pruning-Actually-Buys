"""Novel strategies injected into the adaptive research phase.

All obey the golden rule: `apply(ctx, target_m)` returns exactly `target_m`
neurons, and `target_m` is `round(multiplier * GRIDE_d_l)`. All are closed-form
-- no backward passes, no optimizer.

The idea they share, and how it differs from the seed set
--------------------------------------------------------
Seed strategies 03-06, 09 *merge* neurons: they replace a group by its
barycentre. That is only output-preserving when every member of the group
co-activates on every input (the ReLU exactness condition). On a real trained
net that condition fails badly, and the error is invisible to every downstream
layer -- nothing can undo it.

These strategies instead **select-and-correct**:

  (a) SELECT a subset of neurons, keeping their weights *exactly*, so their ReLU
      responses are exact by construction; then
  (b) CORRECT the consuming layer in closed form against the **dense network's
      own** pre-activation target, with a free intercept.

Two things make (b) stronger than the usual activation-reconstruction map:

  * The target is the dense net's value at layer l+1, not this layer's own
    activations. So the fit absorbs error inherited from layers already
    compressed upstream, instead of compounding it (GPTQ / SparseGPT, adapted
    to structured neuron removal). `Bench.Y_orig` supplies this.
  * The intercept. Deleting a ReLU neuron removes a non-negative contribution
    whose *mean* is non-zero; no linear reweighting of the survivors can
    represent a constant, only a bias can. Leaving it out systematically biases
    every downstream pre-activation.

And (a) is done two ways:

  * SPAN  -- pivoted Cholesky on A^T A: the neurons that best span the
    *activation* space (equivalent to column-pivoted QR of A, far cheaper).
  * OMP   -- greedy selection against A^T Y: the neurons that best span the
    subspace the *next layer actually reads*. The activation space contains
    directions the next layer discards; spending budget on them is waste. This
    is the main departure from prior OT work, where the cost is always a
    geometric distance in parameter or activation space and never references
    the downstream target.
"""

from __future__ import annotations

import numpy as np
import torch

from run_autonomous_compression import (
    Strategy, _f64, ls_consuming, omp_select, pivoted_cholesky_select,
)


# ---------------------------------------------------------------------------
# N1 / N2 -- selection + intercept LS against the dense target
# ---------------------------------------------------------------------------

class N01_SpanLS(Strategy):
    name = "N01_span_ls"
    note = "pivoted-Cholesky spanning selection + intercept LS vs the dense target"

    def apply(self, ctx, m):
        G = ctx.gram()
        keep = pivoted_cholesky_select(G, m)
        V, c = ls_consuming(ctx.A[:, keep], ctx.Y, ctx.ridge)
        return ctx.W[keep], ctx.b[keep], V, c


class N02_OMPLS(Strategy):
    """The headline method."""
    name = "N02_omp_ls"
    note = "output-targeted OMP selection + intercept LS vs the dense target"

    def __init__(self, ridge: float | None = None, tag: str = ""):
        self.ridge = ridge
        if tag:
            self.name = f"N02_omp_ls_{tag}"
            self.note += f" (ridge={ridge})"

    def apply(self, ctx, m):
        keep = omp_select(ctx.gram(), ctx.cross(), m)
        V, c = ls_consuming(ctx.A[:, keep], ctx.Y,
                            ctx.ridge if self.ridge is None else self.ridge)
        return ctx.W[keep], ctx.b[keep], V, c


# ---------------------------------------------------------------------------
# N3 -- gradual (multi-stage) removal
# ---------------------------------------------------------------------------

class N03_GradualOMP(Strategy):
    """Remove neurons over several stages instead of in one shot.

    After each stage the consuming layer is re-fitted, so the *next* stage's
    OMP scores are computed against an already-corrected downstream map. One-shot
    selection has to commit to all removals while the correction is still
    unapplied; staging lets the two interleave. The schedule is geometric in the
    width, and the final stage always lands exactly on `target_m`.
    """
    name = "N03_gradual_omp"
    note = "geometric multi-stage OMP + LS (interleaves selection with correction)"

    def __init__(self, stages: int = 3):
        self.stages = stages
        if stages != 3:
            self.name = f"N03_gradual_omp_s{stages}"

    def apply(self, ctx, m):
        N = ctx.N
        ratios = np.geomspace(N, m, self.stages + 1)[1:]
        schedule = sorted({int(round(r)) for r in ratios} | {m}, reverse=True)
        schedule = [s for s in schedule if m <= s < N] or [m]
        if schedule[-1] != m:
            schedule.append(m)

        W, b, V, c = ctx.W, ctx.b, ctx.V, ctx.c
        for target in schedule:
            A = torch.relu(ctx.H @ W.t() + b)
            Ad = _f64(A)
            keep = omp_select(Ad.t() @ Ad, Ad.t() @ _f64(ctx.Y), target)
            W, b = W[keep], b[keep]
            A = A[:, keep]
            V, c = ls_consuming(A, ctx.Y, ctx.ridge)
        return W, b, V, c


# ---------------------------------------------------------------------------
# N4 -- bias calibration (the only free parameter ReLU leaves us)
# ---------------------------------------------------------------------------

class N04_BiasCalibratedOMP(Strategy):
    """OMP + LS, then alternate closed-form refinement of the kept neurons'
    biases and the consuming layer.

    Rescaling a kept neuron is pointless: relu(a*z + B) = a*relu(z + B/a) for
    a > 0, and the scale is already absorbed by the consuming layer. The **bias
    shift is the one genuinely new degree of freedom** ReLU leaves us, because it
    moves the activation *threshold* -- it changes which inputs the neuron fires
    on, which no linear correction downstream can do.

    For neuron m, hold everything else fixed and let
        u*_t = <Y_t - Yhat_t, v_m> / ||v_m||^2 + A_tm
    be the response that would minimise the downstream residual at sample t.
    Then pick
        beta_m = argmin_beta  sum_t ( relu(z_tm + beta) - u*_t )^2 ,
    a 1-D piecewise-quadratic solved by evaluating a quantile grid of candidates
    (no gradients -- just function evaluations). Sweep all neurons, re-solve the
    consuming layer, repeat.
    """
    name = "N04_bias_calibrated_omp"
    note = "OMP + LS, then alternating closed-form ReLU-threshold (bias) calibration"

    def __init__(self, rounds: int = 2, n_cand: int = 33):
        self.rounds = rounds
        self.n_cand = n_cand

    def apply(self, ctx, m):
        keep = omp_select(ctx.gram(), ctx.cross(), m)
        W, b = ctx.W[keep].clone(), ctx.b[keep].clone()
        Z = ctx.H @ W.t()                      # [T, M] pre-activation minus bias
        A = torch.relu(Z + b)
        V, c = ls_consuming(A, ctx.Y, ctx.ridge)

        Y = ctx.Y
        levels = torch.linspace(0.0, 1.0, self.n_cand, device=Y.device)
        for _ in range(self.rounds):
            Yhat = A @ V.t() + c
            E = Y - Yhat                        # [T, out] current residual
            for j in range(W.shape[0]):
                v = V[:, j]
                s = float(v @ v)
                if s < 1e-12:
                    continue
                u = (E @ v) / s + A[:, j]       # [T] ideal response of neuron j
                z = Z[:, j] + b[j]
                cand = torch.quantile(u - z, levels)
                cand = torch.cat([cand, torch.zeros(1, device=cand.device)])
                err = ((torch.relu(z.unsqueeze(1) + cand.unsqueeze(0))
                        - u.unsqueeze(1)) ** 2).sum(0)
                beta = cand[int(torch.argmin(err))]
                if beta == 0:
                    continue
                a_new = torch.relu(z + beta)
                E = E - torch.outer(a_new - A[:, j], v)   # keep the residual current
                A[:, j] = a_new
                b[j] = b[j] + beta
            V, c = ls_consuming(A, ctx.Y, ctx.ridge)
            A = torch.relu(ctx.H @ W.t() + b)
        V, c = ls_consuming(A, ctx.Y, ctx.ridge)
        return W, b, V, c


# ---------------------------------------------------------------------------
# N5 -- keep the spanning set, but fold the discarded mass in by OT-on-activations
# ---------------------------------------------------------------------------

class N05_OMPplusTransport(Strategy):
    """Hybrid: OMP selection for the producing layer, and for the consuming layer
    a *transport* of every discarded neuron's outgoing weight onto the kept set,
    where the transport plan is the least-squares reconstruction coefficients
    themselves (R = A_keep^+ A_full, column-normalised to be a genuine plan).

    This is the OT view of the correction: R is a transport plan from the full
    neuron measure to the kept one, but its cost is the *activation* residual
    rather than a parameter-space distance. Included to isolate how much of the
    gain comes from the intercept (absent here) versus the selection rule.
    """
    name = "N05_omp_transport"
    note = "OMP selection + LS reconstruction map used as an OT plan (no intercept)"

    def apply(self, ctx, m):
        keep = omp_select(ctx.gram(), ctx.cross(), m)
        A = _f64(ctx.A)
        Ak = A[:, keep]
        G = Ak.t() @ Ak
        lam = ctx.ridge * (G.diagonal().mean() + 1e-12)
        R = torch.linalg.solve(G + lam * torch.eye(len(keep), dtype=A.dtype, device=A.device),
                               Ak.t() @ A)                       # [M, N]
        V = (_f64(ctx.V) @ R.t()).float()
        return ctx.W[keep], ctx.b[keep], V, ctx.c.clone()


STRATEGIES = [
    N01_SpanLS(),
    N02_OMPLS(),
    N02_OMPLS(ridge=1e-6, tag="lowridge"),
    N03_GradualOMP(stages=3),
    N04_BiasCalibratedOMP(rounds=2),
    N05_OMPplusTransport(),
]
