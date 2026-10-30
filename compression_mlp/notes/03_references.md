# References and provenance

**No external code is vendored in this folder.** Everything in `compression_mlp/` was
written from scratch against the project's own modules (`models/`, `utils/`,
`intrinsic_rank/`). The papers below are what the methods relate to — cited so
the contributions and the baselines can be positioned, not copied from.

## Intrinsic dimension

- **GRIDE** — Denti, Doimo, Laio, Mira, *"The generalized ratios intrinsic
  dimension estimator"*, Scientific Reports 12, 2022. arXiv:2104.13832.
  Generalises TwoNN to neighbour-order ratios `(2k, k)` across scales, giving one
  estimate per scale. Implemented in `intrinsic_rank/id_estimator.py::gride`;
  this folder reduces the per-scale array to its **median** (the robust plateau).
- **TwoNN** — Facco, d'Errico, Rodriguez, Laio, *"Estimating the intrinsic
  dimension of datasets by a minimal neighborhood information"*, Scientific
  Reports 7, 2017. Used only as the printed cross-check.
- Why not PCA: PCA-99%-variance rank is a **linear** measure and overcounts a
  curved manifold. Measured on these very checkpoints, PCA gives 200–690 where
  GRIDE gives 15–40 on the same layers. See `feedback-gride-only-widths`.

## Quantization budget

- **Zador's theorem** — Zador, *"Asymptotic quantization error of continuous
  signals and the quantization dimension"*, IEEE Trans. Inf. Theory, 1982. The
  optimal `m`-point quantization error of a measure supported on a set of
  dimension `D` decays as `m^{-1/D}` (exponent `2/D` for `W_2`). This is what
  makes the width rule `M_l ∝ d_l` the right *shape*; the constant is
  empirical, and measuring it is what the multiplier sweep does.

## OT-based neuron merging (the baselines)

- **Intra-Fusion** — Theus, Geimer, Wicke, Hofmann, Anagnostidis, Singh,
  *"Towards Meta-Pruning via Optimal Transport"*, ICLR 2024. arXiv:2402.07839.
  Repo: `github.com/alexandertheus/Intra-Fusion`. Rank neurons by importance,
  keep the top-M, and OT-transport the discarded ones' parameters onto the
  survivors instead of simply dropping them. Reimplemented here as seed
  strategy `04_cosine_ot_intrafusion` (exact EMD via POT, column-normalised
  plan applied to both the producing and consuming dimensions — the form that
  matched the reference in earlier work in `../../Research/existOTtransform/`).
  The project's `models/vgg.py` and `models/resnet.py` were ported from this
  repo.
- **DOTResize** — arXiv:2507.04517. Re-projects *all* N neurons into M new ones
  through an entropy-regularised (Sinkhorn) transport plan computed on
  activation signatures, rather than keeping a subset. Explored in
  `../../Research/dotresize/`; its finding — that re-projecting everything beats
  discarding at heavy compression — is what motivated looking past pure
  importance ranking here.
- **Neuron merging under ReLU** — Kim et al., *"Neuron Merging: Compensating for
  Pruned Neurons"*, NeurIPS 2020. The source of the co-activation exactness
  condition that seed strategies 3, 5, 6 and 9 are built on, and of the
  `outgoing = sum, incoming = weighted mean` decode rule in `merge_by_labels`.

## Layer-wise reconstruction (what the winning method builds on)

- **ThiNet** — Luo, Wu, Lin, ICCV 2017. Channel selection by greedily minimising
  the *next* layer's reconstruction error, plus a least-squares rescale. The
  closest ancestor of `N02_omp_ls`; the differences here are the intercept, the
  dense-network target (rather than the current layer's own output), and OMP
  rather than greedy-subtractive selection.
- **GPTQ** — Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023. arXiv:2210.17323.
- **SparseGPT** — Frantar & Alistarh, ICML 2023. arXiv:2301.00774.
  Both introduce the *sequential* trick this folder adapts: compress layer `l`
  using activations recomputed through the **already-compressed** layers, so each
  solve corrects inherited error instead of compounding it. Both do this for
  weight quantization / unstructured sparsity; applying it to **structured
  neuron removal with a GRIDE-set width** is what is new here.

## Numerical methods used

- **Column-pivoted QR / pivoted Cholesky** — Businger & Golub (1965); Golub &
  Van Loan §5.4. Pivoted Cholesky on `AᵀA` visits columns in the same order as
  pivoted QR of `A` at `O(N·m)` instead of `O(T·N²)`
  (`pivoted_cholesky_select`).
- **Orthogonal matching pursuit** — Pati, Rezaiifar, Krishnaprasad (1993),
  extended to a matrix target here (`omp_select`): the greedy gain
  `‖P_res[j]‖² / d[j]` is evaluated against `AᵀY`, so selection targets what the
  next layer reads.
- **Backward elimination for least squares** — the removal cost
  `‖Θ_j‖²/C_jj` with `C = (ΦᵀΦ+λI)⁻¹` and its rank-1 downdate are standard
  (Miller, *Subset Selection in Regression*, 2002)
  (`forward_backward_select`). Negative result here.
- **Gauss-Newton / Fisher metric** — the layer-wise weighting
  `M_l = E[J_lᵀJ_l]`, with the closed-form ReLU-MLP backward recursion in
  `Bench._jacobian_metrics`. Negative result here; see
  `00_design_decisions.md`.
- **POT (Python Optimal Transport)** — Flamary et al., JMLR 2021. Used for the
  exact EMD in the Intra-Fusion baseline; a greedy nearest-kept fallback runs if
  it is unavailable.
