# The method — stated for the paper

Measured over **96 (checkpoint × GRIDE-budget) cells**: 8 MLP checkpoints across
MNIST / FashionMNIST / CIFAR-10 / CIFAR-100, 10–14 multipliers each, 2 400+ runs.
Everything is training-free: no backward pass, no optimizer, no labels — only
closed-form linear algebra over unlabelled calibration inputs.

**A novel strategy wins 96 of 96 cells. No seed strategy ever wins one.**

---

## The three effects, in order of size

The method is a stack of three ideas. Their contributions are very unequal, and
saying so is more useful than quoting a single winner.

| # | effect | mean gain (heavy, ≤x6) | mean gain (light, >x6) |
|---|---|---|---|
| 1 | **corrected least-squares rebuild** vs. any merging / pruning / OT-transport | **+16.1 overall** (range +0.1 … +47.9) | |
| 2 | **output-targeted (OMP) selection** vs. activation-span selection | +5.09 | +1.22 |
| 3 | **augmented dictionary + forward-backward** vs. plain OMP | +1.27 | +0.25 |

Effect 1 dwarfs the rest. The headline is not "my selection rule is better" — it
is **rebuild the consuming layer in closed form instead of merging neurons**.
Effects 2 and 3 are real, consistent, and matter most where compression is
heaviest, but they are refinements on top.

---

## Setup

Fold BatchNorm so hidden layer *l* is the pure pair `(W_l, b_l)`. Write

- `H_l ∈ R^{T×d}` — the layer's input over `T` unlabelled calibration samples,
  **computed through the layers already compressed**;
- `A_l = relu(H_l W_lᵀ + b_l) ∈ R^{T×N}` — its activations;
- `Y_l ∈ R^{T×p}` — the **dense, uncompressed** network's pre-activation at layer
  `l+1`;
- `M_l = round(multiplier · d_l)`, `d_l` the GRIDE intrinsic dimension.

Layers are processed left to right.

## Effect 1 — rebuild, don't merge

Pick a subset `S`, `|S| = M`, keeping those neurons' weights **exactly**, then
solve

```
    [V'; c']  =  argmin  ‖ [A_l[:,S] , 1] [V'; c']ᵀ  −  Y_l ‖_F²  +  λ‖V'‖_F²
```

— one `(M+1)×(M+1)` linear system. Three details each matter:

**Selection, not averaging.** Section 6.5's exactness condition says a cluster of
neurons may be replaced by its barycentre without error *only if* every member is
simultaneously active or inactive on every input. On a trained network that fails
badly, and the error enters upstream of everything downstream — no later layer
can observe it, let alone undo it. Keeping survivors exact moves the entire
approximation burden into a *linear* problem, which can then be solved optimally.
This is where the +16 points come from.

**The intercept.** Deleting a ReLU unit removes a non-negative contribution with
non-zero *mean*. A reweighting of the survivors spans no constants; only a bias
does. Omitting it puts a systematic offset on every downstream pre-activation
that grows with the compression ratio — precisely the regime the Zador budget
puts us in. The intercept column is never shrunk by the ridge.

**The dense target.** `Y_l` is the *uncompressed* network's value, while `A_l`
comes from the *compressed* stack. Each layer's solve is therefore a correction
step that absorbs inherited error rather than a fresh approximation that
compounds it — GPTQ / SparseGPT's mechanism, moved from weight quantization to
structured neuron removal.

## Effect 2 — select against what the next layer reads

Choose `S` by matrix orthogonal matching pursuit with dictionary `A_l` and target
`Y_l`: greedily maximise the reduction in `‖Y_l − A_l[:,S] (A_l[:,S])⁺ Y_l‖_F²`.
Implemented as a pivoted-Cholesky recursion on `A_lᵀA_l` run simultaneously on
`A_lᵀY_l`, so selection costs `O(N·M·p)` and never factorises `A_l` itself.

Selecting to span the *activation* space instead (pivoted Cholesky alone) loses
5.09 points at heavy compression. The activation space contains directions the
next layer discards; spending a GRIDE-limited budget on them is waste. This is
the operational form of Section 4.5's counterexample — distance between neuron
signatures is not the quantity that governs output error.

## Effect 3 — let the data choose from a richer set of neurons

`N08_augdict_fwdbwd`, the overall winner (31/96 cells), adds two things:

**An augmented dictionary.** A compressed layer may contain *any* neuron, not
only one of the N already present. Build candidates: the originals, barycentric
merges at several granularities, and threshold-shifted copies `relu(w·x + b')`.
Then let the selection decide. A merged atom is kept only if it earns its slot on
calibration data — exactly the check OT merging lacks. The threshold shifts
matter more than they look: several biases along one direction form a spline
basis in that direction, letting M neurons express a piecewise-linear response
where selection alone gives a single kink. (Rescaling is pointless — the
consuming layer absorbs any positive scale — so the threshold is the only
genuinely free parameter ReLU leaves.)

**Exact forward-backward selection.** Forward greedy never reconsiders. Overshoot
to `K = 3M` atoms, then delete one at a time, each time removing the atom whose
removal least increases the least-squares error. For a ridged LS system that
increase is closed-form, `ΔSSE_j = ‖Θ_j‖²/C_jj` with `C = (ΦᵀΦ+λI)⁻¹`, and `C` is
downdated by the matching rank-1 formula — no system is ever refactorised.

---

## What this says about the paper's thesis

**The GRIDE budget is achievable on every checkpoint tested**, training-free, at
63–82% parameter reduction (see `02_results.md`). Accuracy is a smooth, monotone
function of the multiplier on all eight models.

**The required multiplier is task-dependent, not architecture-dependent.** MNIST
reaches the window at ~x3.4, FashionMNIST at ~x5.3–5.7, CIFAR-10 at ~x13.5–18.9,
CIFAR-100 at ~x12.9–18. Within a dataset, the 2-hidden-layer and 3-hidden-layer
models — and the `quad` and `oct` width families, which differ 2× in every
width — need almost the same multiplier. That is exactly what a `M_l ∝ d_l` rule
predicts and what a fixed-sparsity rule does not: the *shape* is set by intrinsic
dimension and only the *scale* is left to the task.

**The Zador shape is already the right allocation.** Phase 3 frees each layer's
multiplier independently at a fixed parameter budget — still GRIDE-legal, since
each width stays `round(mult_l · d_l)` — and coordinate descent returns the
uniform allocation on both CIFAR-10 and CIFAR-100. There is no better way to
spread a budget across layers than proportional to `d_l`.

**Zador's bound fixes a rate, not a constant.** `E_m ~ m^{-1/D}` governs how
quantization error decays; it fixes neither the constant nor the multiplier at
which a downstream classifier stops caring. The sweep measures that constant
empirically, and the striking part is that it is *stable across layers* whose
GRIDE dimensions differ by 2× within the same model.
