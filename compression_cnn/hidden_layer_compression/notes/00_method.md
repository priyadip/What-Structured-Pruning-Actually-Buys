# Method — channel compression for VGG and ResNet

## The object being compressed

For an MLP a "neuron" is a row `w_i` of a weight matrix, and the layer is
`relu(W h + b)`. For a conv layer, flatten each filter to
`w_i ∈ R^(C_in·k·k)` and unfold the input to `U ∈ R^(N·H_out·W_out , C_in·k·k)`;
then the layer is `relu(U W^T + b)` — the same expression. A filter is a neuron
whose "samples" are patches instead of images.

Everything downstream follows from that identity:

| MLP | CNN |
|---|---|
| `A ∈ R^(T, N)` activations | `A ∈ R^(N·H·W, C)` channel activations at every position |
| select `m` columns of `A` | select `m` **groups** of `k·k` columns of `unfold(A)` |
| `min ‖[A_S,1]Θ − Y‖` | identical, with `Y` the consumer's unfolded pre-activation |
| BatchNorm1d folded into `Linear` | BatchNorm2d folded into `Conv2d` |

With `k·k = 1` the group routines reduce to the scalar ones, and
`common/selftest.py` asserts they return identical index sets.

## The unit graph

A **unit** is one producer conv whose output channels are cut, plus the single
consumer that reads them.

**VGG.** `features` is a chain, so every conv is a unit and its consumer is the
next conv — or, for the last one, the classifier (which reads the flattened
1×1×512 map, so `k·k` becomes `H·W`).

**ResNet.** Only the *internal* convs are units:

```
BasicBlock:  conv1 → conv2                 conv1's output is internal
             conv2 → (+ shortcut) → relu   conv2's output is on the stream
Bottleneck:  conv1 → conv2, conv2 → conv3  both internal
```

A residual stream is written by every block in its stage *and* carried by an
identity path. Removing one of its channels removes it from every producer at
once, and no single consumer refit can compensate, because the error is a sum
over producers that never appears in any one consumer's target. Internal
channels have exactly one consumer, so the MLP argument applies unchanged.

This is not a large concession. A BasicBlock costs
`C_in·p·k² + p·C_out·k²` — linear in the internal width `p` — so compressing `p`
scales nearly the whole block. On a ResNet18 the four `layer4` convs alone are
~75% of the parameters, and all but the shortcut are reachable this way.

## Sequential correction

Dense and compressed networks are advanced in **lock-step**. At unit `l`:

* the **design matrix** comes from the compressed path, so the fit sees the error
  already inherited from upstream;
* the **target** is the dense network's pre-activation at the consumer, so that
  error is *corrected* rather than compounded.

This is GPTQ/SparseGPT's idea applied to structured channel removal, and it is
what `Bench.Y_orig` provided in the MLP study. Here it is computed on the fly:
storing every layer's activations for a ResNet34 at the required calibration size
would not fit, but two single-layer buffers do.

## The width rule

```
M_l = round(multiplier × d_l)
```

`d_l` is the GRIDE intrinsic dimension of layer `l`'s **effective neuron cloud**:
the `C_out` filters, flattened to `R^(C_in·k·k)` and projected onto the column
space of the unfolded input — the part of each filter the data can actually
excite. `common/gride_conv.py` also measures two alternatives and logs all three:

* `neuron` — the same projection scaled by the singular values, which makes the
  coordinates literally the channel's pre-activation vector. This is what the MLP
  study used, so it is the continuity check.
* `patch` — the intrinsic dimension of the layer's *input*, which is a different
  question and is what study 2 budgets against.

The projection onto the data's column space is a projection, never a width: it
discards only directions the calibration set does not excite at all. Reading a
width off that rank would be the PCA rule the golden constraint forbids.

## Calibration sizing

The binding constraint is the opposite of the intuitive one. Deep layers have
many channels but a tiny spatial extent, so each image contributes only
`H_out·W_out` rows — four for a VGG11's last unit, one for the classifier. A
1024-image calibration set leaves a `512·9+1`-unknown system underdetermined, and
an underdetermined refit returns a null-space direction that still evaluates to a
plausible accuracy. `plan_calibration` derives the requirement from a shape-only
dry run; `require_overdetermined` refuses to fit below it.

## Strategies

Three families, and the distinction between them is the experiment.

**prune** (`S1_magnitude`) — keep the largest filters, change nothing else.

**merge** (`S6_ot_intrafusion`, `S8_kmeans_merge`) — replace a group of filters by
their barycentre. This is what OT-based pruning does. It is output-preserving only
when every filter in the group co-activates on every *patch* — the ReLU exactness
condition, which for a conv must hold at every spatial position, not merely on
every sample. `S7_ot_then_ls` adds a consumer refit on top.

**select-and-correct** (`S2`–`S5`) — keep a subset of filters exactly, so their
ReLU responses are exact by construction, then re-solve the consumer in closed
form against the dense target with a free intercept. `S4_omp_ls` scores
candidates against `U^T Y`, i.e. against the subspace the next layer actually
reads, rather than against a geometric distance between filters. That is the
departure from OT pruning.

**dictionary** (`S9`, `S10`) — offer merges and threshold-shifted copies as
*candidates* and let the output-targeted selection decide, which restores the
check merging lacks. `relu(w·x + b')` for several `b'` is a spline basis along one
direction; rescaling would be pointless since the consumer absorbs any positive
scale, so the threshold is the one genuinely free parameter.

## Budget allocation

A single global multiplier spends the budget as if every layer converted channels
into accuracy at the same rate. Zador says it does not: distortion falls as
`(m_l/d_l)^(−2/d_l)`, so a layer with small `d_l` improves quickly and one with
large `d_l` barely improves. Minimising `Σ_l w_l D_l(m_l)` under a linear cost
`Σ_l c_l m_l` gives

```
m_l = d_l · ( K · w_l / (d_l² · c_l) ) ^ ( d_l / (2 + d_l) )
```

with one global `K` swept along the budget. Widths remain `multiplier_l × d_l`;
only the split across layers changes. `w_l = tr(Ω_l Σ_l)` measures how much
distortion at layer `l` actually costs at the logits — see
`../data_reduction_and_fit/notes/00_method.md` for `Ω`. Results in
`01_findings.md`.
