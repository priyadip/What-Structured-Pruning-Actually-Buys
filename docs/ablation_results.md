# Results — follow-up experiments

Generated from `ablations/results/*.csv` on 2026-07-27 05:15. Every experiment re-uses the production compression code from `compression_cnn/`; only the variable under test differs. Sections for experiments that have not run say so rather than being omitted.

## E1 — seed variance at the frontier configuration

Each of the 16 saved frontier configurations re-run at seeds 0, 1, 2. The configuration is fixed (same strategy, same multiplier); the seed changes which calibration images are drawn and the GRIDE subsample, hence the measured `d_l` and therefore the widths.

| model | acc mean | s.d. | range | in window | widths identical? |
|---|---|---|---|---|---|
| `cifar100_resnet18` | 77.04 | 0.40 | 76.69–77.48 | 2/3 | **no** |
| `cifar100_resnet34` | 77.30 | 0.12 | 77.19–77.43 | 3/3 | **no** |
| `cifar100_vgg11_bn` | 70.81 | 0.36 | 70.53–71.22 | 3/3 | **no** |
| `cifar100_vgg16_bn` | 73.06 | 0.29 | 72.77–73.35 | 3/3 | **no** |
| `cifar10_resnet18` | 93.12 | 0.15 | 92.95–93.25 | 3/3 | **no** |
| `cifar10_resnet34` | 93.00 | 0.07 | 92.93–93.06 | 1/3 | **no** |
| `cifar10_vgg11_bn` | 90.36 | 0.10 | 90.27–90.47 | 3/3 | **no** |
| `cifar10_vgg16_bn` | 91.67 | 0.16 | 91.49–91.80 | 2/3 | **no** |
| `fashionmnist_resnet18` | 93.16 | 0.06 | 93.09–93.21 | 3/3 | **no** |
| `fashionmnist_resnet34` | 93.12 | 0.08 | 93.03–93.17 | 3/3 | **no** |
| `fashionmnist_vgg11_bn` | 92.13 | 0.10 | 92.03–92.23 | 2/3 | **no** |
| `fashionmnist_vgg16_bn` | 92.51 | 0.05 | 92.45–92.55 | 3/3 | **no** |
| `imagenet32_resnet34` | 56.09 | 0.21 | 55.89–56.31 | 2/3 | **no** |
| `imagenet32_resnet50` | 59.90 | 0.06 | 59.84–59.96 | 3/3 | **no** |
| `imagenet32_vgg16_bn` | 46.41 | 0.23 | 46.14–46.58 | 3/3 | **no** |
| `imagenet32_vgg19_bn` | 46.42 | 0.26 | 46.12–46.60 | 1/3 | **no** |

**Accuracy at a fixed configuration is stable**: median s.d. 0.14 points, max 0.40.

**But 8 of 48 runs fall outside the ±2.5 window**, and the widths differ between seeds on 16 of 16 models. Both follow from what a frontier *is*: the smallest budget that still clears the window, i.e. a point chosen to sit exactly at the boundary. A 0.1–0.4 point perturbation therefore flips a meaningful fraction of them out. This is the same systematic optimism the MLP study found for search subsamples, now shown for seeds.

The width instability traces to GRIDE, not to the least-squares solve: on `cifar100_vgg11_bn` the deepest unit's measured dimension ranges 77.1–90.2 across seeds (14% of its value), which moves `round(α·d)` by tens of channels.

## E3 — a faithful ThiNet control

ThiNet implemented as described — greedy *subtractive* selection against the consumer's **own current** output, then a per-channel scalar rescale — plus two variants that swap in one of our choices at a time, so the three differences can be attributed separately. Paired within (model, multiplier), so nothing is confounded by width.

| variant | Δ vs ours | s.e. | *t* | wins | cells |
|---|---|---|---|---|---|
| ThiNet, faithful (subtractive + own target + rescale) | -41.29 | 5.04 | -8.2 | 0/16 | 16 |
| …but scored against the **dense** target | -26.77 | 3.84 | -7.0 | 0/16 | 16 |
| …ThiNet selection + **our refit** | -4.99 | 1.11 | -4.5 | 0/16 | 16 |
| magnitude selection + our refit | -13.95 | 3.81 | -3.7 | 0/16 | 16 |
| span selection + our refit (ThiNet-like control in the paper) | -6.41 | 1.94 | -3.3 | 1/16 | 16 |
| **ours** (`U^T Y` selection + refit) | +0.00 | 0.00 | +nan | 0/16 | 16 |

**\textsc{ThiNet} as published loses 41.3 points** (*t* = -8.2, 16 cells). Swapping in one of our three choices at a time, holding the other two at ThiNet's:

- its own-output target → the **dense** target: **+14.5** (*t* = +4.7)
- its scalar rescale → our **full refit with intercept**: **+36.3** (*t* = +7.8)
- and with our refit already in place, everything that remains (selection algorithm *and* selection target) is worth only **+5.0**

The first two effects sum to +50.8, more than the 41.3-point total, so they are **sub-additive**: once the consuming layer is re-solved against anything, the choice of target matters much less, and vice versa. We therefore do not attribute a residual to the selection algorithm by subtraction — the contrasts above are what the data isolates, and no cell holds (subtractive selection, dense target, full refit) to separate it cleanly.

**The conclusion is the ablation's, reached independently.** The refit recovers 88% of the gap to a published method on its own; everything about *which* channels are chosen is a refinement on top. It also vindicates the paper's original proxy: `S3_span_ls` (span selection + our refit) sits at -6.41 against ThiNet-selection-plus-our-refit at -4.99, so using it as the ThiNet-like control was reasonable — if marginally unkind to ThiNet.

## E4 — the Gauss–Newton metric with the refit disabled

The experiment the project named as decisive three times and never ran. A 2×2 over (metric / plain) × (refit / no refit). The standing hypothesis — that the metric and the sequential refit correct the same thing twice — predicts the metric **helps** when no refit follows and **hurts or ties** when one does.

**All cells**

| condition | Δ (metric − plain) | s.e. | *t* | wins | cells |
|---|---|---|---|---|---|
| **no** downstream refit | +0.147 | 1.365 | +0.11 | 8/20 | 20 |
| downstream refit | -0.089 | 0.049 | -1.80 | 11/20 | 20 |

**Restricted to cells where the no-refit baseline clears 3× chance (6 cells)**

| condition | Δ (metric − plain) | s.e. | *t* | wins | cells |
|---|---|---|---|---|---|
| **no** downstream refit | -2.298 | 4.019 | -0.57 | 4/6 | 6 |
| downstream refit | +0.070 | 0.015 | +4.82 | 6/6 | 6 |

**The hypothesis is **not supported**.** It predicts a *positive* effect in the frozen arm; the measured effect is -2.298 (*t* = -0.57), and the refit arm is +0.070 (*t* = +4.82). The frozen arm is indistinguishable from zero rather than positive. The predicted ordering (frozen > refitted) does not even hold on this subset. 

So the double-counting story cannot be what is going on, at least on convolutions. The metric's earlier success on MLP input reduction remains unexplained by it, and the paper's Discussion should report the metric as a plain negative result on CNNs rather than as an explained one. Two caveats keep this from being fully decisive: without a refit the compressed models are far from the ±2.5 window at every budget tested, so the comparison happens in a regime the method never operates in; and the metric itself degrades towards the input (rank correlation 0.25–0.58 on early layers), so part of the null could be a bad estimate rather than a bad idea.

## E2 — calibration-set size

Calibration size is swept through `--od-factor`, the over-determination target that sets it, so the sweep is in the units the rule is stated in. `4` is the production default; `2` is the floor the driver accepts.

| model | od | calib | acc | vs default | widths = default? |
|---|---|---|---|---|---|
| `cifar10_resnet18` | 2 | 1024 | 93.06 | -0.10 | no |
| `cifar10_resnet18` | 4 | 1153 | 93.16 | +0.00 | yes |
| `cifar10_resnet18` | 8 | 2305 | 93.21 | +0.05 | no |
| `cifar10_resnet18` | 16 | 4609 | 93.14 | -0.02 | no |
| `cifar10_vgg11_bn` | 2 | 2305 | 90.40 | +0.13 | no |
| `cifar10_vgg11_bn` | 4 | 4609 | 90.27 | +0.00 | yes |
| `cifar10_vgg11_bn` | 8 | 9218 | 90.33 | +0.06 | no |
| `cifar10_vgg11_bn` | 16 | 16384 | 90.60 | +0.33 | no |

Full-range accuracy spread: 0.15–0.33 points across a 8× change in calibration size.

