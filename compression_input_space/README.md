# `compression_input_space/` — input reduction on top of hidden compression

Second compression stage. `../compression_mlp/` shrank the **hidden layers** at a GRIDE
budget; this folder shrinks the **input dimension** and refits the first layer,
with the hidden layers **frozen**.

## The aim

For each hidden-compressed model, find the smallest input dimension `r` such that
the *fully* compressed model — reduced input **and** compressed hidden layers —
stays within **±2.5 points of the ORIGINAL dense accuracy**.

**Result: achieved on all 8 models, at 62.7–91.4% total parameter reduction.**
Every artefact is re-verified from disk by `verify_fully_compressed.py` (8/8).

## The rules, and how they are enforced

- **`r` comes only from GRIDE on the raw input data**, measured at runtime:
  `r = round(multiplier × d_input)`. `core/gride.py` raises rather than falling
  back to any other rule.
- **Only `(W₁, b₁)` and the data are used to fit.** No output, logit, or label of
  the frozen network enters the embedding objective.
- **No fine-tuning.** `load_compressed` sets `requires_grad_(False)` on every
  network parameter; only the landmark coordinates `Z` and the linear map `A`
  are optimised. Verified at runtime: 0/6 params require grad, none changes, none
  carries `.grad`.
- **The final artefacts are fitted on ALL training data** (`fitted_on_n` is 50 000
  / 60 000 in every saved spec); the staged search uses a 20 000-sample subsample
  for speed only.

## Layout

```
core/     geometry.py    chunked + geodesic (kNN-graph) distances
          landmarks.py   centroid_boundary + class_balanced/kmeans/greedy/random
          srgw_mds.py    joint stress + reconstruction; A eliminated in closed form
          oos.py         gower / nystrom / relaxed / linear-distillation
          baselines.py   opt_linear (the provable ceiling), pca, w1_svd, random
          model_io.py    load frozen model, fold in encoder, evaluate, save
          gride.py       GRIDE on raw inputs
novel_strategies/innovations.py   N1 logit-pullback metric, N2 RFF, N3 act-frequency
run_input_compression.py   stages A-D + adaptive rounds + full-data refit
summarize.py               regenerates input_compression_summary.md from the CSV
verify_fully_compressed.py rebuilds every saved model from disk and re-scores it
```

## Running it

```bash
cd <PROJECT_ROOT>

# full staged sweep over every hidden-compressed model
python3 compression_input_space/run_input_compression.py \
    --kind compressed --stages ABCD --exhaustive

# rebuild one configuration on ALL data and save it
python3 compression_input_space/run_input_compression.py \
    --kind compressed --models mnist_mlp3_bn --only N1_jacobian_linear@30

python3 compression_input_space/verify_fully_compressed.py --tag frontier
```

Stages: **A** linear baselines incl. the ceiling · **B** srGW-MDS `r × λ` ·
**C** landmark strategy `× K` · **D** out-of-sample extension · then adaptive
rounds that hot-load `novel_strategies/*.py`, then a full-data refit.

## What the sweep found

Read `notes/00_findings.md`. In one line: **srGW-MDS loses, and the manifold term
is why.**

- Ranked by mean gap to the best encoder at the same `(model, r)`:
  `N1_jacobian_linear` **0.05** < `opt_linear` 0.49 < `N3_actfreq` 0.50 <
  `srgw_gower` 1.01 < `N2_rff` 1.71 < `pca` 3.22 < `srgw_relaxed` 3.56 <
  `w1_svd` 6.81 < `srgw_nystrom` 7.19 < `random` 16.81.
- The λ ranking is monotone in *how much the MDS stress is switched off*
  (100 > 10 > 0.5 > 1.0 > 5.0 > 0.1 > 0.01), and the out-of-sample ordering runs
  the same way (`relaxed` — most faithful to the manifold — is worst).
- The winner, `N1`, is a closed-form reduced-rank regression in the metric the
  frozen network pulls back from its logits. It **folds** into the first layer,
  so it is a real parameter saving, not a reshaped cost.

## Honest accounting

A *linear* encoder folds — `A(Px) = (AP)x` — so input reduction is exactly a
rank-`r` factorisation of `W₁`, kept factored: `r·D + M₁·r` parameters instead of
`M₁·D`. A *distance-based* encoder (gower/nystrom/relaxed) cannot fold and costs
`K·D` flops per sample plus stored landmarks. Both numbers are in the CSV
(`encoder_params`, `encoder_flops`, `folds_into_layer1`); neither is omitted.
