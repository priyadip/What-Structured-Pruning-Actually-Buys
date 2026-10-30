# Technical Supplement

Supporting material for *What Structured Pruning Actually Buys: A Decomposition,
and Why Selection Criteria Stop Mattering*.

The experimental record behind the paper: every result CSV, the full
hyperparameter and hardware description, the ablation outcomes, and the figures
and tables the analysis produced. This part contains no code. The code is in the repository root.

**Start here:** `CLAIM_TO_EVIDENCE.md` maps every quantitative claim in the paper
to the file that carries it and the script that produced it. It is the fastest
way to check whether a particular number is supported, and it names the one claim
this supplement finds weaker than the paper states.

---

## Contents

| Folder | Contents |
| --- | --- |
| `01_experimental_setup/` | Unit definition, the three studies, the analysis dependency order, plus the Transformer study report and findings and the project progress log |
| `02_hyperparameters/` | Every hyperparameter, training and compression kept separate |
| `03_training/` | How the 24 checkpoints were trained, splits, times, one documented exception |
| `04_evaluation/` | Metric definitions, the operating-point rule, statistical treatment |
| `05_hardware/` | CPU, GPU, memory, OS, CUDA and library versions, captured from the machine |
| `06_results_csv/` | All 31 result CSVs in six labelled groups |
| `07_ablations/` | The seven targeted experiments and what each one changed |
| `08_figures/` | Figures in the paper, generated ones that are not, and study diagnostics |
| `09_tables/` | LaTeX tables in the paper and generated ones that are not |
| `10_supporting_analysis/` | The thirteen analysis JSON outputs, and the material cut from the paper for space |
| `CLAIM_TO_EVIDENCE.md` | The evidence map |

Each numbered folder carries its own `README.md`.

---

## What kind of paper this is, and why that shapes the supplement

The paper is a re-analysis. It does not introduce a new compression operator and
does not train a new model to make its point. It takes 3922 compression runs that
already existed across three independent studies, normalises them into a single
record type, and shows that reading them at the deployed operating point rather
than averaged over a budget sweep reverses the received ordering of two
components.

Two consequences follow for anything a reviewer wants to check.

The accuracy-level evidence is a **re-reading of recorded runs**, so the primary
artefacts are the CSVs in `06_results_csv/`. Those files were written by the
original experiment drivers, not by the analysis, and every number in the paper
traces back to one of them.

The mechanism-level evidence is **recomputed from checkpoints** and involves no
accuracy evaluation at all. Section 4 instantiates the quantities of Theorem 1
exactly on 154 layer-budget cells in double precision. That part does not depend
on trusting any recorded run.

---

## Scale of the experimental record

| Quantity | Value |
| --- | --- |
| Compression runs re-analysed | 3922 |
| Checkpoints | 25: 8 fully connected, 16 convolutional, 1 Transformer |
| Datasets | MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNet32, MNLI |
| Layer-budget cells with exact residuals | 154 |
| Compression operators compared | 17 Transformer, 14 vision |
| Result CSVs | 31, totalling 10403 data rows |

The headline count reconciles exactly. `06_results_csv/README.md` shows the nine
files that sum to 3922 and names the files deliberately excluded so that no run
is counted twice.

---

## Deliberate omissions

**Trained weights.** The 24 vision checkpoints total 1066 MiB, past what a
supplement should carry. `trainer/checkpoint/CHECKPOINT_MANIFEST.csv`
lists every checkpoint with its SHA-256 digest, its architecture, and whether
Section 4 needs it, together with the command that retrains it. The eight
checkpoints Section 4 requires total 101 MiB and take about 70 minutes.

**Compressed-model artefacts.** The `*_best.safetensors` and
`*_frontier.safetensors` files written by the drivers are outputs, not inputs,
and nothing in the paper depends on them.

**Raw datasets.** Standard public benchmarks, obtained as described in
`03_training/README.md`.
