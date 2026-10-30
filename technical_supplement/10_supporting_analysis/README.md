# Supporting Analysis

Two kinds of artefact, both supporting claims made in the paper.

## `analysis_json/`

The thirteen JSON files the analysis scripts write. Every number quoted in the
paper is in one of these, and `../CLAIM_TO_EVIDENCE.md` maps them.

| File | Produced by | Carries |
| --- | --- | --- |
| `a1_selection_vs_correction.json` | `a1_selection_vs_correction.py` | the pooled contrast before regime stratification |
| `a2_regime.json` | `a2_regime.py` | contrasts by distance from dense |
| `a3_component_regime.json` | `a3_component_regime.py` | Table 4 |
| `a4_frontier_winners.json` | `a4_who_wins_at_the_frontier.py` | which operator wins where, family spread |
| `a5_collapse_law.json` | `a5_collapse_law.py` | accuracy collapse as the budget tightens |
| `a6_at_the_frontier.json` | `a6_at_the_frontier.py` | **the central result**, every contrast at each frontier |
| `a7_mechanism.json` | `a7_mechanism.py` | exact residuals on 154 layer-budget cells |
| `a8_theory.json` | `a8_theory.py` | Theorem 1 and Corollary 2 verified numerically |
| `a9_spectra.json` | `a9_spectra.py` | spectral tails, feed-forward against attention |
| `a10_dim_vs_rank.json` | `a10_dim_vs_rank.py` | GRIDE dimension against spectral effective rank |
| `fig4_tail.json` | `make_fig_extra.py` | the correlation behind Figure 2 |
| `thinet_comparison.json` | `make_tables.py` | the ThiNet decomposition in Table 5 |
| `paper_numbers.json` | `make_tables.py` | **every headline figure, consolidated** |

`paper_numbers.json` is the fastest way to check the paper's arithmetic without
rerunning anything. Each field corresponds to a number quoted in the text.

## `supplementary_where_repair_fails.md`

Material cut from the paper for space. AAAI 2027 permits no appendix and the
paper is capped at seven pages of content, so the two cases where the repair
provably cannot reach were compressed into a single paragraph in Section 7. This
file holds the full version: the sequential-absorption failure on a residual
architecture with its drift instrumentation, the Gauss-Newton metric that changes
sign with the repair, and the longer form of the implications discussion.

Everything in it is supported by experiments already in this supplement. Nothing
was written for it.

Read it alongside the Gauss-Newton subsection of `../CLAIM_TO_EVIDENCE.md`, which
records that this is the one claim in the paper whose stated magnitude the
artefacts do not support.
