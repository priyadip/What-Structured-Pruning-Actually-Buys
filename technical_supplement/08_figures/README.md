# Figure Index

## In the paper

| Paper number | File | Produced by | Reads |
| --- | --- | --- | --- |
| Figure 1 | `In paper/fig1_mechanism.pdf` | `make_fig_mechanism.py` | `a8_theory.json` for panel (c) |
| Figure 2 | `In paper/fig4_tail.pdf` | `make_fig_extra.py`, `fig4_tail()` | `a7_mechanism.json`, `a8_theory.json` |
| Figure 3 | `In paper/fig3_budget.pdf` | `make_fig_extra.py`, `fig3_budget()` | all three study CSVs via `load_all` |
| Figure 4 | `In paper/fig5_arch.pdf` | `make_fig_extra.py`, `fig5_arch()` | `a6_at_the_frontier.json` |

The file numbering and the paper numbering do not agree, because the figures were
written before the final ordering was settled. The table above is the mapping.

Figure 1 is the only one that is not a plot of measurements. Panels (a) and (b)
are drawn schematics: the compressible unit, and Theorem 1 rendered as an
orthogonal projection onto `span[U_S, 1]`. Panel (c) is measured, on 154 cells.

Figure 2 is the strongest single image in the paper. Every one of 154 points lies
below the diagonal, so Corollary 2 is never violated, and the points track the
bound at r = +0.78 in log-log. The colour gradient shows the bound and the
measured difference collapsing together as more units are retained, which is the
mechanism in one picture.

## Generated but not in the paper

| File | Why it is not included |
| --- | --- |
| `Generated not in paper/fig2_ranking.pdf` | Cut for space at the 7 page limit; both its numbers, the 0.78 against 15.15 spread and the 15 of 25 winner change, are stated in the Section 5 prose |
| `Generated not in paper/fig1_main.pdf` | Superseded by `fig1_mechanism` |

Both are retained so the record is complete and so the figure scripts run to
completion without modification.

## Transformer study figures

`Transformer study/` holds the diagnostic figures produced during the Transformer
work. None appears in the paper. They document the budget ladder, the drift
instrumentation behind the sequential-absorption result, and the per-layer
sensitivity, and they are the fastest way to see what that study actually did.

## Reproducing them

```
cd analysis/scripts
python3 make_fig_mechanism.py     # Figure 1
python3 make_fig_extra.py         # Figures 2, 3, 4
python3 make_figures.py           # the unused ranking figure
```

All four paper figures are written to `analysis/figures/` as both PDF and
PNG. The PDFs are what the paper embeds. Compressed PNG copies, about 70 percent
smaller, are produced by `compress_figures.py`.

Figure heights are set in the scripts rather than in the LaTeX, so regenerating
preserves the paper's page fit. Changing `figsize` will change the pagination.
