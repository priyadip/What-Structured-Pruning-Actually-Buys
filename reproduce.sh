#!/usr/bin/env bash
# Reproduce the paper from the recorded runs, in the order the dependencies
# require. Tier 1 needs no GPU and no checkpoints; tiers 2 and 3 do.
#
#   ./reproduce.sh tables      every table and figure from the shipped CSVs  (~1 min, CPU)
#   ./reproduce.sh mechanism   re-derive Section 4 from checkpoints           (~25 min, GPU)
#   ./reproduce.sh spectra     re-derive Section 6 spectra from DistilBERT    (~10 min, GPU)
#   ./reproduce.sh all         all of the above
set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-tables}"

run_tables() {
  echo "== analyses that read only CSV and JSON"
  ( cd analysis/scripts
    for s in a1_selection_vs_correction a2_regime a3_component_regime \
             a4_who_wins_at_the_frontier a5_collapse_law a6_at_the_frontier; do
      echo "   $s"; python3 "$s.py" > /dev/null
    done
    echo "== tables and figures"
    python3 make_tables.py > /dev/null
    python3 make_figures.py > /dev/null
    python3 make_fig_mechanism.py > /dev/null
    python3 make_fig_extra.py > /dev/null )
  echo "   tables -> analysis/tables/   figures -> analysis/figures/"
}

run_mechanism() {
  echo "== Section 4: exact residuals from checkpoints"
  echo "   requires the 8 fully connected checkpoints listed in"
  echo "   trainer/checkpoint/CHECKPOINT_MANIFEST.csv (101 MiB, not shipped)"
  ( cd analysis/scripts
    python3 a7_mechanism.py
    python3 a8_theory.py
    python3 a10_dim_vs_rank.py )
}

run_spectra() {
  echo "== Section 6: spectral tails (downloads DistilBERT on first run)"
  ( cd analysis/scripts && python3 a9_spectra.py )
}


case "$STAGE" in
  tables)    run_tables ;;
  mechanism) run_mechanism ;;
  spectra)   run_spectra ;;
  all)       run_mechanism; run_spectra; run_tables ;;
  *) echo "unknown stage '$STAGE'"; sed -n '2,12p' "$0"; exit 1 ;;
esac
echo "done: $STAGE"
