#!/usr/bin/env bash
# Wait for E5 and E6, then run E7 on genuinely idle GPUs, then summarise.
# Bracket patterns: an unbracketed pgrep would match this script's own polling
# shell, which is exactly how a chained job idled for an hour on 2026-07-26.
set -u
cd "$(dirname "$0")"
wait_for() { while pgrep -f "$1" >/dev/null; do sleep 20; done; echo "[finish] $2 done $(date +%H:%M:%S)"; }

wait_for "e5_joint[_]budget.py" "E5"
wait_for "e6_id[_]vs_training.py" "E6"
wait_for "chain[.]sh" "chain"

echo "[finish] GPUs idle -> E7 wall-clock (re-run; the chain's copy overlapped E5)"
CUDA_VISIBLE_DEVICES=0 python3 e7_walltime.py --iters 100 > results/e7.log 2>&1
echo "[finish] E7 done"

python3 e1b_robust_frontier.py --k 2.0 > results/e1b.log 2>&1
python3 summarize.py
python3 make_paper_tables.py
echo "[finish] all experiments complete $(date +%H:%M:%S)"
