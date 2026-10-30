#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

wait_for() {   # wait_for <bracketed-pattern> <label>
  while pgrep -f "$1" > /dev/null; do sleep 15; done
  echo "[chain] $2 finished at $(date +%H:%M:%S)"
}

echo "[chain] started $(date +%H:%M:%S)"

wait_for "e1_seed[_]variance.py" "E1"
echo "[chain] GPU0 free -> E4 (metric without refit)"
CUDA_VISIBLE_DEVICES=0 python3 e4_metric_norefit.py > results/e4.log 2>&1
echo "[chain] E4 done"

echo "[chain] GPU0 -> E2 (calibration size)"
CUDA_VISIBLE_DEVICES=0 python3 e2_calibration.py > results/e2.log 2>&1
echo "[chain] E2 done"

wait_for "e3_thinet[.]py" "E3"
echo "[chain] GPU1 free -> E5 (joint budget)"
CUDA_VISIBLE_DEVICES=1 python3 e5_joint_budget.py > results/e5.log 2>&1
echo "[chain] E5 done"

# E7 must run with the GPUs otherwise idle or the timings are meaningless.
wait_for "e4_metric[_]norefit.py" "E4"
wait_for "e2_calibration[.]py" "E2"
wait_for "e5_joint[_]budget.py" "E5"
echo "[chain] all GPU work done -> E7 (wall-clock, needs idle hardware)"
CUDA_VISIBLE_DEVICES=0 python3 e7_walltime.py --iters 100 > results/e7.log 2>&1
echo "[chain] E7 done"

python3 summarize.py
echo "[chain] finished $(date +%H:%M:%S)"
