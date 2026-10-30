#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
while pgrep -f "e7_walltime[.]py" >/dev/null; do sleep 15; done
while pgrep -f "finish[.]sh" >/dev/null; do sleep 15; done
echo "[extend] E7 finished; extending the E5 grid for fashionmnist_vgg11_bn"
CUDA_VISIBLE_DEVICES=0 python3 e5_joint_budget.py \
    --models fashionmnist_vgg11_bn \
    --alpha-ch 9.25 10.0 12.0 16.0 20.0 \
    --alpha-r 4.0 6.0 8.0 12.0 > results/e5_extend.log 2>&1
echo "[extend] done; merging"
python3 merge_e5.py
python3 summarize.py
echo "[extend] complete $(date +%H:%M:%S)"
