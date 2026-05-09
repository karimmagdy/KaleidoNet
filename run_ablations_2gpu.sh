#!/bin/bash
# Run 27 ablation experiments across GPUs 0+1 (NEVER GPU 2 - reserved for other students).
# Variants: cubic_only, masking_only, dual_only
# Datasets: cifar10, cifar100, tiny_imagenet
# Seeds: 1, 2, 3
# Steps: 50,000 (convergence)
# Resume-safe via --skip-existing.

set -e
cd ~/Research/KaleidoNet
mkdir -p results/convergence/logs

echo "============================================================"
echo "KaleidoNet Schedule Add-on Ablation Sweep (2-GPU)"
echo "Started: $(date)"
echo "============================================================"

# Phase 1: cubic_only on GPU 0, masking_only on GPU 1 (parallel, all datasets, all seeds)
CUDA_VISIBLE_DEVICES=0 python3 experiments/run_convergence.py \
  --method cubic_only --dataset all --seeds 1 2 3 --steps 50000 --skip-existing \
  > results/convergence/logs/gpu0_cubic_only.log 2>&1 &
PID0=$!
echo "[Phase 1] GPU 0 (cubic_only): PID $PID0"

CUDA_VISIBLE_DEVICES=1 python3 experiments/run_convergence.py \
  --method masking_only --dataset all --seeds 1 2 3 --steps 50000 --skip-existing \
  > results/convergence/logs/gpu1_masking_only.log 2>&1 &
PID1=$!
echo "[Phase 1] GPU 1 (masking_only): PID $PID1"

wait $PID0 $PID1
echo "[Phase 1] complete at $(date)"

# Phase 2: dual_only split across both GPUs by dataset
CUDA_VISIBLE_DEVICES=0 python3 experiments/run_convergence.py \
  --method dual_only --dataset cifar10 --seeds 1 2 3 --steps 50000 --skip-existing \
  > results/convergence/logs/gpu0_dual_only_cifar10.log 2>&1 &
PID2=$!
echo "[Phase 2] GPU 0 (dual_only/cifar10): PID $PID2"

CUDA_VISIBLE_DEVICES=1 python3 experiments/run_convergence.py \
  --method dual_only --dataset tiny_imagenet --seeds 1 2 3 --steps 50000 --skip-existing \
  > results/convergence/logs/gpu1_dual_only_tinyim.log 2>&1 &
PID3=$!
echo "[Phase 2] GPU 1 (dual_only/tiny_imagenet): PID $PID3"

wait $PID2 $PID3

# Phase 3: dual_only on cifar100
CUDA_VISIBLE_DEVICES=0 python3 experiments/run_convergence.py \
  --method dual_only --dataset cifar100 --seeds 1 2 3 --steps 50000 --skip-existing \
  > results/convergence/logs/gpu0_dual_only_cifar100.log 2>&1
echo "[Phase 3] complete at $(date)"

echo "============================================================"
echo "All 27 ablation runs complete at $(date)"
echo "============================================================"
