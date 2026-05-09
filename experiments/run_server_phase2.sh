#!/usr/bin/env bash
# Phase 2 server experiments for KaleidoNet paper.
#
# GPU 0: Lagrangian baseline on CIFAR-100 (3 seeds, 50k steps)
# GPU 1: All methods on Tiny-ImageNet (3 seeds, 50k steps)
# GPU 2: CUDA latency benchmark
#
# All jobs run via nohup with unbuffered output for real-time logging.
# Usage:
#   chmod +x experiments/run_server_phase2.sh
#   ./experiments/run_server_phase2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/phase2"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "KaleidoNet Phase 2 Server Experiments"
echo "  Project root: $PROJECT_ROOT"
echo "  Log dir:      $LOG_DIR"
echo "  Started:      $(date)"
echo "============================================================"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------
# GPU 0: Lagrangian baseline on CIFAR-100
# ---------------------------------------------------------------
echo "[GPU 0] Lagrangian pruning baseline -- CIFAR-100, 3 seeds, 50k steps"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup python -m experiments.run_convergence \
    --method lagrangian \
    --dataset cifar100 \
    --seeds 1 2 3 \
    --steps 50000 \
    --skip-existing \
    > "$LOG_DIR/lagrangian_cifar100.log" 2>&1 &
PID_LAGRANGIAN=$!
echo "  PID=$PID_LAGRANGIAN  log=$LOG_DIR/lagrangian_cifar100.log"

# ---------------------------------------------------------------
# GPU 1: All methods on Tiny-ImageNet
# ---------------------------------------------------------------
echo "[GPU 1] All methods -- Tiny-ImageNet, 3 seeds, 50k steps"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup python -m experiments.run_convergence \
    --method all \
    --dataset tiny_imagenet \
    --seeds 1 2 3 \
    --steps 50000 \
    --skip-existing \
    > "$LOG_DIR/all_tiny_imagenet.log" 2>&1 &
PID_TINY=$!
echo "  PID=$PID_TINY  log=$LOG_DIR/all_tiny_imagenet.log"

# ---------------------------------------------------------------
# GPU 2: CUDA latency benchmark
# ---------------------------------------------------------------
echo "[GPU 2] CUDA inference latency benchmark"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 nohup python -m experiments.benchmarks.benchmark_inference \
    > "$LOG_DIR/cuda_latency_benchmark.log" 2>&1 &
PID_BENCH=$!
echo "  PID=$PID_BENCH  log=$LOG_DIR/cuda_latency_benchmark.log"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "All jobs launched. Monitor with:"
echo "  tail -f $LOG_DIR/lagrangian_cifar100.log"
echo "  tail -f $LOG_DIR/all_tiny_imagenet.log"
echo "  tail -f $LOG_DIR/cuda_latency_benchmark.log"
echo ""
echo "PIDs: lagrangian=$PID_LAGRANGIAN  tiny_imagenet=$PID_TINY  benchmark=$PID_BENCH"
echo "Kill all:  kill $PID_LAGRANGIAN $PID_TINY $PID_BENCH"
