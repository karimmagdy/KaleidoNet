#!/bin/bash
# Run all pruning baselines across 4 datasets, 3 seeds each
# This produces: magnitude_pruning_*, random_pruning_*, linear_schedule_* JSON files

cd /Users/kmagdy-ma-eg/Workspace/Research/KaleidoNet

for dataset in cifar10 cifar100 tiny_imagenet stl10; do
    for seed in 1 2 3; do
        echo "============================================================"
        echo "Running ALL baselines: dataset=$dataset, seed=$seed"
        echo "============================================================"
        python3 experiments/baselines/pruning_baselines.py \
            --baseline all \
            --dataset "$dataset" \
            --seed "$seed" \
            --steps 5000
    done
done

echo "ALL PRUNING BASELINES COMPLETE"
