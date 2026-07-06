# KaleidoNet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21224908.svg)](https://doi.org/10.5281/zenodo.21224908)

**Stable Hard Pruning in Elastic MoE Vision Transformers via Cubic Sparsity Scheduling**

KaleidoNet explores **differentiable budget-controlled pruning** in Mixture-of-Experts (MoE) Vision Transformers. The core contribution is a **cubic sparsity schedule with gradient masking** that enables stable, selective hard pruning of MoE expert widths — replacing the standard Lagrangian FLOPs penalty, which we show fails at per-neuron granularity.

### Key Results (3 seeds, 5000 steps each)

**CIFAR-100** (32×32, 100 classes):

| Model | Val Acc | Active FLOPs | Params (active/total) |
|-------|---------|-------------|----------------------|
| Dense ViT | 36.07 ± 0.59% | 236.7M | 1.82M |
| KaleidoNet (cubic pruning) | 31.21 ± 0.37% | 132.4M | 2.18M / 5.49M |

**Tiny-ImageNet** (64×64, 200 classes):

| Model | Val Acc | Active FLOPs | Params (active/total) |
|-------|---------|-------------|----------------------|
| Dense ViT | 19.79 ± 0.30% | 242.8M | 1.87M |
| KaleidoNet (cubic pruning) | 16.85 ± 0.23% | 134.9M | 2.23M / 5.54M |

- **~1.80x FLOPs reduction** with 85–87% accuracy retention across both datasets
- **MoE expert width**: pruned to ~30% active neurons (fc1: 231/768, fc2: 58/192)
- **Attention heads**: all 6/6 retained (not pruned in current schedule)
- **Model surgery**: 5.49M → 3.38M parameters after physically removing dormant weights (1.63x compression)

> **Status**: Prototype-stage research. Results are on CIFAR-100 and Tiny-ImageNet. Larger-scale benchmarks are needed.

## Method

The standard approach to differentiable compute control is a **Lagrangian penalty** on FLOPs. We show this fails for selective pruning: per-neuron FLOPs contributions (~10⁻⁴) are too weak relative to task gradients (~10⁻²), causing **uniform collapse** rather than selective pruning.

Our solution:
1. **Cubic sparsity schedule** (Zhu & Gupta, 2017): ramps sparsity from step 500→4000, target 70%
2. **Gradient masking**: zero gradients on pruned logits (set to -100) to prevent recovery
3. **Dual optimizer**: mask parameters at 3x higher LR (Adam, 9e-4) vs. weights (AdamW, 3e-4)
4. **Gumbel-sigmoid masks**: soft during training (straight-through), deterministic hard threshold at inference

### Architecture

```
Image → PatchEmbed → [KaleidoNetBlock × 4] → ClassHead
                         │
                         ├── ElasticAttention (6 heads, learnable head masks)
                         ├── MoE FFN (4 experts, top-1 routing)
                         │     └── ElasticLinear experts (learnable width masks)
                         └── Confidence head (early exit)
```

## Ablation Study (8 configs × 2000 steps)

| Config | Val Acc | Active FLOPs | Notes |
|--------|---------|-------------|-------|
| Dense baseline | 24.23% | 172.5M | Reference |
| MoE only | 24.63% | 172.8M | +0.4pp, no FLOPs cost |
| Early exit only | 24.95% | 172.5M | +0.7pp, regularization effect |
| **MoE + Early exit** | **25.82%** | **172.8M** | **Best at 2000 steps** |
| Elastic only | 20.66% | 152.8M | Pruning hurts short runs |
| All pillars | 20.85% | 153.1M | Recovers with longer training (→31.24% at 5000 steps) |

## Quick Start

```bash
pip install -e ".[full]"

# Dense ViT baseline (seeded)
python experiments/baselines/dense_vit_baseline.py --seed 1

# KaleidoNet with cubic pruning (5000 steps, seeded)
python run.py experiments/baselines/train_cifar100.py --seed 1

# Multi-seed runs (3 seeds, both models)
python experiments/multi_seed_run.py --seeds 1 2 3

# Analyze multi-seed results
python experiments/analyze_multi_seed.py

# Tiny-ImageNet (Dense ViT baseline, seeded)
python experiments/baselines/dense_vit_tiny_imagenet.py --data-dir ./data/tiny-imagenet-200 --seed 1

# Tiny-ImageNet (KaleidoNet, seeded)
python run.py experiments/baselines/train_tiny_imagenet.py --data-dir ./data/tiny-imagenet-200 --seed 1

# Tiny-ImageNet multi-seed
python experiments/multi_seed_run.py --dataset tiny-imagenet --data-dir ./data/tiny-imagenet-200 --seeds 1 2 3

# Analyze Tiny-ImageNet results
python experiments/analyze_multi_seed.py --dataset tiny-imagenet

# Pillar ablation (8 configs × 2000 steps)
python run.py experiments/ablations/pillar_ablation.py

# Inference benchmark (latency + model surgery report)
python run.py experiments/benchmarks/benchmark_inference.py --steps 100
```

## Known Limitations

- **Accuracy gap**: -4.86pp on CIFAR-100, -2.94pp on Tiny-ImageNet vs. Dense ViT. Training budget (5000 steps) is well below convergence; published ViT-Tiny reaches ~75% on CIFAR-100 with full training.
- **No wall-clock speedup**: MoE routing overhead dominates at this model scale on MPS. Sparse forward paths (index_select) are implemented but net latency is higher than Dense ViT.
- **Small-scale datasets only**: CIFAR-100 and Tiny-ImageNet. No ImageNet-1K evaluation yet.
- **Attention pruning inactive**: Head mask logits exist but the cubic schedule only targets ElasticLinear widths. All attention heads survive.

## Project Structure

```
kaleidonet/
  core/elastic.py         # ElasticLinear, ElasticAttention (sparse forward paths)
  routing/moe.py          # MoE layer, TopKRouter
  morphing/lagrangian.py  # Lagrangian budget manager (disabled — fails at neuron granularity)
  backbone/universal.py   # KaleidoNetBlock, UniversalBackbone
  training/trainer.py     # Trainer with cubic pruning, seed control, dual optimizer
  export.py               # Model surgery: physically remove pruned weights
  model.py                # End-to-end KaleidoNet model
experiments/
  baselines/              # Dense ViT and KaleidoNet training scripts
  ablations/              # Pillar ablation study
  benchmarks/             # Inference latency benchmarks
results/                  # JSON result files
```

## Exploratory Components (Future Work)

The following modules exist in the codebase but are **not validated** and **not part of the current contribution**:

- **Pathfinding routing** (`routing/pathfinder.py`): A*-inspired cost-aware expert selection. Not evaluated in ablations.
- **Incremental growth** (`growth/`): FractalNet backbone with variance-transfer widening. Not integrated with pruning.
- **Universal backbone** (`tokenizers/`): Multi-modal tokenizer for vision + text. Only vision path tested.
- **Morph controller** (`morphing/controller.py`): Per-input dynamic width adjustment via Hutchinson trace. Not active in current training.
