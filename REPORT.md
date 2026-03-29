# KaleidoNet: Experiment Report

## Summary

This report documents experiments on **cubic sparsity scheduling for stable hard pruning in elastic MoE Vision Transformers**. The core finding is that the standard Lagrangian FLOPs penalty fails to produce selective pruning at per-neuron granularity, and an explicit cubic schedule with gradient masking provides a stable alternative.

**Primary results** (3 seeds, 5000 steps each):

| Dataset | Dense ViT | KaleidoNet | FLOPs Reduction | Accuracy Retention |
|---------|-----------|------------|-----------------|--------------------|
| CIFAR-100 | 36.07 ± 0.59% at 236.7M | 31.21 ± 0.37% at 132.4M | **1.79x** | 86.5% |
| Tiny-ImageNet | 19.79 ± 0.30% at 242.8M | 16.85 ± 0.23% at 134.9M | **1.80x** | 85.2% |

The FLOPs reduction is consistent across both datasets (~1.8x), with accuracy retention of 85–87%.

> **Caveat**: Results use a short training budget (5000 steps ≪ convergence) on small-scale datasets. Evidence of generalization to larger benchmarks (e.g., ImageNet) is needed before publication.

---

## 1. Experimental Setup

| Property | Value |
|----------|-------|
| Datasets | CIFAR-100 (32×32, 100 classes), Tiny-ImageNet (64×64, 200 classes) |
| Hardware | Apple Silicon MPS |
| Framework | PyTorch 2.8.0, Python 3.9.6 |
| Training | AdamW lr=3e-4, cosine schedule, 5000 steps, batch 64 |
| Mixed Precision | Disabled (MPS compatibility) |

### Model Configurations

| Model | embed_dim | blocks | heads | experts | top_k | elastic | Params (CIFAR-100) | Params (Tiny-ImageNet) |
|-------|-----------|--------|-------|---------|-------|---------|--------------------|-----------------------|
| Dense ViT | 192 | 4 | 6 | — | — | No | 1.82M | 1.87M |
| KaleidoNet | 192 | 4 | 6 | 4 | 1 | Yes | 5.49M (2.18M active) | 5.54M (2.23M active) |

Parameter counts differ slightly due to patch embedding dimensions (CIFAR-100: patch_size=4, Tiny-ImageNet: patch_size=8).

---

## 2. Main Results

### 2.1 CIFAR-100: Dense ViT vs. KaleidoNet (5000 steps, 3 seeds)

| Model | Val Acc (mean ± std) | Active FLOPs | Speedup |
|-------|----------------------|-------------|--------|
| Dense ViT | 36.07 ± 0.59% | 236.7M | 1.00x |
| **KaleidoNet v3f (cubic pruning)** | **31.21 ± 0.37%** | **132.4M** | **1.79x** |

**Per-seed breakdown:**

| Model | Seed 1 | Seed 2 | Seed 3 |
|-------|--------|--------|--------|
| Dense ViT | 35.41% | 36.55% | 36.25% |
| KaleidoNet | 31.40% | 30.79% | 31.45% |

### 2.2 Tiny-ImageNet: Dense ViT vs. KaleidoNet (5000 steps, 3 seeds)

| Model | Val Acc (mean ± std) | Active FLOPs | Speedup |
|-------|----------------------|-------------|--------|
| Dense ViT | 19.79 ± 0.30% | 242.8M | 1.00x |
| **KaleidoNet (cubic pruning)** | **16.85 ± 0.23%** | **134.9M** | **1.80x** |

**Per-seed breakdown:**

| Model | Seed 1 | Seed 2 | Seed 3 |
|-------|--------|--------|--------|
| Dense ViT | 19.70% | 20.12% | 19.54% |
| KaleidoNet | 17.10% | 16.64% | 16.82% |

The Tiny-ImageNet results confirm the CIFAR-100 findings: KaleidoNet achieves a consistent ~1.80x FLOPs reduction with ~85% accuracy retention. The accuracy gap (-2.94pp) is proportionally similar to CIFAR-100 (-4.86pp), and the pruning schedule converges to the same 30% hard-active / 40.3% active-params target.

### 2.3 Compression Breakdown (CIFAR-100, v3f)

| Component | Original | Active | Kept |
|-----------|----------|--------|------|
| Total parameters | 5.49M | 2.18M | 39.8% |
| After surgery (dormant removed) | 5.49M | 3.38M | 61.5% |
| MoE expert fc1 (per expert) | 768 neurons | 231 | 30.1% |
| MoE expert fc2 (per expert) | 192 neurons | 58 | 30.2% |
| Attention heads (per block) | 6 heads | 6 | 100% |

### 2.4 Training Trajectory (CIFAR-100)

| Step | Hard Active | Val Acc | Notes |
|------|-------------|---------|-------|
| 0 | 100% | — | All neurons on |
| 500 | 100% | 10.0% | Pruning begins |
| 1000 | 74% | 15.8% | Cubic ramp accelerating |
| 2000 | 43% | 23.8% | |
| 3000 | 32% | 28.3% | |
| 4000 | 30% | 29.8% | Pruning target reached |
| 5000 | 30% | 31.24% | Still improving (not converged) |

---

## 3. Ablation Study (2000 steps, single seed)

All configs: embed_dim=192, 4 blocks, 6 heads, batch_size=64.

| Config | Val Acc | Active FLOPs (M) | Wall Time (s) |
|--------|---------|-------------------|---------------|
| dense_baseline | 24.23% | 172.5 | 967 |
| moe_only | 24.63% | 172.8 | 3,051 |
| elastic_only | 20.66% | 152.8 | 1,171 |
| early_exit_only | 24.95% | 172.5 | 875 |
| moe+elastic | 20.57% | 153.1 | 3,358 |
| **moe+early_exit** | **25.82%** | **172.8** | **2,191** |
| elastic+early_exit | 20.34% | 152.8 | 1,111 |
| all_pillars | 20.85% | 153.1 | 7,653 |

### Key Findings

1. **MoE adds capacity without FLOPs cost**: +0.4pp accuracy, +0.2% FLOPs (top-1 of 4 experts)
2. **Elastic pruning hurts short runs**: -3.57pp at 2000 steps. The cubic schedule reaches ~50% sparsity by step 2000 — insufficient training time to recover. At 5000 steps, accuracy reaches 31.24%.
3. **Early exit provides free regularization**: +0.72pp with zero FLOPs overhead (confidence threshold not triggered at this model scale, but the auxiliary loss regularizes)
4. **MoE + Early Exit is the best short-run combo**: 25.82%, best of all 8 configs at 2000 steps

---

## 4. Why Lagrangian FLOPs Control Fails

The Lagrangian gradient per mask logit is:

$$\frac{\partial \mathcal{L}}{\partial m_i} = \lambda \cdot \frac{\text{flops}_i}{\text{budget}} \cdot \frac{\sigma'(m_i/\tau)}{\tau}$$

With per-neuron FLOPs ~24K and budget ~119M, each neuron's contribution is ~10⁻⁴ — orders of magnitude weaker than task gradients (~10⁻²). Result: **all logits drift down uniformly** rather than selecting which neurons to prune. Active fraction collapses from 100% → 0% without selectivity.

### Solution: Cubic Schedule + Gradient Masking

- **Cubic schedule**: $s_t = s_f \cdot (1 - (1 - \text{progress})^3)$, from step 500→4000, target 70% sparsity
- **Gradient masking**: pruned logits set to -100, gradients zeroed — no recovery allowed
- **Task loss drives selection**: without Lagrangian pressure, the task loss alone determines which neurons are important
- **Dual optimizer**: mask params (Adam, 9e-4) vs. weights (AdamW, 3e-4) — faster mask adaptation

---

## 5. Inference Infrastructure

### Sparse Forward Paths
- **ElasticLinear**: `weight.index_select(0, active_indices)` — reduced matmul for active neurons only
- **ElasticAttention**: QKV projection + attention + output projection restricted to active heads only
- Both paths produce identical output to dense+mask (verified by tests)

### Model Surgery (`kaleidonet/export.py`)
- Physically removes dormant weights from ElasticLinear and ElasticAttention
- Drops mask logits — produces compact state dict for deployment
- Result: 5.49M → 3.38M parameters (1.63x compression)

### Wall-Clock Benchmark (MPS, batch=64)

| Model | Mean Latency | Throughput |
|-------|-------------|------------|
| Dense ViT | 7.15ms | 8,945 samples/s |
| KaleidoNet (sparse paths) | 131.4ms | 487 samples/s |

KaleidoNet is **~18x slower** than Dense ViT on MPS at this model scale. This is expected: MoE routing overhead (gating network + scatter/gather) dominates when the model is small. Sparse forward paths reduce per-expert compute but cannot offset routing overhead. This limitation would likely diminish at larger model scales or on CUDA with optimized MoE kernels.

---

## 6. Limitations

1. **Small-scale datasets only**: CIFAR-100 (32×32) and Tiny-ImageNet (64×64). No evidence of generalization to full-scale benchmarks (e.g., ImageNet-1K).
2. **Accuracy gap**: -4.86pp on CIFAR-100, -2.94pp on Tiny-ImageNet vs. Dense ViT. Short training budget (5000 steps ≪ convergence).
3. **No wall-clock speedup**: FLOPs savings do not translate to latency savings at toy model scale on MPS.
4. **Attention heads not pruned**: Cubic schedule targets ElasticLinear only. All 6/6 heads survive.
5. **MPS only**: No CUDA testing. FlashAttention, torch.compile not available.

---

## 7. Reproducibility

```bash
# --- CIFAR-100 ---
# Dense ViT baseline (seeded)
python experiments/baselines/dense_vit_baseline.py --seed 1

# KaleidoNet v3f (cubic pruning, 5000 steps, seeded)
python run.py experiments/baselines/train_cifar100.py --seed 1

# Multi-seed runs (3 seeds, both models)
python experiments/multi_seed_run.py --seeds 1 2 3

# Analyze multi-seed results
python experiments/analyze_multi_seed.py

# --- Tiny-ImageNet ---
# Dense ViT baseline (seeded)
python experiments/baselines/dense_vit_tiny_imagenet.py --data-dir ./data/tiny-imagenet-200 --seed 1

# KaleidoNet (cubic pruning, 5000 steps, seeded)
python run.py experiments/baselines/train_tiny_imagenet.py --data-dir ./data/tiny-imagenet-200 --seed 1

# Multi-seed runs (3 seeds, both models)
python experiments/multi_seed_run.py --dataset tiny-imagenet --data-dir ./data/tiny-imagenet-200 --seeds 1 2 3

# Analyze Tiny-ImageNet results
python experiments/analyze_multi_seed.py --dataset tiny-imagenet

# --- Ablation & Benchmarks ---
# Pillar ablation (8 configs × 2000 steps)
python run.py experiments/ablations/pillar_ablation.py

# Inference benchmark + model surgery report
python run.py experiments/benchmarks/benchmark_inference.py --steps 100 --batch-size 64
```

### Key Files

| File | Purpose |
|------|---------|
| `kaleidonet/core/elastic.py` | ElasticLinear, ElasticAttention with sparse forward paths |
| `kaleidonet/training/trainer.py` | Cubic pruning schedule, gradient masking, dual optimizer, seed control |
| `kaleidonet/export.py` | Model surgery — physically remove pruned weights |
| `kaleidonet/routing/moe.py` | MoE layer with top-k routing |
| `experiments/baselines/train_cifar100.py` | KaleidoNet CIFAR-100 training (supports `--seed`) |
| `experiments/baselines/train_tiny_imagenet.py` | KaleidoNet Tiny-ImageNet training (supports `--seed`, `--data-dir`) |
| `experiments/baselines/dense_vit_tiny_imagenet.py` | Dense ViT Tiny-ImageNet baseline |
| `experiments/benchmarks/benchmark_inference.py` | Latency benchmark with surgery stats |
| `results/` | JSON result files for all experiments |
