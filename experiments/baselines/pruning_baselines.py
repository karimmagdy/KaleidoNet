"""Pruning baseline comparisons for KaleidoNet.

Implements 3 pruning baselines:
  1. Magnitude Pruning: one-shot post-training, prune by weight norm
  2. Random Structured Pruning: fixed random mask, train from scratch
  3. Linear Sparsity Schedule: linear ramp instead of cubic

Each baseline is run on the same KaleidoNet architecture so FLOPs are
directly comparable. Results saved to results/ directory.

Run:
    python experiments/baselines/pruning_baselines.py --baseline magnitude --dataset cifar100 --seed 1
    python experiments/baselines/pruning_baselines.py --baseline random --dataset cifar100 --seed 1
    python experiments/baselines/pruning_baselines.py --baseline linear --dataset cifar100 --seed 1
    python experiments/baselines/pruning_baselines.py --baseline all --dataset cifar100 --seed 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaleidonet.model import KaleidoNet
from kaleidonet.core.elastic import ElasticLinear
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig, set_seed


# ---- Dataset loaders ----

def get_loaders(dataset: str, batch_size: int = 64, data_dir: str | None = None):
    if dataset == "cifar10":
        from experiments.baselines.train_cifar10 import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size), 10, 32, 4
    elif dataset == "cifar100":
        from experiments.baselines.train_cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(batch_size=batch_size), 100, 32, 4
    elif dataset == "tiny_imagenet":
        from experiments.baselines.tiny_imagenet_data import get_tiny_imagenet_loaders
        loaders = get_tiny_imagenet_loaders(batch_size=batch_size, data_root=data_dir or "./data/tiny-imagenet-200")
        return loaders, 200, 64, 8
    elif dataset == "stl10":
        from experiments.baselines.stl10_data import get_stl10_loaders
        return get_stl10_loaders(batch_size=batch_size), 10, 96, 8
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_model(num_classes: int, image_size: int, patch_size: int, elastic: bool = True) -> KaleidoNet:
    return KaleidoNet(
        embed_dim=192,
        num_blocks=4,
        num_heads=6,
        num_experts=4,
        top_k=1,
        num_classes=num_classes,
        vocab_size=0,
        image_size=image_size,
        patch_size=patch_size,
        elastic=elastic,
        drop_path_rate=0.1,
    )


def calibrate_flops_budget(model: KaleidoNet, image_size: int, patch_size: int) -> tuple[int, int]:
    counter = FLOPsCounter()
    seq_len = (image_size // patch_size) ** 2
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=seq_len)
    probe_device = next(model.parameters()).device
    with torch.no_grad():
        probe = model({"images": torch.randn(1, 3, image_size, image_size, device=probe_device)})
    init_active = int(probe.get("active_flops", dense_flops))
    return dense_flops, init_active


def measure_final_flops(model: nn.Module, image_size: int) -> int | None:
    model.eval()
    try:
        dev = next(model.parameters()).device
        with torch.no_grad():
            out = model({"images": torch.randn(1, 3, image_size, image_size, device=dev)})
        return int(out["active_flops"])
    except Exception:
        return None


def eval_model(model: nn.Module, val_loader, device: str) -> dict:
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            logits = out["logits"]
            targets = batch["targets"]
            total_loss += criterion(logits, targets).item() * targets.shape[0]
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.shape[0]
    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct / max(total, 1),
    }


# ---- Baseline 1: Magnitude Pruning (one-shot post-training) ----

def run_magnitude_pruning(
    dataset: str, seed: int, steps: int, target_sparsity: float = 0.7,
    finetune_fraction: float = 0.2, data_dir: str | None = None,
):
    """Train full KaleidoNet, then prune by weight magnitude, then fine-tune."""
    print("=" * 60)
    print(f"Baseline: Magnitude Pruning | {dataset} | seed={seed}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Phase 1: Train full model (no cubic pruning — disable by setting start far out)
    config = TrainerConfig(
        lr=3e-4,
        max_steps=steps,
        warmup_steps=250,
        flops_budget=int(init_active * 0.50),
        log_interval=100,
        eval_interval=500,
        use_amp=False,
        seed=seed,
        sparsity_start_step=steps + 1000,  # Disable cubic pruning
        sparsity_end_step=steps + 2000,
    )

    print(f"\nPhase 1: Training full model for {steps} steps (no pruning)...")
    trainer = KaleidoNetTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )
    trainer.train()

    pre_prune_metrics = eval_model(model, val_loader, config.device)
    print(f"Pre-prune accuracy: {pre_prune_metrics['val_accuracy']:.4f}")

    # Phase 2: One-shot magnitude pruning on ElasticLinear layers
    print(f"\nPhase 2: Magnitude pruning (target sparsity={target_sparsity:.0%})...")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ElasticLinear):
                # Score each neuron by L2 norm of its weight row
                scores = m.weight.data.norm(dim=1)
                n = scores.numel()
                n_keep = max(int(n * (1 - target_sparsity)), m.min_width)
                threshold = scores.topk(n_keep, largest=True).values[-1]
                mask = scores >= threshold
                m.mask_logits.data[mask] = 10.0    # keep
                m.mask_logits.data[~mask] = -100.0  # prune

    post_prune_metrics = eval_model(model, val_loader, config.device)
    print(f"Post-prune accuracy: {post_prune_metrics['val_accuracy']:.4f}")

    # Phase 3: Fine-tune
    finetune_steps = int(steps * finetune_fraction)
    print(f"\nPhase 3: Fine-tuning for {finetune_steps} steps...")
    ft_config = TrainerConfig(
        lr=1e-4,  # Lower LR for fine-tuning
        max_steps=finetune_steps,
        warmup_steps=50,
        flops_budget=int(init_active * 0.50),
        log_interval=100,
        eval_interval=500,
        use_amp=False,
        seed=seed,
        sparsity_start_step=finetune_steps + 1000,  # Disable further pruning
        sparsity_end_step=finetune_steps + 2000,
    )
    ft_trainer = KaleidoNetTrainer(
        model=model, config=ft_config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )
    ft_trainer.train()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy: {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,}")

    results = {
        "model": "MagnitudePruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "finetune_steps": finetune_steps,
        "target_sparsity": target_sparsity,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "pre_prune_acc": pre_prune_metrics["val_accuracy"],
        "post_prune_acc": post_prune_metrics["val_accuracy"],
        **final_metrics,
    }
    _save_results(results, f"magnitude_pruning_{dataset}_seed{seed}.json")
    return results


# ---- Baseline 2: Random Structured Pruning ----

def run_random_pruning(
    dataset: str, seed: int, steps: int, target_sparsity: float = 0.7,
    data_dir: str | None = None,
):
    """Train with a fixed random structured mask from scratch."""
    print("=" * 60)
    print(f"Baseline: Random Structured Pruning | {dataset} | seed={seed}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Apply fixed random mask before training
    print(f"Applying random structured mask (target sparsity={target_sparsity:.0%})...")
    rng = torch.Generator().manual_seed(seed + 1000)  # Separate seed for mask
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ElasticLinear):
                n = m.mask_logits.numel()
                n_keep = max(int(n * (1 - target_sparsity)), m.min_width)
                # Random permutation to select which neurons to keep
                perm = torch.randperm(n, generator=rng)
                keep_indices = perm[:n_keep]
                m.mask_logits.data.fill_(-100.0)
                m.mask_logits.data[keep_indices] = 10.0

    # Freeze mask logits — they should not be trained
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            m.mask_logits.requires_grad_(False)

    # Train with fixed mask (disable cubic pruning)
    config = TrainerConfig(
        lr=3e-4,
        max_steps=steps,
        warmup_steps=250,
        flops_budget=int(init_active * 0.50),
        log_interval=100,
        eval_interval=500,
        use_amp=False,
        seed=seed,
        sparsity_start_step=steps + 1000,
        sparsity_end_step=steps + 2000,
    )

    trainer = KaleidoNetTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )
    trainer.train()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy: {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,}")

    results = {
        "model": "RandomPruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_sparsity": target_sparsity,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        **final_metrics,
    }
    _save_results(results, f"random_pruning_{dataset}_seed{seed}.json")
    return results


# ---- Baseline 3: Linear Sparsity Schedule ----

def run_linear_schedule(
    dataset: str, seed: int, steps: int, target_sparsity: float = 0.7,
    data_dir: str | None = None,
):
    """Same as KaleidoNet but with linear sparsity ramp instead of cubic."""
    print("=" * 60)
    print(f"Baseline: Linear Sparsity Schedule | {dataset} | seed={seed}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Use the standard trainer but monkey-patch the pruning method
    config = TrainerConfig(
        lr=3e-4,
        max_steps=steps,
        warmup_steps=250,
        flops_budget=int(init_active * 0.50),
        log_interval=100,
        eval_interval=500,
        use_amp=False,
        seed=seed,
        target_sparsity=target_sparsity,
    )

    trainer = KaleidoNetTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )

    # Override _apply_cubic_pruning with linear schedule
    original_method = trainer._apply_cubic_pruning

    @torch.no_grad()
    def _apply_linear_pruning(self_ref=trainer):
        cfg = self_ref.config
        step = self_ref.global_step

        if step < cfg.sparsity_start_step:
            return
        if step % cfg.sparsity_frequency != 0:
            return

        # Linear schedule: s_t = s_f * (t-t0)/(T-t0)
        t = min((step - cfg.sparsity_start_step) / max(cfg.sparsity_end_step - cfg.sparsity_start_step, 1), 1.0)
        current_sparsity = cfg.target_sparsity * t  # Linear instead of cubic

        for m in self_ref.model.modules():
            if isinstance(m, ElasticLinear):
                logits = m.mask_logits
                n = logits.numel()
                n_prune = max(int(n * current_sparsity), 0)
                n_keep = max(n - n_prune, m.min_width)
                n_prune = n - n_keep

                if n_prune > 0:
                    threshold = logits.topk(n_keep, largest=True).values[-1]
                    mask = logits >= threshold
                    logits.data[~mask] = -100.0

    trainer._apply_cubic_pruning = _apply_linear_pruning

    trainer.train()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy: {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,}")

    results = {
        "model": "LinearSchedule",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_sparsity": target_sparsity,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        **final_metrics,
    }
    _save_results(results, f"linear_schedule_{dataset}_seed{seed}.json")
    return results


def _save_results(results: dict, filename: str):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Pruning baseline comparisons")
    parser.add_argument("--baseline", choices=["magnitude", "random", "linear", "all"],
                        required=True, help="Which baseline to run")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet", "stl10"],
                        default="cifar100", help="Dataset")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--target-sparsity", type=float, default=0.7, help="Target sparsity fraction")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (for Tiny-ImageNet)")
    args = parser.parse_args()

    baselines = {"magnitude", "random", "linear"} if args.baseline == "all" else {args.baseline}

    if "magnitude" in baselines:
        run_magnitude_pruning(args.dataset, args.seed, args.steps, args.target_sparsity, data_dir=args.data_dir)

    if "random" in baselines:
        run_random_pruning(args.dataset, args.seed, args.steps, args.target_sparsity, data_dir=args.data_dir)

    if "linear" in baselines:
        run_linear_schedule(args.dataset, args.seed, args.steps, args.target_sparsity, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
