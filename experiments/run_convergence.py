"""
Convergence training: run ALL methods on ALL datasets with extended training.

Runs 50,000 steps (10x smoke-test budget) with cosine annealing LR and logs
full training curves (loss + accuracy per eval step) for publication plots.

Methods: dense, magnitude, random, linear, lagrangian, kaleidonet
Datasets: cifar10, cifar100, stl10, tiny_imagenet

Usage:
    # Full sweep (all methods x all datasets x 3 seeds)
    python experiments/run_convergence.py

    # Single combo
    python experiments/run_convergence.py --method kaleidonet --dataset cifar100

    # Custom seeds / steps
    python experiments/run_convergence.py --seeds 1 2 3 4 5 --steps 100000

    # Resume-friendly: existing result files are skipped
    python experiments/run_convergence.py --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaleidonet.model import KaleidoNet
from kaleidonet.core.elastic import ElasticLinear
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig, set_seed
from experiments.baselines.lagrangian_pruning import (
    compute_soft_flops_fraction,
    LagrangianPruningTrainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "dense", "magnitude", "random", "linear", "lagrangian", "kaleidonet",
    # Ablation variants of KaleidoNet (cubic schedule, with each add-on toggled)
    "cubic_only",     # cubic schedule only (no gradient masking, no dual-rate)
    "masking_only",   # cubic + gradient masking (no dual-rate)
    "dual_only",      # cubic + dual-rate (no gradient masking)
]
ALL_DATASETS = ["cifar10", "cifar100", "stl10", "tiny_imagenet"]
DEFAULT_STEPS = 50_000
DEFAULT_SEEDS = [1, 2, 3]
RESULTS_DIR = os.path.join(ROOT, "results", "convergence")

# ---------------------------------------------------------------------------
# Dataset helpers (reuse existing loaders)
# ---------------------------------------------------------------------------

def get_loaders(dataset: str, batch_size: int = 64, data_dir: str | None = None):
    """Return (train_loader, val_loader), num_classes, image_size, patch_size."""
    if dataset == "cifar10":
        from experiments.baselines.train_cifar10 import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size), 10, 32, 4
    elif dataset == "cifar100":
        from experiments.baselines.train_cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(batch_size=batch_size), 100, 32, 4
    elif dataset == "stl10":
        from experiments.baselines.stl10_data import get_stl10_loaders
        return get_stl10_loaders(batch_size=batch_size), 10, 96, 8
    elif dataset == "tiny_imagenet":
        from experiments.baselines.tiny_imagenet_data import get_tiny_imagenet_loaders
        loaders = get_tiny_imagenet_loaders(
            batch_size=batch_size,
            data_root=data_dir or "./data/tiny-imagenet-200",
        )
        return loaders, 200, 64, 8
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


# ---------------------------------------------------------------------------
# Training curve logger
# ---------------------------------------------------------------------------

class CurveLogger:
    """Append loss/accuracy rows to a CSV for later plotting."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fp = open(path, "w", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(["step", "train_loss", "task_loss", "val_loss", "val_accuracy", "lr"])
        self._fp.flush()

    def log(self, step: int, train_loss: float, task_loss: float,
            val_loss: float | None, val_accuracy: float | None, lr: float):
        self._writer.writerow([
            step, f"{train_loss:.6f}", f"{task_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{val_accuracy:.6f}" if val_accuracy is not None else "",
            f"{lr:.2e}",
        ])
        self._fp.flush()

    def close(self):
        self._fp.close()


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _make_config(steps: int, seed: int, flops_budget: int, **overrides) -> TrainerConfig:
    """Base config for convergence runs with cosine annealing."""
    # Scale sparsity schedule proportionally to training length
    sparsity_start = int(steps * 0.10)   # 10% warmup before pruning
    sparsity_end = int(steps * 0.80)     # Pruning finishes at 80% of training
    return TrainerConfig(
        lr=3e-4,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),   # 5% warmup
        flops_budget=flops_budget,
        log_interval=100,
        eval_interval=max(steps // 100, 500),  # ~100 eval points
        use_amp=False,
        seed=seed,
        sparsity_start_step=sparsity_start,
        sparsity_end_step=sparsity_end,
        **overrides,
    )


def _train_with_curves(
    model: nn.Module,
    config: TrainerConfig,
    train_loader,
    val_loader,
    curve_logger: CurveLogger,
    task_loss_fn: nn.Module | None = None,
    patch_pruning_fn=None,
) -> dict:
    """Train using KaleidoNetTrainer and log curves. Returns final eval metrics."""
    trainer = KaleidoNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        task_loss_fn=task_loss_fn or nn.CrossEntropyLoss(),
    )

    # Optionally replace pruning method (for linear baseline)
    if patch_pruning_fn is not None:
        trainer._apply_cubic_pruning = patch_pruning_fn(trainer)

    # Monkey-patch the trainer to intercept metrics for curve logging
    _original_train_step = trainer.train_step
    _last_train_metrics: dict = {}

    def _hooked_train_step(batch):
        m = _original_train_step(batch)
        _last_train_metrics.update(m)
        return m

    trainer.train_step = _hooked_train_step

    _original_eval = trainer.eval_step

    def _hooked_eval():
        val_m = _original_eval()
        step = trainer.global_step
        curve_logger.log(
            step=step,
            train_loss=_last_train_metrics.get("total_loss", 0.0),
            task_loss=_last_train_metrics.get("task_loss", 0.0),
            val_loss=val_m.get("val_loss") if val_m else None,
            val_accuracy=val_m.get("val_accuracy") if val_m else None,
            lr=_last_train_metrics.get("lr", 0.0),
        )
        return val_m

    trainer.eval_step = _hooked_eval

    trainer.train()
    return trainer


def run_kaleidonet(dataset, seed, steps, data_dir=None):
    """Standard KaleidoNet with cubic pruning (the full method)."""
    label = f"kaleidonet_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  KaleidoNet | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    config = _make_config(steps, seed, flops_budget=int(init_active * 0.50))
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    _train_with_curves(model, config, train_loader, val_loader, logger)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    results = {
        "model": "KaleidoNet",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "curve_file": curve_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def _run_kaleidonet_variant(
    dataset, seed, steps, label_prefix,
    gradient_masking_enabled, dual_rate_enabled,
    data_dir=None,
):
    """Shared runner for KaleidoNet schedule add-on ablations.

    Reuses the full KaleidoNet pipeline (cubic schedule, MoE routing, early exit)
    but toggles the two schedule add-ons under test:
      - gradient_masking_enabled: when False, removed neurons can drift back
      - dual_rate_enabled: when False, mask logits use the main AdamW LR
    """
    label = f"{label_prefix}_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  {label_prefix} | {dataset} | seed={seed} | {steps} steps\n  "
          f"gradient_masking={gradient_masking_enabled}, dual_rate={dual_rate_enabled}\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    config = _make_config(
        steps, seed,
        flops_budget=int(init_active * 0.50),
        gradient_masking_enabled=gradient_masking_enabled,
        dual_rate_enabled=dual_rate_enabled,
    )
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    _train_with_curves(model, config, train_loader, val_loader, logger)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    results = {
        "model": label_prefix,
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "curve_file": curve_path,
        "gradient_masking_enabled": gradient_masking_enabled,
        "dual_rate_enabled": dual_rate_enabled,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def run_cubic_only(dataset, seed, steps, data_dir=None):
    """Variant (a): cubic schedule only - no gradient masking, no dual-rate."""
    return _run_kaleidonet_variant(
        dataset, seed, steps, label_prefix="cubic_only",
        gradient_masking_enabled=False, dual_rate_enabled=False,
        data_dir=data_dir,
    )


def run_masking_only(dataset, seed, steps, data_dir=None):
    """Variant (b): cubic + gradient masking (no dual-rate)."""
    return _run_kaleidonet_variant(
        dataset, seed, steps, label_prefix="masking_only",
        gradient_masking_enabled=True, dual_rate_enabled=False,
        data_dir=data_dir,
    )


def run_dual_only(dataset, seed, steps, data_dir=None):
    """Variant (c): cubic + dual-rate (no gradient masking)."""
    return _run_kaleidonet_variant(
        dataset, seed, steps, label_prefix="dual_only",
        gradient_masking_enabled=False, dual_rate_enabled=True,
        data_dir=data_dir,
    )


def run_dense(dataset, seed, steps, data_dir=None):
    """Dense KaleidoNet (no pruning, no elasticity)."""
    label = f"dense_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  Dense | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Disable pruning entirely by pushing sparsity window past training
    config = _make_config(
        steps, seed,
        flops_budget=int(init_active * 0.50),
    )
    config.sparsity_start_step = steps + 1000
    config.sparsity_end_step = steps + 2000
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    _train_with_curves(model, config, train_loader, val_loader, logger)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    param_info = model.count_active_params()

    results = {
        "model": "Dense",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": dense_flops,  # No pruning
        "curve_file": curve_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def run_magnitude(dataset, seed, steps, data_dir=None):
    """Magnitude pruning: train dense, prune, fine-tune."""
    label = f"magnitude_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  Magnitude Pruning | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Phase 1: Train full model (no pruning)
    train_steps = int(steps * 0.8)
    config = _make_config(
        train_steps, seed,
        flops_budget=int(init_active * 0.50),
    )
    config.sparsity_start_step = train_steps + 1000
    config.sparsity_end_step = train_steps + 2000
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    print(f"  Phase 1: Training dense for {train_steps} steps...")
    _train_with_curves(model, config, train_loader, val_loader, logger)

    pre_prune_metrics = eval_model(model, val_loader, config.device)
    print(f"  Pre-prune accuracy: {pre_prune_metrics['val_accuracy']:.4f}")

    # Phase 2: One-shot magnitude pruning
    target_sparsity = 0.7
    print(f"  Phase 2: Magnitude pruning (sparsity={target_sparsity:.0%})...")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ElasticLinear):
                scores = m.weight.data.norm(dim=1)
                n = scores.numel()
                n_keep = max(int(n * (1 - target_sparsity)), m.min_width)
                threshold = scores.topk(n_keep, largest=True).values[-1]
                mask = scores >= threshold
                m.mask_logits.data[mask] = 10.0
                m.mask_logits.data[~mask] = -100.0

    # Phase 3: Fine-tune
    finetune_steps = steps - train_steps
    print(f"  Phase 3: Fine-tuning for {finetune_steps} steps...")
    ft_config = _make_config(
        finetune_steps, seed,
        flops_budget=int(init_active * 0.50),
    )
    ft_config.lr = 1e-4
    ft_config.sparsity_start_step = finetune_steps + 1000
    ft_config.sparsity_end_step = finetune_steps + 2000
    _train_with_curves(model, ft_config, train_loader, val_loader, logger)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    results = {
        "model": "MagnitudePruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_sparsity": target_sparsity,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "pre_prune_acc": pre_prune_metrics["val_accuracy"],
        "curve_file": curve_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def run_random(dataset, seed, steps, data_dir=None):
    """Random structured pruning: fixed random mask, train from scratch."""
    label = f"random_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  Random Pruning | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    # Apply fixed random mask
    target_sparsity = 0.7
    rng = torch.Generator().manual_seed(seed + 1000)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ElasticLinear):
                n = m.mask_logits.numel()
                n_keep = max(int(n * (1 - target_sparsity)), m.min_width)
                perm = torch.randperm(n, generator=rng)
                m.mask_logits.data.fill_(-100.0)
                m.mask_logits.data[perm[:n_keep]] = 10.0

    # Freeze mask logits
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            m.mask_logits.requires_grad_(False)

    config = _make_config(
        steps, seed,
        flops_budget=int(init_active * 0.50),
    )
    config.sparsity_start_step = steps + 1000
    config.sparsity_end_step = steps + 2000
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    _train_with_curves(model, config, train_loader, val_loader, logger)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    results = {
        "model": "RandomPruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_sparsity": target_sparsity,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "curve_file": curve_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def run_linear(dataset, seed, steps, data_dir=None):
    """Linear sparsity schedule instead of cubic."""
    label = f"linear_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  Linear Schedule | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    config = _make_config(steps, seed, flops_budget=int(init_active * 0.50))
    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    def make_linear_pruning(trainer):
        @torch.no_grad()
        def _apply_linear_pruning():
            cfg = trainer.config
            step = trainer.global_step
            if step < cfg.sparsity_start_step:
                return
            if step % cfg.sparsity_frequency != 0:
                return
            t = min((step - cfg.sparsity_start_step) / max(cfg.sparsity_end_step - cfg.sparsity_start_step, 1), 1.0)
            current_sparsity = cfg.target_sparsity * t  # Linear
            for m in trainer.model.modules():
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
        return _apply_linear_pruning

    _train_with_curves(model, config, train_loader, val_loader, logger,
                       patch_pruning_fn=make_linear_pruning)
    logger.close()

    final_metrics = eval_model(model, val_loader, config.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    results = {
        "model": "LinearSchedule",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "curve_file": curve_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


def run_lagrangian(dataset, seed, steps, data_dir=None):
    """Lagrangian FLOPs-constrained pruning baseline.

    Uses dual ascent on a Lagrange multiplier to enforce a FLOPs budget.
    The key finding: with per-neuron masks, the Lagrangian penalty produces
    uniform shrinkage across all neurons rather than discrete selection,
    leading to mask collapse.

    Logs mask logit distributions at eval steps to demonstrate uniform collapse.
    """
    label = f"lagrangian_{dataset}_seed{seed}"
    print(f"\n{'='*60}\n  Lagrangian Pruning | {dataset} | seed={seed} | {steps} steps\n{'='*60}")

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size, elastic=True)
    dense_flops, init_active = calibrate_flops_budget(model, img_size, patch_size)

    target_flops_ratio = 0.5
    print(f"  Dense FLOPs:   {dense_flops:,}")
    print(f"  Target ratio:  {target_flops_ratio:.0%}")

    curve_path = os.path.join(RESULTS_DIR, "curves", f"{label}.csv")
    logger = CurveLogger(curve_path)

    # Mask logit distribution log (shows uniform collapse)
    mask_dist_path = os.path.join(RESULTS_DIR, "curves", f"{label}_mask_dist.csv")
    os.makedirs(os.path.dirname(mask_dist_path) or ".", exist_ok=True)
    mask_dist_fp = open(mask_dist_path, "w", newline="")
    mask_dist_writer = csv.writer(mask_dist_fp)
    mask_dist_writer.writerow([
        "step", "layer_idx", "layer_name",
        "mean", "std", "min", "max",
        "frac_active", "num_neurons",
    ])
    mask_dist_fp.flush()

    eval_interval = max(steps // 100, 500)

    trainer = LagrangianPruningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_flops_ratio=target_flops_ratio,
        lr=3e-4,
        lambda_init=0.01,
        lambda_lr=0.01,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),
        seed=seed,
        log_interval=100,
        eval_interval=eval_interval,
    )

    # Run training with curve + mask distribution logging
    train_iter = iter(train_loader)
    import time as _time
    start_time = _time.time()

    for step in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        metrics = trainer.train_step(batch)

        # Standard logging
        if step % 100 == 0:
            elapsed = _time.time() - start_time
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"Step {step:6d} | "
                f"loss={metrics['total_loss']:.4f} | "
                f"task={metrics['task_loss']:.4f} | "
                f"flops_frac={metrics['flops_fraction']:.2%} | "
                f"lambda={metrics['lambda']:.4f} | "
                f"{sps:.1f} steps/s"
            )

        # Eval + curve logging + mask distribution logging
        if step % eval_interval == 0 and step > 0:
            val_m = eval_model(model, val_loader, trainer.device)
            print(
                f"  [EVAL] val_loss={val_m['val_loss']:.4f} | "
                f"val_acc={val_m['val_accuracy']:.4f}"
            )
            logger.log(
                step=step,
                train_loss=metrics["total_loss"],
                task_loss=metrics["task_loss"],
                val_loss=val_m["val_loss"],
                val_accuracy=val_m["val_accuracy"],
                lr=metrics["lr"],
            )

            # Log mask logit distributions per layer
            with torch.no_grad():
                for idx, m in enumerate(model.modules()):
                    if isinstance(m, ElasticLinear):
                        logits = m.mask_logits
                        soft_mask = torch.sigmoid(logits)
                        mask_dist_writer.writerow([
                            step, idx, type(m).__name__,
                            f"{logits.mean().item():.4f}",
                            f"{logits.std().item():.4f}",
                            f"{logits.min().item():.4f}",
                            f"{logits.max().item():.4f}",
                            f"{(soft_mask > 0.5).float().mean().item():.4f}",
                            logits.numel(),
                        ])
                mask_dist_fp.flush()

    logger.close()
    mask_dist_fp.close()

    total_time = _time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    # Final evaluation
    final_metrics = eval_model(model, val_loader, trainer.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    # Final mask logit summary (key evidence of uniform collapse)
    mask_summary = []
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ElasticLinear):
                logits = m.mask_logits
                mask_summary.append({
                    "mean": round(logits.mean().item(), 4),
                    "std": round(logits.std().item(), 4),
                    "min": round(logits.min().item(), 4),
                    "max": round(logits.max().item(), 4),
                    "frac_active": round((torch.sigmoid(logits) > 0.5).float().mean().item(), 4),
                })

    print(f"\nFinal accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"Final lambda:   {trainer.lambda_val.item():.4f}")
    if final_flops:
        print(f"Final FLOPs:    {final_flops:,} ({final_flops/dense_flops:.0%} of dense)")
    print(f"Mask logit distributions (evidence of uniform collapse):")
    for i, ms in enumerate(mask_summary):
        print(f"  Layer {i}: mean={ms['mean']:.2f} std={ms['std']:.2f} "
              f"range=[{ms['min']:.2f}, {ms['max']:.2f}] active={ms['frac_active']:.0%}")

    results = {
        "model": "LagrangianPruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_flops_ratio": target_flops_ratio,
        "final_lambda": trainer.lambda_val.item(),
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "mask_logit_summary": mask_summary,
        "curve_file": curve_path,
        "mask_dist_file": mask_dist_path,
        **final_metrics,
    }
    _save(results, f"{label}.json")
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "dense": run_dense,
    "magnitude": run_magnitude,
    "random": run_random,
    "linear": run_linear,
    "lagrangian": run_lagrangian,
    "kaleidonet": run_kaleidonet,
    # Schedule add-on ablation variants
    "cubic_only": run_cubic_only,
    "masking_only": run_masking_only,
    "dual_only": run_dual_only,
}


def _save(results: dict, filename: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def result_exists(method: str, dataset: str, seed: int) -> bool:
    path = os.path.join(RESULTS_DIR, f"{method}_{dataset}_seed{seed}.json")
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convergence training: all methods x all datasets x multiple seeds",
    )
    parser.add_argument("--method", choices=ALL_METHODS + ["all"], default="all",
                        help="Which method to run (default: all)")
    parser.add_argument("--dataset", choices=ALL_DATASETS + ["all"], default="all",
                        help="Which dataset to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 1 2 3)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data root for Tiny-ImageNet")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose result JSON already exists")
    args = parser.parse_args()

    methods = ALL_METHODS if args.method == "all" else [args.method]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    os.makedirs(os.path.join(RESULTS_DIR, "curves"), exist_ok=True)

    total_runs = len(methods) * len(datasets) * len(args.seeds)
    completed = 0
    failed = 0
    skipped = 0
    run_log = []

    print("=" * 60)
    print("KaleidoNet Convergence Training")
    print(f"  Methods:  {methods}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Steps:    {args.steps}")
    print(f"  Total runs: {total_runs}")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 60)

    wall_start = time.time()

    for method in methods:
        runner = METHOD_RUNNERS[method]
        for dataset in datasets:
            for seed in args.seeds:
                tag = f"{method}/{dataset}/seed{seed}"

                if args.skip_existing and result_exists(method, dataset, seed):
                    print(f"\n[SKIP] {tag} -- result already exists")
                    skipped += 1
                    continue

                t0 = time.time()
                try:
                    result = runner(dataset, seed, args.steps, data_dir=args.data_dir)
                    elapsed = time.time() - t0
                    completed += 1
                    run_log.append({
                        "tag": tag, "status": "ok", "time_s": elapsed,
                        "val_accuracy": result.get("val_accuracy"),
                    })
                    print(f"  [{tag}] OK in {elapsed:.0f}s  acc={result.get('val_accuracy', 0):.4f}")
                except Exception as exc:
                    elapsed = time.time() - t0
                    failed += 1
                    run_log.append({"tag": tag, "status": "failed", "time_s": elapsed, "error": str(exc)})
                    print(f"  [{tag}] FAILED in {elapsed:.0f}s: {exc}")

    wall_elapsed = time.time() - wall_start

    # Summary
    print("\n" + "=" * 60)
    print("Convergence Training Summary")
    print("=" * 60)
    print(f"  Completed: {completed}/{total_runs}  (skipped: {skipped}, failed: {failed})")
    print(f"  Wall time: {wall_elapsed / 60:.1f} min")

    for entry in run_log:
        status = "OK" if entry["status"] == "ok" else "FAILED"
        acc_str = f"  acc={entry['val_accuracy']:.4f}" if entry.get("val_accuracy") is not None else ""
        print(f"    {entry['tag']:40s} {status} ({entry['time_s']:.0f}s){acc_str}")

    # Save run log
    log_path = os.path.join(RESULTS_DIR, "run_log.json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nRun log saved to {log_path}")


if __name__ == "__main__":
    main()
