"""Lagrangian FLOPs-constrained pruning baseline.

Implements standard Lagrangian relaxation for structured pruning as a
comparison to KaleidoNet's cubic sparsity schedule.

  L_total = L_task + lambda * max(0, FLOPs(model) - target_FLOPs)
  lambda  <- lambda + lambda_lr * (FLOPs(model) - target_FLOPs)   (dual ascent)

Masks are magnitude-based soft masks with straight-through estimators
on each ElasticLinear layer. The Lagrangian multiplier drives the model
to meet the FLOPs constraint without a fixed pruning schedule.

Uses the same ViT backbone (KaleidoNet with elastic=True) so that
architecture, param count, and FLOPs are directly comparable.

Run:
    python experiments/baselines/lagrangian_pruning.py --dataset cifar100 --seed 1
    python experiments/baselines/lagrangian_pruning.py --dataset cifar100 --seed 1 --target-flops-ratio 0.3
    python experiments/baselines/lagrangian_pruning.py --dataset cifar100 --seed 1 --steps 50000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaleidonet.model import KaleidoNet
from kaleidonet.core.elastic import ElasticLinear
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.training.trainer import set_seed, _detect_device

# ---------------------------------------------------------------------------
# Dataset helpers (same as pruning_baselines.py)
# ---------------------------------------------------------------------------

def get_loaders(dataset: str, batch_size: int = 64, data_dir: str | None = None):
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


def build_model(num_classes: int, image_size: int, patch_size: int) -> KaleidoNet:
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
        elastic=True,
        drop_path_rate=0.1,
    )


def calibrate_flops(model: KaleidoNet, image_size: int, patch_size: int) -> tuple[int, int]:
    counter = FLOPsCounter()
    seq_len = (image_size // patch_size) ** 2
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=seq_len)
    dev = next(model.parameters()).device
    with torch.no_grad():
        probe = model({"images": torch.randn(1, 3, image_size, image_size, device=dev)})
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


def eval_model(model: nn.Module, val_loader, device) -> dict:
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
# Straight-through magnitude mask
# ---------------------------------------------------------------------------

class STEMagnitudeMask(torch.autograd.Function):
    """Hard threshold mask with straight-through gradient estimator.

    Forward:  mask = (score > threshold).float()
    Backward: grad flows through as if mask = sigmoid(score - threshold)
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, threshold: torch.Tensor):
        mask = (scores >= threshold).float()
        ctx.save_for_backward(scores, threshold)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, threshold = ctx.saved_tensors
        # STE: pass gradient through sigmoid proxy
        proxy = torch.sigmoid(10.0 * (scores - threshold))
        grad_scores = grad_output * proxy * (1 - proxy) * 10.0
        return grad_scores, None


def compute_soft_flops_fraction(model: nn.Module) -> torch.Tensor:
    """Differentiable estimate of active FLOPs fraction via sigmoid masks.

    Returns a scalar tensor (fraction of total neurons that are active).
    This serves as a differentiable proxy for the actual FLOPs ratio.
    """
    total = 0
    active = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            soft_mask = torch.sigmoid(m.mask_logits)
            active = active + soft_mask.sum()
            total += m.mask_logits.numel()
    if total == 0:
        return torch.tensor(1.0, device=active.device)
    return active / total


# ---------------------------------------------------------------------------
# Lagrangian Trainer
# ---------------------------------------------------------------------------

class LagrangianPruningTrainer:
    """Train a KaleidoNet model with Lagrangian FLOPs constraint.

    Unlike the cubic schedule, the pruning pressure comes entirely from
    the Lagrangian penalty on FLOPs, and the dual variable lambda is
    updated via gradient ascent to enforce the constraint.
    """

    def __init__(
        self,
        model: KaleidoNet,
        train_loader,
        val_loader,
        target_flops_ratio: float = 0.5,
        lr: float = 3e-4,
        lambda_init: float = 0.01,
        lambda_lr: float = 0.01,
        lambda_max: float = 100.0,
        max_steps: int = 5000,
        warmup_steps: int = 250,
        seed: int | None = None,
        log_interval: int = 100,
        eval_interval: int = 500,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        device: str | None = None,
    ):
        self.device = torch.device(device or _detect_device())
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_flops_ratio = target_flops_ratio
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.grad_clip = grad_clip
        self.seed = seed

        # Lagrangian dual variable (non-negative)
        self.lambda_val = torch.tensor(lambda_init, device=self.device, requires_grad=False)
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max

        # Separate mask params from weight params
        mask_params = []
        weight_params = []
        for name, p in model.named_parameters():
            if "mask_logits" in name or "head_mask_logits" in name:
                mask_params.append(p)
            else:
                weight_params.append(p)

        # Weight optimizer with cosine annealing
        self.optimizer = torch.optim.AdamW(
            weight_params, lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.01,
        )

        # Mask optimizer (higher LR, no weight decay)
        self.mask_optimizer = torch.optim.Adam(
            mask_params, lr=lr * 3,
        ) if mask_params else None

        self.task_loss_fn = nn.CrossEntropyLoss()
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _warmup_lr_scale(self) -> float:
        if self.global_step >= self.warmup_steps:
            return 1.0
        return self.global_step / max(self.warmup_steps, 1)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward
        outputs = self.model(batch)
        logits = outputs["logits"]
        targets = batch["targets"]

        # Task loss
        if logits.dim() == 3:
            task_loss = self.task_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            task_loss = self.task_loss_fn(logits, targets)

        # Differentiable FLOPs proxy: fraction of active neurons
        flops_fraction = compute_soft_flops_fraction(self.model)
        constraint_violation = flops_fraction - self.target_flops_ratio

        # Lagrangian penalty: lambda * max(0, violation)
        flops_penalty = self.lambda_val * F.relu(constraint_violation)

        # Total loss
        total_loss = task_loss + flops_penalty

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        if self.mask_optimizer:
            self.mask_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        # Zero grads on hard-pruned mask logits (already dead)
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                if m.mask_logits.grad is not None:
                    m.mask_logits.grad[m.mask_logits.data <= -50] = 0.0

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Apply warmup LR scaling
        warmup_scale = self._warmup_lr_scale()
        if warmup_scale < 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_scale

        self.optimizer.step()
        if self.mask_optimizer:
            self.mask_optimizer.step()
        self.scheduler.step()

        # Dual ascent: update lambda
        with torch.no_grad():
            self.lambda_val += self.lambda_lr * constraint_violation.detach()
            self.lambda_val.clamp_(min=0.0, max=self.lambda_max)

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "flops_penalty": flops_penalty.item(),
            "flops_fraction": flops_fraction.item(),
            "constraint_violation": constraint_violation.item(),
            "lambda": self.lambda_val.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def eval_step(self) -> dict:
        if self.val_loader is None:
            return {}
        return eval_model(self.model, self.val_loader, self.device)

    def train(self):
        if self.seed is not None:
            set_seed(self.seed)
        print(f"Starting Lagrangian pruning training on {self.device}")
        print(f"  Target FLOPs ratio: {self.target_flops_ratio:.0%}")
        print(f"  Max steps: {self.max_steps}")
        print()

        train_iter = iter(self.train_loader)
        start_time = time.time()

        for step in range(self.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)

            # Logging
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step:6d} | "
                    f"loss={metrics['total_loss']:.4f} | "
                    f"task={metrics['task_loss']:.4f} | "
                    f"flops_pen={metrics['flops_penalty']:.4f} | "
                    f"flops_frac={metrics['flops_fraction']:.2%} | "
                    f"viol={metrics['constraint_violation']:+.4f} | "
                    f"lambda={metrics['lambda']:.4f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"{sps:.1f} steps/s"
                )

            # Evaluation
            if step % self.eval_interval == 0 and step > 0:
                val_metrics = self.eval_step()
                if val_metrics:
                    print(
                        f"  [EVAL] val_loss={val_metrics['val_loss']:.4f} | "
                        f"val_acc={val_metrics['val_accuracy']:.4f}"
                    )
                    val_loss = val_metrics["val_loss"]
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best val loss: {self.best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lagrangian(
    dataset: str,
    seed: int,
    steps: int,
    target_flops_ratio: float = 0.5,
    lambda_init: float = 0.01,
    lambda_lr: float = 0.01,
    data_dir: str | None = None,
) -> dict:
    """Run Lagrangian FLOPs-constrained pruning baseline."""
    print("=" * 60)
    print(f"Baseline: Lagrangian Pruning | {dataset} | seed={seed}")
    print(f"  target_flops_ratio={target_flops_ratio:.0%} | lambda_init={lambda_init} | lambda_lr={lambda_lr}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)
    model = build_model(num_classes, img_size, patch_size)
    dense_flops, init_active = calibrate_flops(model, img_size, patch_size)

    print(f"Dense FLOPs:        {dense_flops:,}")
    print(f"Init active FLOPs:  {init_active:,}")
    target_flops = int(dense_flops * target_flops_ratio)
    print(f"Target FLOPs:       {target_flops:,} ({target_flops_ratio:.0%} of dense)")
    print()

    trainer = LagrangianPruningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_flops_ratio=target_flops_ratio,
        lr=3e-4,
        lambda_init=lambda_init,
        lambda_lr=lambda_lr,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),
        seed=seed,
        log_interval=100,
        eval_interval=max(steps // 100, 500),
    )

    trainer.train()

    # Final evaluation
    final_metrics = eval_model(model, val_loader, trainer.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy: {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,} ({final_flops/dense_flops:.0%} of dense)")

    results = {
        "model": "LagrangianPruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_flops_ratio": target_flops_ratio,
        "target_flops": target_flops,
        "lambda_init": lambda_init,
        "lambda_lr": lambda_lr,
        "final_lambda": trainer.lambda_val.item(),
        "params": param_info,
        "dense_flops": dense_flops,
        "init_active_flops": init_active,
        "active_flops": final_flops,
        **final_metrics,
    }

    os.makedirs("results", exist_ok=True)
    path = f"results/lagrangian_pruning_{dataset}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Lagrangian FLOPs-constrained pruning baseline")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10", "tiny_imagenet"],
                        default="cifar100", help="Dataset")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--target-flops", type=float, default=None,
                        help="Absolute target FLOPs (overrides --target-flops-ratio)")
    parser.add_argument("--target-flops-ratio", type=float, default=0.5,
                        help="Target FLOPs as fraction of dense (default: 0.5)")
    parser.add_argument("--lambda-init", type=float, default=0.01,
                        help="Initial Lagrangian multiplier")
    parser.add_argument("--lambda-lr", type=float, default=0.01,
                        help="Learning rate for dual variable")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory for Tiny-ImageNet")
    args = parser.parse_args()

    run_lagrangian(
        dataset=args.dataset,
        seed=args.seed,
        steps=args.steps,
        target_flops_ratio=args.target_flops_ratio,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
