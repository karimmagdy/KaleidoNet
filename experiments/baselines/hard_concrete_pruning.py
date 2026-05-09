"""L0 / Hard-Concrete differentiable-sparsity baseline (Louizos et al. 2018).

Tier B response to peer review of KaleidoNet SR (W8, Q4). The
reviewer asked why we did not include L0/Hard-Concrete or stochastic
gates as a differentiable-sparsity baseline. Hard-Concrete sidesteps
the gradient-scale mismatch that the paper diagnoses by penalising
the *expected L0 norm* of the mask rather than the differentiable
FLOPs ratio --- so the penalty gradient is on the same scale as the
location logits themselves, not 1/B times smaller.

Implementation strategy: re-use the existing ElasticLinear backbone
unchanged --- the `mask_logits` parameter is reinterpreted as the
location logit `log_alpha` of a Hard-Concrete distribution, and the
forward-time sampler is monkey-patched to sample from the
Hard-Concrete distribution instead of Gumbel-sigmoid. The model's
parameter count, dense FLOPs, and active-FLOPs counter are
unchanged; only the mask-training rule differs from the cubic
schedule and Lagrangian baselines.

Loss: L_total = L_task + l0_weight * E[||z||_0_total]

where E[||z||_0_total] = sum over all ElasticLinear layers of
sum_i sigmoid(log_alpha_i - beta * log(-gamma/zeta)).

The l0_weight is binary-search-tuned (or set explicitly via
--l0-weight) to land at the same active-FLOPs target ratio as the
KaleidoNet reference (default 1/1.80 = 0.556 -> target ratio 0.50
in our active-FLOPs convention).

Run examples:

    # Smoke test (zero L0 weight should reach dense accuracy)
    python experiments/baselines/hard_concrete_pruning.py \\
        --dataset cifar10 --seed 1 --steps 200 --l0-weight 0

    # Full sweep at canonical 1.80x FLOPs reduction
    python experiments/baselines/hard_concrete_pruning.py \\
        --dataset cifar100 --seed 42 --steps 50000 --l0-weight 1e-4
"""
from __future__ import annotations

import argparse
import json
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
from kaleidonet.core.elastic import ElasticLinear, _apply_floor_mask
from kaleidonet.morphing.hard_concrete import (
    hard_concrete_sample, expected_l0, expected_active_fraction,
    L0_LOG_CORRECTION,
)
from kaleidonet.training.trainer import set_seed, _detect_device
from experiments.baselines.lagrangian_pruning import (
    get_loaders, build_model, calibrate_flops, measure_final_flops, eval_model,
)


# ---------------------------------------------------------------------------
# Monkey-patch: replace ElasticLinear.get_mask with Hard-Concrete sampling
# ---------------------------------------------------------------------------

def _hard_concrete_get_mask(self) -> torch.Tensor:
    """Replacement for ElasticLinear.get_mask that samples from the
    Hard-Concrete distribution rather than Gumbel-sigmoid.

    Uses self.mask_logits as the Hard-Concrete location parameter
    log_alpha, preserving parameter count and the existing
    floor-mask semantics for min_width.
    """
    if not self.training:
        # Deterministic Hard-Concrete expectation
        z = hard_concrete_sample(self.mask_logits, training=False)
        # Hard-binarize at 0.5 to match the Gumbel-sigmoid inference behavior
        mask = (z > 0.5).to(dtype=z.dtype)
        return _apply_floor_mask(self.mask_logits, mask, self.min_width)
    z = hard_concrete_sample(self.mask_logits, training=True)
    return _apply_floor_mask(self.mask_logits, z, self.min_width)


def patch_to_hard_concrete(model: nn.Module) -> int:
    """Replace get_mask on every ElasticLinear with Hard-Concrete
    sampling. Returns the number of layers patched.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            # Bind the function to the instance
            m.get_mask = _hard_concrete_get_mask.__get__(m, ElasticLinear)
            n += 1
    return n


def total_expected_l0(model: nn.Module) -> torch.Tensor:
    """Sum E[||z||_0] over every ElasticLinear in the model. Returns a
    scalar tensor for use in the loss.
    """
    device = next(model.parameters()).device
    out = torch.tensor(0.0, device=device)
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            out = out + expected_l0(m.mask_logits)
    return out


def total_expected_active_fraction(model: nn.Module) -> torch.Tensor:
    """Mean expected-active rate across all ElasticLinear layers.
    Returns a scalar in [0, 1] for diagnostic logging.
    """
    device = next(model.parameters()).device
    total_active = torch.tensor(0.0, device=device)
    total_count = 0
    for m in model.modules():
        if isinstance(m, ElasticLinear):
            total_active = total_active + expected_l0(m.mask_logits)
            total_count += m.mask_logits.numel()
    return total_active / max(total_count, 1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HardConcreteTrainer:
    """Train a KaleidoNet model with L0 / Hard-Concrete sparsity.

    Identical optimizer grouping, dual-rate (3x mask LR), gradient
    clipping, and Gumbel-style early-exit / balance-loss
    configuration as lagrangian_pruning.py and the cubic-schedule
    KaleidoNet method, so the comparison is on the mask-training
    rule alone.
    """

    def __init__(
        self,
        model: KaleidoNet,
        train_loader,
        val_loader,
        l0_weight: float = 1e-4,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_steps: int = 5000,
        warmup_steps: int = 250,
        log_interval: int = 100,
        eval_interval: int = 500,
        grad_clip: float = 1.0,
        seed: int | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(device or _detect_device())
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.l0_weight = l0_weight
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.grad_clip = grad_clip
        self.seed = seed
        self.global_step = 0
        self.best_val_loss = float("inf")

        n_patched = patch_to_hard_concrete(self.model)
        print(f"  Patched {n_patched} ElasticLinear layers to Hard-Concrete sampling.")

        # Separate mask params from weight params (same grouping as Lagrangian / KaleidoNet)
        mask_params, weight_params = [], []
        for name, p in model.named_parameters():
            if "mask_logits" in name or "head_mask_logits" in name:
                mask_params.append(p)
            else:
                weight_params.append(p)

        self.optimizer = torch.optim.AdamW(
            weight_params, lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.01,
        )
        self.mask_optimizer = torch.optim.Adam(
            mask_params, lr=lr * 3,
        ) if mask_params else None

        self.task_loss_fn = nn.CrossEntropyLoss()

    def _warmup_lr_scale(self) -> float:
        if self.global_step >= self.warmup_steps:
            return 1.0
        return self.global_step / max(self.warmup_steps, 1)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        outputs = self.model(batch)
        logits = outputs["logits"]
        targets = batch["targets"]

        if logits.dim() == 3:
            task_loss = self.task_loss_fn(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            )
        else:
            task_loss = self.task_loss_fn(logits, targets)

        l0_norm = total_expected_l0(self.model)
        l0_penalty = self.l0_weight * l0_norm
        active_frac = total_expected_active_fraction(self.model).item()

        total_loss = task_loss + l0_penalty

        self.optimizer.zero_grad(set_to_none=True)
        if self.mask_optimizer:
            self.mask_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        # Hard-Concrete uses real-valued log_alpha; no need to zero
        # grads on dead masks (the L0 penalty itself drives them to
        # the saturation regime where sigmoid -> 0).

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        warmup_scale = self._warmup_lr_scale()
        if warmup_scale < 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_scale

        self.optimizer.step()
        if self.mask_optimizer:
            self.mask_optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "l0_penalty": l0_penalty.item(),
            "l0_norm": l0_norm.item(),
            "active_frac": active_frac,
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
        print(f"Starting Hard-Concrete pruning training on {self.device}")
        print(f"  L0 weight:    {self.l0_weight:.2e}")
        print(f"  Max steps:    {self.max_steps}")
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

            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step:6d} | "
                    f"loss={metrics['total_loss']:.4f} | "
                    f"task={metrics['task_loss']:.4f} | "
                    f"l0_pen={metrics['l0_penalty']:.4f} | "
                    f"E[L0]={metrics['l0_norm']:.1f} | "
                    f"active={metrics['active_frac']:.2%} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"{sps:.1f} steps/s"
                )

            if step % self.eval_interval == 0 and step > 0:
                val_metrics = self.eval_step()
                if val_metrics:
                    print(
                        f"  [EVAL] val_loss={val_metrics['val_loss']:.4f} | "
                        f"val_acc={val_metrics['val_accuracy']:.4f}"
                    )
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best val loss: {self.best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    dataset: str,
    seed: int,
    steps: int,
    l0_weight: float = 1e-4,
    data_dir: str | None = None,
) -> dict:
    print("=" * 60)
    print(f"Baseline: L0 / Hard-Concrete Pruning | {dataset} | seed={seed}")
    print(f"  l0_weight={l0_weight:.2e}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(
        dataset, data_dir=data_dir,
    )
    model = build_model(num_classes, img_size, patch_size)
    dense_flops, init_active = calibrate_flops(model, img_size, patch_size)
    print(f"Dense FLOPs:        {dense_flops:,}")
    print(f"Init active FLOPs:  {init_active:,}")

    trainer = HardConcreteTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        l0_weight=l0_weight,
        lr=3e-4,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),
        seed=seed,
        log_interval=100,
        eval_interval=max(steps // 100, 500),
    )
    trainer.train()

    final_metrics = eval_model(model, val_loader, trainer.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy:     {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,} ({final_flops/dense_flops:.0%} of dense)")

    results = {
        "model": "HardConcretePruning",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "l0_weight": l0_weight,
        "params": param_info,
        "dense_flops": dense_flops,
        "init_active_flops": init_active,
        "active_flops": final_flops,
        **final_metrics,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/hard_concrete_pruning_{dataset}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="L0 / Hard-Concrete differentiable-sparsity pruning baseline"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10", "tiny_imagenet"],
                        default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--l0-weight", type=float, default=1e-4,
                        help="L0 regularisation weight; tune to land at the desired "
                             "active-FLOPs target ratio. Set to 0 for a smoke test "
                             "(no L0 pressure -> dense-comparable accuracy).")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    run(
        dataset=args.dataset,
        seed=args.seed,
        steps=args.steps,
        l0_weight=args.l0_weight,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
