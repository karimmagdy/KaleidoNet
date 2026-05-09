"""Strengthened Lagrangian FLOPs-constrained pruning baselines (W1, Q1-Q3).

Tier B response to peer review of KaleidoNet SR. The reviewer asked
whether the Lagrangian failure is specific to the global-scalar
formulation, or if it persists under richer dual variables and
update rules. We evaluate four strengthened variants alongside the
canonical global baseline:

  1. global       --- single shared lambda (matches lagrangian_pruning.py)
  2. per_layer    --- one lambda per ElasticLinear layer (one violation
                      term per layer, dual ascent per layer)
  3. per_expert   --- one lambda per (block, expert) pair --- the most
                      granular level above per-neuron, named explicitly
                      in the reviewer's Q2
  4. augmented    --- adds (mu/2) * violation^2 to the loss alongside
                      the lambda * violation term (augmented Lagrangian)
  5. rescaled     --- multiplies the FLOPs gradient by a constant
                      g_scale (default 3.0, matching the dual-rate factor)
                      to test whether explicit penalty-gradient rescaling
                      recovers selectivity

Per-layer / per-expert FLOPs fractions are computed by mapping each
ElasticLinear layer to its (block_idx, expert_idx) via name parsing
of `model.named_modules()` paths like
`backbone.blocks.{block_idx}.moe.experts.{expert_idx}.fc{1,2}`.

Run examples:

    python experiments/baselines/lagrangian_strengthened.py \\
        --variant per_expert --dataset cifar100 --seed 42 --steps 50000

    python experiments/baselines/lagrangian_strengthened.py \\
        --variant augmented --dataset cifar10 --seed 1 --steps 200  # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

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
from experiments.baselines.lagrangian_pruning import (
    get_loaders, build_model, calibrate_flops, measure_final_flops, eval_model,
)


VARIANT_CHOICES = ["global", "per_layer", "per_expert", "augmented", "rescaled"]


# ---------------------------------------------------------------------------
# (block, expert) discovery
# ---------------------------------------------------------------------------

# Module-name regex: e.g. "backbone.blocks.2.moe.experts.3.fc1"
_EXPERT_PATH = re.compile(r"blocks\.(\d+)\.moe\.experts\.(\d+)")


def discover_elastic_layers(model: nn.Module) -> dict:
    """Map each ElasticLinear layer to (block_idx, expert_idx) where
    detectable. Layers outside the MoE expert path get assigned the
    sentinel index (-1, -1) and are pooled into a "shared" bucket.

    Returns a dict with:
        layers: list of (name, module, block_idx, expert_idx)
        n_blocks: max block_idx + 1 found
        n_experts: max expert_idx + 1 found
        shared_count: number of layers outside the MoE expert path
    """
    layers = []
    max_block, max_expert = -1, -1
    shared_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, ElasticLinear):
            continue
        match = _EXPERT_PATH.search(name)
        if match:
            b, e = int(match.group(1)), int(match.group(2))
            max_block = max(max_block, b)
            max_expert = max(max_expert, e)
            layers.append((name, module, b, e))
        else:
            shared_count += 1
            layers.append((name, module, -1, -1))
    return {
        "layers": layers,
        "n_blocks": max_block + 1 if max_block >= 0 else 0,
        "n_experts": max_expert + 1 if max_expert >= 0 else 0,
        "shared_count": shared_count,
    }


# ---------------------------------------------------------------------------
# Per-(block, expert) FLOPs fraction
# ---------------------------------------------------------------------------

def compute_per_expert_fractions(
    layers_info: dict, device: torch.device,
) -> dict:
    """Differentiable per-(block, expert) and per-layer FLOPs fractions.

    Returns dict with:
        per_layer: list of scalar tensors, one per ElasticLinear layer
        per_expert: tensor of shape (n_blocks, n_experts), zeros where
                    no layer maps to that index
        global: scalar tensor (sum of all soft masks / total)
        per_layer_count: list of mask numel per layer (for normalization)
        per_expert_count: tensor of shape (n_blocks, n_experts)
    """
    n_blocks = max(layers_info["n_blocks"], 1)
    n_experts = max(layers_info["n_experts"], 1)
    per_expert_active = torch.zeros(n_blocks, n_experts, device=device)
    per_expert_total = torch.zeros(n_blocks, n_experts, device=device)
    per_layer = []
    per_layer_count = []
    total_active = torch.tensor(0.0, device=device)
    total_count = 0

    for name, module, b, e in layers_info["layers"]:
        soft_mask = torch.sigmoid(module.mask_logits)
        active = soft_mask.sum()
        count = module.mask_logits.numel()
        per_layer.append(active / max(count, 1))
        per_layer_count.append(count)
        total_active = total_active + active
        total_count += count
        if b >= 0:
            per_expert_active[b, e] = per_expert_active[b, e] + active
            per_expert_total[b, e] = per_expert_total[b, e] + count

    per_expert_frac = per_expert_active / per_expert_total.clamp(min=1.0)
    global_frac = total_active / max(total_count, 1)

    return {
        "per_layer": per_layer,
        "per_expert": per_expert_frac,
        "global": global_frac,
        "per_layer_count": per_layer_count,
        "per_expert_count": per_expert_total,
    }


# ---------------------------------------------------------------------------
# Strengthened Lagrangian Trainer
# ---------------------------------------------------------------------------

@dataclass
class StrengthenedConfig:
    variant: str = "per_expert"
    target_flops_ratio: float = 0.5
    lambda_init: float = 0.01
    lambda_lr: float = 0.01
    lambda_max: float = 100.0
    augmented_mu: float = 0.1   # quadratic coefficient for augmented variant
    flops_grad_scale: float = 3.0  # for "rescaled" variant (matches dual-rate factor)
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 5000
    warmup_steps: int = 250
    log_interval: int = 100
    eval_interval: int = 500


class StrengthenedLagrangianTrainer:
    """Train a KaleidoNet model with one of five Lagrangian variants.

    The variant is selected via cfg.variant in {"global", "per_layer",
    "per_expert", "augmented", "rescaled"}. The backbone, optimizer
    grouping (Adam for masks at 3x the task LR), gradient clipping,
    Gumbel-sigmoid temperature, and early-exit / balance-loss
    configuration are identical to lagrangian_pruning.py so that the
    only difference is the dual-variable shape and update rule.
    """

    def __init__(
        self,
        model: KaleidoNet,
        train_loader,
        val_loader,
        cfg: StrengthenedConfig,
        seed: int | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(device or _detect_device())
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.seed = seed
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Discover (block, expert) structure
        self.layers_info = discover_elastic_layers(self.model)
        n_blocks = max(self.layers_info["n_blocks"], 1)
        n_experts = max(self.layers_info["n_experts"], 1)
        n_layers = len(self.layers_info["layers"])
        print(f"  Discovered {n_layers} ElasticLinear layers across "
              f"{n_blocks} blocks x {n_experts} experts "
              f"(plus {self.layers_info['shared_count']} shared layers).")

        # Initialise dual variable(s) according to variant
        if cfg.variant == "global":
            self.lambda_val = torch.tensor(cfg.lambda_init, device=self.device)
        elif cfg.variant == "per_layer":
            self.lambda_val = torch.full(
                (n_layers,), cfg.lambda_init, device=self.device,
            )
        elif cfg.variant in ("per_expert", "augmented", "rescaled"):
            # per_expert/augmented/rescaled all use per-(block, expert) lambdas;
            # they differ only in the loss / gradient computation, not in lambda shape
            self.lambda_val = torch.full(
                (n_blocks, n_experts), cfg.lambda_init, device=self.device,
            )
        else:
            raise ValueError(f"Unknown variant: {cfg.variant}; choices: {VARIANT_CHOICES}")

        # Separate mask params from weight params (same as the reference)
        mask_params, weight_params = [], []
        for name, p in model.named_parameters():
            if "mask_logits" in name or "head_mask_logits" in name:
                mask_params.append(p)
            else:
                weight_params.append(p)

        self.optimizer = torch.optim.AdamW(
            weight_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_steps, eta_min=cfg.lr * 0.01,
        )
        self.mask_optimizer = torch.optim.Adam(
            mask_params, lr=cfg.lr * 3,
        ) if mask_params else None

        self.task_loss_fn = nn.CrossEntropyLoss()

    def _warmup_lr_scale(self) -> float:
        if self.global_step >= self.cfg.warmup_steps:
            return 1.0
        return self.global_step / max(self.cfg.warmup_steps, 1)

    def _compute_penalty(self, fracs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lagrangian penalty term(s) to add to the task loss
        and the violation tensor used for the dual update.

        Returns (penalty_scalar, violation_tensor). The shape of
        violation_tensor matches lambda_val (scalar / (n_layers,) /
        (n_blocks, n_experts)).
        """
        target = self.cfg.target_flops_ratio
        v = self.cfg.variant

        if v == "global":
            violation = fracs["global"] - target
            # ReLU is the standard "no reward for under-budget" Lagrangian
            penalty = self.lambda_val * F.relu(violation)
            return penalty, violation.detach()

        if v == "per_layer":
            # Each layer compared to the same global target ratio
            per_layer = torch.stack(fracs["per_layer"])  # (n_layers,)
            violation = per_layer - target
            penalty = (self.lambda_val * F.relu(violation)).sum()
            return penalty, violation.detach()

        if v in ("per_expert", "rescaled"):
            violation = fracs["per_expert"] - target  # (n_blocks, n_experts)
            penalty = (self.lambda_val * F.relu(violation)).sum()
            if v == "rescaled":
                # Multiply the FLOPs-gradient by g_scale (in-loss equivalent:
                # scale the entire penalty term, since the only path from
                # the masks to the loss is via this term).
                penalty = penalty * self.cfg.flops_grad_scale
            return penalty, violation.detach()

        if v == "augmented":
            violation = fracs["per_expert"] - target
            # Augmented Lagrangian: lambda * violation_+ + (mu/2) * violation^2
            # Both terms penalise overshoot; the quadratic term pushes
            # the constraint to zero even when lambda is small.
            linear = (self.lambda_val * F.relu(violation)).sum()
            quad = 0.5 * self.cfg.augmented_mu * (violation ** 2).sum()
            return linear + quad, violation.detach()

        raise ValueError(f"Unknown variant: {v}")

    def _dual_step(self, violation: torch.Tensor):
        """Update the dual variable(s) via dual ascent on the same-shaped
        violation tensor. Clamped to [0, lambda_max].
        """
        with torch.no_grad():
            self.lambda_val.add_(self.cfg.lambda_lr * violation)
            self.lambda_val.clamp_(min=0.0, max=self.cfg.lambda_max)

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

        # Per-(layer, expert) FLOPs fractions
        fracs = compute_per_expert_fractions(self.layers_info, self.device)
        penalty, violation = self._compute_penalty(fracs)

        total_loss = task_loss + penalty

        self.optimizer.zero_grad(set_to_none=True)
        if self.mask_optimizer:
            self.mask_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        # Zero grads on hard-pruned mask logits (same as reference)
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                if m.mask_logits.grad is not None:
                    m.mask_logits.grad[m.mask_logits.data <= -50] = 0.0

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        warmup_scale = self._warmup_lr_scale()
        if warmup_scale < 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_scale

        self.optimizer.step()
        if self.mask_optimizer:
            self.mask_optimizer.step()
        self.scheduler.step()

        # Dual ascent
        self._dual_step(violation)

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "penalty": penalty.item(),
            "global_frac": fracs["global"].item(),
            "violation_max": violation.max().item() if violation.numel() else 0.0,
            "lambda_max_val": self.lambda_val.max().item(),
            "lambda_mean_val": self.lambda_val.mean().item(),
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
        cfg = self.cfg
        print(f"Starting strengthened Lagrangian (variant={cfg.variant}) on {self.device}")
        print(f"  Target FLOPs ratio: {cfg.target_flops_ratio:.0%}")
        print(f"  Max steps: {cfg.max_steps}")
        if cfg.variant == "augmented":
            print(f"  Augmented quadratic coefficient mu: {cfg.augmented_mu}")
        if cfg.variant == "rescaled":
            print(f"  FLOPs-gradient rescale factor: {cfg.flops_grad_scale}")
        print()

        train_iter = iter(self.train_loader)
        start_time = time.time()
        for step in range(cfg.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)

            if step % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step:6d} | "
                    f"loss={metrics['total_loss']:.4f} | "
                    f"task={metrics['task_loss']:.4f} | "
                    f"pen={metrics['penalty']:.4f} | "
                    f"frac={metrics['global_frac']:.2%} | "
                    f"viol_max={metrics['violation_max']:+.4f} | "
                    f"lam_mean={metrics['lambda_mean_val']:.4f} | "
                    f"lam_max={metrics['lambda_max_val']:.4f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"{sps:.1f} steps/s"
                )

            if step % cfg.eval_interval == 0 and step > 0:
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
    variant: str,
    dataset: str,
    seed: int,
    steps: int,
    target_flops_ratio: float = 0.5,
    augmented_mu: float = 0.1,
    flops_grad_scale: float = 3.0,
    lambda_init: float = 0.01,
    lambda_lr: float = 0.01,
    data_dir: str | None = None,
) -> dict:
    print("=" * 60)
    print(f"Variant: {variant} | Dataset: {dataset} | Seed: {seed}")
    print("=" * 60)

    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(
        dataset, data_dir=data_dir,
    )
    model = build_model(num_classes, img_size, patch_size)
    dense_flops, init_active = calibrate_flops(model, img_size, patch_size)
    print(f"Dense FLOPs:        {dense_flops:,}")
    print(f"Init active FLOPs:  {init_active:,}")
    target_flops = int(dense_flops * target_flops_ratio)
    print(f"Target FLOPs:       {target_flops:,} ({target_flops_ratio:.0%} of dense)")

    cfg = StrengthenedConfig(
        variant=variant,
        target_flops_ratio=target_flops_ratio,
        lambda_init=lambda_init,
        lambda_lr=lambda_lr,
        augmented_mu=augmented_mu,
        flops_grad_scale=flops_grad_scale,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),
        log_interval=100,
        eval_interval=max(steps // 100, 500),
    )
    trainer = StrengthenedLagrangianTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, seed=seed,
    )
    trainer.train()

    final_metrics = eval_model(model, val_loader, trainer.device)
    final_flops = measure_final_flops(model, img_size)
    param_info = model.count_active_params()

    print(f"\nFinal accuracy:     {final_metrics['val_accuracy']:.4f}")
    if final_flops:
        print(f"Final active FLOPs: {final_flops:,} ({final_flops/dense_flops:.0%} of dense)")

    results = {
        "model": "LagrangianStrengthened",
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_flops_ratio": target_flops_ratio,
        "target_flops": target_flops,
        "lambda_init": lambda_init,
        "lambda_lr": lambda_lr,
        "augmented_mu": augmented_mu,
        "flops_grad_scale": flops_grad_scale,
        "final_lambda_max": trainer.lambda_val.max().item(),
        "final_lambda_mean": trainer.lambda_val.mean().item(),
        "final_lambda_shape": list(trainer.lambda_val.shape) or [1],
        "params": param_info,
        "dense_flops": dense_flops,
        "init_active_flops": init_active,
        "active_flops": final_flops,
        **final_metrics,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/lagrangian_strengthened_{variant}_{dataset}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Strengthened Lagrangian FLOPs-constrained pruning baselines"
    )
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="per_expert",
                        help="Lagrangian variant (W1 / Q1-Q3 from the reviewer).")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10", "tiny_imagenet"],
                        default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--target-flops-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-init", type=float, default=0.01)
    parser.add_argument("--lambda-lr", type=float, default=0.01)
    parser.add_argument("--augmented-mu", type=float, default=0.1,
                        help="Quadratic coefficient for the augmented variant.")
    parser.add_argument("--flops-grad-scale", type=float, default=3.0,
                        help="FLOPs-gradient rescale factor for the rescaled variant.")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    run(
        variant=args.variant,
        dataset=args.dataset,
        seed=args.seed,
        steps=args.steps,
        target_flops_ratio=args.target_flops_ratio,
        augmented_mu=args.augmented_mu,
        flops_grad_scale=args.flops_grad_scale,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
