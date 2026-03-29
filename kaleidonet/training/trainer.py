"""
KaleidoNet Trainer: orchestrates training with all pillars active.

Handles:
- Forward pass through tokenizer → backbone → task head
- Multi-objective loss computation
- Lagrangian dual step for compute budget
- Morph controller updates
- Growth scheduler checks
- Temperature annealing for Gumbel-sigmoid masks
- Logging to W&B (optional)
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kaleidonet.morphing.lagrangian import LagrangianBudgetManager
from kaleidonet.morphing.controller import MorphController, LayerStatsCollector
from kaleidonet.growth.scheduler import GrowthScheduler
from kaleidonet.training.loss import KaleidoNetLoss
from kaleidonet.metrics.flops import FLOPsCounter


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NumPy is optional — only seed if imported
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _detect_device() -> str:
    """Detect best device: TPU/XLA > CUDA > MPS > CPU."""
    try:
        import torch_xla.core.xla_model as xm
        return str(xm.xla_device())
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainerConfig:
    """Training configuration."""
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Compute budget
    flops_budget: int = 1_000_000_000  # 1 GFLOPs target
    lambda_init: float = 0.01
    lambda_lr: float = 0.01

    # Loss weights
    lambda_balance: float = 0.01
    lambda_ponder: float = 0.01
    lambda_distill: float = 0.5

    # Growth
    growth_patience: int = 500
    growth_factor: float = 1.5
    max_growth_events: int = 5

    # Temperature annealing for elastic masks
    tau_start: float = 1.0
    tau_end: float = 0.1
    tau_anneal_steps: int = 2500

    # Sparsity schedule (cubic ramp, Zhu & Gupta 2017)
    target_sparsity: float = 0.7       # Final fraction of neurons to prune
    sparsity_start_step: int = 500     # When pruning begins
    sparsity_end_step: int = 4000      # When pruning reaches target
    sparsity_frequency: int = 100      # Re-compute pruning mask every N steps

    # Logging
    log_interval: int = 50
    eval_interval: int = 200
    use_wandb: bool = False
    wandb_project: str = "kaleidonet"

    # Mixed precision
    use_amp: bool = True

    # Reproducibility
    seed: int | None = None  # Set for deterministic training

    # Device
    device: str = field(default_factory=_detect_device)


class KaleidoNetTrainer:
    """
    Orchestrates KaleidoNet training with all compound efficiency mechanisms.

    Args:
        model: The full KaleidoNet model (tokenizer + backbone + head).
        config: TrainerConfig.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        task_loss_fn: Task-specific loss function (e.g., nn.CrossEntropyLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        task_loss_fn: nn.Module | None = None,
    ):
        self.model = model.to(torch.device(config.device))
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)

        # Task loss
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()

        # Multi-objective loss
        self.kaleidonet_loss = KaleidoNetLoss(
            lambda_balance=config.lambda_balance,
            lambda_ponder=config.lambda_ponder,
            lambda_distill=config.lambda_distill,
        )

        # Lagrangian budget
        self.budget_manager = LagrangianBudgetManager(
            flops_budget=config.flops_budget,
            lambda_init=config.lambda_init,
            lambda_lr=config.lambda_lr,
            lambda_max=100.0,
        ).to(torch.device(config.device))

        # FLOPs counter
        self.flops_counter = FLOPsCounter()

        # Growth scheduler
        self.growth_scheduler = GrowthScheduler(
            patience=config.growth_patience,
            growth_factor=config.growth_factor,
            max_growth_events=config.max_growth_events,
        )

        # Separate mask parameters from other parameters
        mask_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'mask_logits' in name or 'head_mask_logits' in name:
                mask_params.append(param)
            else:
                other_params.append(param)

        # Main optimizer (for weights) with cosine schedule
        self.optimizer = torch.optim.AdamW(
            other_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Mask optimizer with 3x higher LR and no weight decay
        self.mask_optimizer = torch.optim.Adam(
            mask_params,
            lr=config.lr * 3,
        ) if mask_params else None

        # LR scheduler (cosine with warmup)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=config.max_steps,
            pct_start=config.warmup_steps / config.max_steps,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(enabled=config.use_amp and "cuda" in config.device)

        # wandb integration (optional)
        self._wandb = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    config={
                        "lr": config.lr,
                        "max_steps": config.max_steps,
                        "flops_budget": config.flops_budget,
                        "target_sparsity": config.target_sparsity,
                        "tau_start": config.tau_start,
                        "tau_end": config.tau_end,
                        "seed": config.seed,
                    },
                )
                self._wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, logging disabled")

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _anneal_temperature(self):
        """Anneal Gumbel-sigmoid temperature from tau_start to tau_end."""
        cfg = self.config
        progress = min(self.global_step / max(cfg.tau_anneal_steps, 1), 1.0)
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * progress

        for module in self.model.modules():
            if hasattr(module, "tau"):
                module.tau = tau

    def _get_amp_dtype(self):
        dev_str = str(self.device)
        if "cuda" in dev_str or "xla" in dev_str:
            return torch.bfloat16
        return torch.float32

    def train_step(self, batch: dict) -> dict:
        """Execute a single training step."""
        self.model.train()
        self._anneal_temperature()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        dev_type = self.device.type  # 'xla', 'cuda', 'cpu', etc.
        amp_enabled = self.config.use_amp and dev_type == "cuda"
        with torch.amp.autocast(device_type=dev_type, dtype=self._get_amp_dtype(), enabled=amp_enabled):
            # Forward pass
            outputs = self.model(batch)
            logits = outputs["logits"]
            backbone_aux = outputs["backbone_aux"]
            targets = batch["targets"]

            # Task loss
            if logits.dim() == 3:
                # Sequence prediction: (B, N, vocab) -> (B*N, vocab)
                task_loss = self.task_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                task_loss = self.task_loss_fn(logits, targets)

            # FLOPs measurement
            actual_flops = outputs.get("active_flops", self.config.flops_budget)

            # Combined loss (task only; cubic schedule handles compression)
            total_loss, breakdown = self.kaleidonet_loss(
                task_loss=task_loss,
                backbone_aux=backbone_aux,
                flops_penalty=None,
            )

        # Backward + optimizer step
        self.optimizer.zero_grad(set_to_none=True)
        if self.mask_optimizer:
            self.mask_optimizer.zero_grad(set_to_none=True)

        _is_xla = self.device.type == "xla"
        if _is_xla:
            # XLA/TPU: no scaler, direct backward + xm.optimizer_step
            total_loss.backward()
            for m in self.model.modules():
                if hasattr(m, 'mask_logits') and m.mask_logits.grad is not None:
                    m.mask_logits.grad[m.mask_logits.data <= -50] = 0.0
                if hasattr(m, 'head_mask_logits') and m.head_mask_logits.grad is not None:
                    m.head_mask_logits.grad[m.head_mask_logits.data <= -50] = 0.0
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(self.optimizer)
            if self.mask_optimizer:
                xm.optimizer_step(self.mask_optimizer)
        else:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.mask_optimizer:
                self.scaler.unscale_(self.mask_optimizer)
            for m in self.model.modules():
                if hasattr(m, 'mask_logits') and m.mask_logits.grad is not None:
                    m.mask_logits.grad[m.mask_logits.data <= -50] = 0.0
                if hasattr(m, 'head_mask_logits') and m.head_mask_logits.grad is not None:
                    m.head_mask_logits.grad[m.head_mask_logits.data <= -50] = 0.0
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            if self.mask_optimizer:
                self.scaler.step(self.mask_optimizer)
            self.scaler.update()
        self.scheduler.step()

        # Cubic sparsity schedule: gradually prune lowest-score logits
        self._apply_cubic_pruning()

        self.global_step += 1
        breakdown["lr"] = self.scheduler.get_last_lr()[0]
        breakdown["tau"] = next(
            (m.tau for m in self.model.modules() if hasattr(m, "tau")), 0.0
        )
        breakdown["lambda"] = self.budget_manager.lambda_val.item()
        breakdown["active_flops"] = actual_flops if isinstance(actual_flops, int) else actual_flops.item()

        return breakdown

    @torch.no_grad()
    def eval_step(self) -> dict:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_blocks_used = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = self.model(batch)
            logits = outputs["logits"]
            targets = batch["targets"]
            backbone_aux = outputs["backbone_aux"]

            if logits.dim() == 3:
                loss = self.task_loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).float().sum().item()
                total_samples += targets.numel()
            else:
                loss = self.task_loss_fn(logits, targets)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.shape[0]

            total_loss += loss.item() * targets.shape[0]
            total_blocks_used += backbone_aux.get("blocks_used", 0)

        n = max(total_samples, 1)
        return {
            "val_loss": total_loss / n,
            "val_accuracy": total_correct / n,
            "val_avg_blocks": total_blocks_used / max(len(self.val_loader), 1),
        }

    def train(self):
        """Full training loop."""
        if self.config.seed is not None:
            set_seed(self.config.seed)
            print(f"Seed set to {self.config.seed}")
        print(f"Starting KaleidoNet training on {self.device}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  FLOPs budget: {self.config.flops_budget:,}")
        print(f"  Max steps: {self.config.max_steps}")
        print()

        train_iter = iter(self.train_loader)
        start_time = time.time()

        for step in range(self.config.max_steps):
            # Get batch (cycle through data)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            metrics = self.train_step(batch)

            # Logging
            if step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                # Compute active fraction for visibility
                act_frac = self._get_active_fraction()
                hard_frac = self._get_hard_active_fraction()
                # Mask logit statistics
                logit_stats = self._get_mask_logit_stats()
                print(
                    f"Step {step:6d} | "
                    f"loss={metrics['total_loss']:.4f} | "
                    f"task={metrics['task_loss']:.4f} | "
                    f"soft={act_frac:.0%} hard={hard_frac:.0%} | "
                    f"logits=[{logit_stats['min']:.2f},{logit_stats['mean']:.2f},{logit_stats['max']:.2f}] | "
                    f"λ={metrics['lambda']:.4f} | "
                    f"τ={metrics['tau']:.3f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )
                if self._wandb is not None:
                    self._wandb.log({
                        "train/loss": metrics["total_loss"],
                        "train/task_loss": metrics["task_loss"],
                        "train/lr": metrics["lr"],
                        "train/tau": metrics["tau"],
                        "train/lambda": metrics["lambda"],
                        "train/active_flops": metrics["active_flops"],
                        "train/soft_active_frac": act_frac,
                        "train/hard_active_frac": hard_frac,
                        "train/logit_min": logit_stats["min"],
                        "train/logit_mean": logit_stats["mean"],
                        "train/logit_max": logit_stats["max"],
                        "train/steps_per_sec": steps_per_sec,
                    }, step=step)

            # Evaluation
            if step % self.config.eval_interval == 0 and step > 0:
                val_metrics = self.eval_step()
                if val_metrics:
                    print(
                        f"  [EVAL] val_loss={val_metrics['val_loss']:.4f} | "
                        f"val_acc={val_metrics['val_accuracy']:.4f} | "
                        f"avg_blocks={val_metrics['val_avg_blocks']:.1f}"
                    )
                    if self._wandb is not None:
                        self._wandb.log({
                            "val/loss": val_metrics["val_loss"],
                            "val/accuracy": val_metrics["val_accuracy"],
                            "val/avg_blocks": val_metrics["val_avg_blocks"],
                        }, step=step)

                    # Save checkpoint
                    self.save_checkpoint("checkpoints/latest.pt")

                    val_loss = val_metrics["val_loss"]

                    # Growth check
                    if self.growth_scheduler.step(val_loss):
                        print(f"  [GROWTH] Triggering growth event #{self.growth_scheduler.growth_events + 1}...")
                        event = self.growth_scheduler.execute_growth(
                            self.model, self.optimizer,
                        )
                        print(f"  [GROWTH] {event}")
                        # Rebuild optimizer with new params
                        self.optimizer = torch.optim.AdamW(
                            self.model.parameters(),
                            lr=self.config.lr,
                            weight_decay=self.config.weight_decay,
                        )

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        if self._wandb is not None:
            self._wandb.finish()

    def _get_active_fraction(self) -> float:
        """Compute fraction of active parameters across elastic layers (soft sigmoid)."""
        from kaleidonet.core.elastic import ElasticLinear
        total = 0
        active = 0
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                mask = torch.sigmoid(m.mask_logits)
                active += mask.sum().item()
                total += mask.numel()
        return active / total if total > 0 else 1.0

    def _get_hard_active_fraction(self) -> float:
        """Compute fraction of active parameters across elastic layers (hard threshold)."""
        from kaleidonet.core.elastic import ElasticLinear
        total = 0
        active = 0
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                active += (m.mask_logits >= 0).sum().item()
                total += m.mask_logits.numel()
        return active / total if total > 0 else 1.0

    def _get_mask_logit_stats(self) -> dict:
        """Get min/mean/max of all mask logits for diagnostics."""
        from kaleidonet.core.elastic import ElasticLinear
        all_logits = []
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                all_logits.append(m.mask_logits.detach())
        if not all_logits:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        cat = torch.cat(all_logits)
        return {"min": cat.min().item(), "mean": cat.mean().item(), "max": cat.max().item()}

    @torch.no_grad()
    def _apply_cubic_pruning(self):
        """Apply cubic sparsity schedule: gradually prune lowest-score neurons."""
        from kaleidonet.core.elastic import ElasticLinear
        cfg = self.config
        step = self.global_step

        # Only prune at specified frequency and after start step
        if step < cfg.sparsity_start_step:
            return
        if step % cfg.sparsity_frequency != 0:
            return

        # Cubic schedule: s_t = s_f * (1 - (1 - (t-t0)/(T-t0))^3)
        t = min((step - cfg.sparsity_start_step) / max(cfg.sparsity_end_step - cfg.sparsity_start_step, 1), 1.0)
        current_sparsity = cfg.target_sparsity * (1 - (1 - t) ** 3)

        # Apply per-layer pruning
        for m in self.model.modules():
            if isinstance(m, ElasticLinear):
                logits = m.mask_logits
                n = logits.numel()
                n_prune = max(int(n * current_sparsity), 0)
                n_keep = max(n - n_prune, m.min_width)
                n_prune = n - n_keep

                if n_prune > 0:
                    # Find the threshold: prune logits below the (n_prune)-th value
                    threshold = logits.topk(n_keep, largest=True).values[-1]
                    # Set pruned logits to -100 (hard off)
                    mask = logits >= threshold
                    logits.data[~mask] = -100.0

    def save_checkpoint(self, path: str):
        """Save model + optimizer + scheduler state."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "lambda_val": self.budget_manager.lambda_val.item(),
        }, path)

    def load_checkpoint(self, path: str):
        """Resume from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.budget_manager.lambda_val.fill_(ckpt["lambda_val"])
        print(f"Resumed from step {self.global_step}")
