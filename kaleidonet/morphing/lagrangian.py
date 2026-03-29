"""
Lagrangian compute budget manager.

Self-regulating mechanism: a learnable dual variable λ penalizes FLOPs overshoot.
If total FLOPs exceed budget B, λ rises, which penalizes wide layers, which causes
the morph controller to compress them. No manual threshold tuning needed.

Augmented loss: L_total = L_task + λ * (FLOPs(shape) - B)
Dual update: λ ← max(0, λ + η_dual * (FLOPs(shape) - B))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LagrangianBudgetManager(nn.Module):
    """
    Manages the compute budget via Lagrangian dual ascent.

    Maintains a dual variable λ that automatically balances task loss against
    compute cost. When actual FLOPs exceed the budget, λ increases to pressure
    the morph controller toward narrower shapes. When under budget, λ decreases
    to allow fuller capacity.

    Args:
        flops_budget: Target FLOPs budget per forward pass.
        lambda_init: Initial value of the dual variable.
        lambda_lr: Learning rate for dual ascent (separate from main optimizer).
        lambda_max: Maximum value for λ (prevents instability).
    """

    def __init__(
        self,
        flops_budget: int,
        lambda_init: float = 0.01,
        lambda_lr: float = 0.001,
        lambda_max: float = 10.0,
    ):
        super().__init__()
        self.flops_budget = flops_budget
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max

        # λ is NOT a learned parameter — updated by dual ascent, not gradient descent
        self.register_buffer("lambda_val", torch.tensor(lambda_init))
        # Track constraint violation history for diagnostics
        self.register_buffer("violation_ema", torch.tensor(0.0))
        self.ema_decay = 0.99

    def compute_penalty(self, actual_flops: int | torch.Tensor) -> torch.Tensor:
        """
        Compute the Lagrangian penalty: λ * (actual_FLOPs - budget).

        Args:
            actual_flops: Current active FLOPs (int or differentiable tensor).
        Returns:
            Penalty term to add to the loss.
        """
        if isinstance(actual_flops, int):
            actual_flops = torch.tensor(float(actual_flops), device=self.lambda_val.device)
        else:
            actual_flops = actual_flops.float()

        budget = torch.tensor(float(self.flops_budget), device=self.lambda_val.device)
        # Normalize violation as fraction of budget so penalty scale is ~O(1)
        violation = (actual_flops - budget) / budget
        penalty = self.lambda_val * violation

        return penalty

    @torch.no_grad()
    def dual_step(self, actual_flops: int | torch.Tensor):
        """
        Update the dual variable λ via dual ascent.

        Called once per batch (after the main optimizer step).

        Args:
            actual_flops: Measured active FLOPs for this batch.
        """
        if isinstance(actual_flops, int):
            actual_flops = float(actual_flops)
        else:
            actual_flops = actual_flops.item()

        # Normalized violation (fraction of budget)
        violation = (actual_flops - self.flops_budget) / self.flops_budget
        # Dual ascent: λ ← max(0, λ + η * violation)
        new_lambda = self.lambda_val + self.lambda_lr * violation
        self.lambda_val.copy_(new_lambda.clamp(0.0, self.lambda_max))

        # Update EMA of violation for diagnostics
        self.violation_ema.copy_(
            self.ema_decay * self.violation_ema + (1 - self.ema_decay) * violation
        )

    @property
    def is_over_budget(self) -> bool:
        return self.violation_ema.item() > 0

    def state_dict_extra(self) -> dict:
        return {
            "lambda_val": self.lambda_val.item(),
            "violation_ema": self.violation_ema.item(),
            "flops_budget": self.flops_budget,
        }

    def extra_repr(self) -> str:
        return (
            f"budget={self.flops_budget:,}, λ={self.lambda_val.item():.4f}, "
            f"violation_ema={self.violation_ema.item():.1f}"
        )
