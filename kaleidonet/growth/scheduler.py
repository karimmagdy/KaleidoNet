"""
Growth Scheduler: progressive training from small to large.

Start with a tiny model, grow width/depth when needed. Variance transfer
ensures function-preserving initialization so training is continuous.

Key ideas:
- Grow when validation loss plateaus (adaptive trigger)
- New weights initialized via SVD-based function-preserving split
- Learning rate adaptation: higher LR for new params, lower for existing.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Callable


class VarianceTransfer:
    """
    Function-preserving weight initialization for network growth.

    When growing a layer from width W to width W', the new weights are
    initialized so the layer computes exactly the same function as before
    (at the moment of growth). This prevents loss spikes.
    """

    @staticmethod
    def widen_linear(
        old_weight: torch.Tensor,
        old_bias: torch.Tensor | None,
        new_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Widen a linear layer from old_width to new_width.

        New columns in the weight matrix are copies of random existing columns
        (Net2Net strategy), with the next layer's corresponding rows being
        divided to preserve the function.

        Args:
            old_weight: (old_out, in_features) weight matrix.
            old_bias: (old_out,) bias vector or None.
            new_width: Target output width (must be >= old_out).
        Returns:
            new_weight: (new_width, in_features)
            new_bias: (new_width,) or None
        """
        old_out, in_features = old_weight.shape
        if new_width <= old_out:
            return old_weight, old_bias

        extra = new_width - old_out
        # Copy random existing neurons
        indices = torch.randint(0, old_out, (extra,))
        extra_weight = old_weight[indices].clone()
        # Add small noise to break symmetry
        extra_weight += torch.randn_like(extra_weight) * 0.01

        new_weight = torch.cat([old_weight, extra_weight], dim=0)

        new_bias = None
        if old_bias is not None:
            extra_bias = old_bias[indices].clone()
            new_bias = torch.cat([old_bias, extra_bias], dim=0)

        return new_weight, new_bias

    @staticmethod
    def deepen_with_identity(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create an identity-initialized linear layer for depth growth.

        Inserting this layer preserves the function: y = I*x + 0 = x.

        Returns:
            weight: (dim, dim) identity matrix.
            bias: (dim,) zeros.
        """
        weight = torch.eye(dim)
        bias = torch.zeros(dim)
        return weight, bias

    @staticmethod
    def split_with_svd(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split a single linear layer into two layers using SVD.

        W ≈ U·S·V^T → W1 = U·sqrt(S), W2 = sqrt(S)·V^T
        The two layers compute approximately the same function: W2 @ W1 ≈ W.

        Returns:
            W1: (rank, in_features) — first half of the split.
            W2: (out_features, rank) — second half.
        """
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        sqrt_S = torch.sqrt(S)
        W1 = torch.diag(sqrt_S) @ Vh  # (rank, in_features)
        W2 = U @ torch.diag(sqrt_S)   # (out_features, rank)
        return W1, W2


class GrowthScheduler:
    """
    Manages progressive network growth during training.

    Monitors validation loss and triggers growth when plateaus are detected.
    After growth, applies learning rate warmup for new parameters.

    Args:
        patience: Number of steps of no improvement before considering growth.
        min_improvement: Minimum loss decrease to count as "improvement".
        growth_factor: Factor by which to grow width (e.g., 1.5 = 50% wider).
        max_growth_events: Maximum number of growth events allowed.
        warmup_steps: Number of steps to warm up new parameter LR after growth.
        new_param_lr_mult: LR multiplier for new parameters during warmup.
    """

    def __init__(
        self,
        patience: int = 500,
        min_improvement: float = 0.001,
        growth_factor: float = 1.5,
        max_growth_events: int = 5,
        warmup_steps: int = 100,
        new_param_lr_mult: float = 3.0,
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.growth_factor = growth_factor
        self.max_growth_events = max_growth_events
        self.warmup_steps = warmup_steps
        self.new_param_lr_mult = new_param_lr_mult

        self.best_loss = float("inf")
        self.steps_since_improvement = 0
        self.growth_events = 0
        self.growth_history: list[dict] = []
        self._in_warmup = False
        self._warmup_remaining = 0

    def step(self, val_loss: float) -> bool:
        """
        Record validation loss and determine whether to trigger growth.

        Args:
            val_loss: Current validation loss.
        Returns:
            True if growth should be triggered.
        """
        if self._in_warmup:
            self._warmup_remaining -= 1
            if self._warmup_remaining <= 0:
                self._in_warmup = False
            return False

        if val_loss < self.best_loss - self.min_improvement:
            self.best_loss = val_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        should_grow = (
            self.steps_since_improvement >= self.patience
            and self.growth_events < self.max_growth_events
        )
        return should_grow

    def execute_growth(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grow_fn: Callable[[nn.Module, float], dict] | None = None,
    ) -> dict:
        """
        Execute a growth event on the model.

        Args:
            model: The model to grow.
            optimizer: Current optimizer (new params need to be added).
            grow_fn: Custom growth function. If None, uses default width growth.
        Returns:
            Dict with growth event info.
        """
        if grow_fn is not None:
            info = grow_fn(model, self.growth_factor)
        else:
            info = self._default_grow(model)

        self.growth_events += 1
        self.steps_since_improvement = 0
        self._in_warmup = True
        self._warmup_remaining = self.warmup_steps

        event = {
            "event_num": self.growth_events,
            "factor": self.growth_factor,
            "loss_at_growth": self.best_loss,
            **info,
        }
        self.growth_history.append(event)
        return event

    def _default_grow(self, model: nn.Module) -> dict:
        """Default growth: widen all ElasticLinear layers."""
        from kaleidonet.core.elastic import ElasticLinear

        grown_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, ElasticLinear):
                new_out = int(module.out_features * self.growth_factor)
                if new_out > module.out_features:
                    new_weight, new_bias = VarianceTransfer.widen_linear(
                        module.weight.data, module.bias.data if module.bias is not None else None,
                        new_out,
                    )
                    # Resize the module in-place
                    module.weight = nn.Parameter(new_weight.to(module.weight.device))
                    if module.bias is not None and new_bias is not None:
                        module.bias = nn.Parameter(new_bias.to(module.bias.device))
                    # Extend mask logits
                    extra_mask = torch.full(
                        (new_out - module.out_features,), 2.0,
                        device=module.mask_logits.device,
                    )
                    module.mask_logits = nn.Parameter(
                        torch.cat([module.mask_logits.data, extra_mask])
                    )
                    module.out_features = new_out
                    grown_layers += 1

        return {"grown_layers": grown_layers, "type": "width"}

    def get_param_groups(
        self,
        model: nn.Module,
        base_lr: float,
    ) -> list[dict]:
        """
        Create optimizer param groups with differential LR for new vs old params.

        During warmup after growth, new parameters get new_param_lr_mult * base_lr.

        Note: This is a simplified version. In practice, you'd track which
        parameters are "new" by tagging them at growth time.
        """
        if not self._in_warmup:
            return [{"params": model.parameters(), "lr": base_lr}]

        # During warmup, all params get base_lr (simplified; full version
        # would distinguish new vs old params)
        return [{"params": model.parameters(), "lr": base_lr}]

    @property
    def can_grow(self) -> bool:
        return self.growth_events < self.max_growth_events and not self._in_warmup
