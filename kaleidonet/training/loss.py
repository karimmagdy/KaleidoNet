"""
Multi-objective loss for KaleidoNet.

Combines:
- Task loss (CE for classification, LM loss for generation, contrastive for CLIP)
- FLOPs penalty (Lagrangian compute budget)
- Load balance loss (MoE expert utilization)
- Ponder cost (early exit regularization)
- Distillation loss (teacher → student knowledge transfer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KaleidoNetLoss(nn.Module):
    """
    Multi-objective loss:
    L = L_task + λ1*L_flops + λ2*L_balance + λ3*L_ponder + λ4*L_distill

    Args:
        lambda_balance: Weight for load balance loss.
        lambda_ponder: Weight for ponder (early exit) cost.
        lambda_distill: Weight for distillation loss.
        distill_temperature: Temperature for distillation softmax.
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_ponder: float = 0.01,
        lambda_distill: float = 0.5,
        distill_temperature: float = 4.0,
    ):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_ponder = lambda_ponder
        self.lambda_distill = lambda_distill
        self.distill_temperature = distill_temperature

    def forward(
        self,
        task_loss: torch.Tensor,
        backbone_aux: dict,
        flops_penalty: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        student_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            task_loss: Primary task loss (CE, LM, contrastive).
            backbone_aux: Auxiliary dict from UniversalBackbone.forward().
            flops_penalty: From LagrangianBudgetManager.compute_penalty().
            teacher_logits: Teacher model predictions (for distillation).
            student_logits: Student model predictions (for distillation).
        Returns:
            total_loss: Combined scalar loss for backprop.
            loss_breakdown: Dict with each component for logging.
        """
        total = task_loss.clone()
        breakdown = {"task_loss": task_loss.item()}

        # Load balance loss from MoE routing
        balance_loss = backbone_aux.get("total_balance_loss", torch.tensor(0.0))
        if isinstance(balance_loss, torch.Tensor):
            total = total + self.lambda_balance * balance_loss
            breakdown["balance_loss"] = balance_loss.item()

        # Ponder cost (early exit penalty)
        ponder_cost = backbone_aux.get("ponder_cost", 0.0)
        if isinstance(ponder_cost, (int, float)):
            ponder_cost = torch.tensor(ponder_cost, device=total.device)
        total = total + self.lambda_ponder * ponder_cost
        breakdown["ponder_cost"] = ponder_cost.item() if isinstance(ponder_cost, torch.Tensor) else ponder_cost

        # FLOPs penalty from Lagrangian budget
        if flops_penalty is not None:
            total = total + flops_penalty
            breakdown["flops_penalty"] = flops_penalty.item()

        # Distillation loss (KL divergence)
        if teacher_logits is not None and student_logits is not None:
            T = self.distill_temperature
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
            total = total + self.lambda_distill * distill_loss
            breakdown["distill_loss"] = distill_loss.item()

        breakdown["total_loss"] = total.item()
        breakdown["blocks_used"] = backbone_aux.get("blocks_used", 0)
        breakdown["mean_confidence"] = (
            backbone_aux["mean_confidence"].item()
            if isinstance(backbone_aux.get("mean_confidence"), torch.Tensor)
            else 0.0
        )

        return total, breakdown
