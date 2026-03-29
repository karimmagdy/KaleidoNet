"""
Universal Backbone: shared fractal-MoE transformer for all modalities.

This is the core architecture of KaleidoNet. It combines:
- ElasticAttention with dynamic head count
- MoE FFN with dynamic expert routing
- Fractal block structure with drop-path
- Early exit capability (anytime inference)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kaleidonet.core.elastic import ElasticAttention
from kaleidonet.routing.moe import MoELayer
from kaleidonet.growth.fractal import drop_path as stochastic_drop_path


class KaleidoNetBlock(nn.Module):
    """
    Single transformer block with elastic attention + MoE FFN.

    Pre-norm architecture:
    x → LayerNorm → ElasticAttention → residual → LayerNorm → MoE → residual

    Includes:
    - Confidence head for early exit decisions
    - Drop-path for regularization

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Max attention heads.
        num_experts: Number of MoE experts.
        expert_hidden_dim: Hidden dim per expert.
        top_k: Experts per token.
        elastic: Use elastic (morphable) sub-layers.
        drop_path_rate: Stochastic depth rate for this block.
        dropout: Attention/FFN dropout.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_experts: int = 8,
        expert_hidden_dim: int | None = None,
        top_k: int = 1,
        elastic: bool = True,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ElasticAttention(
            embed_dim, num_heads, min_heads=1, dropout=dropout,
        ) if elastic else nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.elastic_attn = elastic

        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe = MoELayer(
            embed_dim, num_experts, expert_hidden_dim, top_k,
            elastic_experts=elastic, dropout=dropout,
        )

        self.drop_path_rate = drop_path_rate

        # Confidence head for early exit (lightweight: single linear)
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_heads: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            target_heads: Morph controller target for attention heads.
            attn_mask: Optional attention mask.
        Returns:
            x: Output tensor.
            block_aux: Dict with MoE aux info + confidence scores.
        """
        # Attention
        residual = x
        x_norm = self.norm1(x)
        if self.elastic_attn:
            attn_out = self.attn(x_norm, target_heads=target_heads, attn_mask=attn_mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        attn_out = stochastic_drop_path(attn_out, self.drop_path_rate, self.training)
        x = residual + attn_out

        # MoE FFN
        residual = x
        x_norm = self.norm2(x)
        moe_out, moe_aux = self.moe(x_norm)
        moe_out = stochastic_drop_path(moe_out, self.drop_path_rate, self.training)
        x = residual + moe_out

        # Confidence score (mean over sequence)
        confidence = self.confidence_head(x.mean(dim=1)).squeeze(-1)  # (batch,)

        block_aux = {
            **moe_aux,
            "confidence": confidence,
        }
        return x, block_aux


class UniversalBackbone(nn.Module):
    """
    Full KaleidoNet backbone: stack of KaleidoNetBlocks with early exit.

    Args:
        embed_dim: Token embedding dimension.
        num_blocks: Number of transformer blocks.
        num_heads: Max attention heads per block.
        num_experts: MoE experts per block.
        expert_hidden_dim: Hidden dim per expert.
        top_k: Experts activated per token.
        elastic: Use elastic sub-layers.
        drop_path_rate: Maximum drop-path rate (linearly scaled per block).
        dropout: Dropout rate.
        confidence_threshold: Early exit confidence threshold (inference only).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        num_experts: int = 8,
        expert_hidden_dim: int | None = None,
        top_k: int = 1,
        elastic: bool = True,
        drop_path_rate: float = 0.1,
        dropout: float = 0.0,
        confidence_threshold: float = 0.95,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.confidence_threshold = confidence_threshold

        # Linearly increasing drop-path rates
        dpr = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]

        self.blocks = nn.ModuleList([
            KaleidoNetBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                expert_hidden_dim=expert_hidden_dim,
                top_k=top_k,
                elastic=elastic,
                drop_path_rate=dpr[i],
                dropout=dropout,
            )
            for i in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Ponder cost: tracks computation spent per sample for ACT regularization
        self._ponder_cost = 0.0

    def forward(
        self,
        x: torch.Tensor,
        target_heads_per_block: list[int] | None = None,
        attn_mask: torch.Tensor | None = None,
        allow_early_exit: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, embed_dim) — tokenized input from UniversalTokenizer.
            target_heads_per_block: Morph controller targets per block.
            attn_mask: Optional attention mask.
            allow_early_exit: Whether to enable early exit during inference.
        Returns:
            x: (batch, seq_len, embed_dim) — final hidden states.
            backbone_aux: Aggregated auxiliary info.
        """
        all_aux = []
        blocks_used = 0
        total_balance_loss = torch.tensor(0.0, device=x.device)
        total_confidence = []

        for i, block in enumerate(self.blocks):
            target_heads = target_heads_per_block[i] if target_heads_per_block else None
            x, block_aux = block(x, target_heads=target_heads, attn_mask=attn_mask)

            blocks_used += 1
            total_balance_loss = total_balance_loss + block_aux["load_balance_loss"]
            total_confidence.append(block_aux["confidence"])
            all_aux.append(block_aux)

            # Early exit during inference
            if (
                allow_early_exit
                and not self.training
                and block_aux["confidence"].min().item() > self.confidence_threshold
                and i >= 1  # Use at least 2 blocks
            ):
                break

        x = self.final_norm(x)

        # Ponder cost: proportion of total blocks used
        ponder_cost = blocks_used / self.num_blocks

        backbone_aux = {
            "blocks_used": blocks_used,
            "total_blocks": self.num_blocks,
            "ponder_cost": ponder_cost,
            "total_balance_loss": total_balance_loss,
            "per_block_aux": all_aux,
            "mean_confidence": torch.stack(total_confidence).mean() if total_confidence else torch.tensor(0.0),
        }
        return x, backbone_aux
