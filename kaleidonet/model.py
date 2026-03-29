"""
KaleidoNet: Full end-to-end model wiring tokenizer → backbone → task heads.

This is the main user-facing class that composes all pillars into a single
trainable module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kaleidonet.tokenizers.universal import UniversalTokenizer
from kaleidonet.backbone.universal import UniversalBackbone
from kaleidonet.metrics.flops import FLOPsCounter


class KaleidoNet(nn.Module):
    """
    Complete KaleidoNet model for multi-modal multi-task learning.

    Composes:
    1. UniversalTokenizer — converts any modality to shared token space
    2. UniversalBackbone — fractal-MoE transformer with all dynamic mechanisms
    3. Task heads — classification, generation, contrastive

    Args:
        embed_dim: Shared embedding dimension.
        num_blocks: Number of backbone blocks.
        num_heads: Max attention heads.
        num_experts: MoE experts per block.
        top_k: Active experts per token.
        num_classes: Number of classes for classification head (0 to disable).
        vocab_size: Vocabulary size for text/generation.
        image_size: Input image size.
        patch_size: Patch size for vision tokenizer.
        max_seq_len: Max text sequence length.
        elastic: Use elastic (morphable) layers.
        drop_path_rate: Drop-path rate.
        dropout: Dropout rate.
        confidence_threshold: Early exit threshold.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k: int = 1,
        num_classes: int = 100,
        vocab_size: int = 32000,
        image_size: int = 32,
        patch_size: int = 4,
        max_seq_len: int = 512,
        elastic: bool = True,
        drop_path_rate: float = 0.1,
        dropout: float = 0.0,
        confidence_threshold: float = 0.95,
    ):
        super().__init__()

        # Tokenizer
        self.tokenizer = UniversalTokenizer(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )

        # Backbone
        self.backbone = UniversalBackbone(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            elastic=elastic,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            confidence_threshold=confidence_threshold,
        )

        # Task heads
        self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        self.lm_head = nn.Linear(embed_dim, vocab_size) if vocab_size > 0 else None

        # Task embedding (tells backbone which task is active)
        num_tasks = 3  # classification, generation, contrastive
        self.task_embed = nn.Embedding(num_tasks, embed_dim)

        self._flops_counter = FLOPsCounter()

    def differentiable_flops(self, seq_len: int = 1) -> torch.Tensor:
        """Compute differentiable FLOPs estimate for Lagrangian penalty gradient flow."""
        from kaleidonet.core.elastic import ElasticLinear, ElasticAttention
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        counted = set()
        for name, m in self.named_modules():
            if any(name.startswith(p + ".") for p in counted):
                continue
            if isinstance(m, (ElasticLinear, ElasticAttention)) and hasattr(m, "soft_active_flops"):
                total = total + m.soft_active_flops(batch_size=1, seq_len=seq_len)
                counted.add(name)
            elif isinstance(m, nn.Linear):
                total = total + 2 * m.in_features * m.out_features * seq_len
        return total

    def forward(self, batch: dict) -> dict:
        """
        Unified forward pass for any task.

        Args:
            batch: Dict with keys:
                - 'images': (B, C, H, W) optional
                - 'token_ids': (B, N) optional
                - 'targets': (B,) or (B, N) — labels
                - 'task': str — 'classify', 'generate', or 'contrastive'
        Returns:
            Dict with 'logits', 'backbone_aux', 'active_flops'.
        """
        images = batch.get("images")
        token_ids = batch.get("token_ids")
        task = batch.get("task", "classify")

        # Tokenize
        tokens, tok_info = self.tokenizer(images=images, token_ids=token_ids)

        # Add task embedding
        task_id = {"classify": 0, "generate": 1, "contrastive": 2}.get(task, 0)
        task_emb = self.task_embed(torch.tensor(task_id, device=tokens.device))
        tokens = tokens + task_emb.unsqueeze(0).unsqueeze(0)

        # Backbone forward
        hidden, backbone_aux = self.backbone(tokens)

        # Task head
        if task == "classify" and self.cls_head is not None:
            # Global average pooling → classification
            pooled = hidden.mean(dim=1)  # (B, embed_dim)
            logits = self.cls_head(pooled)  # (B, num_classes)
        elif task == "generate" and self.lm_head is not None:
            logits = self.lm_head(hidden)  # (B, N, vocab_size)
        else:
            # Contrastive: return hidden states directly
            logits = hidden

        # Compute active FLOPs for this forward pass (per-sample)
        seq_len = tokens.shape[1]
        flops_info = self._flops_counter.count(self, batch_size=1, seq_len=seq_len)
        diff_flops = self.differentiable_flops(seq_len=seq_len)

        return {
            "logits": logits,
            "hidden": hidden,
            "backbone_aux": backbone_aux,
            "tokenizer_info": tok_info,
            "active_flops": flops_info["total_active_flops"],
            "diff_flops": diff_flops,
        }

    def count_active_params(self) -> dict:
        """Count active vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        active = total  # Start with total, subtract dormant

        # Count inactive neurons in elastic layers
        from kaleidonet.core.elastic import ElasticLinear, ElasticAttention, ElasticConv2d
        dormant = 0
        for module in self.modules():
            if isinstance(module, ElasticLinear):
                inactive = module.out_features - module.active_width
                dormant += inactive * module.in_features
                if module.bias is not None:
                    dormant += inactive
            elif isinstance(module, ElasticAttention):
                inactive_heads = module.num_heads - module.active_heads
                dormant += inactive_heads * module.head_dim * module.embed_dim * 4  # Q,K,V,O per head approx

        active = total - dormant
        return {
            "total_params": total,
            "active_params": active,
            "dormant_params": dormant,
            "active_fraction": active / max(total, 1),
        }
