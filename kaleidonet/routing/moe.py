"""
Mixture-of-Experts layer with top-k routing and load balancing.

Implements Switch Transformer-style top-k routing where each token is dispatched
to the top-k experts out of N total. Only the selected experts compute on each
token, giving O(k/N) activation sparsity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaleidonet.core.elastic import ElasticLinear


class TopKRouter(nn.Module):
    """
    Learned router that assigns tokens to experts via top-k selection.

    The router is a single linear projection from token embeddings to expert
    scores, followed by top-k selection. During training, a load-balancing
    auxiliary loss prevents expert collapse.

    Args:
        embed_dim: Token embedding dimension.
        num_experts: Number of expert sub-networks.
        top_k: Number of experts activated per token.
        capacity_factor: Multiplier for expert buffer capacity (prevents overflow).
        jitter_noise: Multiplicative noise added during training for exploration.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: (batch * seq_len, embed_dim) — flattened token embeddings.
        Returns:
            dispatch_mask: (batch*seq_len, num_experts) — soft routing weights (top-k nonzero).
            expert_indices: (batch*seq_len, top_k) — indices of selected experts per token.
            expert_weights: (batch*seq_len, top_k) — normalized weights for selected experts.
            aux: Dict with 'load_balance_loss' and 'expert_utilization'.
        """
        # Optional jitter for exploration during training
        if self.training and self.jitter_noise > 0:
            x = x * (1.0 + self.jitter_noise * torch.randn_like(x))

        # Router logits and probabilities
        logits = self.gate(x)  # (num_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Normalize weights among selected experts
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Build dispatch mask (sparse)
        dispatch_mask = torch.zeros_like(probs)
        dispatch_mask.scatter_(1, top_k_indices, top_k_weights)

        # Load balancing auxiliary loss (Switch Transformer formulation)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability for expert i
        # loss = N * sum(f_i * P_i)
        tokens_per_expert = dispatch_mask.sum(dim=0)  # (num_experts,)
        f = tokens_per_expert / (x.shape[0] + 1e-8)
        p = probs.mean(dim=0)
        load_balance_loss = self.num_experts * (f * p).sum()

        # Expert utilization stats
        with torch.no_grad():
            expert_counts = (dispatch_mask > 0).float().sum(dim=0)
            utilization = (expert_counts > 0).float().mean()

        aux = {
            "load_balance_loss": load_balance_loss,
            "expert_utilization": utilization,
            "tokens_per_expert": tokens_per_expert.detach(),
            "router_entropy": -(probs * (probs + 1e-8).log()).sum(dim=-1).mean(),
        }

        return dispatch_mask, top_k_indices, top_k_weights, aux


class ExpertFFN(nn.Module):
    """Single expert feed-forward network (optionally elastic)."""

    def __init__(self, embed_dim: int, hidden_dim: int, elastic: bool = False, dropout: float = 0.0):
        super().__init__()
        if elastic:
            self.fc1 = ElasticLinear(embed_dim, hidden_dim)
            self.fc2 = ElasticLinear(hidden_dim, embed_dim)
        else:
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer.

    Replaces a standard FFN block. Each token is routed to top-k experts,
    processed by those experts, and the outputs are combined with routing weights.

    Args:
        embed_dim: Token embedding dimension.
        num_experts: Number of expert FFNs.
        expert_hidden_dim: Hidden dimension within each expert.
        top_k: Number of experts per token.
        elastic_experts: Whether experts use ElasticLinear (morphable width).
        capacity_factor: Expert buffer capacity multiplier.
        dropout: Dropout rate within experts.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int | None = None,
        top_k: int = 1,
        elastic_experts: bool = False,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * embed_dim

        self.router = TopKRouter(embed_dim, num_experts, top_k, capacity_factor)
        self.experts = nn.ModuleList([
            ExpertFFN(embed_dim, expert_hidden_dim, elastic=elastic_experts, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim) — MoE-processed tokens.
            aux: Routing auxiliary info (load balance loss, utilization, etc.).
        """
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        dispatch_mask, top_k_indices, top_k_weights, aux = self.router(x_flat)

        # Process tokens through selected experts
        output = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]  # (B*N,)
            weight = top_k_weights[:, k_idx]       # (B*N,)

            for e_id in range(self.num_experts):
                token_mask = (expert_idx == e_id)
                if not token_mask.any():
                    continue
                tokens = x_flat[token_mask]
                expert_out = self.experts[e_id](tokens)
                output[token_mask] += weight[token_mask].unsqueeze(-1) * expert_out

        output = output.reshape(B, N, C)
        return output, aux

    def active_flops(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Approximate active FLOPs: only top-k experts compute per token."""
        # Each token uses top_k experts; each expert is an FFN
        expert = self.experts[0]
        if hasattr(expert.fc1, "active_flops"):
            per_expert = expert.fc1.active_flops() + expert.fc2.active_flops()
        else:
            per_expert = (
                2 * expert.fc1.in_features * expert.fc1.out_features +
                2 * expert.fc2.in_features * expert.fc2.out_features
            )
        # Router FLOPs
        router_flops = 2 * self.embed_dim * self.num_experts * batch_size * seq_len
        # Expert FLOPs
        expert_flops = per_expert * self.top_k * batch_size * seq_len
        return router_flops + expert_flops
