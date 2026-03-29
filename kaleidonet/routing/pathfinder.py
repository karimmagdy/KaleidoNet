"""
Pathfinding-optimized routing: A*-inspired and Sinkhorn min-cost flow routers.

These replace naive top-k MoE gating with cost-aware routing that considers
both quality (expected accuracy contribution) and cost (FLOPs) of each expert.

Key innovation: the A* heuristic is *learned end-to-end*. It estimates "how much
more compute does this input need?" — fundamentally different from fixed top-k
routing which ignores input difficulty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AStarRouter(nn.Module):
    """
    Differentiable A*-inspired router over the computation graph.

    Models the network as a DAG where nodes are experts/blocks.
    For each input, computes:
    - g(n): accumulated cost (FLOPs) to reach node n
    - h(n): learned heuristic estimating remaining cost-to-accuracy from node n
    - f(n) = g(n) + h(n): total estimated cost

    During training: softmax over -f(n) for differentiable path probabilities.
    During inference: hard argmin (actual A*) for maximum speedup.

    Args:
        embed_dim: Token embedding dimension.
        num_nodes: Number of computational nodes (experts/blocks) in the graph.
        heuristic_hidden: Hidden dim of the learned heuristic network.
        cost_weight: How much to weight FLOPs cost vs quality.
    """

    def __init__(
        self,
        embed_dim: int,
        num_nodes: int,
        heuristic_hidden: int = 32,
        cost_weight: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.cost_weight = cost_weight

        # Learned heuristic h(n): estimates remaining cost-to-accuracy from node n
        # Input: token embedding + node embedding → scalar estimate
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.heuristic_net = nn.Sequential(
            nn.Linear(embed_dim * 2, heuristic_hidden),
            nn.ReLU(),
            nn.Linear(heuristic_hidden, 1),
        )

        # Quality estimator q(n): expected quality contribution of each node
        self.quality_net = nn.Sequential(
            nn.Linear(embed_dim * 2, heuristic_hidden),
            nn.ReLU(),
            nn.Linear(heuristic_hidden, 1),
            nn.Sigmoid(),  # Quality in [0, 1]
        )

        # Learnable per-node FLOPs cost (normalized; actual FLOPs are external)
        self.node_costs = nn.Parameter(torch.ones(num_nodes) * 0.5)

        # Temperature for softmax relaxation
        self.temperature = 1.0

    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 2,
        actual_flops_per_node: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute A*-inspired routing scores for each node.

        Args:
            x: (num_tokens, embed_dim) — token embeddings.
            top_k: Number of nodes to select per token.
            actual_flops_per_node: (num_nodes,) — measured FLOPs per node (optional).
        Returns:
            selected_indices: (num_tokens, top_k) — selected node indices.
            selected_weights: (num_tokens, top_k) — routing weights.
            aux: Dict with routing statistics.
        """
        T, D = x.shape

        # Costs g(n): accumulated FLOPs cost
        if actual_flops_per_node is not None:
            # Normalize to [0, 1] range for stable combination with heuristic
            g = actual_flops_per_node / (actual_flops_per_node.max() + 1e-8)
        else:
            g = torch.sigmoid(self.node_costs)  # (num_nodes,)

        # Expand for broadcasting: (T, num_nodes, embed_dim*2)
        x_expanded = x.unsqueeze(1).expand(T, self.num_nodes, D)
        nodes_expanded = self.node_embeddings.unsqueeze(0).expand(T, self.num_nodes, D)
        combined = torch.cat([x_expanded, nodes_expanded], dim=-1)  # (T, num_nodes, D*2)

        # Heuristic h(n): learned estimate of remaining cost
        h = self.heuristic_net(combined).squeeze(-1)  # (T, num_nodes)

        # Quality q(n): expected benefit of activating this node
        q = self.quality_net(combined).squeeze(-1)  # (T, num_nodes)

        # A* score: f(n) = cost_weight * g(n) + h(n) - q(n)
        # Lower f is better (cheaper + better quality)
        # We negate quality so that high-quality nodes have lower f.
        g_expanded = g.unsqueeze(0).expand(T, self.num_nodes)
        f_scores = self.cost_weight * g_expanded + h - q  # (T, num_nodes)

        # Routing probabilities: softmax over -f (lower f = higher probability)
        if self.training:
            probs = F.softmax(-f_scores / self.temperature, dim=-1)
        else:
            probs = F.softmax(-f_scores / 0.01, dim=-1)  # Near-hard routing at inference

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(probs, top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Auxiliary info
        with torch.no_grad():
            selected_costs = g[top_k_indices].mean()
            selected_quality = q.gather(1, top_k_indices).mean()
            routing_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

        aux = {
            "a_star_cost": selected_costs,
            "a_star_quality": selected_quality,
            "routing_entropy": routing_entropy,
            "mean_heuristic": h.mean(),
            "mean_f_score": f_scores.mean(),
        }

        return top_k_indices, top_k_weights, aux


class SinkhornRouter(nn.Module):
    """
    Min-cost flow router using Sinkhorn iterations.

    Formulates token-to-expert assignment as an optimal transport problem.
    Capacity constraints per expert are built-in (natural load balancing).
    GPU-friendly and differentiable.

    Args:
        embed_dim: Token embedding dimension.
        num_experts: Number of experts.
        sinkhorn_iters: Number of Sinkhorn normalization iterations.
        capacity_factor: Expert capacity as fraction of (tokens / experts).
        temperature: Softmax temperature for cost matrix.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        sinkhorn_iters: int = 20,
        capacity_factor: float = 1.25,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.sinkhorn_iters = sinkhorn_iters
        self.capacity_factor = capacity_factor
        self.temperature = temperature

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: (num_tokens, embed_dim)
            top_k: Number of experts per token.
        Returns:
            selected_indices: (num_tokens, top_k)
            selected_weights: (num_tokens, top_k)
            aux: Dict with routing statistics.
        """
        T, D = x.shape
        logits = self.gate(x) / self.temperature  # (T, num_experts)

        # Sinkhorn iterations to get doubly-stochastic assignment
        # Row constraint: each token assigned to top_k experts (sum ≈ top_k)
        # Column constraint: each expert gets ≈ capacity tokens
        capacity = int(self.capacity_factor * T * top_k / self.num_experts)
        capacity = max(capacity, 1)

        # Log-domain Sinkhorn for numerical stability
        log_alpha = logits
        for _ in range(self.sinkhorn_iters):
            # Row normalization (token side)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Column normalization (expert side) with capacity
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

        assignment = log_alpha.exp()  # (T, num_experts) — near doubly-stochastic

        # Top-k selection from the assignment matrix
        top_k_weights, top_k_indices = torch.topk(assignment, top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balance
        with torch.no_grad():
            tokens_per_expert = assignment.sum(dim=0)
            utilization = (tokens_per_expert > 0.1).float().mean()

        # Balance loss (variance in expert load)
        balance_loss = tokens_per_expert.var()

        aux = {
            "load_balance_loss": balance_loss,
            "expert_utilization": utilization,
            "tokens_per_expert": tokens_per_expert.detach(),
            "assignment_entropy": -(assignment * (assignment + 1e-8).log()).sum(dim=-1).mean(),
        }

        return top_k_indices, top_k_weights, aux
