"""
FractalNet blocks: self-similar recursive columns with drop-path regularization.

A fractal block at depth D contains:
- A direct path (single operation)
- A recursive path (two sub-fractal blocks of depth D-1 joined)

Drop-path randomly disables entire sub-paths during training, giving the network
an "anytime" property: shallow paths give fast answers, full depth gives best.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaleidonet.core.elastic import ElasticLinear


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastically drop entire paths (applied per-sample in a batch)."""
    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
    return x * mask / (keep_prob + 1e-8)


class FractalBlock(nn.Module):
    """
    Self-similar fractal block.

    At depth 1: just a single operation (ElasticLinear + activation).
    At depth D: combines a direct path with a recursive path of two FractalBlock(D-1).
    A learned join weight interpolates between paths.

    Args:
        dim: Feature dimension (in = out for residual compatibility).
        depth: Recursion depth of the fractal.
        drop_path_rate: Probability of dropping each sub-path during training.
        elastic: Whether to use ElasticLinear (morphable) or standard Linear.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 3,
        drop_path_rate: float = 0.15,
        elastic: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.drop_path_rate = drop_path_rate

        # Direct path: simple transform
        if elastic:
            self.direct = nn.Sequential(ElasticLinear(dim, dim), nn.GELU())
        else:
            self.direct = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

        # Recursive path (only if depth > 1)
        if depth > 1:
            self.left = FractalBlock(dim, depth - 1, drop_path_rate, elastic)
            self.right = FractalBlock(dim, depth - 1, drop_path_rate, elastic)
            # Learned join weight for combining direct and recursive paths
            self.join_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.left = None
            self.right = None

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            Output of same shape.
        """
        residual = x
        direct_out = self.direct(x)
        direct_out = drop_path(direct_out, self.drop_path_rate, self.training)

        if self.left is not None and self.right is not None:
            recursive_out = self.right(self.left(x))
            recursive_out = drop_path(recursive_out, self.drop_path_rate, self.training)
            # Interpolate between direct and recursive paths
            alpha = torch.sigmoid(self.join_weight)
            out = alpha * direct_out + (1 - alpha) * recursive_out
        else:
            out = direct_out

        return self.norm(out + residual)

    @property
    def num_paths(self) -> int:
        """Total number of possible paths through this fractal block."""
        if self.depth <= 1:
            return 1
        return 1 + self.left.num_paths * self.right.num_paths


class FractalNet(nn.Module):
    """
    Stack of fractal blocks forming the backbone.

    Args:
        dim: Feature dimension.
        num_blocks: Number of fractal blocks stacked sequentially.
        fractal_depth: Recursion depth of each block.
        drop_path_rate: Drop-path rate.
        elastic: Use elastic (morphable) layers.
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 4,
        fractal_depth: int = 3,
        drop_path_rate: float = 0.15,
        elastic: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FractalBlock(dim, fractal_depth, drop_path_rate, elastic)
            for _ in range(num_blocks)
        ])
        # Auxiliary heads at each depth for anytime prediction
        self.aux_heads: nn.ModuleList | None = None

    def add_aux_heads(self, num_classes: int):
        """Add auxiliary classification heads at each block for anytime inference."""
        dim = self.blocks[0].direct[0].in_features if hasattr(self.blocks[0].direct[0], "in_features") else self.blocks[0].direct[0].in_features
        self.aux_heads = nn.ModuleList([
            nn.Linear(dim, num_classes) for _ in self.blocks
        ])

    def forward(
        self, x: torch.Tensor, exit_after: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (..., dim)
            exit_after: If set, exit after this many blocks (anytime inference).
        Returns:
            final_out: Output after all (or exit_after) blocks.
            aux_logits: List of auxiliary predictions from each block (empty if no aux_heads).
        """
        aux_logits = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if self.aux_heads is not None:
                aux_logits.append(self.aux_heads[i](x.mean(dim=-2) if x.dim() > 2 else x))

            if exit_after is not None and i + 1 >= exit_after:
                break

        return x, aux_logits
