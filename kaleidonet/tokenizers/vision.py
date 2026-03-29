"""
Vision tokenizer: converts images into patch token sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """
    Convert images into a sequence of patch embeddings.

    Splits an image into non-overlapping patches and projects each patch
    into the shared embedding space. Adds positional and modality embeddings.

    Args:
        image_size: Input image size (assumes square).
        patch_size: Patch size (assumes square).
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Output embedding dimension (shared token space).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, channels, H, W)
        Returns:
            tokens: (batch, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed + self.modality_embed
        return self.norm(x)
