"""
Text tokenizer: wraps BPE tokenization and produces embeddings in shared token space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextTokenizer(nn.Module):
    """
    Convert token IDs into embeddings in the shared token space.

    Assumes an external BPE/WordPiece tokenizer produces integer IDs.
    This module handles the embedding lookup + positional + modality embedding.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        max_seq_len: Maximum sequence length.
        embed_dim: Output embedding dimension (shared token space).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer token IDs.
        Returns:
            tokens: (batch, seq_len, embed_dim)
        """
        B, N = token_ids.shape
        x = self.token_embed(token_ids)
        x = x + self.pos_embed[:, :N, :] + self.modality_embed
        return self.norm(x)
