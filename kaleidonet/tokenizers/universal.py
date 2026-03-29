"""
Universal tokenizer: routes different modalities through appropriate tokenizers
and concatenates them into a unified token sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kaleidonet.tokenizers.vision import PatchTokenizer
from kaleidonet.tokenizers.text import TextTokenizer


class UniversalTokenizer(nn.Module):
    """
    Accepts multi-modal inputs and produces a unified token sequence.

    Each modality has its own tokenizer that projects into the same embedding
    space. The outputs are concatenated along the sequence dimension. A [SEP]
    token separates modalities.

    Args:
        embed_dim: Shared embedding dimension.
        image_size: Image input size (for vision tokenizer).
        patch_size: Patch size (for vision tokenizer).
        vocab_size: Vocabulary size (for text tokenizer).
        max_seq_len: Maximum text sequence length.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_size: int = 32,
        patch_size: int = 4,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.vision_tokenizer = PatchTokenizer(image_size, patch_size, 3, embed_dim)
        self.text_tokenizer = TextTokenizer(vocab_size, max_seq_len, embed_dim)

        # Separator token between modalities
        self.sep_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        images: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Tokenize available modalities and concatenate.

        Args:
            images: (batch, 3, H, W) or None.
            token_ids: (batch, seq_len) or None.
        Returns:
            tokens: (batch, total_seq_len, embed_dim)
            info: Dict with per-modality token counts and positions.
        """
        parts = []
        info = {"modality_ranges": {}}
        pos = 0

        if images is not None:
            vision_tokens = self.vision_tokenizer(images)
            B = vision_tokens.shape[0]
            n_vis = vision_tokens.shape[1]
            parts.append(vision_tokens)
            info["modality_ranges"]["vision"] = (pos, pos + n_vis)
            pos += n_vis

            # Add separator
            parts.append(self.sep_token.expand(B, -1, -1))
            pos += 1

        if token_ids is not None:
            text_tokens = self.text_tokenizer(token_ids)
            B = text_tokens.shape[0]
            n_txt = text_tokens.shape[1]
            parts.append(text_tokens)
            info["modality_ranges"]["text"] = (pos, pos + n_txt)
            pos += n_txt

        if not parts:
            raise ValueError("At least one modality (images or token_ids) must be provided.")

        tokens = torch.cat(parts, dim=1)
        info["total_tokens"] = tokens.shape[1]
        return tokens, info
