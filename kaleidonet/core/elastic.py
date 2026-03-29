"""
Elastic (shape-shifting) layer primitives.

These layers support dynamic width adjustment during training and inference.
The effective width is controlled by a continuous mask (Gumbel-sigmoid during
training, hard threshold at inference) enabling the morph controller to reshape
the network on-the-fly per input.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoid(torch.autograd.Function):
    """Gumbel-sigmoid with straight-through estimator for hard masks."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        # Gumbel-sigmoid sampling
        u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        gumbel_noise = -torch.log(-torch.log(u))
        y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through: forward uses hard, backward uses soft gradient
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        ctx.save_for_backward(y_soft)
        ctx.tau = tau
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        (y_soft,) = ctx.saved_tensors
        # Gradient of sigmoid
        grad_input = grad_output * y_soft * (1 - y_soft) / ctx.tau
        return grad_input, None, None


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    return GumbelSigmoid.apply(logits, tau, hard)


def _apply_floor_mask(logits: torch.Tensor, mask: torch.Tensor, min_active: int) -> torch.Tensor:
    """Ensure a minimum number of units remain active."""
    if min_active <= 0:
        return mask
    _, topk_indices = logits.topk(min_active)
    floor_mask = torch.zeros_like(mask)
    floor_mask[topk_indices] = 1.0
    return torch.max(mask, floor_mask)


def _deterministic_hard_mask(logits: torch.Tensor, min_active: int) -> torch.Tensor:
    """Deterministic inference mask based on the learned logit threshold."""
    mask = (logits >= 0).to(dtype=logits.dtype)
    return _apply_floor_mask(logits, mask, min_active)


class ElasticLinear(nn.Module):
    """
    Linear layer with learnable dynamic width.

    The layer has a maximum width (out_features) but can operate at any
    effective width from min_width to out_features. A learnable mask determines
    which output neurons are active. During training, the mask is soft
    (Gumbel-sigmoid); during inference, it is hard (binary).

    Args:
        in_features: Input dimension.
        out_features: Maximum output dimension.
        min_width: Minimum active output neurons (floor).
        bias: Whether to include bias.
        tau_init: Initial Gumbel-sigmoid temperature (annealed during training).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        min_width: int = 1,
        bias: bool = True,
        tau_init: float = 5.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.min_width = min_width

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable mask logits — one per output neuron
        # Init at +0.5: sigmoid(0.5)≈0.62, all neurons start ON.
        # The sparsity regularizer + Lagrangian will push unneeded ones OFF.
        self.mask_logits = nn.Parameter(torch.full((out_features,), 0.5))
        self.tau = tau_init

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_mask(self) -> torch.Tensor:
        """Get the current width mask (soft during training, hard during eval)."""
        if not self.training:
            return _deterministic_hard_mask(self.mask_logits, self.min_width)

        mask = gumbel_sigmoid(self.mask_logits, tau=self.tau, hard=self.tau < 0.5)
        return _apply_floor_mask(self.mask_logits, mask, self.min_width)

    def _get_active_indices(self, target_width: int | None = None) -> torch.Tensor:
        """Return the active output indices for deterministic inference/surgery."""
        if target_width is not None:
            target_width = max(self.min_width, min(target_width, self.out_features))
            indices = self.mask_logits.topk(target_width).indices
            return indices.sort().values

        mask = self.get_mask() > 0
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            indices = self.mask_logits.topk(self.min_width).indices
        return indices.sort().values

    def _sparse_forward(self, x: torch.Tensor, active_indices: torch.Tensor) -> torch.Tensor:
        """Reduced compute path used when the active set is known deterministically."""
        active_weight = self.weight.index_select(0, active_indices)
        active_bias = self.bias.index_select(0, active_indices) if self.bias is not None else None
        active_out = F.linear(x, active_weight, active_bias)

        out_shape = (*x.shape[:-1], self.out_features)
        out = x.new_zeros(out_shape)
        out.index_copy_(-1, active_indices, active_out)
        return out

    def forward(self, x: torch.Tensor, target_width: int | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., in_features).
            target_width: If set by morph controller, override mask to use exactly
                          this many neurons (used during controlled morphing).
        Returns:
            Output tensor (..., out_features) with inactive neurons zeroed.
        """
        deterministic = (not self.training) or (target_width is not None)
        if deterministic:
            active_indices = self._get_active_indices(target_width)
            if active_indices.numel() < self.out_features:
                return self._sparse_forward(x, active_indices)

        mask = self.get_mask()
        out = F.linear(x, self.weight, self.bias)
        out = out * mask
        return out

    @property
    def active_width(self) -> int:
        """Current number of active neurons (hard count)."""
        with torch.no_grad():
            return int((self.mask_logits >= 0).sum().item())

    @property
    def active_fraction(self) -> float:
        """Fraction of active neurons."""
        return self.active_width / self.out_features

    def active_flops(self, batch_size: int = 1, seq_len: int = 1) -> int:
        """Compute active FLOPs (only counting active output neurons)."""
        width = self.active_width
        # FLOPs for linear: 2 * in * out (multiply + add) per element
        return 2 * self.in_features * width * batch_size * seq_len

    def soft_active_flops(self, batch_size: int = 1, seq_len: int = 1) -> torch.Tensor:
        """Differentiable FLOPs estimate using soft mask values."""
        soft_width = torch.sigmoid(self.mask_logits / max(self.tau, 0.01)).sum()
        return 2 * self.in_features * soft_width * batch_size * seq_len

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"min_width={self.min_width}, active={self.active_width}/{self.out_features}, "
            f"tau={self.tau:.3f}"
        )


class ElasticAttention(nn.Module):
    """
    Multi-head attention with dynamic number of active heads.

    The morph controller can activate/deactivate attention heads, reducing
    compute for inputs that don't need full attention capacity.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Maximum number of attention heads.
        min_heads: Minimum active heads.
        dropout: Attention dropout rate.
        tau_init: Initial Gumbel-sigmoid temperature.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        min_heads: int = 1,
        dropout: float = 0.0,
        tau_init: float = 5.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.min_heads = min_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable mask logits — one per head
        self.head_mask_logits = nn.Parameter(torch.full((num_heads,), 0.5))
        self.tau = tau_init
        self.scale = self.head_dim ** -0.5

    def get_head_mask(self) -> torch.Tensor:
        if not self.training:
            return _deterministic_hard_mask(self.head_mask_logits, self.min_heads)

        mask = gumbel_sigmoid(self.head_mask_logits, tau=self.tau, hard=self.tau < 0.5)
        return _apply_floor_mask(self.head_mask_logits, mask, self.min_heads)

    def _get_active_head_indices(self, target_heads: int | None = None) -> torch.Tensor:
        """Return sorted indices of active heads for deterministic inference/surgery."""
        if target_heads is not None:
            target_heads = max(self.min_heads, min(target_heads, self.num_heads))
            return self.head_mask_logits.topk(target_heads).indices.sort().values
        mask = self.get_head_mask() > 0
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            indices = self.head_mask_logits.topk(self.min_heads).indices
        return indices.sort().values

    def _sparse_forward(
        self,
        x: torch.Tensor,
        active_head_indices: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduced-compute path: only project and attend over active heads."""
        B, N, C = x.shape
        h = active_head_indices.numel()
        D = self.head_dim

        # Build row indices for active heads in QKV weight (3*C, C)
        # Q rows [0:C], K rows [C:2C], V rows [2C:3C]; head i => rows [i*D:(i+1)*D]
        offsets = active_head_indices * D  # (h,)
        per_head = torch.arange(D, device=x.device)  # (D,)
        head_rows = (offsets.unsqueeze(1) + per_head.unsqueeze(0)).reshape(-1)  # (h*D,)
        active_rows = torch.cat([head_rows, head_rows + C, head_rows + 2 * C])  # (3*h*D,)

        # Sparse QKV projection
        active_qkv_w = self.qkv.weight.index_select(0, active_rows)
        active_qkv_b = self.qkv.bias.index_select(0, active_rows) if self.qkv.bias is not None else None
        qkv = F.linear(x, active_qkv_w, active_qkv_b)  # (B, N, 3*h*D)

        qkv = qkv.reshape(B, N, 3, h, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, h, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # (B, h, N, D) → (B, N, h*D)
        out_active = (attn @ v).transpose(1, 2).reshape(B, N, h * D)

        # Sparse output projection: only active-head input columns contribute
        active_proj_w = self.proj.weight.index_select(1, head_rows)  # (C, h*D)
        out = F.linear(out_active, active_proj_w, self.proj.bias)  # (B, N, C)
        return out

    def forward(
        self,
        x: torch.Tensor,
        target_heads: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            target_heads: If set, use exactly this many heads.
            attn_mask: Optional attention mask (batch, seq_len, seq_len) or broadcastable.
        Returns:
            Output tensor (batch, seq_len, embed_dim).
        """
        deterministic = (not self.training) or (target_heads is not None)
        if deterministic:
            active = self._get_active_head_indices(target_heads)
            if active.numel() < self.num_heads:
                return self._sparse_forward(x, active, attn_mask)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Head mask
        head_mask = self.get_head_mask()

        # Apply head mask: (num_heads,) -> (1, num_heads, 1, 1)
        head_mask = head_mask.view(1, self.num_heads, 1, 1)
        attn = attn * head_mask

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

    @property
    def active_heads(self) -> int:
        with torch.no_grad():
            return int((self.head_mask_logits >= 0).sum().item())

    def active_flops(self, batch_size: int = 1, seq_len: int = 1) -> int:
        h = self.active_heads
        # QKV projection + attention scores + attention @ V + output projection
        qkv_flops = 2 * self.embed_dim * 3 * self.embed_dim * batch_size * seq_len
        attn_flops = 2 * h * self.head_dim * seq_len * seq_len * batch_size
        proj_flops = 2 * self.embed_dim * self.embed_dim * batch_size * seq_len
        return qkv_flops + attn_flops + proj_flops

    def soft_active_flops(self, batch_size: int = 1, seq_len: int = 1) -> torch.Tensor:
        """Differentiable FLOPs estimate using soft head mask."""
        soft_heads = torch.sigmoid(self.head_mask_logits / max(self.tau, 0.01)).sum()
        qkv_flops = 2 * self.embed_dim * 3 * self.embed_dim * batch_size * seq_len
        attn_flops = 2 * soft_heads * self.head_dim * seq_len * seq_len * batch_size
        proj_flops = 2 * self.embed_dim * self.embed_dim * batch_size * seq_len
        return qkv_flops + attn_flops + proj_flops


class ElasticConv2d(nn.Module):
    """
    Conv2d with dynamic output channel count.

    Similar to ElasticLinear but for convolutional layers — the morph controller
    can adjust the number of active output channels.

    Args:
        in_channels: Input channels.
        out_channels: Maximum output channels.
        kernel_size: Convolution kernel size.
        min_channels: Minimum active output channels.
        stride: Convolution stride.
        padding: Convolution padding.
        tau_init: Initial Gumbel-sigmoid temperature.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        min_channels: int = 1,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        tau_init: float = 5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_channels = min_channels

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.mask_logits = nn.Parameter(torch.full((out_channels,), 0.5))
        self.tau = tau_init

    def get_mask(self) -> torch.Tensor:
        if not self.training:
            return _deterministic_hard_mask(self.mask_logits, self.min_channels)

        mask = gumbel_sigmoid(self.mask_logits, tau=self.tau, hard=self.tau < 0.5)
        return _apply_floor_mask(self.mask_logits, mask, self.min_channels)

    def forward(self, x: torch.Tensor, target_channels: int | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
            target_channels: If set, activate exactly this many output channels.
        """
        out = self.conv(x)

        if target_channels is not None:
            target_channels = max(self.min_channels, min(target_channels, self.out_channels))
            _, idx = self.mask_logits.topk(target_channels)
            mask = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            mask[idx] = 1.0
        else:
            mask = self.get_mask()

        # (out_channels,) -> (1, out_channels, 1, 1)
        out = out * mask.view(1, -1, 1, 1)
        return out

    @property
    def active_channels(self) -> int:
        with torch.no_grad():
            return int((self.mask_logits >= 0).sum().item())
