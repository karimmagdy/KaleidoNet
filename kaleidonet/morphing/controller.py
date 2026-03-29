"""
Morph Controller: decides the shape (width/heads) of each elastic layer per batch.

Observes per-layer statistics (Hessian trace, gradient norm, activation stats)
and outputs target widths. Tiny MLP (~0 overhead) that runs once per batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def hutchinson_trace_estimate(
    loss: torch.Tensor,
    params: list[torch.Tensor],
    num_samples: int = 5,
) -> list[torch.Tensor]:
    """
    Estimate the trace of the Hessian for each parameter using Hutchinson's method.

    Uses random Rademacher vectors and Hessian-vector products (one backward per sample).
    Cost: num_samples Hessian-vector products. Typically 5 samples is enough.

    Args:
        loss: Scalar loss tensor (must have grad_fn).
        params: List of parameter tensors to estimate Hessian trace for.
        num_samples: Number of random vectors for the estimate.
    Returns:
        List of scalar trace estimates, one per parameter.
    """
    traces = [torch.zeros(1, device=p.device) for p in params]

    for _ in range(num_samples):
        # Rademacher random vectors (+1 or -1 with equal probability)
        vs = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

        # First-order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

        # Hessian-vector products: Hv = d/dp (grad . v)
        grad_v = sum(
            (g * v).sum()
            for g, v in zip(grads, vs)
            if g is not None
        )
        hvps = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)

        for i, (hvp, v) in enumerate(zip(hvps, vs)):
            if hvp is not None:
                # Trace estimate: E[v^T H v] = Tr(H)
                traces[i] = traces[i] + (hvp * v).sum().detach()

    traces = [t / num_samples for t in traces]
    return traces


class LayerStatsCollector(nn.Module):
    """
    Collects per-layer statistics for the morph controller.

    Attaches hooks to elastic layers and records:
    - Activation mean and variance
    - Gradient norm (populated after backward)
    - Current active width / heads
    """

    def __init__(self):
        super().__init__()
        self._stats: dict[str, dict] = {}
        self._hooks: list = []

    def register_layer(self, name: str, module: nn.Module):
        """Register an elastic layer for statistics collection."""
        self._stats[name] = {
            "act_mean": 0.0,
            "act_var": 0.0,
            "grad_norm": 0.0,
            "active_fraction": 1.0,
        }

        def fwd_hook(mod, inp, out, layer_name=name):
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
            with torch.no_grad():
                self._stats[layer_name]["act_mean"] = out_tensor.mean().item()
                self._stats[layer_name]["act_var"] = out_tensor.var().item()
                if hasattr(mod, "active_fraction"):
                    self._stats[layer_name]["active_fraction"] = mod.active_fraction
                elif hasattr(mod, "active_heads"):
                    self._stats[layer_name]["active_fraction"] = mod.active_heads / mod.num_heads

        def bwd_hook(mod, grad_input, grad_output, layer_name=name):
            if grad_output[0] is not None:
                with torch.no_grad():
                    self._stats[layer_name]["grad_norm"] = grad_output[0].norm().item()

        self._hooks.append(module.register_forward_hook(fwd_hook))
        self._hooks.append(module.register_full_backward_hook(bwd_hook))

    def get_stats_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Flatten all layer stats into a single tensor for the controller MLP.
        Returns: (num_layers * 4,) tensor.
        """
        feats = []
        for stats in self._stats.values():
            feats.extend([
                stats["act_mean"],
                stats["act_var"],
                stats["grad_norm"],
                stats["active_fraction"],
            ])
        return torch.tensor(feats, device=device, dtype=torch.float32)

    @property
    def num_layers(self) -> int:
        return len(self._stats)

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class MorphController(nn.Module):
    """
    Tiny MLP that decides target widths/heads for each elastic layer.

    Input: concatenated per-layer statistics (4 features per layer).
    Output: target width fraction [0, 1] for each layer (scaled to actual width by caller).

    Cost: ~0 relative to the main model (2-layer MLP, 32 hidden units).

    Args:
        num_layers: Number of elastic layers to control.
        hidden_dim: Hidden dimension of the controller MLP.
        features_per_layer: Number of statistics per layer (default: 4).
    """

    def __init__(self, num_layers: int, hidden_dim: int = 32, features_per_layer: int = 4):
        super().__init__()
        self.num_layers = num_layers
        input_dim = num_layers * features_per_layer

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Sigmoid(),  # Output ∈ [0, 1] — fraction of max width to use
        )

    def forward(self, layer_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_stats: (num_layers * features_per_layer,) tensor from LayerStatsCollector.
        Returns:
            width_fractions: (num_layers,) tensor in [0, 1].
        """
        return self.mlp(layer_stats)

    def get_target_widths(
        self,
        layer_stats: torch.Tensor,
        max_widths: list[int],
        min_widths: list[int],
    ) -> list[int]:
        """
        Compute integer target widths for each layer.

        Args:
            layer_stats: Stats tensor from collector.
            max_widths: Maximum width for each layer.
            min_widths: Minimum width for each layer.
        Returns:
            List of integer target widths.
        """
        fractions = self.forward(layer_stats)
        targets = []
        for frac, max_w, min_w in zip(fractions, max_widths, min_widths):
            w = int(min_w + frac.item() * (max_w - min_w))
            w = max(min_w, min(w, max_w))
            targets.append(w)
        return targets
