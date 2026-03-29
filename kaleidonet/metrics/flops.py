"""
Active FLOPs counter and speedup calculator.

Tracks *active* compute (only the neurons/experts/heads that actually fire),
not total model parameter count. This is the correct measure for sparse/dynamic
architectures where most of the model is dormant per input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FLOPsCounter:
    """
    Counts active FLOPs for KaleidoNet models.

    Traverses the model and sums active FLOPs from all layers that report them.
    For layers without active_flops(), falls back to parameter-based estimation.
    """

    def count(self, model: nn.Module, batch_size: int = 1, seq_len: int = 1) -> dict:
        """
        Count active FLOPs for the model.

        Args:
            model: KaleidoNet model.
            batch_size: Current batch size.
            seq_len: Current sequence length.
        Returns:
            Dict with total and per-layer FLOPs breakdown.
        """
        total_flops = 0
        breakdown = {}

        for name, module in model.named_modules():
            # Skip children of already-counted parents (avoids double counting)
            if any(name.startswith(parent + ".") for parent in breakdown):
                continue

            if hasattr(module, "active_flops"):
                flops = module.active_flops(batch_size, seq_len)
                breakdown[name] = flops
                total_flops += flops
            elif isinstance(module, nn.Linear):
                flops = 2 * module.in_features * module.out_features * batch_size * seq_len
                breakdown[name] = flops
                total_flops += flops

        return {
            "total_active_flops": total_flops,
            "breakdown": breakdown,
        }

    @staticmethod
    def count_dense(model: nn.Module, batch_size: int = 1, seq_len: int = 1) -> int:
        """Count total FLOPs assuming all parameters are active (dense baseline)."""
        from kaleidonet.core.elastic import ElasticLinear, ElasticAttention
        from kaleidonet.routing.moe import MoELayer

        total = 0
        counted_prefixes: list[str] = []

        for name, module in model.named_modules():
            # Skip children of already-counted compound modules
            if any(name.startswith(p + ".") for p in counted_prefixes):
                continue

            if isinstance(module, MoELayer):
                # Dense: ALL experts fire on every token (no routing sparsity)
                for expert in module.experts:
                    if hasattr(expert, 'fc1'):
                        fc1 = expert.fc1
                        fc2 = expert.fc2
                        in1 = fc1.in_features
                        out1 = fc1.out_features
                        in2 = fc2.in_features
                        out2 = fc2.out_features
                        total += (2 * in1 * out1 + 2 * in2 * out2) * batch_size * seq_len
                # Router gate
                gate = module.router.gate
                total += 2 * gate.in_features * gate.out_features * batch_size * seq_len
                counted_prefixes.append(name)
            elif isinstance(module, ElasticAttention):
                # Dense: all heads active; Q/K/V/O projections at full width
                d = module.embed_dim
                total += 4 * (2 * d * d) * batch_size * seq_len  # Q, K, V, O
                counted_prefixes.append(name)
            elif isinstance(module, ElasticLinear):
                # Dense: full out_features (no mask)
                total += 2 * module.in_features * module.out_features * batch_size * seq_len
                counted_prefixes.append(name)
            elif isinstance(module, nn.Linear):
                total += 2 * module.in_features * module.out_features * batch_size * seq_len
                counted_prefixes.append(name)
            elif isinstance(module, nn.Conv2d):
                out_h = out_w = 1
                total += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w * batch_size
                counted_prefixes.append(name)
        return total


class SpeedupCalculator:
    """
    Computes speedup factor by comparing baseline and method FLOPs-to-target-accuracy.

    Usage:
        calc = SpeedupCalculator()
        calc.record_baseline(flops=1e12, accuracy=0.92)
        calc.record_method(flops=5e10, accuracy=0.92)
        print(calc.compute_speedup())  # ~20x
    """

    def __init__(self):
        self.baseline_records: list[dict] = []
        self.method_records: list[dict] = []

    def record_baseline(self, flops: float, accuracy: float, wall_time: float = 0.0):
        self.baseline_records.append({"flops": flops, "accuracy": accuracy, "wall_time": wall_time})

    def record_method(self, flops: float, accuracy: float, wall_time: float = 0.0):
        self.method_records.append({"flops": flops, "accuracy": accuracy, "wall_time": wall_time})

    def compute_speedup(self, target_accuracy: float | None = None) -> dict:
        """
        Compute speedup at a target accuracy level.

        If target_accuracy is None, uses the best accuracy achieved by the method.
        Interpolates FLOPs for the baseline at that accuracy level.

        Returns:
            Dict with flops_speedup, wall_time_speedup, and details.
        """
        if not self.baseline_records or not self.method_records:
            return {"flops_speedup": 0.0, "wall_time_speedup": 0.0}

        method_best = max(self.method_records, key=lambda r: r["accuracy"])
        if target_accuracy is None:
            target_accuracy = method_best["accuracy"]

        # Find baseline FLOPs to reach target accuracy (use closest record)
        baseline_candidates = [r for r in self.baseline_records if r["accuracy"] >= target_accuracy]
        if baseline_candidates:
            baseline_at_target = min(baseline_candidates, key=lambda r: r["flops"])
        else:
            # Baseline never reached target accuracy — use its best
            baseline_at_target = max(self.baseline_records, key=lambda r: r["accuracy"])

        # Find method FLOPs to reach target accuracy
        method_candidates = [r for r in self.method_records if r["accuracy"] >= target_accuracy]
        if method_candidates:
            method_at_target = min(method_candidates, key=lambda r: r["flops"])
        else:
            method_at_target = method_best

        flops_speedup = baseline_at_target["flops"] / max(method_at_target["flops"], 1)
        wall_speedup = (
            baseline_at_target["wall_time"] / max(method_at_target["wall_time"], 1e-6)
            if baseline_at_target["wall_time"] > 0 and method_at_target["wall_time"] > 0
            else 0.0
        )

        return {
            "flops_speedup": flops_speedup,
            "wall_time_speedup": wall_speedup,
            "target_accuracy": target_accuracy,
            "baseline_flops": baseline_at_target["flops"],
            "method_flops": method_at_target["flops"],
            "baseline_accuracy": baseline_at_target["accuracy"],
            "method_accuracy": method_at_target["accuracy"],
        }

    def summary(self) -> str:
        s = self.compute_speedup()
        return (
            f"Speedup: {s['flops_speedup']:.1f}x FLOPs, {s['wall_time_speedup']:.1f}x wall-clock\n"
            f"  Target accuracy: {s['target_accuracy']:.4f}\n"
            f"  Baseline: {s['baseline_flops']:.2e} FLOPs @ {s['baseline_accuracy']:.4f}\n"
            f"  Method:   {s['method_flops']:.2e} FLOPs @ {s['method_accuracy']:.4f}"
        )
