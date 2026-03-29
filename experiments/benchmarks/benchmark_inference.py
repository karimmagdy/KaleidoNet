"""Benchmark inference latency for DenseViT vs pruned KaleidoNet.

Run:
    python run.py experiments/benchmarks/benchmark_inference.py
    python run.py experiments/benchmarks/benchmark_inference.py --steps 100 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from experiments.baselines.dense_vit_baseline import DenseViT
from kaleidonet.model import KaleidoNet
from kaleidonet.export import export_pruned_state


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_kaleidonet(checkpoint_path: str, device: str) -> KaleidoNet:
    model = KaleidoNet(
        embed_dim=192,
        num_blocks=4,
        num_heads=6,
        num_experts=4,
        top_k=1,
        num_classes=100,
        vocab_size=0,
        image_size=32,
        patch_size=4,
        elastic=True,
        drop_path_rate=0.1,
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model.eval()
    return model


def load_dense_vit(device: str) -> DenseViT:
    model = DenseViT(
        embed_dim=192,
        num_heads=6,
        num_layers=4,
        num_classes=100,
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def benchmark_model(model: torch.nn.Module, batch: dict, device: str, warmup: int, steps: int) -> dict:
    for _ in range(warmup):
        _ = model(batch)
    synchronize(device)

    latencies_ms = []
    for _ in range(steps):
        start = time.perf_counter()
        _ = model(batch)
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    sorted_ms = sorted(latencies_ms)
    median_ms = sorted_ms[len(sorted_ms) // 2]
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_samples_per_s": batch["images"].shape[0] * 1000.0 / mean_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DenseViT vs pruned KaleidoNet inference")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    batch = {
        "images": torch.randn(args.batch_size, 3, 32, 32, device=device),
        "task": "classify",
    }

    dense_vit = load_dense_vit(device)
    kaleidonet = load_kaleidonet(args.checkpoint, device)

    # Run export surgery before benchmarking
    surgery = export_pruned_state(kaleidonet)

    dense_stats = benchmark_model(dense_vit, batch, device, args.warmup, args.steps)
    kaleidonet_stats = benchmark_model(kaleidonet, batch, device, args.warmup, args.steps)
    kaleidonet_param_info = kaleidonet.count_active_params()
    kaleidonet_flops = kaleidonet(batch)["active_flops"]

    results = {
        "device": device,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "steps": args.steps,
        "dense_vit": {
            "params": sum(p.numel() for p in dense_vit.parameters()),
            **dense_stats,
        },
        "kaleidonet": {
            **kaleidonet_param_info,
            "active_flops": int(kaleidonet_flops),
            **kaleidonet_stats,
        },
        "surgery": {
            "original_params": surgery["original_params"],
            "pruned_params": surgery["pruned_params"],
            "compression_ratio": surgery["original_params"] / max(surgery["pruned_params"], 1),
            "per_layer": surgery["surgery_summary"],
        },
    }
    results["latency_speedup_vs_dense"] = dense_stats["mean_ms"] / max(kaleidonet_stats["mean_ms"], 1e-9)
    results["throughput_speedup_vs_dense"] = kaleidonet_stats["throughput_samples_per_s"] / max(dense_stats["throughput_samples_per_s"], 1e-9)

    print("=" * 60)
    print("Inference Benchmark")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    os.makedirs("results", exist_ok=True)
    with open("results/inference_benchmark.json", "w") as handle:
        json.dump(results, handle, indent=2)
    print("Saved results to results/inference_benchmark.json")


if __name__ == "__main__":
    main()