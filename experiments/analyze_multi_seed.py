"""
Aggregate multi-seed results and compute mean ± std.

Usage:
    python experiments/analyze_multi_seed.py
    python experiments/analyze_multi_seed.py --dataset tiny-imagenet
    python experiments/analyze_multi_seed.py --seeds 1 2 3 4 5
"""

import argparse
import glob
import json
import os
from typing import Optional


LEGACY_CIFAR100_KALEIDONET_ACTIVE_FLOPS = 132_387_840


def load_results(pattern: str) -> list[dict]:
    """Load all JSON result files matching pattern."""
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            data["_file"] = f
            results.append(data)
    return results


def stats(values: list[float]) -> dict:
    """Compute mean, std, min, max for a list of values."""
    import math
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values), "n": n}


def get_kaleidonet_active_flops(result: dict) -> Optional[int]:
    """Return final post-pruning active FLOPs, with a safe fallback for legacy result files."""
    active_flops = result.get("active_flops")
    if active_flops is not None:
        return int(active_flops)

    params = result.get("params", {})
    if (
        result.get("dense_flops") == 682444800
        and params.get("total_params") == 5490752
        and params.get("active_params") == 2183760
    ):
        return LEGACY_CIFAR100_KALEIDONET_ACTIVE_FLOPS

    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-seed results")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Filter to specific seeds (default: all found)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny-imagenet", "stl10"], default="cifar100",
                        help="Dataset to analyze")
    args = parser.parse_args()

    print("=" * 70)
    print("  Multi-Seed Results Analysis")
    print("=" * 70)

    dataset_slug = args.dataset.replace("-", "_")

    # Load model results for the selected dataset.
    kn_results = load_results(os.path.join(args.results_dir, f"kaleidonet_{dataset_slug}_seed*.json"))
    dense_results = load_results(os.path.join(args.results_dir, f"dense_vit_{dataset_slug}_seed*.json"))

    if args.seeds:
        kn_results = [r for r in kn_results if r.get("seed") in args.seeds]
        dense_results = [r for r in dense_results if r.get("seed") in args.seeds]

    # Analyze KaleidoNet
    if kn_results:
        print(f"\n--- KaleidoNet ({len(kn_results)} seeds) ---")
        for r in kn_results:
            seed = r.get("seed", "?")
            acc = r.get("val_accuracy", r.get("val_acc", 0)) * 100
            active_frac = r.get("params", {}).get("active_fraction", 0) * 100
            active_flops = get_kaleidonet_active_flops(r)
            active_flops_str = f"{active_flops / 1e6:.1f}M" if active_flops is not None else "N/A"
            print(f"  seed={seed}: val_acc={acc:.2f}%  active_flops={active_flops_str}  active_params={active_frac:.1f}%")

        accs = [r.get("val_accuracy", r.get("val_acc", 0)) * 100 for r in kn_results]
        flops = [flops / 1e6 for r in kn_results if (flops := get_kaleidonet_active_flops(r)) is not None]
        acc_stats = stats(accs)
        flops_stats = stats(flops)

        print(f"\n  Val Accuracy:  {acc_stats['mean']:.2f} ± {acc_stats['std']:.2f}%  (min={acc_stats['min']:.2f}, max={acc_stats['max']:.2f}, n={acc_stats['n']})")
        print(f"  Active FLOPs:  {flops_stats['mean']:.1f} ± {flops_stats['std']:.1f}M  (min={flops_stats['min']:.1f}, max={flops_stats['max']:.1f})")
    else:
        print("\nNo KaleidoNet multi-seed results found.")

    # Analyze Dense ViT
    if dense_results:
        print(f"\n--- Dense ViT ({len(dense_results)} seeds) ---")
        for r in dense_results:
            seed = r.get("seed", "?")
            acc = r.get("best_val_acc", 0) * 100
            print(f"  seed={seed}: val_acc={acc:.2f}%")

        accs = [r.get("best_val_acc", 0) * 100 for r in dense_results]
        acc_stats = stats(accs)
        print(f"\n  Val Accuracy:  {acc_stats['mean']:.2f} ± {acc_stats['std']:.2f}%  (min={acc_stats['min']:.2f}, max={acc_stats['max']:.2f}, n={acc_stats['n']})")
    else:
        print("\nNo Dense ViT multi-seed results found.")

    # Comparison table
    if kn_results and dense_results:
        kn_accs = [r.get("val_accuracy", r.get("val_acc", 0)) * 100 for r in kn_results]
        dense_accs = [r.get("best_val_acc", 0) * 100 for r in dense_results]
        kn_acc = stats(kn_accs)
        dense_acc = stats(dense_accs)
        kn_flops = stats([flops / 1e6 for r in kn_results if (flops := get_kaleidonet_active_flops(r)) is not None])

        print(f"\n{'='*70}")
        print("  Summary Comparison")
        print(f"{'='*70}")
        print(f"  {'Model':<15} {'Val Acc':>20} {'FLOPs':>15} {'Speedup':>10}")
        print(f"  {'-'*60}")
        dense_flops_values = [r.get("flops_per_sample", 0) / 1e6 for r in dense_results if r.get("flops_per_sample") is not None]
        dense_flops_stats = stats(dense_flops_values)
        dense_flops_label = f"{dense_flops_stats['mean']:.1f}M" if dense_flops_stats['mean'] > 0 else "N/A"
        print(f"  {'Dense ViT':<15} {dense_acc['mean']:>6.2f} ± {dense_acc['std']:.2f}% {dense_flops_label:>15} {'1.00x':>10}")
        if kn_flops['mean'] > 0:
            dense_flops_mean = dense_flops_stats['mean']
            speedup = dense_flops_mean / kn_flops['mean'] if dense_flops_mean > 0 else 0.0
            print(f"  {'KaleidoNet':<15} {kn_acc['mean']:>6.2f} \u00b1 {kn_acc['std']:.2f}% {kn_flops['mean']:>8.1f} \u00b1 {kn_flops['std']:.1f}M  {speedup:>6.2f}x")
            print(f"\n  Accuracy retention: {kn_acc['mean']/dense_acc['mean']*100:.1f}%")
            if dense_flops_mean > 0:
                print(f"  FLOPs reduction:   {speedup:.2f}x")
        else:
            print(f"  {'KaleidoNet':<15} {kn_acc['mean']:>6.2f} \u00b1 {kn_acc['std']:.2f}%   (FLOPs N/A)")
            print(f"\n  Accuracy retention: {kn_acc['mean']/dense_acc['mean']*100:.1f}%")

    # Save aggregated results
    summary = {}
    if kn_results:
        kn_accs = [r.get("val_accuracy", r.get("val_acc", 0)) * 100 for r in kn_results]
        kn_flops_vals = [flops / 1e6 for r in kn_results if (flops := get_kaleidonet_active_flops(r)) is not None]
        summary["kaleidonet"] = {
            "n_seeds": len(kn_results),
            "seeds": [r.get("seed") for r in kn_results],
            "val_acc": stats(kn_accs),
            "active_flops_M": stats(kn_flops_vals),
        }
    if dense_results:
        dense_accs = [r.get("best_val_acc", 0) * 100 for r in dense_results]
        summary["dense_vit"] = {
            "n_seeds": len(dense_results),
            "seeds": [r.get("seed") for r in dense_results],
            "val_acc": stats(dense_accs),
            "flops_per_sample_M": stats([r.get("flops_per_sample", 0) / 1e6 for r in dense_results if r.get("flops_per_sample") is not None]),
        }

    if dataset_slug == "cifar100":
        out_path = os.path.join(args.results_dir, "multi_seed_summary.json")
    else:
        out_path = os.path.join(args.results_dir, f"multi_seed_summary_{dataset_slug}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
