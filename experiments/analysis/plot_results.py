"""Visualization pipeline for KaleidoNet paper figures.

Generates publication-quality figures from experiment results.

Figures:
  1. FLOPs vs Accuracy Pareto curves (all datasets, all methods)
  2. Pruning evolution (active fraction vs step: cubic vs linear)
  3. Sparsity target sweep (target vs accuracy, with FLOPs)
  4. Per-layer pruning heatmap
  5. Mask LR sensitivity

Run:
    python experiments/analysis/plot_results.py
    python experiments/analysis/plot_results.py --fig 1     # Only figure 1
    python experiments/analysis/plot_results.py --outdir figures/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

DATASETS = ["cifar10", "cifar100", "tiny_imagenet", "stl10"]
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
    "stl10": "STL-10",
}
METHOD_COLORS = {
    "DenseViT": "#2196F3",
    "KaleidoNet": "#4CAF50",
    "MagnitudePruning": "#FF9800",
    "RandomPruning": "#9C27B0",
    "LinearSchedule": "#F44336",
}
METHOD_MARKERS = {
    "DenseViT": "o",
    "KaleidoNet": "s",
    "MagnitudePruning": "^",
    "RandomPruning": "D",
    "LinearSchedule": "v",
}


def load_all_results() -> list[dict]:
    """Load all JSON result files (only seed-numbered files for consistency)."""
    import re
    results = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if "summary" in f or "sweep_combined" in f or "ablation" in f:
            continue
        # Only include seed-numbered files (e.g., *_seed1.json)
        if not re.search(r"_seed\d+\.json$", f):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["_file"] = f
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def get_accuracy(r: dict) -> float | None:
    acc = r.get("val_accuracy") or r.get("best_val_acc")
    if acc is not None:
        return float(acc) if acc > 1.0 else float(acc) * 100
    return None


def get_flops_m(r: dict) -> float | None:
    flops = r.get("active_flops") or r.get("flops_per_sample")
    if flops is not None:
        return float(flops) / 1e6
    return None


def get_dataset(r: dict) -> str | None:
    ds = r.get("dataset")
    if ds:
        return ds
    fname = os.path.basename(r.get("_file", ""))
    # Check longer slugs first to avoid "cifar10" matching "cifar100"
    for slug in sorted(DATASETS, key=len, reverse=True):
        if slug in fname:
            return slug
    return None


def get_model(r: dict) -> str:
    return r.get("model", "Unknown")


# ---- Figure 1: FLOPs vs Accuracy Pareto ----

def fig1_pareto(results: list[dict], outdir: str):
    """FLOPs vs accuracy for all datasets and methods."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4), squeeze=False)
    axes = axes[0]

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        ds_results = [r for r in results if get_dataset(r) == ds]

        for r in ds_results:
            model = get_model(r)
            acc = get_accuracy(r)
            flops = get_flops_m(r)
            if acc is None or flops is None:
                continue

            color = METHOD_COLORS.get(model, "#999999")
            marker = METHOD_MARKERS.get(model, "x")
            ax.scatter(flops, acc, color=color, marker=marker, s=60, zorder=5,
                       edgecolors="white", linewidths=0.5)

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        ax.set_xlabel("Active FLOPs (M)")
        if idx == 0:
            ax.set_ylabel("Val Accuracy (%)")
        ax.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker=METHOD_MARKERS[m], color="w",
                    markerfacecolor=METHOD_COLORS[m], markersize=8, label=m)
        for m in METHOD_COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHOD_COLORS),
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("FLOPs vs Accuracy (All Methods)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_pareto.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # Also save PNG
    fig2, axes2 = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4), squeeze=False)
    axes2 = axes2[0]
    for idx, ds in enumerate(DATASETS):
        ax = axes2[idx]
        ds_results = [r for r in results if get_dataset(r) == ds]
        for r in ds_results:
            model = get_model(r)
            acc = get_accuracy(r)
            flops = get_flops_m(r)
            if acc is None or flops is None:
                continue
            color = METHOD_COLORS.get(model, "#999999")
            marker = METHOD_MARKERS.get(model, "x")
            ax.scatter(flops, acc, color=color, marker=marker, s=60, zorder=5,
                       edgecolors="white", linewidths=0.5)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11)
        ax.set_xlabel("Active FLOPs (M)")
        if idx == 0:
            ax.set_ylabel("Val Accuracy (%)")
        ax.grid(True, alpha=0.3)
    fig2.legend(handles=handles, loc="lower center", ncol=len(METHOD_COLORS),
                bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig2.suptitle("FLOPs vs Accuracy (All Methods)", fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "fig1_pareto.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)


# ---- Figure 2: Pruning Evolution (cubic vs linear sparsity curves) ----

def fig2_pruning_evolution(outdir: str):
    """Visualize cubic vs linear sparsity schedules theoretically."""
    steps = np.linspace(0, 5000, 500)
    t0, t1, sf = 500, 4000, 0.7

    cubic = np.zeros_like(steps)
    linear = np.zeros_like(steps)
    for i, s in enumerate(steps):
        if s >= t0:
            t = min((s - t0) / (t1 - t0), 1.0)
            cubic[i] = sf * (1 - (1 - t) ** 3)
            linear[i] = sf * t

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, cubic * 100, color="#4CAF50", linewidth=2, label="Cubic (KaleidoNet)")
    ax.plot(steps, linear * 100, color="#F44336", linewidth=2, linestyle="--", label="Linear")
    ax.axhline(y=sf * 100, color="gray", linestyle=":", alpha=0.5, label=f"Target ({sf:.0%})")
    ax.axvline(x=t0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=t1, color="gray", linestyle=":", alpha=0.3)
    ax.annotate("Start", (t0, -2), fontsize=8, ha="center", color="gray")
    ax.annotate("End", (t1, -2), fontsize=8, ha="center", color="gray")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Sparsity Schedule: Cubic vs Linear")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 80)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_pruning_evolution.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, "fig2_pruning_evolution.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_pruning_evolution.pdf/png")


# ---- Figure 3: Sparsity Target Sweep ----

def fig3_sparsity_sweep(outdir: str):
    """Sparsity target vs accuracy (with FLOPs on second y-axis)."""
    pattern = os.path.join(RESULTS_DIR, "sparsity_sweep_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  [SKIP] No sparsity sweep results found")
        return

    data = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
            data.append(r)

    targets = [r["target_sparsity"] for r in data]
    accs = [get_accuracy(r) or 0 for r in data]
    flops = [(r.get("active_flops") or 0) / 1e6 for r in data]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "#4CAF50"
    ax1.plot(targets, accs, "o-", color=color1, linewidth=2, markersize=8, label="Val Accuracy")
    ax1.set_xlabel("Target Sparsity")
    ax1.set_ylabel("Val Accuracy (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#2196F3"
    ax2.plot(targets, flops, "s--", color=color2, linewidth=2, markersize=8, label="Active FLOPs")
    ax2.set_ylabel("Active FLOPs (M)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    ax1.set_title("Sparsity Target Sensitivity")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_sparsity_sweep.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, "fig3_sparsity_sweep.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_sparsity_sweep.pdf/png")


# ---- Figure 4: Per-Layer Pruning Heatmap ----

def fig4_layer_heatmap(outdir: str):
    """Load a trained KaleidoNet checkpoint and visualize per-layer active fractions."""
    ckpt_path = os.path.join(ROOT, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] No checkpoint found for heatmap")
        return

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    # Extract mask logits from state dict
    layer_names = []
    active_fracs = []
    for key in sorted(state.keys()):
        if "mask_logits" in key:
            logits = state[key]
            frac = (logits >= 0).float().mean().item()
            # Simplify name for display
            short = key.replace("backbone.", "").replace(".mask_logits", "")
            layer_names.append(short)
            active_fracs.append(frac)

    if not layer_names:
        print("  [SKIP] No mask logits found in checkpoint")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(layer_names) * 0.3)))
    colors = ["#F44336" if f < 0.3 else "#FF9800" if f < 0.5 else "#4CAF50" for f in active_fracs]
    y_pos = np.arange(len(layer_names))
    ax.barh(y_pos, [f * 100 for f in active_fracs], color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("Active Fraction (%)")
    ax.set_title("Per-Layer Active Neuron Fraction (After Pruning)")
    ax.axvline(x=30, color="gray", linestyle=":", alpha=0.5, label="30% target")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_layer_heatmap.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, "fig4_layer_heatmap.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved fig4_layer_heatmap.pdf/png")


# ---- Figure 5: Mask LR Sensitivity ----

def fig5_mask_lr(outdir: str):
    """Mask learning rate scale vs accuracy."""
    pattern = os.path.join(RESULTS_DIR, "mask_lr_sweep_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  [SKIP] No mask LR sweep results found")
        return

    data = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
            data.append(r)

    scales = [r["mask_lr_scale"] for r in data]
    accs = [get_accuracy(r) or 0 for r in data]
    flops = [(r.get("active_flops") or 0) / 1e6 for r in data]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "#4CAF50"
    ax1.plot(scales, accs, "o-", color=color1, linewidth=2, markersize=8, label="Val Accuracy")
    ax1.set_xlabel("Mask LR Scale (× base LR)")
    ax1.set_ylabel("Val Accuracy (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#2196F3"
    ax2.plot(scales, flops, "s--", color=color2, linewidth=2, markersize=8, label="Active FLOPs")
    ax2.set_ylabel("Active FLOPs (M)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax1.set_title("Mask Learning Rate Sensitivity")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_mask_lr.pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, "fig5_mask_lr.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved fig5_mask_lr.pdf/png")


# ---- Cross-Dataset Summary Table ----

def summary_table(results: list[dict], outdir: str):
    """Generate a summary comparison table as a text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("Cross-Dataset Summary")
    lines.append("=" * 80)
    header = f"{'Dataset':<15} {'Model':<20} {'Seeds':>5} {'Val Acc (%)':>15} {'FLOPs (M)':>12} {'Speedup':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for ds in DATASETS:
        ds_results = [r for r in results if get_dataset(r) == ds]
        if not ds_results:
            continue

        # Group by model
        by_model = {}
        for r in ds_results:
            m = get_model(r)
            by_model.setdefault(m, []).append(r)

        for model_name in ["DenseViT", "KaleidoNet", "MagnitudePruning", "RandomPruning", "LinearSchedule"]:
            if model_name not in by_model:
                continue
            runs = by_model[model_name]
            accs = [get_accuracy(r) for r in runs if get_accuracy(r) is not None]
            flops_vals = [get_flops_m(r) for r in runs if get_flops_m(r) is not None]

            if accs:
                acc_mean = np.mean(accs)
                acc_std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                acc_str = f"{acc_mean:.2f} ± {acc_std:.2f}"
            else:
                acc_str = "N/A"

            if flops_vals:
                flops_str = f"{np.mean(flops_vals):.1f}"
            else:
                flops_str = "N/A"

            # Compute speedup relative to DenseViT
            speedup_str = ""
            if model_name != "DenseViT" and "DenseViT" in by_model:
                dense_flops = [get_flops_m(r) for r in by_model["DenseViT"] if get_flops_m(r) is not None]
                if dense_flops and flops_vals:
                    speedup = np.mean(dense_flops) / np.mean(flops_vals)
                    speedup_str = f"{speedup:.2f}x"

            ds_label = DATASET_LABELS.get(ds, ds)
            lines.append(f"{ds_label:<15} {model_name:<20} {len(runs):>5} {acc_str:>15} {flops_str:>12} {speedup_str:>8}")

        lines.append("")

    table_text = "\n".join(lines)
    print(table_text)

    path = os.path.join(outdir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(table_text)
    print(f"\nSaved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--fig", type=int, nargs="*", default=None, help="Which figures to generate (1-5, default: all)")
    parser.add_argument("--outdir", default=FIGURES_DIR, help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = load_all_results()
    print(f"Loaded {len(results)} result files\n")

    figs = set(args.fig) if args.fig else {1, 2, 3, 4, 5}

    if 1 in figs:
        print("Figure 1: FLOPs vs Accuracy Pareto...")
        fig1_pareto(results, args.outdir)

    if 2 in figs:
        print("Figure 2: Pruning Evolution (Cubic vs Linear)...")
        fig2_pruning_evolution(args.outdir)

    if 3 in figs:
        print("Figure 3: Sparsity Target Sweep...")
        fig3_sparsity_sweep(args.outdir)

    if 4 in figs:
        print("Figure 4: Per-Layer Pruning Heatmap...")
        fig4_layer_heatmap(args.outdir)

    if 5 in figs:
        print("Figure 5: Mask LR Sensitivity...")
        fig5_mask_lr(args.outdir)

    print("\nSummary Table:")
    summary_table(results, args.outdir)

    print("\nDone! All figures saved to", args.outdir)


if __name__ == "__main__":
    main()
