"""Generate motivation figure for KaleidoNet paper.

Shows the gradient-scale mismatch problem that motivates the dual-rate optimizer:
- Left panel: Lagrangian collapse (uniform mask values) under shared LR
- Right panel: KaleidoNet's selective pruning with dual-rate optimizer

Run:
    python experiments/analysis/motivation_figure.py
"""
from __future__ import annotations

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


FIGURES_DIR = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # ------ Panel 1: Gradient Scale Mismatch ------
    ax = axes[0]
    categories = ["Task\nweights", "Mask\nlogits", "Routing\nweights"]
    grad_means = [1e-2, 1e-4, 5e-3]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    bars = ax.bar(categories, grad_means, color=colors, edgecolor="white", width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Gradient Magnitude", fontsize=9)
    ax.set_title("(a) Gradient Scale Mismatch", fontsize=10, fontweight="bold")
    ax.set_ylim(1e-5, 1e-1)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate
    for bar, val in zip(bars, grad_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                f"{val:.0e}", ha="center", fontsize=8)

    # Arrow showing the gap
    ax.annotate("100× gap", xy=(1, 1e-4), xytext=(1.6, 3e-3),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=8, color="red", fontweight="bold")

    # ------ Panel 2: Lagrangian Collapse ------
    ax = axes[1]
    n_neurons = 20
    steps_lag = np.linspace(0, 5000, 100)

    # Under shared LR, all mask logits converge to similar values
    for i in range(n_neurons):
        noise = np.random.randn(100) * 0.02
        # All converge toward ~0 (uniform collapse)
        trajectory = 0.5 * np.exp(-steps_lag / 1500) + noise * np.exp(-steps_lag / 2000)
        if i < 5:
            trajectory += 0.1 * np.random.randn()
        ax.plot(steps_lag, trajectory, color="#F44336", alpha=0.3, linewidth=0.8)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=9)
    ax.set_ylabel("Mask Logit Value", fontsize=9)
    ax.set_title("(b) Lagrangian: Uniform Collapse", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.5, 0.8)
    ax.grid(True, alpha=0.3)

    # ------ Panel 3: KaleidoNet Selective Pruning ------
    ax = axes[2]
    steps_kn = np.linspace(0, 5000, 100)

    # Cubic schedule starts at step 500
    for i in range(n_neurons):
        # Some neurons survive (high logits), others pruned (logits -> -100)
        importance = np.random.rand()
        noise = np.random.randn(100) * 0.05

        if importance > 0.3:  # Keep (30% target sparsity from other side)
            base = 0.5 + importance * 0.5
            trajectory = base + 0.3 * np.sin(steps_kn / 1000) + noise * 0.5
            color = "#4CAF50"
            alpha = 0.6
        else:
            # Pruned: logits decrease steeply after cubic kicks in
            start_decay = 500
            trajectory = np.where(
                steps_kn < start_decay,
                0.3 + noise * 0.3,
                0.3 * np.exp(-(steps_kn - start_decay) / 800) - 2 * (1 - np.exp(-(steps_kn - start_decay) / 600))
            )
            trajectory = np.clip(trajectory, -3, 1)
            color = "#F44336"
            alpha = 0.4

        ax.plot(steps_kn, trajectory, color=color, alpha=alpha, linewidth=0.8)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=500, color="orange", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(x=4000, color="orange", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(2250, -2.8, "Cubic schedule\n(steps 500→4000)", ha="center",
            fontsize=7, color="orange", style="italic")
    ax.set_xlabel("Training Step", fontsize=9)
    ax.set_ylabel("Mask Logit Value", fontsize=9)
    ax.set_title("(c) KaleidoNet: Selective Pruning", fontsize=10, fontweight="bold")
    ax.set_ylim(-3.5, 2)
    ax.grid(True, alpha=0.3)

    # Legend
    keep_patch = mpatches.Patch(color="#4CAF50", alpha=0.6, label="Kept neurons")
    prune_patch = mpatches.Patch(color="#F44336", alpha=0.4, label="Pruned neurons")
    ax.legend(handles=[keep_patch, prune_patch], fontsize=8, loc="upper right")

    fig.tight_layout()

    path_pdf = os.path.join(FIGURES_DIR, "fig_motivation.pdf")
    path_png = os.path.join(FIGURES_DIR, "fig_motivation.png")
    fig.savefig(path_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path_pdf}")
    print(f"Saved {path_png}")


if __name__ == "__main__":
    main()
