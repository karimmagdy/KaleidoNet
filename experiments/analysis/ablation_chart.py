"""Generate ablation bar chart for KaleidoNet paper.

Creates a grouped bar chart comparing 8 ablation configurations
(Dense, MoE only, Early exit only, Elastic only, MoE+EE, MoE+Elastic,
Elastic+EE, All pillars) at 2000 steps on CIFAR-100.

Run:
    python experiments/analysis/ablation_chart.py
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
import numpy as np


FIGURES_DIR = os.path.join(ROOT, "figures")

# Data from ablation study (2000 steps, CIFAR-100, single seed)
CONFIGS = [
    ("Dense",       24.23, 172.5),
    ("MoE",         24.63, 172.8),
    ("EE",          24.95, 172.5),
    ("Elastic",     20.66, 152.8),
    ("MoE+EE",      25.82, 172.8),
    ("MoE+Elas.",   20.57, 153.1),
    ("Elas.+EE",    20.34, 152.8),
    ("All",         20.85, 153.1),
]

labels = [c[0] for c in CONFIGS]
accs   = [c[1] for c in CONFIGS]
flops  = [c[2] for c in CONFIGS]

# Colors: lighter for non-elastic, darker for elastic-based
colors_acc = []
for c in CONFIGS:
    if "Elas" in c[0] or c[0] == "All":
        colors_acc.append("#E57373")  # Red-ish for elastic (accuracy drops)
    elif "MoE" in c[0] or "EE" in c[0]:
        colors_acc.append("#81C784")  # Green for capacity-boosting
    else:
        colors_acc.append("#90CAF9")  # Blue for baseline


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(labels))
    width = 0.6

    # Top: Accuracy
    bars1 = ax1.bar(x, accs, width, color=colors_acc, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Val Accuracy (%)", fontsize=10)
    ax1.set_title("Ablation Study: CIFAR-100, 2k Steps", fontsize=12)
    ax1.axhline(y=accs[0], color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax1.set_ylim(18, 28)
    ax1.grid(True, axis="y", alpha=0.3)

    # Annotate accuracy values
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

    # Bottom: FLOPs
    colors_flops = ["#FFB74D" if f < 160 else "#64B5F6" for f in flops]
    bars2 = ax2.bar(x, flops, width, color=colors_flops, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Active FLOPs (M)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.axhline(y=flops[0], color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax2.set_ylim(140, 180)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    path_pdf = os.path.join(FIGURES_DIR, "fig_ablation.pdf")
    path_png = os.path.join(FIGURES_DIR, "fig_ablation.png")
    fig.savefig(path_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path_pdf}")
    print(f"Saved {path_png}")


if __name__ == "__main__":
    main()
