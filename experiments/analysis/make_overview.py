"""Fig. 1 — methodology + contributions overview (3 panels).

(a) the setting: elastic MoE ViT with per-neuron masks inside each expert
(b) the two pruning pathways: gated Lagrangian penalty -> selectivity
    collapse, vs deterministic schedule + dual-rate -> selective pruning
(c) the three contributions

Run: python experiments/analysis/make_overview.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figures")

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#666666"


def box(ax, x, y, w, h, text, fc="#f4f6f8", ec=GREY, fs=8.2, weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw))


def panel_a(ax):
    ax.set_title("(a) Setting: per-neuron masks inside MoE experts",
                 fontsize=9.5, fontweight="bold")
    box(ax, 0.05, 0.80, 0.90, 0.13, "image patches $\\to$ self-attention", fc="#eaf1fa", ec=BLUE)
    arrow(ax, 0.50, 0.80, 0.50, 0.72)
    box(ax, 0.30, 0.60, 0.40, 0.12, "router (top-1)", fc="#eaf1fa", ec=BLUE)
    # four experts, each with a mask strip
    for i, x in enumerate((0.03, 0.27, 0.51, 0.75)):
        arrow(ax, 0.50, 0.60, x + 0.11, 0.50)
        box(ax, x, 0.28, 0.22, 0.22, f"expert {i+1}\nFFN", fc="white", ec=GREY, fs=7.5)
        # neuron mask dots: mix of kept/pruned
        for j in range(6):
            kept = (i + j) % 3 != 0
            ax.plot(x + 0.035 + j * 0.031, 0.315, "o", ms=3.4,
                    color=GREEN if kept else "#cccccc",
                    mec=GREY, mew=0.4, zorder=5)
    box(ax, 0.05, 0.06, 0.90, 0.13,
        "each neuron $i$ carries a mask logit $m_i$:\nwhich neurons survive, and who decides?",
        fc="#fdf6e3", ec="#b8860b", fs=8.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def panel_b(ax):
    ax.set_title("(b) Two pathways for the pruning decision",
                 fontsize=9.5, fontweight="bold")
    # top: Lagrangian
    box(ax, 0.02, 0.76, 0.43, 0.17,
        "Lagrangian penalty\n$\\lambda\\,\\mathrm{relu}(\\hat{s}-s^{*})$ on mask\ngradients", fc="#fdecea", ec=RED)
    arrow(ax, 0.45, 0.845, 0.53, 0.845, color=RED)
    box(ax, 0.53, 0.76, 0.44, 0.17,
        "gate closed on 80–88% of steps;\nnon-selective when open\n(measured, Fig. 2)", fc="#fdecea", ec=RED, fs=7.6)
    arrow(ax, 0.75, 0.76, 0.75, 0.66, color=RED)
    box(ax, 0.53, 0.52, 0.44, 0.14,
        "selectivity collapse\n10.4% on CIFAR-100", fc=RED, ec=RED, tc="white", weight="bold")
    # bottom: schedule
    box(ax, 0.02, 0.26, 0.43, 0.17,
        "KaleidoNet: deterministic cubic\nschedule + dual-rate mask LR\n($3\\times$; the active ingredient)", fc="#eaf7ea", ec=GREEN)
    arrow(ax, 0.45, 0.345, 0.53, 0.345, color=GREEN)
    box(ax, 0.53, 0.26, 0.44, 0.17,
        "selection by task-driven\nimportance; schedule forces\neach decision", fc="#eaf7ea", ec=GREEN, fs=7.6)
    arrow(ax, 0.75, 0.26, 0.75, 0.16, color=GREEN)
    box(ax, 0.53, 0.02, 0.44, 0.14,
        "59.1% on CIFAR-100 at\n$1.80\\times$ fewer active FLOPs", fc=GREEN, ec=GREEN, tc="white", weight="bold")
    ax.text(0.02, 0.575, "same backbone,\nsame budget,\nsame optimiser\ngrouping",
            fontsize=7.6, color=GREY, style="italic", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def panel_c(ax):
    ax.set_title("(c) Contributions", fontsize=9.5, fontweight="bold")
    box(ax, 0.04, 0.68, 0.92, 0.24,
        "1 — Diagnosis (measured)\nthe per-neuron penalty is gated to zero for most of training\n"
        "and cannot prefer one neuron over another when active", fc="#eaf1fa", ec=BLUE, fs=7.8)
    box(ax, 0.04, 0.37, 0.92, 0.24,
        "2 — Validation\n$\\lambda\\times\\tau$ probe grid, matched executed budgets, five seeds,\n"
        "non-MoE ViT controls, loss-scaling ablation", fc="#eaf1fa", ec=BLUE, fs=7.8)
    box(ax, 0.04, 0.06, 0.92, 0.24,
        "3 — Instantiation\nKaleidoNet: cubic schedule + dual-rate optimisation\n"
        "(gradient-masking safeguard shown inert by ablation)", fc="#eaf1fa", ec=BLUE, fs=7.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.tight_layout(w_pad=1.4)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_overview.{ext}"), bbox_inches="tight", dpi=170)
    print("saved fig_overview.pdf/png")


if __name__ == "__main__":
    main()
