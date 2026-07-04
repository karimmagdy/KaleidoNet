"""Mechanism + collapse figures for the v2 revision (R2.2, R3.1, R6.6, R5).

Fig A (fig_mechanism.pdf), 3 panels:
  (a) measured |dL_task/dm| vs |dL_penalty/dm| over training (protocol replica)
      -> the penalty is ReLU-gated to zero for >80% of steps
  (b) final accuracy of all 8 Lagrangian probe configs (lambda x tau grid)
      vs the cubic-schedule reference at the same 5k budget
  (c) lambda and soft-active-fraction trajectories -> constraint hovers at
      target, dual ascent plateaus, budget never selectively enforced

Fig B (fig_collapse_50k.pdf), 2 panels, from the 50k mask_dist logs:
  (a) per-layer soft-active-fraction trajectories (3 seeds, CIFAR-100)
      -> dispersion + whole-layer death, not uniform drift
  (b) final per-layer mask-logit mean +/- std -> wide separation without
      task-aligned selectivity

Run: python experiments/analysis/plot_mechanism.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRADSCALE = os.path.join(ROOT, "results", "gradscale")
CONV = os.path.join(ROOT, "results", "convergence")
OUT = os.path.join(ROOT, "figures")


def read_probe_csv(path):
    """Return per-step medians across layers: g_task, g_pen, lambda, frac."""
    by_step = {}
    for r in csv.DictReader(open(path)):
        try:
            s = int(r["step"])
        except (ValueError, TypeError):
            continue
        d = by_step.setdefault(s, {"gt": [], "gp": [], "lam": [], "frac": []})
        d["gt"].append(float(r["g_task_abs_mean"]))
        d["gp"].append(float(r["g_penalty_abs_mean"]))
        d["lam"].append(float(r["lambda"]))
        d["frac"].append(float(r["flops_fraction"]))
    steps = sorted(by_step)
    med = lambda v: float(np.median(v))
    return (np.array(steps),
            np.array([med(by_step[s]["gt"]) for s in steps]),
            np.array([med(by_step[s]["gp"]) for s in steps]),
            np.array([med(by_step[s]["lam"]) for s in steps]),
            np.array([med(by_step[s]["frac"]) for s in steps]))


def fig_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # --- (a) gradient components, protocol replica (lambda=0.01, tau fixed 5) ---
    ax = axes[0]
    steps, gt, gp, lam, frac = read_probe_csv(
        os.path.join(GRADSCALE, "gradscale_cifar100_d192_taufixed5_seed1.csv"))
    floor = 1e-10
    ax.semilogy(steps, np.maximum(gt, floor), color="#1f77b4", lw=1.5,
                label=r"$|\partial\mathcal{L}_{\rm task}/\partial m|$")
    ax.semilogy(steps, np.maximum(gp, floor), color="#d62728", lw=1.2,
                label=r"$|\partial\mathcal{L}_{\rm penalty}/\partial m|$")
    zero_frac = float((gp == 0).mean()) * 100
    ax.set_ylim(floor, None)
    ax.set_xlabel("training step")
    ax.set_ylabel("mask-logit gradient (median over layers)")
    ax.set_title(f"(a) penalty gradient is gated to zero\n({zero_frac:.0f}% of steps)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # --- (b) accuracy across the lambda x tau grid ---
    ax = axes[1]
    labels, accs = [], []
    for p in sorted(glob.glob(os.path.join(GRADSCALE, "*.json"))):
        d = json.load(open(p))
        lam0 = p.split("lam")[1].split("_")[0] if "lam" in os.path.basename(p) else "0.01"
        tau = "anneal" if d["anneal_tau"] else "fix5"
        labels.append(f"$\\lambda$={lam0}\n$\\tau$ {tau}")
        accs.append(d["val_accuracy"] * 100)
    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    accs = [accs[i] for i in order]
    x = np.arange(len(labels))
    ax.bar(x, accs, color="#d62728", alpha=0.8, label="Lagrangian (probe)")
    ref_path = os.path.join(CONV, "cubic_only_cifar100_seed9.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["val_accuracy"] * 100
        ax.axhline(ref, color="#2ca02c", lw=2, ls="--",
                   label=f"cubic schedule, same budget ({ref:.1f}%)")
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="chance (1%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("CIFAR-100 accuracy @ 5k steps (%)")
    ax.set_title("(b) collapse across $\\lambda\\in[10^{-3},1]$\nand both $\\tau$ protocols")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- (c) lambda + soft fraction trajectories ---
    ax = axes[2]
    ax.plot(steps, frac * 100, color="#9467bd", lw=1.5, label="soft active fraction (%)")
    ax.axhline(50, color="#9467bd", lw=1, ls=":", label="target (50%)")
    ax.set_xlabel("training step")
    ax.set_ylabel("soft active fraction (%)", color="#9467bd")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(steps, lam, color="#ff7f0e", lw=1.5, label=r"$\lambda$")
    ax2.set_ylabel(r"dual variable $\lambda$", color="#ff7f0e")
    ax.set_title("(c) constraint hovers at target;\ndual ascent plateaus")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_mechanism.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved fig_mechanism.pdf/png")


def load_mask_dist(path):
    """Defensive parse of *_mask_dist.csv (drops corrupted lines, steps>50k)."""
    per_layer = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 9:
                continue
            try:
                step, layer = int(parts[0]), int(parts[1])
                frac = float(parts[7])
            except ValueError:
                continue
            if step > 50000:
                continue
            per_layer.setdefault(layer, []).append((step, frac))
    return per_layer


def fig_collapse():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # --- (a) per-layer frac_active trajectories, 3 seeds ---
    ax = axes[0]
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for seed in (1, 2, 3):
        pl = load_mask_dist(os.path.join(
            CONV, "curves", f"lagrangian_cifar100_seed{seed}_mask_dist.csv"))
        for layer, pts in pl.items():
            pts.sort()
            s = [p[0] for p in pts]
            v = [p[1] * 100 for p in pts]
            ax.plot(s, v, color=colors[seed], alpha=0.25, lw=0.6)
        ax.plot([], [], color=colors[seed], lw=1.5, label=f"seed {seed}")
    ax.set_xlabel("training step")
    ax.set_ylabel("per-layer soft active fraction (%)")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_title("(a) per-layer trajectories disperse\n(some layers die at 0%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (b) final per-layer logit mean +/- std (seed 1, 50k) ---
    ax = axes[1]
    d = json.load(open(os.path.join(CONV, "lagrangian_cifar100_seed1.json")))
    ms = d["mask_logit_summary"]
    idx = np.arange(len(ms))
    means = np.array([m["mean"] for m in ms])
    stds = np.array([m["std"] for m in ms])
    ax.errorbar(idx, means, yerr=stds, fmt="o", ms=3, lw=1, capsize=2,
                color="#d62728", ecolor="#d62728", alpha=0.85)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("elastic layer index")
    ax.set_ylabel("final mask logits (mean $\\pm$ std)")
    ax.set_title("(b) final state: wide within-layer spread,\nnot uniform suppression")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_collapse_50k.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved fig_collapse_50k.pdf/png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_mechanism()
    fig_collapse()
