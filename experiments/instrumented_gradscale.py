"""Instrumented gradient-scale measurement for the Lagrangian baseline (R3.1 / R1.Q2).

Runs a (short) Lagrangian training and, every --measure-interval steps, logs on
the current batch:
  - mean |dL_task/dm|      (task-loss gradient on mask logits, per layer)
  - mean |dL_penalty/dm|   (Lagrangian-penalty gradient on mask logits, per layer)
  - their ratio            (the gradient-scale mismatch, measured not derived)
  - Adam second-moment sqrt(v_hat) and the effective update magnitude
    lr * m_hat / (sqrt(v_hat) + eps) for mask logits
This shows that Adam's per-coordinate normalisation rescales the SUMMED
gradient but cannot amplify the penalty component drowned inside it.

Also reusable at other widths for the scaling probe (R1/R2.1):
    python experiments/instrumented_gradscale.py --embed-dim 384 --num-heads 6 ...

Output: results/gradscale/gradscale_{dataset}_d{dim}_seed{seed}.csv (+ .json summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaleidonet.model import KaleidoNet
from kaleidonet.core.elastic import ElasticLinear
from kaleidonet.training.trainer import set_seed
from experiments.baselines.lagrangian_pruning import (
    LagrangianPruningTrainer,
    compute_soft_flops_fraction,
    eval_model,
    get_loaders,
)


def _mask_modules(model):
    return [(i, m) for i, m in enumerate(model.modules()) if isinstance(m, ElasticLinear)]


def _grad_stats(model):
    """Per-layer mean |grad| on mask logits (None-safe)."""
    out = {}
    for idx, m in _mask_modules(model):
        g = m.mask_logits.grad
        out[idx] = float(g.abs().mean().item()) if g is not None else 0.0
    return out


def measure_components(trainer, model, batch):
    """Two extra backward passes on the same batch: task-only and penalty-only."""
    model.train()
    batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # --- task-only ---
    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    logits = outputs["logits"]
    task_loss = trainer.task_loss_fn(logits, batch["targets"])
    task_loss.backward()
    g_task = _grad_stats(model)

    # --- penalty-only ---
    model.zero_grad(set_to_none=True)
    flops_fraction = compute_soft_flops_fraction(model)
    violation = flops_fraction - trainer.target_flops_ratio
    penalty = trainer.lambda_val * F.relu(violation)
    if penalty.requires_grad:
        penalty.backward()
    g_pen = _grad_stats(model)

    model.zero_grad(set_to_none=True)
    return g_task, g_pen, float(flops_fraction.item()), float(violation.item())


def adam_state_stats(trainer, model):
    """Mean sqrt(v_hat) and effective update magnitude for mask logits."""
    opt = trainer.mask_optimizer
    if opt is None:
        return {}
    out = {}
    step_lr = opt.param_groups[0]["lr"]
    beta1, beta2 = opt.param_groups[0].get("betas", (0.9, 0.999))
    eps = opt.param_groups[0].get("eps", 1e-8)
    id_to_idx = {id(m.mask_logits): idx for idx, m in _mask_modules(model)}
    for p in opt.param_groups[0]["params"]:
        st = opt.state.get(p)
        if not st or "exp_avg_sq" not in st:
            continue
        t = st.get("step", 1)
        t = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        bc1 = 1 - beta1 ** max(t, 1)
        bc2 = 1 - beta2 ** max(t, 1)
        m_hat = st["exp_avg"] / bc1
        v_hat = st["exp_avg_sq"] / bc2
        upd = step_lr * m_hat / (v_hat.sqrt() + eps)
        idx = id_to_idx.get(id(p), -1)
        out[idx] = {
            "sqrt_v_hat_mean": float(v_hat.sqrt().mean().item()),
            "update_abs_mean": float(upd.abs().mean().item()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Instrumented Lagrangian gradient-scale run")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--measure-interval", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--target-flops-ratio", type=float, default=0.5)
    ap.add_argument("--lambda-init", type=float, default=0.01)
    ap.add_argument("--lambda-lr", type=float, default=0.01)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--anneal-tau", action="store_true",
                    help="Anneal Gumbel tau 1.0->0.1 over the first 2500 steps "
                         "(the schedule methods' protocol) instead of the "
                         "Lagrangian family's fixed tau=5.0 — tests the tau confound")
    args = ap.parse_args()

    set_seed(args.seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(
        args.dataset, data_dir=args.data_dir)

    model = KaleidoNet(
        embed_dim=args.embed_dim, num_blocks=4, num_heads=args.num_heads,
        num_experts=4, top_k=1, num_classes=num_classes,
        vocab_size=0, image_size=img_size, patch_size=patch_size,
        elastic=True, drop_path_rate=0.1,
    )

    trainer = LagrangianPruningTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        target_flops_ratio=args.target_flops_ratio,
        lr=3e-4, lambda_init=args.lambda_init, lambda_lr=args.lambda_lr,
        max_steps=args.steps, warmup_steps=int(args.steps * 0.05),
        seed=args.seed, log_interval=100,
        eval_interval=max(args.steps // 20, 100),
    )

    out_dir = os.path.join(ROOT, "results", "gradscale")
    os.makedirs(out_dir, exist_ok=True)
    tau_tag = "tauanneal" if args.anneal_tau else "taufixed5"
    lam_tag = "" if args.lambda_init == 0.01 else f"_lam{args.lambda_init:g}"
    tag = f"gradscale_{args.dataset}_d{args.embed_dim}_{tau_tag}{lam_tag}_seed{args.seed}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    fp = open(csv_path, "w", newline="")
    w = csv.writer(fp)
    w.writerow([
        "step", "layer_idx",
        "g_task_abs_mean", "g_penalty_abs_mean", "penalty_to_task_ratio",
        "sqrt_v_hat_mean", "update_abs_mean",
        "lambda", "flops_fraction", "violation",
    ])

    ratios_all = []
    train_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if args.anneal_tau:
            progress = min(step / 2500.0, 1.0)
            tau = 1.0 + (0.1 - 1.0) * progress
            for m in model.modules():
                if hasattr(m, "tau"):
                    m.tau = tau

        if step % args.measure_interval == 0:
            g_task, g_pen, frac, viol = measure_components(trainer, model, batch)
            adam_stats = adam_state_stats(trainer, model)
            for idx in g_task:
                gt, gp = g_task[idx], g_pen[idx]
                ratio = gp / gt if gt > 0 else float("nan")
                st = adam_stats.get(idx, {})
                w.writerow([
                    step, idx,
                    f"{gt:.3e}", f"{gp:.3e}",
                    f"{ratio:.3e}" if ratio == ratio else "nan",
                    f"{st.get('sqrt_v_hat_mean', 0.0):.3e}",
                    f"{st.get('update_abs_mean', 0.0):.3e}",
                    f"{float(trainer.lambda_val.item()):.4f}",
                    f"{frac:.4f}", f"{viol:.4f}",
                ])
                if gt > 0 and gp > 0:
                    ratios_all.append(ratio)
            fp.flush()

        trainer.train_step(batch)

        if step % 500 == 0:
            print(f"step {step:6d} | lambda={trainer.lambda_val.item():.4f} | "
                  f"measured ratios so far: n={len(ratios_all)}")

    fp.close()

    final_metrics = eval_model(model, val_loader, trainer.device)
    final_logits = []
    with torch.no_grad():
        for idx, m in _mask_modules(model):
            lg = m.mask_logits
            final_logits.append({
                "layer": idx,
                "mean": float(lg.mean()), "std": float(lg.std()),
                "min": float(lg.min()), "max": float(lg.max()),
                "frac_active": float((torch.sigmoid(lg) > 0.5).float().mean()),
            })

    summary = {
        "dataset": args.dataset,
        "embed_dim": args.embed_dim,
        "anneal_tau": args.anneal_tau,
        "final_lambda": float(trainer.lambda_val.item()),
        **final_metrics,
        "final_mask_logits": final_logits,
        "seed": args.seed,
        "steps": args.steps,
        "n_measurements": len(ratios_all),
        "penalty_to_task_ratio_median": (
            float(sorted(ratios_all)[len(ratios_all) // 2]) if ratios_all else None),
        "penalty_to_task_ratio_mean": (
            float(sum(ratios_all) / len(ratios_all)) if ratios_all else None),
        "csv": csv_path,
    }
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
