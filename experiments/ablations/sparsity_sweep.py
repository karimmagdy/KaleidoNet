"""Sparsity target and mask LR sweep for sensitivity analysis.

Sweeps:
  - Target sparsity: [0.50, 0.60, 0.70, 0.80, 0.90]
  - Mask LR scale:   [1, 2, 3, 5] (multiplier of base LR)

Run:
    python experiments/ablations/sparsity_sweep.py --sweep sparsity --dataset cifar100 --seed 1
    python experiments/ablations/sparsity_sweep.py --sweep mask-lr --dataset cifar100 --seed 1
    python experiments/ablations/sparsity_sweep.py --sweep both --dataset cifar100 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaleidonet.model import KaleidoNet
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig, set_seed


def get_loaders(dataset: str, batch_size: int = 64, data_dir: str | None = None):
    if dataset == "cifar10":
        from experiments.baselines.train_cifar10 import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size), 10, 32, 4
    elif dataset == "cifar100":
        from experiments.baselines.train_cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(batch_size=batch_size), 100, 32, 4
    elif dataset == "tiny_imagenet":
        from experiments.baselines.tiny_imagenet_data import get_tiny_imagenet_loaders
        loaders = get_tiny_imagenet_loaders(batch_size=batch_size, data_root=data_dir or "./data/tiny-imagenet-200")
        return loaders, 200, 64, 8
    elif dataset == "stl10":
        from experiments.baselines.stl10_data import get_stl10_loaders
        return get_stl10_loaders(batch_size=batch_size), 10, 96, 8
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_single(
    dataset: str, seed: int, steps: int,
    target_sparsity: float, mask_lr_scale: float,
    data_dir: str | None = None,
    t0: int | None = None, t1: int | None = None,
) -> dict:
    """Run a single KaleidoNet training with specified sparsity target and mask LR scale.

    t0/t1 default to the same 10%/80%-of-budget window used by
    experiments/run_convergence.py so long-budget sweep runs are directly
    comparable to the main convergence results.
    """
    set_seed(seed)
    (train_loader, val_loader), num_classes, img_size, patch_size = get_loaders(dataset, data_dir=data_dir)

    model = KaleidoNet(
        embed_dim=192, num_blocks=4, num_heads=6,
        num_experts=4, top_k=1, num_classes=num_classes,
        vocab_size=0, image_size=img_size, patch_size=patch_size,
        elastic=True, drop_path_rate=0.1,
    )

    counter = FLOPsCounter()
    seq_len = (img_size // patch_size) ** 2
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=seq_len)
    probe_device = next(model.parameters()).device
    with torch.no_grad():
        probe = model({"images": torch.randn(1, 3, img_size, img_size, device=probe_device)})
    init_active = int(probe.get("active_flops", dense_flops))

    config = TrainerConfig(
        lr=3e-4,
        max_steps=steps,
        warmup_steps=int(steps * 0.05),
        flops_budget=int(init_active * 0.50),
        log_interval=100,
        eval_interval=max(steps // 100, 500),
        use_amp=False,
        seed=seed,
        target_sparsity=target_sparsity,
        sparsity_start_step=t0 if t0 is not None else int(steps * 0.10),
        sparsity_end_step=t1 if t1 is not None else int(steps * 0.80),
    )

    # Build trainer then override mask optimizer LR
    trainer = KaleidoNetTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )

    # Update mask optimizer LR based on scale
    if trainer.mask_optimizer is not None:
        for pg in trainer.mask_optimizer.param_groups:
            pg["lr"] = config.lr * mask_lr_scale

    trainer.train()

    # Final eval
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            val_loss += criterion(out["logits"], batch["targets"]).item() * batch["targets"].shape[0]
            correct += (out["logits"].argmax(1) == batch["targets"]).sum().item()
            total += batch["targets"].shape[0]

    final_acc = correct / max(total, 1)
    final_flops = None
    try:
        with torch.no_grad():
            fp = model({"images": torch.randn(1, 3, img_size, img_size, device=config.device)})
        final_flops = int(fp["active_flops"])
    except Exception:
        pass

    param_info = model.count_active_params()

    return {
        "model": "KaleidoNet",
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "target_sparsity": target_sparsity,
        "mask_lr_scale": mask_lr_scale,
        "t0": config.sparsity_start_step,
        "t1": config.sparsity_end_step,
        "params": param_info,
        "dense_flops": dense_flops,
        "active_flops": final_flops,
        "val_accuracy": final_acc,
        "val_loss": val_loss / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Sparsity & mask LR sweep")
    parser.add_argument("--sweep", choices=["sparsity", "mask-lr", "schedule", "both"], default="both")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet", "stl10"], default="cifar100")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--targets", type=float, nargs="+", default=None,
                        help="Explicit target_sparsity list for --sweep sparsity (overrides default grid)")
    parser.add_argument("--t0-list", type=int, nargs="+", default=None,
                        help="Schedule start steps for --sweep schedule (each at default t1)")
    parser.add_argument("--t1-list", type=int, nargs="+", default=None,
                        help="Schedule end steps for --sweep schedule (each at default t0)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if its result JSON already exists")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_results = []

    def _run_and_save(fname: str, **kwargs):
        path = os.path.join("results", fname)
        if args.skip_existing and os.path.exists(path):
            print(f"  [skip-existing] {fname}")
            with open(path) as f:
                r = json.load(f)
            all_results.append(r)
            return r
        r = run_single(args.dataset, args.seed, args.steps, data_dir=args.data_dir, **kwargs)
        all_results.append(r)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  -> acc={r['val_accuracy']:.4f}, flops={r['active_flops']}  [{fname}]")
        return r

    # Sparsity sweep (mask_lr_scale fixed at 3 — the default)
    if args.sweep in ("sparsity", "both"):
        targets = args.targets if args.targets else [0.50, 0.60, 0.70, 0.80, 0.90]
        for target in targets:
            print(f"\n{'='*60}")
            print(f"Sparsity sweep: target={target:.2f} | {args.dataset} | seed={args.seed}")
            print(f"{'='*60}")
            _run_and_save(
                f"sparsity_sweep_{target:.2f}_{args.dataset}_seed{args.seed}_steps{args.steps}.json",
                target_sparsity=target, mask_lr_scale=3.0,
            )

    # Schedule-window sweep (one-factor-at-a-time around the 10%/80% defaults)
    if args.sweep == "schedule":
        for t0 in (args.t0_list or []):
            print(f"\n{'='*60}")
            print(f"Schedule sweep: t0={t0} (default t1) | {args.dataset} | seed={args.seed}")
            print(f"{'='*60}")
            _run_and_save(
                f"schedule_sweep_t0_{t0}_{args.dataset}_seed{args.seed}_steps{args.steps}.json",
                target_sparsity=0.70, mask_lr_scale=3.0, t0=t0,
            )
        for t1 in (args.t1_list or []):
            print(f"\n{'='*60}")
            print(f"Schedule sweep: t1={t1} (default t0) | {args.dataset} | seed={args.seed}")
            print(f"{'='*60}")
            _run_and_save(
                f"schedule_sweep_t1_{t1}_{args.dataset}_seed{args.seed}_steps{args.steps}.json",
                target_sparsity=0.70, mask_lr_scale=3.0, t1=t1,
            )

    # Mask LR sweep (target_sparsity fixed at 0.70 — the default)
    if args.sweep in ("mask-lr", "both"):
        scales = [1.0, 2.0, 3.0, 5.0]
        for scale in scales:
            print(f"\n{'='*60}")
            print(f"Mask LR sweep: scale={scale:.1f}x | {args.dataset} | seed={args.seed}")
            print(f"{'='*60}")
            _run_and_save(
                f"mask_lr_sweep_{scale:.1f}x_{args.dataset}_seed{args.seed}.json",
                target_sparsity=0.70, mask_lr_scale=scale,
            )

    # Summary
    print(f"\n{'='*60}")
    print("Sweep Summary")
    print(f"{'='*60}")
    print(f"{'Target':>8} {'MaskLR':>8} {'Val Acc':>10} {'FLOPs':>12}")
    print(f"{'-'*42}")
    for r in all_results:
        flops_str = f"{r['active_flops']/1e6:.1f}M" if r['active_flops'] else "N/A"
        print(f"{r['target_sparsity']:>8.2f} {r['mask_lr_scale']:>7.1f}x {r['val_accuracy']:>10.4f} {flops_str:>12}")

    # Save combined results
    combined_path = os.path.join("results", f"sweep_combined_{args.dataset}_seed{args.seed}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
