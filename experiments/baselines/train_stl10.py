"""Train KaleidoNet on STL-10 classification.

Run:
    python run.py experiments/baselines/train_stl10.py --seed 1 --steps 5000
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.baselines.stl10_data import get_stl10_loaders
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.model import KaleidoNet
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train KaleidoNet on STL-10")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--image-size", type=int, default=96, help="Input image size")
    parser.add_argument("--patch-size", type=int, default=8, help="Patch size")
    args = parser.parse_args()

    print("=" * 60)
    print("KaleidoNet STL-10 Training")
    print("=" * 60)

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed: {args.seed}")

    config = TrainerConfig(
        lr=3e-4,
        max_steps=args.steps,
        warmup_steps=250,
        flops_budget=400_000_000,
        log_interval=50,
        eval_interval=500,
        use_amp=False,
        seed=args.seed,
    )

    train_loader, val_loader = get_stl10_loaders(
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.image_size,
    )

    model = KaleidoNet(
        embed_dim=192,
        num_blocks=4,
        num_heads=6,
        num_experts=4,
        top_k=1,
        num_classes=10,
        vocab_size=0,
        image_size=args.image_size,
        patch_size=args.patch_size,
        elastic=True,
        drop_path_rate=0.1,
    )

    param_info = model.count_active_params()
    print(f"Total params:  {param_info['total_params']:,}")
    print(f"Active params: {param_info['active_params']:,}")
    print(f"Active fraction: {param_info['active_fraction']:.1%}")
    print()

    counter = FLOPsCounter()
    seq_len = (args.image_size // args.patch_size) ** 2
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=seq_len)
    probe_device = next(model.parameters()).device
    with torch.no_grad():
        probe = model({"images": torch.randn(1, 3, args.image_size, args.image_size, device=probe_device)})
    init_active_flops = int(probe.get("active_flops", dense_flops))
    target_fraction = 0.50
    config.flops_budget = int(init_active_flops * target_fraction)
    print(f"Dense FLOPs (per sample):   {dense_flops:,}")
    print(f"Active FLOPs at init:       {init_active_flops:,}")
    print(f"FLOPs budget (50% active):  {config.flops_budget:,}")
    print()

    trainer = KaleidoNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )

    trainer.train()

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_metrics = trainer.eval_step()
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    param_info = model.count_active_params()
    print(f"\nFinal active params: {param_info['active_params']:,} / {param_info['total_params']:,} ({param_info['active_fraction']:.1%})")

    final_active_flops = None
    model.eval()
    try:
        probe_device = next(model.parameters()).device
        with torch.no_grad():
            final_probe = model({"images": torch.randn(1, 3, args.image_size, args.image_size, device=probe_device)})
        final_active_flops = int(final_probe["active_flops"])
        print(f"Final active FLOPs: {final_active_flops:,}")
    except RuntimeError as exc:
        print(f"Warning: final active FLOPs probe failed: {exc}")

    results = {
        "model": "KaleidoNet",
        "dataset": "stl10",
        "seed": args.seed,
        "steps": config.max_steps,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "params": param_info,
        "flops_budget": config.flops_budget,
        "dense_flops": dense_flops,
        "init_active_flops": init_active_flops,
        "active_flops": final_active_flops,
        **final_metrics,
    }
    os.makedirs("results", exist_ok=True)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    result_path = f"results/kaleidonet_stl10{suffix}.json"
    with open(result_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
