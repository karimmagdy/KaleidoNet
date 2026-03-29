"""Train KaleidoNet on CIFAR-10 classification.

Run:
    python run.py experiments/baselines/train_cifar10.py --seed 1 --steps 5000
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kaleidonet.model import KaleidoNet
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig, set_seed
from kaleidonet.metrics.flops import FLOPsCounter


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size: int = 64, num_workers: int = 0):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return {
            "images": torch.stack(images),
            "targets": torch.tensor(labels),
            "task": "classify",
        }

    pin = torch.cuda.is_available() and not hasattr(torch, '_xla_device')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train KaleidoNet on CIFAR-10")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps")
    args = parser.parse_args()

    print("=" * 60)
    print("KaleidoNet CIFAR-10 Training")
    print("=" * 60)

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed: {args.seed}")

    config = TrainerConfig(
        lr=3e-4,
        max_steps=args.steps,
        warmup_steps=250,
        flops_budget=200_000_000,
        log_interval=50,
        eval_interval=500,
        use_amp=False,
        seed=args.seed,
    )

    train_loader, val_loader = get_cifar10_loaders(batch_size=64)

    model = KaleidoNet(
        embed_dim=192,
        num_blocks=4,
        num_heads=6,
        num_experts=4,
        top_k=1,
        num_classes=10,
        vocab_size=0,
        image_size=32,
        patch_size=4,
        elastic=True,
        drop_path_rate=0.1,
    )

    param_info = model.count_active_params()
    print(f"Total params:  {param_info['total_params']:,}")
    print(f"Active params: {param_info['active_params']:,}")
    print(f"Active fraction: {param_info['active_fraction']:.1%}")
    print()

    counter = FLOPsCounter()
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=64)
    probe_device = next(model.parameters()).device
    _dummy = torch.randn(1, 3, 32, 32, device=probe_device)
    _out = model({"images": _dummy})
    init_active_flops = int(_out.get("active_flops", dense_flops))
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
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    param_info = model.count_active_params()
    print(f"\nFinal active params: {param_info['active_params']:,} / {param_info['total_params']:,} ({param_info['active_fraction']:.1%})")

    final_active_flops = None
    model.eval()
    try:
        probe_device = next(model.parameters()).device
        with torch.no_grad():
            final_probe = model({"images": torch.randn(1, 3, 32, 32, device=probe_device)})
        final_active_flops = int(final_probe["active_flops"])
        print(f"Final active FLOPs: {final_active_flops:,}")
    except RuntimeError as exc:
        print(f"Warning: final active FLOPs probe failed: {exc}")

    results = {
        "model": "KaleidoNet",
        "dataset": "cifar10",
        "seed": args.seed,
        "steps": config.max_steps,
        "params": param_info,
        "flops_budget": config.flops_budget,
        "dense_flops": dense_flops,
        "init_active_flops": init_active_flops,
        "active_flops": final_active_flops,
        **final_metrics,
    }
    os.makedirs("results", exist_ok=True)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    result_path = f"results/kaleidonet_cifar10{suffix}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
