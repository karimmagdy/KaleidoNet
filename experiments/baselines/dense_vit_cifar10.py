"""Dense ViT baseline for CIFAR-10 — no KaleidoNet mechanisms.

Run:
    python experiments/baselines/dense_vit_cifar10.py --seed 1 --steps 5000
"""

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.baselines.dense_vit_baseline import DenseViT


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
        return {"images": torch.stack(images), "targets": torch.tensor(labels), "task": "classify"}

    pin = torch.cuda.is_available() and not hasattr(torch, '_xla_device')
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin),
    )


def main():
    parser = argparse.ArgumentParser(description="Dense ViT baseline for CIFAR-10")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=5000, help="Max training steps")
    args = parser.parse_args()

    print("=" * 60)
    print("Dense ViT Baseline — CIFAR-10")
    print("=" * 60)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    try:
        import torch_xla.core.xla_model as xm
        device = str(xm.xla_device())
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = DenseViT(embed_dim=192, num_heads=6, num_layers=4, num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Device: {device}")

    train_loader, val_loader = get_cifar10_loaders(batch_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    criterion = nn.CrossEntropyLoss()

    max_steps = args.steps
    log_interval = 50
    eval_interval = 500
    best_acc = 0.0
    start_time = time.time()

    model.train()
    train_iter = iter(train_loader)

    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        loss = criterion(out["logits"], batch["targets"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if "xla" in str(device):
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:6d} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {(step+1)/elapsed:.1f} steps/s")

        if step % eval_interval == 0 and step > 0:
            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vbatch.items()}
                    out = model(vbatch)
                    val_loss += criterion(out["logits"], vbatch["targets"]).item() * vbatch["targets"].shape[0]
                    correct += (out["logits"].argmax(1) == vbatch["targets"]).sum().item()
                    total += vbatch["targets"].shape[0]
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"  [EVAL] val_loss={val_loss/total:.4f} | val_acc={acc:.4f} | best_acc={best_acc:.4f}")
            model.train()

    total_time = time.time() - start_time
    flops_per_sample = 2 * total_params * 65
    print(f"\nDense baseline complete in {total_time:.1f}s")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"FLOPs per sample (approx): {flops_per_sample:,}")

    results = {
        "model": "DenseViT",
        "dataset": "cifar10",
        "seed": args.seed,
        "steps": max_steps,
        "total_params": total_params,
        "best_val_acc": best_acc,
        "flops_per_sample": flops_per_sample,
        "total_time_s": total_time,
    }
    os.makedirs("results", exist_ok=True)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    result_path = f"results/dense_vit_cifar10{suffix}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
