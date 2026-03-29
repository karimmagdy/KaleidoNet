"""
Ablation study: measure contribution of each KaleidoNet pillar.

Tests all combinations of pillars on/off to produce the compound speedup
decomposition figure for the paper.

Run: python experiments/ablations/pillar_ablation.py
"""

import itertools
import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kaleidonet.model import KaleidoNet
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig
from kaleidonet.metrics.flops import FLOPsCounter


@dataclass
class AblationConfig:
    name: str
    elastic: bool = True       # Pillar 1: Dynamic morphing
    moe: bool = True           # Pillar 3: Sparse MoE
    early_exit: bool = True    # Part of Pillar 3: Early exit
    growth: bool = False       # Pillar 4: Incremental growth (off by default for short runs)
    num_experts: int = 4
    top_k: int = 1


ABLATION_CONFIGS = [
    AblationConfig(name="dense_baseline", elastic=False, moe=False, early_exit=False, num_experts=1, top_k=1),
    AblationConfig(name="moe_only", elastic=False, moe=True, early_exit=False),
    AblationConfig(name="elastic_only", elastic=True, moe=False, early_exit=False, num_experts=1, top_k=1),
    AblationConfig(name="early_exit_only", elastic=False, moe=False, early_exit=True, num_experts=1, top_k=1),
    AblationConfig(name="moe+elastic", elastic=True, moe=True, early_exit=False),
    AblationConfig(name="moe+early_exit", elastic=False, moe=True, early_exit=True),
    AblationConfig(name="elastic+early_exit", elastic=True, moe=False, early_exit=True, num_experts=1, top_k=1),
    AblationConfig(name="all_pillars", elastic=True, moe=True, early_exit=True),
]


def get_cifar100_loaders(batch_size: int = 64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return {"images": torch.stack(images), "targets": torch.tensor(labels), "task": "classify"}

    pin = torch.cuda.is_available() and not hasattr(torch, '_xla_device')
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=pin),
    )


def run_ablation(ablation: AblationConfig, train_loader, val_loader, max_steps: int = 2000):
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation.name}")
    print(f"  elastic={ablation.elastic}, moe={ablation.moe}, early_exit={ablation.early_exit}")
    print(f"{'='*60}")

    model = KaleidoNet(
        embed_dim=192,
        num_blocks=4,
        num_heads=6,
        num_experts=ablation.num_experts if ablation.moe else 1,
        top_k=ablation.top_k if ablation.moe else 1,
        num_classes=100,
        vocab_size=0,
        image_size=32,
        patch_size=4,
        elastic=ablation.elastic,
        drop_path_rate=0.1,
        confidence_threshold=0.9 if ablation.early_exit else 1.0,
    )

    config = TrainerConfig(
        lr=3e-4,
        max_steps=max_steps,
        warmup_steps=100,
        flops_budget=200_000_000,  # Placeholder, calibrated below
        log_interval=100,
        eval_interval=500,
        use_amp=False,
    )

    # Calibrate FLOPs budget from init active FLOPs
    _dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _out = model({"images": _dummy, "task": "classify"})
    init_active = int(_out.get("active_flops", 200_000_000))
    config.flops_budget = int(init_active * 0.5)
    print(f"  Init active FLOPs: {init_active:,}, budget (50%): {config.flops_budget:,}")

    trainer = KaleidoNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )

    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    final_metrics = trainer.eval_step()
    param_info = model.count_active_params()

    # Compute inference FLOPs
    model.eval()
    counter = FLOPsCounter()
    active_flops_info = counter.count(model, batch_size=1, seq_len=64)
    dense_flops = counter.count_dense(model, batch_size=1, seq_len=64)

    result = {
        "name": ablation.name,
        "config": {
            "elastic": ablation.elastic,
            "moe": ablation.moe,
            "early_exit": ablation.early_exit,
            "num_experts": ablation.num_experts,
            "top_k": ablation.top_k,
        },
        "wall_time_seconds": wall_time,
        "total_params": param_info["total_params"],
        "active_params": param_info["active_params"],
        "active_fraction": param_info["active_fraction"],
        "active_flops": active_flops_info["total_active_flops"],
        "dense_flops": dense_flops,
        **final_metrics,
    }

    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    return result


def main():
    print("KaleidoNet Pillar Ablation Study")
    print("=" * 60)

    train_loader, val_loader = get_cifar100_loaders(batch_size=64)
    results = []

    for ablation in ABLATION_CONFIGS:
        result = run_ablation(ablation, train_loader, val_loader, max_steps=2000)
        results.append(result)

    # Summary table
    print("\n\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'Val Acc':>8} {'Val Loss':>9} {'Active %':>9} {'FLOPs (M)':>10} {'Time (s)':>9}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<25} "
            f"{r.get('val_accuracy', 0):>8.4f} "
            f"{r.get('val_loss', 0):>9.4f} "
            f"{r['active_fraction']:>8.1%} "
            f"{r.get('active_flops', 0) / 1e6:>10.1f} "
            f"{r['wall_time_seconds']:>9.1f}"
        )

    # Save results
    with open("experiments/ablations/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to experiments/ablations/results.json")


if __name__ == "__main__":
    main()
