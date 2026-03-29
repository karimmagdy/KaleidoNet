"""
Quick 100-step CIFAR-100 smoke test to verify training loop works end-to-end.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kaleidonet.model import KaleidoNet
from kaleidonet.training.trainer import KaleidoNetTrainer, TrainerConfig


def main():
    print("=" * 60)
    print("KaleidoNet Smoke Test (synthetic data, 100 steps)")
    print("=" * 60)

    # Synthetic CIFAR-100-like data (skip download for speed)
    N_train, N_val = 640, 128
    train_images = torch.randn(N_train, 3, 32, 32)
    train_labels = torch.randint(0, 100, (N_train,))
    val_images = torch.randn(N_val, 3, 32, 32)
    val_labels = torch.randint(0, 100, (N_val,))

    def collate_fn(batch):
        imgs, labels = [], []
        for img, lbl in batch:
            imgs.append(img)
            labels.append(lbl)
        return {
            "images": torch.stack(imgs),
            "targets": torch.stack(labels),
            "task": "classify",
        }

    train_ds = TensorDataset(train_images, train_labels)
    val_ds = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    config = TrainerConfig(
        lr=3e-4,
        max_steps=100,
        warmup_steps=10,
        flops_budget=500_000_000,
        log_interval=10,
        eval_interval=50,
        use_amp=False,
    )

    model = KaleidoNet(
        embed_dim=192, num_blocks=4, num_heads=6, num_experts=4,
        top_k=1, num_classes=100, image_size=32, patch_size=4, elastic=True,
        drop_path_rate=0.1,
    )

    param_info = model.count_active_params()
    print(f"Total params: {param_info['total_params']:,}")
    print()

    trainer = KaleidoNetTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        task_loss_fn=nn.CrossEntropyLoss(),
    )

    trainer.train()

    print("\nSmoke test PASSED!")
    final = trainer.eval_step()
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
