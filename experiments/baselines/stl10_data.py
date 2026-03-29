"""STL-10 data utilities.

STL-10 is 96x96 with 10 classes, 5000 train + 8000 test images.
We resize to a configurable image_size for the ViT tokenizer.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


def get_stl10_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 96,
):
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
    ])

    train_ds = datasets.STL10(root="./data", split="train", download=True, transform=transform_train)
    test_ds = datasets.STL10(root="./data", split="test", download=True, transform=transform_test)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return {
            "images": torch.stack(images),
            "targets": torch.tensor(labels),
            "task": "classify",
        }

    pin = torch.cuda.is_available() and not hasattr(torch, '_xla_device')
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    return train_loader, test_loader
