"""Tiny-ImageNet data utilities.

Supports the standard Tiny-ImageNet layout:

    tiny-imagenet-200/
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG
      val/val_annotations.txt
      wnids.txt

The validation loader reads labels from val_annotations.txt directly, so the
validation images do not need to be reorganized into class subfolders.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyImageNetValDataset(Dataset):
    """Validation dataset for the standard Tiny-ImageNet layout."""

    def __init__(self, root: str, class_to_idx: dict[str, int], transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        annotations_path = os.path.join(root, "val", "val_annotations.txt")
        images_dir = os.path.join(root, "val", "images")

        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"Tiny-ImageNet validation annotations not found: {annotations_path}"
            )
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Tiny-ImageNet validation images directory not found: {images_dir}"
            )

        self.samples: list[tuple[str, int]] = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                image_name, wnid, *_ = line.strip().split("\t")
                if wnid not in class_to_idx:
                    continue
                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    self.samples.append((image_path, class_to_idx[wnid]))

        if not self.samples:
            raise RuntimeError(
                "No Tiny-ImageNet validation samples were loaded. "
                "Check the dataset path and standard directory layout."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _resolve_dataset_root(data_root: str) -> str:
    """Allow passing either the dataset root or its parent directory."""
    if os.path.basename(os.path.normpath(data_root)) == "tiny-imagenet-200":
        root = data_root
    else:
        candidate = os.path.join(data_root, "tiny-imagenet-200")
        root = candidate if os.path.isdir(candidate) else data_root

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            "Tiny-ImageNet dataset not found. Expected either "
            f"'{data_root}/tiny-imagenet-200' or '{data_root}' to contain train/ and val/."
        )
    return root


def _classification_collate_fn() -> Callable:
    def collate_fn(batch):
        images, labels = zip(*batch)
        return {
            "images": torch.stack(images),
            "targets": torch.tensor(labels),
            "task": "classify",
        }

    return collate_fn


def get_tiny_imagenet_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    data_root: str = "./data/tiny-imagenet-200",
    image_size: int = 64,
):
    root = _resolve_dataset_root(data_root)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(image_size + 8),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=transform_train)
    val_ds = TinyImageNetValDataset(root, class_to_idx=train_ds.class_to_idx, transform=transform_val)

    pin = torch.cuda.is_available() and not hasattr(torch, '_xla_device')
    collate_fn = _classification_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    return train_loader, val_loader
