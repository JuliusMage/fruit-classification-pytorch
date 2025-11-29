# src/dataset.py
"""
Dataset helpers for fruit-classification-pytorch.

Expect the data folder to be structured as:
data/
  cherry/
    img001.jpg
    ...
  tomato/
  strawberry/

Functions:
 - get_dataloaders(data_dir, img_size, batch_size, val_split, seed)
"""

import os
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEFAULT_MEAN = [0.5469, 0.4078, 0.3354]
DEFAULT_STD = [0.2291, 0.2381, 0.2302]


def set_seed(seed: int = 309):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int = 300, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        ])


def get_dataloaders(data_dir: str,
                    img_size: int = 300,
                    batch_size: int = 32,
                    val_split: float = 0.2,
                    seed: int = 309,
                    num_workers: int = 4
                    ) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create training and validation dataloaders from ImageFolder dataset.

    Returns (train_loader, val_loader, class_to_idx)
    """
    assert os.path.isdir(data_dir), f"data_dir not found: {data_dir}"

    set_seed(seed)

    # Load full dataset with train transforms initially — we'll split then apply transforms per subset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

    # Compute split sizes
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Data directory too small or val_split too large.")

    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Set transforms for each subset
    train_ds.dataset.transform = get_transforms(img_size=img_size, train=True)
    val_ds.dataset.transform = get_transforms(img_size=img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_to_idx = full_dataset.class_to_idx

    return train_loader, val_loader, class_to_idx
