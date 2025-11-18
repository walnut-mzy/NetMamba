"""Dataset and DataLoader helpers for SD-MKD."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetMeta:
    """Metadata describing a processed dataset split."""

    split: str
    num_classes: int
    height: int
    width: int


class FlowDataset(Dataset):
    """Torch dataset for byte-level flow images."""

    def __init__(self, flows: Sequence[np.ndarray], labels: Sequence[int], transform: Optional[Callable] = None, height: Optional[int] = None, width: Optional[int] = None):
        self.flows = flows
        self.labels = labels
        self.transform = transform
        self.height = height
        self.width = width

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self.labels)

    def __getitem__(self, idx: int):
        arr = self.flows[idx].astype(np.float32) / 255.0
        if arr.ndim == 1 and self.height and self.width and arr.size == self.height * self.width:
            x = torch.tensor(arr.reshape(1, self.height, self.width), dtype=torch.float32)
        elif arr.ndim == 3:
            x = torch.tensor(arr, dtype=torch.float32)
        else:
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        y = int(self.labels[idx])
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(y, dtype=torch.long)


def create_dataloaders(dataset: FlowDataset, val_ratio: float = 0.15, test_ratio: float = 0.15, batch_size: int = 512, num_workers: int = 4, pin_memory: bool = True):
    """Split a dataset into train/val/test loaders."""

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


__all__ = ["FlowDataset", "DatasetMeta", "create_dataloaders"]
