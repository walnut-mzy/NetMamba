"""Data splitting utilities for SD-MKD preprocessing."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def train_val_test_split(indices: Sequence[int], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split indices into train/val/test subsets."""

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    n_total = len(indices)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]
    return train_idx, val_idx, test_idx


def stacking_split(indices: Sequence[int], stacking_ratio: float = 0.3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split training indices into base-model and stacking subsets."""

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    split = int(len(indices) * (1 - stacking_ratio))
    return indices[:split], indices[split:]


def generate_kfold_splits(indices: Sequence[int], labels: Sequence[int], k: int = 5, stratified: bool = True, seed: int = 42):
    """Yield k-fold train/val index pairs, optionally stratified."""

    if stratified:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(indices, labels):
            yield np.array(indices)[train_idx], np.array(indices)[val_idx]
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(indices):
            yield np.array(indices)[train_idx], np.array(indices)[val_idx]


__all__ = ["train_val_test_split", "stacking_split", "generate_kfold_splits"]
