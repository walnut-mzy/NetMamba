"""Label encoding utilities for encrypted traffic datasets."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


class LabelEncoder:
    """Bidirectional mapping between string labels and integer indices."""

    def __init__(self, labels: Sequence[str]):
        unique = sorted(set(labels))
        self.label_to_index: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique)}
        self.index_to_label: Dict[int, str] = {i: lbl for lbl, i in self.label_to_index.items()}

    def encode(self, labels: Sequence[str]) -> List[int]:
        return [self.label_to_index[lbl] for lbl in labels]

    def decode(self, indices: Sequence[int]) -> List[str]:
        return [self.index_to_label[int(idx)] for idx in indices]

    def get_num_classes(self) -> int:
        return len(self.label_to_index)


def calculate_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""

    counts = Counter(labels)
    num_classes = max(labels) + 1 if labels else 0
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


__all__ = ["LabelEncoder", "calculate_class_weights"]
