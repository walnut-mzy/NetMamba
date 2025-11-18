"""Dataset statistics and visualization helpers."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_logger


class DatasetStatistics:
    """Compute descriptive statistics for flows and labels."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger()

    def summarize(self, flows: Sequence[np.ndarray], labels: Sequence[int]) -> Dict[str, float]:
        lengths = [len(arr) for arr in flows]
        stats = {
            "num_flows": len(flows),
            "num_classes": len(set(labels)),
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "median_length": float(np.median(lengths)) if lengths else 0.0,
            "min_length": int(np.min(lengths)) if lengths else 0,
            "max_length": int(np.max(lengths)) if lengths else 0,
        }
        counts = Counter(labels)
        stats.update({f"class_{k}_count": v for k, v in counts.items()})
        self.logger.info(f"Statistics: {stats}")
        return stats

    def plot_class_distribution(self, labels: Sequence[int], path: str | None = None) -> None:
        counts = Counter(labels)
        classes, values = zip(*sorted(counts.items())) if counts else ([], [])
        plt.figure(figsize=(8, 4))
        plt.bar(classes, values)
        plt.xlabel("Class index")
        plt.ylabel("Count")
        plt.title("Class distribution")
        if path:
            plt.savefig(path, bbox_inches="tight")
        else:  # pragma: no cover - visualization side effect
            plt.show()

    def plot_length_histogram(self, flows: Sequence[np.ndarray], path: str | None = None) -> None:
        lengths = [len(arr) for arr in flows]
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=30)
        plt.xlabel("Bytes per flow")
        plt.ylabel("Frequency")
        plt.title("Flow length distribution")
        if path:
            plt.savefig(path, bbox_inches="tight")
        else:  # pragma: no cover
            plt.show()


__all__ = ["DatasetStatistics"]
