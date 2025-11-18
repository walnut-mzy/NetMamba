"""Data validation utilities for preprocessed flows."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .utils import get_logger


class DataValidator:
    """Perform sanity checks on flows and labels."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger()

    def validate(self, flows: Sequence[np.ndarray], labels: Sequence[int], num_classes: int) -> bool:
        """Validate flow arrays and labels.

        Parameters
        ----------
        flows: Sequence[np.ndarray]
            Iterable of byte arrays or image-shaped arrays.
        labels: Sequence[int]
            Integer labels corresponding to flows.
        num_classes: int
            Expected number of classes; labels must fall in ``[0, num_classes)``.
        """

        if len(flows) != len(labels):
            raise ValueError("Flows and labels length mismatch")
        for lbl in labels:
            if not 0 <= int(lbl) < num_classes:
                raise ValueError(f"Label {lbl} outside [0, {num_classes})")
        for idx, arr in enumerate(flows):
            if arr is None or len(arr) == 0:
                self.logger.warning(f"Empty flow at index {idx}")
        return True


__all__ = ["DataValidator"]
