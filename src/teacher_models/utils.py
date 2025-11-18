"""Utility helpers for teacher model training and evaluation."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


@contextmanager
def inference_mode(*modules: nn.Module) -> Iterator[None]:
    """Temporarily switch modules to eval + no_grad mode."""

    prev = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        with torch.no_grad():
            yield
    finally:
        for m, was_training in zip(modules, prev):
            m.train(was_training)


__all__ = ["freeze_module", "compute_accuracy", "softmax_probs", "inference_mode"]
