"""Lightweight ensemble wrapper for three teachers."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .base_models import build_all_teachers
from .utils import freeze_module


class TeacherEnsemble(nn.Module):
    """Wrap three teachers and expose `forward_all` returning logits."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        self.t1, self.t2, self.t3 = build_all_teachers(num_classes, pretrained)

    @torch.no_grad()
    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.t1(x), self.t2(x), self.t3(x)

    def freeze(self) -> None:
        for m in (self.t1, self.t2, self.t3):
            freeze_module(m)


__all__ = ["TeacherEnsemble"]
