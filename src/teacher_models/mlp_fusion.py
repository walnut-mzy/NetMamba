"""MLP fusion head for stacking ensemble teachers."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


class MLPFusion(nn.Module):
    """Two-layer MLP used as stacking meta-learner.

    Args:
        num_classes: Number of target classes C.
        hidden_dims: Sequence of hidden dimensions. Defaults to (3*C, 2*C).
        dropout: Dropout probability applied after each hidden layer.
        use_batchnorm: Whether to include BatchNorm1d after each Linear.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.4,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (3 * num_classes, 2 * num_classes)
        dims: List[int] = [3 * num_classes, *hidden_dims, num_classes]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != num_classes:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, logits3: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([logits1, logits2, logits3], dim=-1)
        return self.net(concat)


StackingModel = MLPFusion  # Backward compatibility alias

__all__ = ["MLPFusion", "StackingModel"]
