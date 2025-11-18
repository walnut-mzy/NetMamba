"""Compatibility shims matching the earlier teacher_models API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .mlp_fusion import MLPFusion as StackingModel
from .utils import compute_accuracy


@dataclass
class TrainResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            accs.append(compute_accuracy(logits, y))
    return float(sum(losses) / max(1, len(losses))), float(sum(accs) / max(1, len(accs)))


def train_single_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
) -> Iterable[TrainResult]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(sum(losses) / max(1, len(losses)))
        val_loss, val_acc = _evaluate(model, val_loader, device)
        yield TrainResult(epoch, train_loss, val_loss, val_acc)


def train_stacking_model(
    ensemble,
    stacking_model: StackingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Iterable[TrainResult]:
    for p in ensemble.parameters():
        p.requires_grad_(False)
    ensemble.to(device).eval()
    stacking_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(stacking_model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        stacking_model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                l1, l2, l3 = ensemble.forward_all(x)
            logits = stacking_model(l1, l2, l3)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(sum(losses) / max(1, len(losses)))
        val_loss, val_acc = _evaluate(lambda z: stacking_model(*ensemble.forward_all(z)), val_loader, device)  # type: ignore[arg-type]
        yield TrainResult(epoch, train_loss, val_loss, val_acc)


__all__ = ["TrainResult", "train_single_teacher", "train_stacking_model"]
