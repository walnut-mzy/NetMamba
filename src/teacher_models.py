"""
Teacher models and stacking ensemble for SD-MKD.

This module defines three torchvision-based teachers, along with training
routines for individual teachers (Phase I) and the stacking ensemble (Phase II).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader


def _adjust_first_conv(module: nn.Module, in_channels: int = 1):
    """Replace the first convolution to accept a different number of channels."""

    if isinstance(module, models.ResNet):
        conv = module.conv1
        new_conv = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        if conv.weight.shape[1] == 3:
            new_conv.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
        module.conv1 = new_conv
    elif isinstance(module, models.DenseNet):
        conv = module.features.conv0
        new_conv = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        if conv.weight.shape[1] == 3:
            new_conv.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
        module.features.conv0 = new_conv
    elif isinstance(module, models.MobileNetV3):
        conv = module.features[0][0]
        new_conv = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        if conv.weight.shape[1] == 3:
            new_conv.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
        module.features[0][0] = new_conv
    else:
        raise ValueError(f"Unsupported model type: {type(module)}")


class ResNet50Teacher(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        _adjust_first_conv(base, in_channels=1)
        in_dim = base.fc.in_features
        base.fc = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class MobileNetV3LargeTeacher(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        _adjust_first_conv(base, in_channels=1)
        in_dim = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class DenseNet121Teacher(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        _adjust_first_conv(base, in_channels=1)
        in_dim = base.classifier.in_features
        base.classifier = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class TeacherEnsemble(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.t1 = ResNet50Teacher(num_classes, pretrained)
        self.t2 = MobileNetV3LargeTeacher(num_classes, pretrained)
        self.t3 = DenseNet121Teacher(num_classes, pretrained)

    @torch.no_grad()
    def forward_all(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1 = self.t1(x)
        l2 = self.t2(x)
        l3 = self.t3(x)
        return l1, l2, l3


class StackingModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        in_dim = 3 * num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits1, logits2, logits3):
        x = torch.cat([logits1, logits2, logits3], dim=-1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += (preds == y).sum().item()
    val_loss = float(sum(losses) / max(1, len(losses)))
    acc = correct / max(1, total)
    return val_loss, acc


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
    """Train a single teacher using cross-entropy."""

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        val_loss, val_acc = _evaluate(model, val_loader, device)
        yield TrainResult(epoch, train_loss, val_loss, val_acc)


def train_stacking_model(
    ensemble: TeacherEnsemble,
    stacking_model: StackingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Iterable[TrainResult]:
    """Train the stacking model on frozen teachers."""

    ensemble.eval()
    for p in ensemble.parameters():
        p.requires_grad_(False)
    ensemble.to(device)
    stacking_model.to(device)

    optimizer = torch.optim.AdamW(stacking_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        stacking_model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                l1, l2, l3 = ensemble.forward_all(x)
            logits = stacking_model(l1, l2, l3)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        val_loss, val_acc = _evaluate(lambda z: stacking_model(*ensemble.forward_all(z)), val_loader, device)  # type: ignore[arg-type]
        yield TrainResult(epoch, train_loss, val_loss, val_acc)


__all__ = [
    "ResNet50Teacher",
    "MobileNetV3LargeTeacher",
    "DenseNet121Teacher",
    "TeacherEnsemble",
    "StackingModel",
    "train_single_teacher",
    "train_stacking_model",
    "TrainResult",
]

