"""Training utilities for stacking teacher ensemble."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from .base_models import DenseNet121Teacher, MobileNetV3LargeTeacher, ResNet50Teacher
from .mlp_fusion import MLPFusion
from .utils import compute_accuracy, freeze_module, softmax_probs


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    accs: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            accs.append(compute_accuracy(logits, y))
    return float(sum(losses) / max(1, len(losses))), float(sum(accs) / max(1, len(accs)))


class BaseModelTrainer:
    """Trainer for a single teacher backbone with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        if optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> Tuple[List[EpochMetrics], nn.Module]:
        best_state = None
        best_val = float("inf")
        patience_ctr = 0
        history: List[EpochMetrics] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            losses: List[float] = []
            for x, y in tqdm(train_loader, desc=f"Teacher epoch {epoch}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            train_loss = float(sum(losses) / max(1, len(losses)))

            val_loss, val_acc = evaluate_classifier(self.model, val_loader, self.device)
            history.append(EpochMetrics(epoch, train_loss, val_loss, val_acc))

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history, self.model


def stacking_split(dataset: Dataset, split_ratio: float = 0.7, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Split dataset into D1 (for teachers) and D2 (for meta-learner)."""

    total = len(dataset)
    len_d1 = int(total * split_ratio)
    len_d2 = total - len_d1
    generator = torch.Generator().manual_seed(seed)
    d1, d2 = random_split(dataset, [len_d1, len_d2], generator=generator)
    return d1, d2


def generate_meta_features(
    teachers: Sequence[nn.Module],
    loader: DataLoader,
    device: torch.device,
    apply_softmax: bool = True,
) -> TensorDataset:
    """Generate Meta-D2 dataset by concatenating teachers' predictions."""

    for t in teachers:
        freeze_module(t)
        t.to(device)

    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Generating meta features", leave=False):
            x = x.to(device)
            preds = [t(x) for t in teachers]
            if apply_softmax:
                preds = [softmax_probs(p) for p in preds]
            concat = torch.cat(preds, dim=-1).cpu()
            features.append(concat)
            labels.append(y)
    return TensorDataset(torch.cat(features, dim=0), torch.cat(labels, dim=0))


def _fusion_forward_from_concat(fusion: MLPFusion, feats: torch.Tensor) -> torch.Tensor:
    chunks = torch.chunk(feats, 3, dim=-1)
    return fusion(*chunks)


def train_mlp_fusion(
    fusion: MLPFusion,
    meta_train_loader: DataLoader,
    meta_val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> List[EpochMetrics]:
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    fusion.to(device)

    history: List[EpochMetrics] = []
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        fusion.train()
        losses: List[float] = []
        for feats, y in tqdm(meta_train_loader, desc=f"Fusion epoch {epoch}", leave=False):
            feats, y = feats.to(device), y.to(device)
            optimizer.zero_grad()
            logits = _fusion_forward_from_concat(fusion, feats)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(sum(losses) / max(1, len(losses)))

        val_loss, val_acc = evaluate_classifier(
            lambda z: _fusion_forward_from_concat(fusion, z), meta_val_loader, device
        )  # type: ignore[arg-type]
        history.append(EpochMetrics(epoch, train_loss, val_loss, val_acc))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in fusion.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        fusion.load_state_dict(best_state)
    return history


def train_stacking_ensemble(
    dataset: Dataset,
    num_classes: int,
    device: torch.device,
    split_ratio: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 2,
    base_epochs: int = 50,
    fusion_epochs: int = 30,
    patience: int = 8,
    lr_base: float = 1e-3,
    lr_fusion: float = 1e-3,
) -> Tuple[List[nn.Module], MLPFusion]:
    """Complete stacking pipeline: train base teachers then the fusion head."""

    d1, d2 = stacking_split(dataset, split_ratio=split_ratio)
    train_len = int(len(d1) * 0.9)
    val_len = len(d1) - train_len
    d1_train, d1_val = random_split(d1, [train_len, val_len])
    train_loader = DataLoader(d1_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(d1_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    teachers = [ResNet50Teacher(num_classes), MobileNetV3LargeTeacher(num_classes), DenseNet121Teacher(num_classes)]
    trained_teachers: List[nn.Module] = []
    for teacher in teachers:
        trainer = BaseModelTrainer(teacher, device=device, lr=lr_base, patience=patience)
        _history, model = trainer.train(train_loader, val_loader, epochs=base_epochs)
        trained_teachers.append(model)

    meta_loader = DataLoader(d2, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meta_dataset = generate_meta_features(trained_teachers, meta_loader, device=device)
    meta_len = len(meta_dataset)
    val_meta_len = max(1, int(meta_len * 0.2))
    train_meta_len = meta_len - val_meta_len
    meta_train, meta_val = random_split(meta_dataset, [train_meta_len, val_meta_len])
    meta_train_loader = DataLoader(meta_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    meta_val_loader = DataLoader(meta_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    fusion = MLPFusion(num_classes=num_classes)
    train_mlp_fusion(
        fusion,
        meta_train_loader,
        meta_val_loader,
        device=device,
        epochs=fusion_epochs,
        lr=lr_fusion,
        patience=patience,
    )

    return trained_teachers, fusion


__all__ = [
    "BaseModelTrainer",
    "EpochMetrics",
    "evaluate_classifier",
    "generate_meta_features",
    "stacking_split",
    "train_mlp_fusion",
    "train_stacking_ensemble",
]
