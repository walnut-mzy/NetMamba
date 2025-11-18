"""
Complete SD-MKD (Stacking-based Distillation with Multi-level Knowledge and
Sinkhorn Distances) training scaffold.

The code focuses on model structures and training logic. Data preprocessing,
logging, and hyper-parameter search are intentionally lightweight so the module
can serve as a readable reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import EncryptedFlowDataset
from loss_functions import class_cost_matrix, distillation_loss
from teacher_models import (
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
    ResNet50Teacher,
    StackingModel,
    TeacherEnsemble,
    TrainResult,
    train_single_teacher,
    train_stacking_model,
)
from student_model import StudentNet


# ---------------------------------------------------------------------------
# Student model with Agent Attention
# ---------------------------------------------------------------------------

# The reference student architecture is implemented in ``student_model.py``
# (StudentNet) and imported above for use in the distillation phase.


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def train_student(
    student: StudentNet,
    teachers: TeacherEnsemble,
    stacking: StackingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
) -> Iterable[TrainResult]:
    """Phase III: distill student from frozen teachers + stacking."""

    for p in teachers.parameters():
        p.requires_grad_(False)
    teachers.eval().to(device)
    stacking.eval().to(device)

    student.to(device)
    cost_matrix = class_cost_matrix(student.fc.out_features, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        student.train()
        batch_losses: List[float] = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                l1, l2, l3 = teachers.forward_all(x)
                teacher_logits = stacking(l1, l2, l3)

            logits = student(x)
            loss = distillation_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=y,
                T=temperature,
                lamb_ce=lamb_ce,
                lamb_f=lamb_f,
                lamb_r=lamb_r,
                lamb_s=lamb_s,
                cost_matrix=cost_matrix,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
        val_loss, val_acc = _evaluate_student(student, val_loader, device)
        yield TrainResult(epoch, train_loss, val_loss, val_acc)


def _evaluate_student(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    total = 0
    correct = 0
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


# ---------------------------------------------------------------------------
# Minimal runnable example
# ---------------------------------------------------------------------------


def _make_dummy_loader(num_samples: int, num_classes: int, height: int, width: int, batch_size: int) -> DataLoader:
    x = torch.rand(num_samples, 1, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    height = width = 32

    train_A = _make_dummy_loader(32, num_classes, height, width, batch_size=8)
    val_A = _make_dummy_loader(16, num_classes, height, width, batch_size=8)
    train_B = _make_dummy_loader(32, num_classes, height, width, batch_size=8)
    val_B = _make_dummy_loader(16, num_classes, height, width, batch_size=8)
    train_C = _make_dummy_loader(32, num_classes, height, width, batch_size=8)
    val_C = _make_dummy_loader(16, num_classes, height, width, batch_size=8)

    # Phase I: train three teachers independently
    teachers = TeacherEnsemble(num_classes=num_classes, pretrained=False)
    teacher_ckpts: List[Tuple[str, nn.Module]] = [
        ("teacher_resnet50.pth", teachers.t1),
        ("teacher_mbv3.pth", teachers.t2),
        ("teacher_densenet.pth", teachers.t3),
    ]

    for name, model in teacher_ckpts:
        print(f"Training {name}...")
        for res in train_single_teacher(model, train_A, val_A, device, epochs=1):
            print(f"  epoch {res.epoch}: train={res.train_loss:.3f} val={res.val_loss:.3f} acc={res.val_acc:.3f}")
        torch.save(model.state_dict(), name)

    # Phase II: stacking on frozen teachers
    stacking = StackingModel(num_classes=num_classes)
    print("Training stacking model...")
    for res in train_stacking_model(teachers, stacking, train_B, val_B, device, epochs=1):
        print(f"  epoch {res.epoch}: train={res.train_loss:.3f} val={res.val_loss:.3f} acc={res.val_acc:.3f}")
    torch.save(stacking.state_dict(), "stacking_model.pth")

    # Phase III: distill student
    student = StudentNet(num_classes=num_classes)
    print("Training student with SD-MKD distillation...")
    for res in train_student(student, teachers, stacking, train_C, val_C, device, epochs=1):
        print(f"  epoch {res.epoch}: train={res.train_loss:.3f} val={res.val_loss:.3f} acc={res.val_acc:.3f}")
    torch.save(student.state_dict(), "student_sd_mkd.pth")


if __name__ == "__main__":
    run_demo()

