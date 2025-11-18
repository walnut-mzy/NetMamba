"""Three-stage training pipeline for SD-MKD.

This module orchestrates:
1) Teacher pretraining on dataset A.
2) Stacking ensemble training on dataset B.
3) Student distillation on dataset B using CE + FKL + RKL + Sinkhorn losses.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from loss_functions import class_cost_matrix, distillation_loss
from student_model import StudentNet
from teacher_models import (
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
    ResNet50Teacher,
    StackingModel,
    TeacherEnsemble,
    train_single_teacher,
    train_stacking_model,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _validate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += (preds == y).sum().item()
            losses.append(loss.item())
    val_loss = float(sum(losses) / max(1, len(losses)))
    val_acc = correct / max(1, total)
    return val_loss, val_acc


# ---------------------------------------------------------------------------
# Stage I: teacher pretraining
# ---------------------------------------------------------------------------


def train_teachers(
    train_loader_A: DataLoader,
    val_loader_A: DataLoader,
    num_classes: int,
    device: torch.device,
    num_epochs_teacher: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, str]:
    """Train three teachers independently and save checkpoints.

    Returns a mapping from teacher name to checkpoint path for convenience.
    """

    teachers = {
        "resnet50": ResNet50Teacher(num_classes, pretrained=False).to(device),
        "mbv3": MobileNetV3LargeTeacher(num_classes, pretrained=False).to(device),
        "densenet121": DenseNet121Teacher(num_classes, pretrained=False).to(device),
    }
    ckpt_paths: Dict[str, str] = {}
    for name, model in teachers.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for res in train_single_teacher(
            model, train_loader_A, val_loader_A, device, epochs=num_epochs_teacher, lr=lr, weight_decay=weight_decay
        ):
            print(
                f"[Teacher {name}] epoch={res.epoch} train_loss={res.train_loss:.4f} "
                f"val_loss={res.val_loss:.4f} val_acc={res.val_acc:.4f}"
            )
        path = f"{name}_teacher.pth"
        torch.save(model.state_dict(), path)
        ckpt_paths[name] = path
    return ckpt_paths


# ---------------------------------------------------------------------------
# Stage II: stacking ensemble
# ---------------------------------------------------------------------------


def train_stacking_model_stage(
    train_loader_B: DataLoader,
    val_loader_B: DataLoader,
    num_classes: int,
    device: torch.device,
    num_epochs_stacking: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    teacher_ckpts: Dict[str, str] | None = None,
) -> str:
    """Train the stacking MLP on top of frozen teachers.

    Returns the stacking checkpoint path.
    """

    # Load teachers
    t1 = ResNet50Teacher(num_classes, pretrained=False)
    t2 = MobileNetV3LargeTeacher(num_classes, pretrained=False)
    t3 = DenseNet121Teacher(num_classes, pretrained=False)
    if teacher_ckpts:
        t1.load_state_dict(torch.load(teacher_ckpts["resnet50"], map_location="cpu"))
        t2.load_state_dict(torch.load(teacher_ckpts["mbv3"], map_location="cpu"))
        t3.load_state_dict(torch.load(teacher_ckpts["densenet121"], map_location="cpu"))

    for t in (t1, t2, t3):
        t.to(device).eval()
        for p in t.parameters():
            p.requires_grad = False

    ensemble = TeacherEnsemble(num_classes=num_classes)
    ensemble.t1 = t1
    ensemble.t2 = t2
    ensemble.t3 = t3

    stacking = StackingModel(num_classes=num_classes).to(device)
    for res in train_stacking_model(
        ensemble, stacking, train_loader_B, val_loader_B, device, epochs=num_epochs_stacking, lr=lr, weight_decay=weight_decay
    ):
        print(
            f"[Stacking] epoch={res.epoch} train_loss={res.train_loss:.4f} "
            f"val_loss={res.val_loss:.4f} val_acc={res.val_acc:.4f}"
        )

    stacking_path = "stacking_model.pth"
    torch.save(stacking.state_dict(), stacking_path)
    return stacking_path


# ---------------------------------------------------------------------------
# Stage III: student distillation
# ---------------------------------------------------------------------------


def train_student_stage(
    train_loader_B: DataLoader,
    val_loader_B: DataLoader,
    num_classes: int,
    device: torch.device,
    T: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
    num_epochs_student: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    teacher_ckpts: Dict[str, str] | None = None,
    stacking_ckpt: str | None = None,
) -> str:
    """Distill the student from frozen teachers + stacking ensemble."""

    # Load frozen teachers and stacking
    t1 = ResNet50Teacher(num_classes, pretrained=False)
    t2 = MobileNetV3LargeTeacher(num_classes, pretrained=False)
    t3 = DenseNet121Teacher(num_classes, pretrained=False)
    stacking = StackingModel(num_classes=num_classes)

    if teacher_ckpts:
        t1.load_state_dict(torch.load(teacher_ckpts["resnet50"], map_location="cpu"))
        t2.load_state_dict(torch.load(teacher_ckpts["mbv3"], map_location="cpu"))
        t3.load_state_dict(torch.load(teacher_ckpts["densenet121"], map_location="cpu"))
    if stacking_ckpt:
        stacking.load_state_dict(torch.load(stacking_ckpt, map_location="cpu"))

    for m in (t1, t2, t3, stacking):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

    student = StudentNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    cost = class_cost_matrix(num_classes, device=device)

    for epoch in range(1, num_epochs_student + 1):
        student.train()
        losses = []
        for x, y in train_loader_B:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                l1 = t1(x)
                l2 = t2(x)
                l3 = t3(x)
                teacher_logits = stacking(l1, l2, l3)
            student_logits = student(x)
            loss = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=y,
                T=T,
                lamb_ce=lamb_ce,
                lamb_f=lamb_f,
                lamb_r=lamb_r,
                lamb_s=lamb_s,
                cost_matrix=cost,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(sum(losses) / max(1, len(losses)))
        val_loss, val_acc = _validate_classifier(student, val_loader_B, device)
        print(
            f"[Student] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    student_path = "student_sd_mkd.pth"
    torch.save(student.state_dict(), student_path)
    return student_path


# ---------------------------------------------------------------------------
# CLI entry (also reusable by main.py)
# ---------------------------------------------------------------------------


def _make_dummy_loader(num_samples: int, num_classes: int, height: int, width: int, batch_size: int) -> DataLoader:
    x = torch.rand(num_samples, 1, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def run_demo_pipeline(mode: str, device: torch.device):
    num_classes = 5
    h = w = 32
    batch_size = 8

    # Dummy loaders for demonstration
    train_A = _make_dummy_loader(32, num_classes, h, w, batch_size)
    val_A = _make_dummy_loader(16, num_classes, h, w, batch_size)
    train_B = _make_dummy_loader(32, num_classes, h, w, batch_size)
    val_B = _make_dummy_loader(16, num_classes, h, w, batch_size)

    teacher_ckpts: Dict[str, str] | None = None
    stacking_ckpt: str | None = None

    def _ensure_teachers() -> Dict[str, str]:
        paths = {
            "resnet50": "resnet50_teacher.pth",
            "mbv3": "mbv3_teacher.pth",
            "densenet121": "densenet121_teacher.pth",
        }
        if all(os.path.exists(p) for p in paths.values()):
            return paths
        print("[Demo] Training teachers because checkpoints were not found.")
        return train_teachers(train_A, val_A, num_classes, device)

    def _ensure_stacking(existing_teachers: Dict[str, str]) -> str:
        path = "stacking_model.pth"
        if os.path.exists(path):
            return path
        print("[Demo] Training stacking model because checkpoint was not found.")
        return train_stacking_model_stage(
            train_B,
            val_B,
            num_classes,
            device,
            teacher_ckpts=existing_teachers,
        )

    if mode == "train_teachers":
        train_teachers(train_A, val_A, num_classes, device)
    elif mode == "train_stacking":
        teacher_ckpts = _ensure_teachers()
        train_stacking_model_stage(train_B, val_B, num_classes, device, teacher_ckpts=teacher_ckpts)
    elif mode == "train_student":
        teacher_ckpts = _ensure_teachers()
        stacking_ckpt = _ensure_stacking(teacher_ckpts)
        train_student_stage(
            train_B,
            val_B,
            num_classes,
            device,
            teacher_ckpts=teacher_ckpts,
            stacking_ckpt=stacking_ckpt,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="SD-MKD three-stage training")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_student",
        choices=["train_teachers", "train_stacking", "train_student"],
        help="Which training phase to run",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_demo_pipeline(args.mode, device)


if __name__ == "__main__":
    main()
