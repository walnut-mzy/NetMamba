"""Ablation experiment scaffolding for SD-MKD.

This module defines simple configurations that toggle individual distillation
components and a driver to run them sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from evaluation import evaluate_model
from student_model import StudentNet
from train import train_student_stage


@dataclass
class AblationConfig:
    name: str
    lamb_ce: float = 1.0
    lamb_f: float = 0.0
    lamb_r: float = 0.0
    lamb_s: float = 0.0
    temperature: float = 4.0


@dataclass
class AblationResult:
    config: AblationConfig
    checkpoint: str
    val_acc: Optional[float] = None
    val_f1: Optional[float] = None
    val_cm: Optional[torch.Tensor] = None


def default_ablation_configs() -> List[AblationConfig]:
    """Return a set of standard configurations.

    - baseline: CE only
    - +FKL: CE + forward KL
    - +FKL+RKL: adds reverse KL
    - full: adds Sinkhorn
    """

    return [
        AblationConfig(name="baseline", lamb_ce=1.0, lamb_f=0.0, lamb_r=0.0, lamb_s=0.0),
        AblationConfig(name="fkl", lamb_ce=1.0, lamb_f=0.5, lamb_r=0.0, lamb_s=0.0),
        AblationConfig(name="fkl_rkl", lamb_ce=1.0, lamb_f=0.5, lamb_r=0.5, lamb_s=0.0),
        AblationConfig(name="full", lamb_ce=1.0, lamb_f=0.5, lamb_r=0.5, lamb_s=0.1),
    ]


def run_ablation_suite(
    train_loader,
    val_loader,
    num_classes: int,
    device: torch.device,
    configs: Optional[Iterable[AblationConfig]] = None,
    num_epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    teacher_ckpts: Optional[Dict[str, str]] = None,
    stacking_ckpt: Optional[str] = None,
    evaluate: bool = True,
) -> List[AblationResult]:
    """Train a series of students under different distillation settings."""

    configs = list(configs) if configs is not None else default_ablation_configs()
    results: List[AblationResult] = []

    for cfg in configs:
        print(f"Running ablation: {cfg.name} -> lambdas CE={cfg.lamb_ce} F={cfg.lamb_f} R={cfg.lamb_r} S={cfg.lamb_s}")
        ckpt = train_student_stage(
            train_loader_B=train_loader,
            val_loader_B=val_loader,
            num_classes=num_classes,
            device=device,
            T=cfg.temperature,
            lamb_ce=cfg.lamb_ce,
            lamb_f=cfg.lamb_f,
            lamb_r=cfg.lamb_r,
            lamb_s=cfg.lamb_s,
            num_epochs_student=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            teacher_ckpts=teacher_ckpts,
            stacking_ckpt=stacking_ckpt,
        )

        result = AblationResult(config=cfg, checkpoint=ckpt)
        if evaluate:
            student = StudentNet(num_classes=num_classes)
            state = torch.load(ckpt, map_location="cpu")
            student.load_state_dict(state)
            student.to(device)
            acc, f1, cm = evaluate_model(student, val_loader, device)
            result.val_acc = acc
            result.val_f1 = f1
            result.val_cm = cm
            print(f"Ablation {cfg.name}: acc={acc:.4f} f1={f1:.4f}")
        results.append(result)

    return results
