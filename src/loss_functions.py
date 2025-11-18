"""Distillation losses for SD-MKD.

This module implements the composite loss used in SD-MKD training, combining
cross-entropy, forward/reverse KL, and Sinkhorn distances between teacher and
student distributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from student_model import softmax_with_temperature


def ce_loss(logits_s: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss for student logits.

    Args:
        logits_s: Student logits of shape ``[B, C]``.
        labels: Ground-truth labels of shape ``[B]``.
    """
    return F.cross_entropy(logits_s, labels)


def forward_kl(P_t: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
    """Forward KL divergence KL(P_t || P_s).

    Both inputs are assumed to be valid probability distributions.
    """
    log_P_s = (P_s + 1e-8).log()
    return F.kl_div(log_P_s, P_t, reduction="batchmean")


def reverse_kl(P_t: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
    """Reverse KL divergence KL(P_s || P_t)."""
    log_P_t = (P_t + 1e-8).log()
    return F.kl_div(log_P_t, P_s, reduction="batchmean")


def class_cost_matrix(num_classes: int, device=None) -> torch.Tensor:
    """Squared-distance cost matrix between class indices."""
    idx = torch.arange(num_classes, device=device, dtype=torch.float32)
    i = idx.unsqueeze(0)
    j = idx.unsqueeze(1)
    return (i - j).pow(2)


def sinkhorn_distance(
    P_t: torch.Tensor,
    P_s: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Compute Sinkhorn distance between teacher and student distributions.

    Args:
        P_t: Teacher probability distributions ``[B, C]``.
        P_s: Student probability distributions ``[B, C]``.
        C: Cost matrix ``[C, C]``.
        epsilon: Entropic regularization coefficient.
        n_iters: Number of Sinkhorn iterations.
    Returns:
        Mean Sinkhorn cost over the batch.
    """
    B, C_dim = P_t.shape
    K = torch.exp(-C / epsilon)

    u = torch.ones(B, C_dim, device=P_t.device, dtype=P_t.dtype) / C_dim
    v = torch.ones(B, C_dim, device=P_t.device, dtype=P_t.dtype) / C_dim

    for _ in range(n_iters):
        Kv = torch.matmul(v, K.T)
        u = P_t / (Kv + 1e-8)
        Ku = torch.matmul(u, K)
        v = P_s / (Ku + 1e-8)

    pi = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)
    cost = (pi * C.unsqueeze(0)).sum(dim=(1, 2))
    return cost.mean()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    cost_matrix: torch.Tensor,
    T: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
) -> torch.Tensor:
    """Composite distillation loss for SD-MKD.

    Args:
        student_logits: Student logits ``[B, C]``.
        teacher_logits: Teacher logits ``[B, C]``.
        labels: Ground-truth labels ``[B]``.
        T: Softmax temperature for soft targets.
        lamb_ce: Weight for cross-entropy loss.
        lamb_f: Weight for forward KL loss.
        lamb_r: Weight for reverse KL loss.
        lamb_s: Weight for Sinkhorn loss.
        cost_matrix: Precomputed class cost matrix ``[C, C]``.
    """
    L_ce = ce_loss(student_logits, labels)

    P_t_T = softmax_with_temperature(teacher_logits, T).detach()
    P_s_T = softmax_with_temperature(student_logits, T)

    L_f = forward_kl(P_t_T, P_s_T)
    L_r = reverse_kl(P_t_T, P_s_T)
    L_s = sinkhorn_distance(P_t_T, P_s_T, cost_matrix)

    return lamb_ce * L_ce + lamb_f * L_f + lamb_r * L_r + lamb_s * L_s
