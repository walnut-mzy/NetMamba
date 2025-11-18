"""Unified stacking teacher wrapper combining base models and fusion head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .mlp_fusion import MLPFusion
from .utils import inference_mode, softmax_probs


@dataclass
class TeacherCheckpoint:
    num_classes: int
    fusion_hidden: Tuple[int, ...]
    state_dict: Dict[str, Dict[str, torch.Tensor]]
    metadata: Dict[str, float]


class StackingTeacherModel(nn.Module):
    """Wrapper around three frozen teachers and an MLP fusion head."""

    def __init__(
        self,
        teachers: List[nn.Module],
        fusion_head: MLPFusion,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if len(teachers) != 3:
            raise ValueError("Expected exactly three teachers for stacking")
        self.teachers = nn.ModuleList(teachers)
        self.fusion = fusion_head
        if freeze:
            for t in self.teachers:
                for p in t.parameters():
                    p.requires_grad_(False)
                t.eval()

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward through three frozen teachers and the fusion head.

        Returns logits (or probabilities if ``return_probs``) to stay compatible
        with downstream validation loops that expect ``model(x) -> logits``. If
        ``return_intermediate`` is True, an additional dictionary with
        concatenated teacher probabilities and fusion logits is returned.
        """

        logits_list = [t(x) for t in self.teachers]
        probs_list = [softmax_probs(l) for l in logits_list]
        fused_logits = self.fusion(*probs_list)
        output = softmax_probs(fused_logits) if return_probs else fused_logits

        if return_intermediate:
            intermediates = {
                "teacher_probs": torch.cat(probs_list, dim=-1),
                "fusion_logits": fused_logits,
            }
            return output, intermediates
        return output

    def predict(self, loader, device: torch.device) -> torch.Tensor:
        self.to(device)
        preds: List[torch.Tensor] = []
        with inference_mode(self):
            for x, _ in loader:
                x = x.to(device)
                out = self.forward(x)
                logits = out[0] if isinstance(out, tuple) else out
                preds.append(logits.cpu())
        return torch.cat(preds, dim=0)

    def save(self, path: str, val_accs: Optional[List[float]] = None) -> None:
        payload = {
            "num_classes": self.fusion.net[-1].out_features if hasattr(self.fusion, "net") else None,
            "fusion_hidden": tuple(
                m.out_features
                for m in self.fusion.net
                if isinstance(m, nn.Linear) and m.out_features != self.fusion.net[-1].out_features
            ),
            "state_dict": {
                "fusion": self.fusion.state_dict(),
                **{f"teacher_{i}": t.state_dict() for i, t in enumerate(self.teachers)},
            },
            "metadata": {"val_acc_teacher_avg": float(sum(val_accs) / len(val_accs))} if val_accs else {},
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, teachers: List[nn.Module], fusion_head: Optional[MLPFusion] = None) -> "StackingTeacherModel":
        checkpoint = torch.load(path, map_location="cpu")
        fusion_hidden = checkpoint.get("fusion_hidden")
        if fusion_head is None:
            num_classes = checkpoint.get("num_classes")
            if num_classes is None:
                raise ValueError("num_classes missing in checkpoint")
            fusion_head = MLPFusion(num_classes=num_classes, hidden_dims=fusion_hidden)
        fusion_head.load_state_dict(checkpoint["state_dict"]["fusion"])
        for i, teacher in enumerate(teachers):
            teacher.load_state_dict(checkpoint["state_dict"][f"teacher_{i}"])
        return StackingTeacherModel(teachers, fusion_head, freeze=True)


__all__ = ["StackingTeacherModel", "TeacherCheckpoint"]
