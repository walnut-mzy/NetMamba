"""Entry point to run SD-MKD training and evaluation phases."""

from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from student_model import StudentNet
from train import run_demo_pipeline


def _make_dummy_loader(num_samples: int, num_classes: int, height: int, width: int, batch_size: int) -> DataLoader:
    x = torch.rand(num_samples, 1, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def evaluate_student(model_path: str, num_classes: int, device: torch.device) -> None:
    """Evaluate a saved student checkpoint on dummy data as a placeholder."""

    h = w = 32
    loader = _make_dummy_loader(32, num_classes, h, w, batch_size=8)
    student = StudentNet(num_classes=num_classes)
    student.load_state_dict(torch.load(model_path, map_location="cpu"))
    student.to(device).eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = student(x)
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += (preds == y).sum().item()
    acc = correct / max(1, total)
    print(f"[Eval] accuracy={acc:.4f} on dummy data")


def main():
    parser = argparse.ArgumentParser(description="SD-MKD runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_student",
        choices=["train_teachers", "train_stacking", "train_student", "eval"],
        help="Which pipeline stage to execute",
    )
    parser.add_argument("--student_ckpt", type=str, default="student_sd_mkd.pth", help="Student checkpoint for eval mode")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in {"train_teachers", "train_stacking", "train_student"}:
        run_demo_pipeline(args.mode, device)
    elif args.mode == "eval":
        evaluate_student(args.student_ckpt, num_classes=5, device=device)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
