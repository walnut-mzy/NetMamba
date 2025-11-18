"""Teacher backbone definitions with single-channel adaptation.

This module defines three ImageNet-pretrained backbones (ResNet-50,
MobileNetV3-Large, DenseNet-121) with their input stems adjusted for
single-channel flow images and classification heads resized to the target
number of classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


def _average_first_conv_weights(weight: torch.Tensor) -> torch.Tensor:
    """Average RGB weights into a single-channel kernel.

    Args:
        weight: Original convolution weight of shape [out_channels, 3, kH, kW].

    Returns:
        A weight tensor with shape [out_channels, 1, kH, kW] obtained by
        averaging over the RGB channels.
    """

    if weight.shape[1] == 1:
        return weight
    return weight.mean(dim=1, keepdim=True)


def adjust_first_conv(module: nn.Module, in_channels: int = 1) -> None:
    """Replace the first convolution of a torchvision model for 1-channel input."""

    if isinstance(module, models.ResNet):
        conv = module.conv1
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        new_conv.weight.data = _average_first_conv_weights(conv.weight.data)
        module.conv1 = new_conv
    elif isinstance(module, models.DenseNet):
        conv = module.features.conv0
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        new_conv.weight.data = _average_first_conv_weights(conv.weight.data)
        module.features.conv0 = new_conv
    elif isinstance(module, models.MobileNetV3):
        conv = module.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        new_conv.weight.data = _average_first_conv_weights(conv.weight.data)
        module.features[0][0] = new_conv
    else:
        raise ValueError(f"Unsupported model type: {type(module)}")


@dataclass
class TeacherConfig:
    """Configuration describing dataset-specific class counts."""

    dataset: Literal["ISCXVPN2016", "ISCXTor2016", "USTCTFC2016"]
    num_classes: int

    @staticmethod
    def from_dataset_name(name: str) -> "TeacherConfig":
        name_up = name.upper()
        mapping = {
            "ISCXVPN2016": 7,
            "ISCXTOR2016": 8,
            "USTCTFC2016": 19,
        }
        if name_up not in mapping:
            raise ValueError(f"Unsupported dataset: {name}")
        return TeacherConfig(dataset=name, num_classes=mapping[name_up])


class ResNet50Teacher(nn.Module):
    """ResNet-50 teacher adapted for 1-channel inputs."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        adjust_first_conv(base, in_channels=1)
        in_dim = base.fc.in_features
        base.fc = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


class MobileNetV3LargeTeacher(nn.Module):
    """MobileNetV3-Large teacher adapted for 1-channel inputs."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        adjust_first_conv(base, in_channels=1)
        in_dim = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


class DenseNet121Teacher(nn.Module):
    """DenseNet-121 teacher adapted for 1-channel inputs."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        base = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        adjust_first_conv(base, in_channels=1)
        in_dim = base.classifier.in_features
        base.classifier = nn.Linear(in_dim, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


def build_all_teachers(num_classes: int, pretrained: bool = False) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Convenience helper to instantiate all three teachers."""

    return (
        ResNet50Teacher(num_classes=num_classes, pretrained=pretrained),
        MobileNetV3LargeTeacher(num_classes=num_classes, pretrained=pretrained),
        DenseNet121Teacher(num_classes=num_classes, pretrained=pretrained),
    )


__all__ = [
    "ResNet50Teacher",
    "MobileNetV3LargeTeacher",
    "DenseNet121Teacher",
    "TeacherConfig",
    "build_all_teachers",
    "adjust_first_conv",
]
