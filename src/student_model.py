"""
Student network for SD-MKD: ShuffleNet v2 0.5x backbone with AgentAttention2D.

This module focuses on the lightweight student used in the third training stage
(distillation). Inputs are expected to be single-channel traffic images of shape
[B, 1, H, W]. The backbone is adapted to accept one channel by replacing the
first convolution when needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class ShuffleNetBackbone(nn.Module):
    """ShuffleNet v2 0.5x feature extractor for single-channel inputs."""

    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        base = shufflenet_v2_x0_5(weights=weights)

        # Adapt the stem to match desired number of input channels
        if in_channels != 3:
            conv1 = base.conv1[0]
            new_conv = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            if conv1.weight.shape[1] == 3:
                new_conv.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
            base.conv1[0] = new_conv

        # Recompose features into a single sequential block
        self.features = nn.Sequential(
            base.conv1,
            base.maxpool,
            base.stage2,
            base.stage3,
            base.stage4,
            base.conv5,
        )
        self.out_channels = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AgentAttention2D(nn.Module):
    """Agent attention block with aggregate and broadcast stages.

    Complexity is approximately O(N*M) where N is the number of spatial tokens
    and M is the number of agent tokens, providing a lightweight alternative to
    full self-attention.
    """

    def __init__(self, dim: int, num_heads: int = 4, num_agents: int = 8):
        super().__init__()
        self.num_agents = num_agents
        self.dim = dim
        self.agent_tokens = nn.Parameter(torch.randn(num_agents, dim))
        self.norm_x = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.attn_agg = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_broadcast = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply agent attention.

        Args:
            x: Input tensor of shape [B, C, H, W].
        Returns:
            Tensor with the same shape as ``x`` after attention and residual.
        """

        B, C, H, W = x.shape
        # Flatten spatial dimensions -> [B, N, C]
        # Using flatten keeps shape safety even if the input is non-contiguous
        X = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        X_norm = self.norm_x(X)

        # Agent tokens replicated for each batch -> [B, M, C]
        A = self.agent_tokens.unsqueeze(0).expand(B, -1, -1)
        A_norm = self.norm_a(A)

        # 1) Aggregate: agents query spatial tokens
        A_agg, _ = self.attn_agg(query=A_norm, key=X_norm, value=X_norm)

        # 2) Broadcast: spatial tokens query aggregated agents
        X_out, _ = self.attn_broadcast(query=X_norm, key=A_agg, value=A_agg)

        # 3) Residual projection
        X_out = self.proj_out(X_out)
        X_out = X_out + X

        # Reshape back to [B, C, H, W]
        X_out = X_out.permute(0, 2, 1).view(B, C, H, W)
        return X_out


class StudentNet(nn.Module):
    """ShuffleNet v2 student with AgentAttention2D and global pooling."""

    def __init__(
        self,
        num_classes: int,
        num_heads: int = 4,
        num_agents: int = 8,
        in_channels: int = 1,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = ShuffleNetBackbone(in_channels=in_channels, pretrained=pretrained_backbone)
        C = self.backbone.out_channels
        self.agent_attn = AgentAttention2D(dim=C, num_heads=num_heads, num_agents=num_agents)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # [B, C, H', W']
        feat = self.agent_attn(feat)  # [B, C, H', W']
        pooled = self.pool(feat).flatten(1)  # [B, C]
        logits = self.fc(pooled)  # [B, num_classes]
        return logits


def softmax_with_temperature(logits: torch.Tensor, T: float):
    return torch.softmax(logits / T, dim=-1)
