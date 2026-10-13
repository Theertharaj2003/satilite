from __future__ import annotations

import torch
from torch import nn


class WeightedBiFPN(nn.Module):
    """Compact BiFPN-style weighted fusion block for three feature scales.

    This is a research adapter rather than a drop-in replacement for every
    RT-DETR neck implementation. It is designed to be independently tested.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(3))
        self.w2 = nn.Parameter(torch.ones(3))
        self.eps = 1e-4
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def _norm_weights(self, w: torch.Tensor) -> torch.Tensor:
        return torch.relu(w) / (torch.sum(torch.relu(w)) + self.eps)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("WeightedBiFPN expects exactly three feature maps")
        target = features[0].shape[-2:]
        resized = [features[0]]
        resized.append(torch.nn.functional.interpolate(features[1], size=target, mode="bilinear", align_corners=False))
        resized.append(torch.nn.functional.interpolate(features[2], size=target, mode="bilinear", align_corners=False))
        w1 = self._norm_weights(self.w1)
        fused = w1[0] * resized[0] + w1[1] * resized[1] + w1[2] * resized[2]
        return self.conv(fused)
