from __future__ import annotations

import torch
from torch import nn


class FusionReconstructionLoss(nn.Module):
    """Stable starter loss for training the fusion adapter.

    It encourages the fused image to retain information from both modalities.
    Later, a detector-aware joint loss can be added when an end-to-end
    multimodal RT-DETR branch is implemented.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, fused: torch.Tensor, eo: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        mean_target = 0.5 * (eo + ir)
        max_target = torch.maximum(eo, ir)
        return self.alpha * self.l1(fused, mean_target) + (1 - self.alpha) * self.l1(fused, max_target)
