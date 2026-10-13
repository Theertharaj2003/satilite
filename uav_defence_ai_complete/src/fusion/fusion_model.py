from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from .bifpn import WeightedBiFPN
from .cross_attention import CrossModalAttention


class CrossModalFusionNet(nn.Module):
    """Trainable EO/IR fusion adapter that outputs a 3-channel image.

    The output is intentionally image-like so it can feed the standard
    3-channel RT-DETR detector without modifying the detector API.
    """

    def __init__(self, base_channels: int = 32, heads: int = 4):
        super().__init__()
        # Downsample before attention so token count stays manageable.
        self.eo_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, stride=4, padding=2),
            nn.BatchNorm2d(base_channels), nn.SiLU(),
        )
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, stride=4, padding=2),
            nn.BatchNorm2d(base_channels), nn.SiLU(),
        )
        self.attn = CrossModalAttention(base_channels, heads)
        self.bifpn = WeightedBiFPN(base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels // 2, 3, 1), nn.Sigmoid(),
        )

    def forward(self, eo: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        eo_f = self.eo_encoder(eo)
        ir_f = self.ir_encoder(ir)
        fused = self.attn(eo_f, ir_f)
        multi = [fused, nn.functional.avg_pool2d(fused, 2), nn.functional.avg_pool2d(fused, 4)]
        out = self.bifpn(multi)
        out = self.head(out)
        out = nn.functional.interpolate(out, size=(eo.shape[-2], eo.shape[-1]), mode="bilinear", align_corners=False)
        return out


def preprocess_image(image: np.ndarray, size: int = 512) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def postprocess_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    x = (x * 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: str = "cpu") -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
