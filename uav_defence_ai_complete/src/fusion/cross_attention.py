from __future__ import annotations

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """Lightweight cross-modal attention block.

    Query comes from EO features; keys/values come from IR features.
    A reverse branch is added and the two outputs are fused.
    """

    def __init__(self, channels: int = 32, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.eo_norm = nn.LayerNorm(channels)
        self.ir_norm = nn.LayerNorm(channels)
        self.eo_to_ir = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ir_to_eo = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, eo: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        b, c, h, w = eo.shape
        eo_tokens = eo.flatten(2).transpose(1, 2)
        ir_tokens = ir.flatten(2).transpose(1, 2)
        eo_q = self.eo_norm(eo_tokens)
        ir_kv = self.ir_norm(ir_tokens)
        eo_attn, _ = self.eo_to_ir(eo_q, ir_kv, ir_kv, need_weights=False)
        ir_attn, _ = self.ir_to_eo(ir_kv, eo_q, eo_q, need_weights=False)
        fused = self.proj(torch.cat([eo_attn, ir_attn], dim=-1))
        fused = fused + eo_tokens + ir_tokens
        return fused.transpose(1, 2).reshape(b, c, h, w)
