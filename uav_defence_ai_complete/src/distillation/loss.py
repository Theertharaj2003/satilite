from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        t = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        ) * (t * t)
        hard = self.ce(student_logits, labels)
        return self.alpha * soft + (1 - self.alpha) * hard
