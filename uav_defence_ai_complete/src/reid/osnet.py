from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class ReIDResult:
    track_id: int
    similarity: float
    matched: bool


class OSNetAdapter:
    """Optional OSNet adapter with a deterministic appearance fallback.

    For a thesis experiment, install a supported OSNet/Torchreid stack and use
    its pretrained weights. The fallback is only for pipeline development.
    """

    def __init__(self, cosine_threshold: float = 0.55, checkpoint: str = ""):
        self.cosine_threshold = cosine_threshold
        self.model = None
        self.device = "cpu"
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        if checkpoint:
            self._try_load_osnet(checkpoint)

    def _try_load_osnet(self, checkpoint: str) -> None:
        try:
            import torchreid  # type: ignore
            self.model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
            self.model.to(self.device).eval()
        except Exception:
            self.model = None

    @staticmethod
    def _fallback_embedding(crop: np.ndarray) -> np.ndarray:
        crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm else hist

    def embedding(self, crop: np.ndarray) -> np.ndarray:
        # Keep the optional OSNet route isolated because checkpoint/preprocessing
        # conventions vary by torchreid version.
        if self.model is not None:
            try:
                import torch
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
                x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                x = x.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(x)
                feat = feat.detach().cpu().numpy().reshape(-1).astype(np.float32)
                norm = np.linalg.norm(feat)
                return feat / norm if norm else feat
            except Exception:
                pass
        return self._fallback_embedding(crop)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))
