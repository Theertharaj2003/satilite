from __future__ import annotations

import cv2
import numpy as np


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    a = a.astype(np.float32)
    b = cv2.resize(b, (a.shape[1], a.shape[0])).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / den) if den > 0 else 0.0
