from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    boxes: Iterable[Iterable[float]],
    labels: Iterable[str],
    confidences: Iterable[float] | None = None,
    track_ids: Iterable[int] | None = None,
) -> np.ndarray:
    out = image.copy()
    confidences = list(confidences) if confidences is not None else [None] * len(list(boxes))
    boxes = list(boxes)
    labels = list(labels)
    track_ids = list(track_ids) if track_ids is not None else [None] * len(boxes)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        txt = labels[i]
        if track_ids[i] is not None:
            txt = f"ID {track_ids[i]} | {txt}"
        if confidences[i] is not None:
            txt += f" {confidences[i]:.2f}"
        cv2.putText(out, txt, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, txt, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return out


def save_image(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not write image: {path}")
    return path
