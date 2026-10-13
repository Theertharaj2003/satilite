from __future__ import annotations

from pathlib import Path
from typing import Any


def train_rtdetr(
    data_yaml: str,
    model: str = "rtdetr-l.pt",
    epochs: int = 50,
    image_size: int = 640,
    device: str = "0",
    project: str = "outputs/detection",
    name: str = "rtdetr_baseline",
    **kwargs: Any,
):
    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: pip install -r requirements.txt") from exc
    Path(project).mkdir(parents=True, exist_ok=True)
    detector = RTDETR(model)
    return detector.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        **kwargs,
    )
