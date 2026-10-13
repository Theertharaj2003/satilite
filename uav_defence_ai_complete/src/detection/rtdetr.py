from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def load_rtdetr(model_path: str = "rtdetr-l.pt"):
    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return RTDETR(model_path)


def predict(model: Any, image, confidence: float = 0.25, image_size: int = 640, device: str | int = "cpu") -> list[Detection]:
    results = model.predict(image, conf=confidence, imgsz=image_size, device=device, verbose=False)
    if not results:
        return []
    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return []
    names = result.names
    detections: list[Detection] = []
    for box in boxes:
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cid = int(box.cls[0])
        detections.append(Detection(xyxy, conf, cid, str(names.get(cid, cid))))
    return detections
