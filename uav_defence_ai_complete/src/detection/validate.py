from __future__ import annotations


def validate_rtdetr(model_path: str, data_yaml: str, image_size: int = 640, device: str = "0"):
    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: pip install -r requirements.txt") from exc
    model = RTDETR(model_path)
    return model.val(data=data_yaml, imgsz=image_size, device=device)
