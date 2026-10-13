from __future__ import annotations

from pathlib import Path


def export_rtdetr_to_onnx(model_path: str, output_dir: str = "outputs/deployment", half: bool = False) -> Path:
    try:
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: pip install -r requirements.txt") from exc
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = RTDETR(model_path)
    exported = model.export(format="onnx", half=half, opset=17)
    return Path(str(exported))
