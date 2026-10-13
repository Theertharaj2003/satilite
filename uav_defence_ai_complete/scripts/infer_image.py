from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import cv2

from src.detection.rtdetr import load_rtdetr, predict
from src.utils.visualization import draw_detections, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--output", default="outputs/visualizations/detection.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    model = load_rtdetr(args.model)
    detections = predict(model, image, args.conf, 640, args.device)
    vis = draw_detections(image, [d.xyxy for d in detections], [d.class_name for d in detections], [d.confidence for d in detections])
    save_image(vis, args.output)
    print(f"detections={len(detections)} output={args.output}")


if __name__ == "__main__":
    main()
