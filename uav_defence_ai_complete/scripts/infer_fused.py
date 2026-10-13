from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
import cv2
import torch

from src.fusion.fusion_model import CrossModalFusionNet, preprocess_image, postprocess_image, load_checkpoint
from src.detection.rtdetr import load_rtdetr, predict
from src.utils.visualization import draw_detections, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--detector", default="rtdetr-l.pt")
    ap.add_argument("--fusion-checkpoint", default="outputs/checkpoints/fusion_best.pt")
    ap.add_argument("--output", default="outputs/visualizations/fused_detection.jpg")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eo = cv2.imread(args.eo)
    ir = cv2.imread(args.ir)
    if eo is None or ir is None:
        raise SystemExit("Could not read EO/IR input")
    ir = cv2.resize(ir, (eo.shape[1], eo.shape[0]), interpolation=cv2.INTER_LINEAR)

    fusion = CrossModalFusionNet().eval()
    if Path(args.fusion_checkpoint).exists():
        load_checkpoint(fusion, args.fusion_checkpoint)
        print(f"Loaded fusion checkpoint: {args.fusion_checkpoint}")
    else:
        print("Fusion checkpoint not found; using untrained fusion adapter for smoke testing.")
    with torch.no_grad():
        fused = postprocess_image(fusion(preprocess_image(eo), preprocess_image(ir)))

    detector = load_rtdetr(args.detector)
    dets = predict(detector, fused, 0.25, 640, args.device)
    vis = draw_detections(fused, [d.xyxy for d in dets], [d.class_name for d in dets], [d.confidence for d in dets])
    save_image(vis, args.output)
    print(f"detections={len(dets)} output={args.output}")


if __name__ == "__main__":
    main()
