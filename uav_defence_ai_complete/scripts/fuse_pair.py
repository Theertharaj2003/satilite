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

from src.fusion.fusion_model import CrossModalFusionNet, preprocess_image, postprocess_image, load_checkpoint, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--output", default="outputs/visualizations/fused.png")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    eo = cv2.imread(args.eo)
    ir = cv2.imread(args.ir)
    if eo is None or ir is None:
        raise SystemExit("Could not read input images")
    ir = cv2.resize(ir, (eo.shape[1], eo.shape[0]), interpolation=cv2.INTER_LINEAR)
    model = CrossModalFusionNet()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.eval()
    with torch.no_grad():
        out = model(preprocess_image(eo, args.size), preprocess_image(ir, args.size))
    fused = postprocess_image(out)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), fused):
        raise IOError(f"Could not write fused image: {output_path}")
    print(f"Saved fused image: {args.output}")


if __name__ == "__main__":
    main()
