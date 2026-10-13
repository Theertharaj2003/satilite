from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

from src.detection.rtdetr import load_rtdetr, predict
from src.deployment.benchmark import benchmark_callable, load_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="data/processed/images/val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--output", default="outputs/reports/benchmark.json")
    args = ap.parse_args()
    model = load_rtdetr(args.model)
    images = load_images(args.images, args.repeats)
    if not images:
        raise SystemExit("No benchmark images found")
    result = benchmark_callable(lambda image: predict(model, image, 0.25, 640, args.device), images, repeats=min(args.repeats, len(images)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
