from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from src.detection.train import train_rtdetr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="config/dataset.yaml")
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    train_rtdetr(args.data, args.model, args.epochs, args.imgsz, args.device)


if __name__ == "__main__":
    main()
