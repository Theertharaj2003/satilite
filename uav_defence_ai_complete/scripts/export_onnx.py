from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from src.deployment.export_onnx import export_rtdetr_to_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default="outputs/deployment")
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()
    print(export_rtdetr_to_onnx(args.model, args.output_dir, args.half))


if __name__ == "__main__":
    main()
