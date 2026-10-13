from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
import cv2

from src.registration.align import align_ecc, save_alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--output", default="data/registered/aligned_ir.png")
    args = ap.parse_args()
    eo = cv2.imread(args.eo, cv2.IMREAD_COLOR)
    ir = cv2.imread(args.ir, cv2.IMREAD_COLOR)
    if eo is None or ir is None:
        raise SystemExit("Could not read EO or IR image")
    aligned, _, cc = align_ecc(ir, eo)
    save_alignment(aligned, args.output)
    print(f"Saved aligned IR: {args.output}")
    print(f"ECC correlation: {cc:.4f}")


if __name__ == "__main__":
    main()
