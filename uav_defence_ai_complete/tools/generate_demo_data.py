from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path
import cv2
import numpy as np


def make_pair(out_eo: Path, out_ir: Path, size=(640, 384)):
    w, h = size
    rng = np.random.default_rng(42)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (28, 32, 38)
    # Ground plane and synthetic objects.
    cv2.rectangle(base, (20, 60), (w - 20, h - 20), (55, 60, 70), -1)
    for i, (x, y) in enumerate([(140, 140), (320, 200), (510, 120), (430, 290)]):
        cv2.rectangle(base, (x, y), (x + 55, y + 35), (110, 130, 150), -1)
        cv2.putText(base, str(i + 1), (x + 18, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    eo = cv2.GaussianBlur(base, (3, 3), 0)
    eo = np.clip(eo.astype(np.int16) + rng.normal(0, 7, eo.shape), 0, 255).astype(np.uint8)
    ir_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    ir_gray = cv2.normalize(ir_gray, None, 0, 255, cv2.NORM_MINMAX)
    ir = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)
    # Small geometric shift makes registration non-trivial.
    matrix = np.float32([[1, 0, 4], [0, 1, 2]])
    ir = cv2.warpAffine(ir, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    out_eo.parent.mkdir(parents=True, exist_ok=True)
    out_ir.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_eo), eo)
    cv2.imwrite(str(out_ir), ir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_pair(root / "data/raw/eo/demo_eo.png", root / "data/raw/ir/demo_ir.png")
    print("Generated demo EO/IR pair.")
