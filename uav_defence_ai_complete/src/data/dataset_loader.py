from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_pairs import pair_images


class PairedEODataset:
    """Minimal paired EO/IR loader for registration and fusion experiments."""

    def __init__(self, eo_dir: str | Path, ir_dir: str | Path, size: tuple[int, int] | None = None):
        self.pairs = list(pair_images(eo_dir, ir_dir))
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        eo = cv2.imread(str(pair.eo), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(pair.ir), cv2.IMREAD_COLOR)
        if eo is None or ir is None:
            raise IOError(f"Could not read pair: {pair}")
        if self.size:
            eo = cv2.resize(eo, self.size, interpolation=cv2.INTER_AREA)
            ir = cv2.resize(ir, self.size, interpolation=cv2.INTER_AREA)
        return {"key": pair.key, "eo": eo, "ir": ir}
