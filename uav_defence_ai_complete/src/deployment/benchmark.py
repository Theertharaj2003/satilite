from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2


def benchmark_callable(fn: Callable, inputs: list, warmup: int = 3, repeats: int = 20) -> dict[str, float]:
    for item in inputs[:warmup]:
        fn(item)
    start = time.perf_counter()
    calls = 0
    for item in inputs[:repeats]:
        fn(item)
        calls += 1
    elapsed = time.perf_counter() - start
    return {"calls": calls, "total_seconds": elapsed, "latency_ms": elapsed / max(calls, 1) * 1000.0, "fps": calls / elapsed if elapsed else 0.0}


def load_images(folder: str, limit: int = 20):
    paths = []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = cv2.imread(str(p))
            if image is not None:
                paths.append(image)
            if len(paths) >= limit:
                break
    return paths
