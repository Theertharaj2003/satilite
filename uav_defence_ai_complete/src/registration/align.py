from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def align_ecc(moving: np.ndarray, fixed: np.ndarray, max_iterations: int = 100, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """Align a moving image to a fixed image using ECC affine registration.

    Returns aligned_image, warp_matrix, correlation_coefficient.
    """
    if moving is None or fixed is None:
        raise ValueError("Both images are required")
    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY) if fixed.ndim == 3 else fixed
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving
    moving_gray = cv2.resize(moving_gray, (fixed_gray.shape[1], fixed_gray.shape[0]))
    moving_resized = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]))
    matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)
    try:
        cc, matrix = cv2.findTransformECC(fixed_gray, moving_gray, matrix, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(
            moving_resized,
            matrix,
            (fixed.shape[1], fixed.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, matrix, float(cc)
    except cv2.error:
        # Fallback keeps the pipeline operational while exposing the failure to the caller.
        return moving_resized, matrix, 0.0


def save_alignment(aligned: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), aligned):
        raise IOError(f"Failed to save alignment: {path}")
    return path
