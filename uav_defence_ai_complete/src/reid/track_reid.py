from __future__ import annotations

import cv2
import numpy as np

from ..tracking.bytetrack import Track
from .matching import IdentityMatcher


def crop_track(image: np.ndarray, track: Track) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in track.xyxy]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def identify_tracks(image: np.ndarray, tracks: list[Track], matcher: IdentityMatcher) -> list[dict]:
    results = []
    for track in tracks:
        crop = crop_track(image, track)
        emb = matcher.model.embedding(crop)
        iid, score, matched = matcher.match(emb, track.track_id)
        results.append({"track_id": track.track_id, "identity_id": iid, "identity_similarity": score, "reidentified": matched})
    return results
