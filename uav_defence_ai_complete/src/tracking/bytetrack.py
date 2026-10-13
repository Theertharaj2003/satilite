from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..detection.rtdetr import Detection
from ..utils.metrics import iou_xyxy


@dataclass
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    age: int = 1
    missed: int = 0


class IoUFallbackTracker:
    """Deterministic fallback tracker for development when external trackers are unavailable.

    Research runs should use a real ByteTrack implementation for published results.
    """

    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        detections = list(detections)
        unmatched = set(range(len(detections)))
        used_tracks: set[int] = set()
        matches: list[tuple[int, int]] = []
        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_idx = None
            for idx in unmatched:
                det = detections[idx]
                if det.class_id != track.class_id:
                    continue
                score = iou_xyxy(np.asarray(track.xyxy), np.asarray(det.xyxy))
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None:
                matches.append((tid, best_idx))
                unmatched.remove(best_idx)
                used_tracks.add(tid)

        for tid, idx in matches:
            det = detections[idx]
            self.tracks[tid] = Track(tid, det.xyxy, det.class_id, det.class_name, det.confidence, self.tracks[tid].age + 1, 0)

        for tid in list(self.tracks):
            if tid not in used_tracks and all(tid != m[0] for m in matches):
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        for idx in unmatched:
            det = detections[idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, det.xyxy, det.class_id, det.class_name, det.confidence)

        return list(self.tracks.values())
