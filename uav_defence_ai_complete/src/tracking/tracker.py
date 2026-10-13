from __future__ import annotations

from typing import Iterable

import numpy as np

from .bytetrack import IoUFallbackTracker, Track
from ..detection.rtdetr import Detection


class ByteTrackAdapter:
    """Canonical ByteTrack adapter with a development fallback.

    The preferred path uses the current `trackers` package (ByteTrackTracker),
    which accepts detector-agnostic `supervision.Detections`. The fallback is
    intentionally labeled and must not be used for thesis metrics.
    """

    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 20):
        self.fallback = IoUFallbackTracker(iou_threshold, max_missed)
        self.external = None
        self.external_ok = False
        try:
            import supervision as sv
            from trackers import ByteTrackTracker
            self.sv = sv
            self.external = ByteTrackTracker()
            self.external_ok = True
        except Exception:
            self.sv = None

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        detections = list(detections)
        if not self.external_ok:
            return self.fallback.update(detections)

        if not detections:
            return []
        boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array([d.class_id for d in detections], dtype=np.int32)
        sv_dets = self.sv.Detections(xyxy=boxes, confidence=conf, class_id=cls)
        tracked = self.external.update(sv_dets)
        ids = tracked.tracker_id
        result: list[Track] = []
        if ids is None:
            return result
        for i, tid in enumerate(ids):
            if tid is None:
                continue
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else int(cls[i])
            conf_i = float(tracked.confidence[i]) if tracked.confidence is not None else float(conf[i])
            class_name = detections[i].class_name if i < len(detections) else str(class_id)
            box = tuple(float(x) for x in tracked.xyxy[i].tolist())
            result.append(Track(int(tid), box, class_id, class_name, conf_i))
        return result
