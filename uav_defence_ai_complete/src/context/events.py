from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import fabs


@dataclass
class ContextEvent:
    timestamp: str
    track_id: int
    event: str
    confidence: float
    details: dict


class EventDetector:
    def __init__(self, stop_speed: float = 1.0, direction_change: float = 60.0, dwell_seconds: float = 15.0):
        self.stop_speed = stop_speed
        self.direction_change = direction_change
        self.dwell_seconds = dwell_seconds
        self.previous_direction: dict[int, float] = {}

    def detect(self, motion, fps: float = 10.0) -> list[ContextEvent]:
        events: list[ContextEvent] = []
        ts = datetime.now(timezone.utc).isoformat()
        if motion.speed <= self.stop_speed and motion.duration_frames >= int(self.dwell_seconds * fps):
            events.append(ContextEvent(ts, motion.track_id, "long_dwell", 0.75, {"speed": motion.speed}))
        prev = self.previous_direction.get(motion.track_id)
        if prev is not None:
            diff = abs((motion.direction_deg - prev + 180.0) % 360.0 - 180.0)
            if diff >= self.direction_change:
                events.append(ContextEvent(ts, motion.track_id, "direction_change", 0.70, {"change_deg": diff}))
        self.previous_direction[motion.track_id] = motion.direction_deg
        return events
