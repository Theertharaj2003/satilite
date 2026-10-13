from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import atan2, degrees, hypot


@dataclass
class MotionState:
    track_id: int
    x: float
    y: float
    speed: float
    direction_deg: float
    duration_frames: int


class TrajectoryStore:
    def __init__(self, history_size: int = 30):
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.frames = defaultdict(int)

    def update(self, track_id: int, xyxy, fps: float = 10.0) -> MotionState:
        x1, y1, x2, y2 = xyxy
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.history[track_id].append(point)
        self.frames[track_id] += 1
        pts = self.history[track_id]
        speed = 0.0
        direction = 0.0
        if len(pts) >= 2:
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            speed = hypot(dx, dy) * fps
            direction = (degrees(atan2(-dy, dx)) + 360.0) % 360.0
        return MotionState(track_id, point[0], point[1], speed, direction, self.frames[track_id])
