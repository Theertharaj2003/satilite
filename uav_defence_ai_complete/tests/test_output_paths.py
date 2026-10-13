from __future__ import annotations

import numpy as np

import scripts.run_video as run_video


class FakeCapture:
    def __init__(self, *args, **kwargs):
        self._frame_count = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == __import__("cv2").CAP_PROP_FPS:
            return 25.0
        if prop == __import__("cv2").CAP_PROP_FRAME_WIDTH:
            return 10
        if prop == __import__("cv2").CAP_PROP_FRAME_HEIGHT:
            return 10
        return 0

    def read(self):
        if self._frame_count == 0:
            self._frame_count += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8)
        return False, None

    def release(self):
        pass


class FakeWriter:
    def __init__(self, path, *args, **kwargs):
        self.path = path
        if not __import__("pathlib").Path(path).parent.exists():
            raise RuntimeError("missing parent directory")

    def write(self, frame):
        return None

    def release(self):
        pass


class FakeStore:
    def add_event(self, *args, **kwargs):
        pass

    def upsert_track(self, *args, **kwargs):
        pass

    def close(self):
        pass


def test_run_video_creates_parent_dir_for_output(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output = tmp_path / "nested" / "tracked.mp4"

    monkeypatch.setattr(run_video.cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture())
    monkeypatch.setattr(run_video.cv2, "VideoWriter_fourcc", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_video.cv2, "VideoWriter", FakeWriter)
    class FakeModel:
        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setattr(run_video, "load_rtdetr", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(run_video, "ByteTrackAdapter", lambda *args, **kwargs: type("Tracker", (), {"update": lambda self, dets: []})())
    monkeypatch.setattr(
        run_video,
        "TrajectoryStore",
        lambda *args, **kwargs: type(
            "TrajectoryStore",
            (),
            {
                "update": lambda self, *a, **k: type(
                    "Motion",
                    (),
                    {"track_id": 1, "speed": 0.0, "direction_deg": 0.0, "duration_frames": 1},
                )()
            },
        )(),
    )
    monkeypatch.setattr(run_video, "EventDetector", lambda *args, **kwargs: type("Detector", (), {"detect": lambda self, motion, fps: []})())
    monkeypatch.setattr(run_video, "EventStore", lambda *args, **kwargs: FakeStore())
    monkeypatch.setattr(run_video, "draw_detections", lambda frame, *args, **kwargs: frame)
    monkeypatch.setattr(run_video.sys, "argv", ["run_video.py", "--source", str(source), "--output", str(output)])

    run_video.main()

    assert output.parent.exists()
