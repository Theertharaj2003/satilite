from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from datetime import datetime, timezone
import cv2

from src.detection.rtdetr import load_rtdetr, predict
from src.tracking.tracker import ByteTrackAdapter
from src.context.trajectory import TrajectoryStore
from src.context.events import EventDetector
from src.confidence.confidence import combined_confidence
from src.confidence.review_rules import review_required, review_reason
from src.state.event_store import EventStore
from src.utils.visualization import draw_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--output", default="outputs/predictions/tracked.mp4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = load_rtdetr(args.model)
    tracker = ByteTrackAdapter()
    trajectories = TrajectoryStore(history_size=30)
    events = EventDetector()
    store = EventStore()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = predict(model, frame, args.conf, 640, args.device)
        tracks = tracker.update(dets)
        boxes, labels, confs, ids = [], [], [], []
        now = datetime.now(timezone.utc).isoformat()
        for t in tracks:
            motion = trajectories.update(t.track_id, t.xyxy, fps)
            for event in events.detect(motion, fps):
                store.add_event(event.timestamp, event.track_id, event.event, event.confidence, event.details)
            combined = combined_confidence(t.confidence)
            store.upsert_track(t.track_id, now, t.class_name, combined, {"speed_px_s": motion.speed, "direction_deg": motion.direction_deg, "review": review_required(combined)})
            boxes.append(t.xyxy)
            label = f"{t.class_name} | {'REVIEW' if review_required(combined) else 'OK'}"
            labels.append(label)
            confs.append(combined)
            ids.append(t.track_id)
        vis = draw_detections(frame, boxes, labels, confs, ids)
        out.write(vis)

    cap.release()
    out.release()
    store.close()
    print(f"Saved tracked video: {args.output}")


if __name__ == "__main__":
    main()
