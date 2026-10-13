from __future__ import annotations

from typing import Any

from ..state.event_store import EventStore


class AssistantTools:
    def __init__(self, store: EventStore):
        self.store = store

    def get_scene_summary(self) -> dict[str, Any]:
        tracks = self.store.active_tracks()
        events = self.store.recent_events(10)
        classes: dict[str, int] = {}
        for t in tracks:
            classes[t["class_name"]] = classes.get(t["class_name"], 0) + 1
        return {"active_track_count": len(tracks), "classes": classes, "recent_events": events}

    def get_active_tracks(self) -> list[dict[str, Any]]:
        return self.store.active_tracks()

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.recent_events(limit)

    def get_track(self, track_id: int) -> dict[str, Any] | None:
        for t in self.store.active_tracks():
            if int(t["track_id"]) == int(track_id):
                return t
        return None
