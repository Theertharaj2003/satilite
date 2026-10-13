from pathlib import Path
from src.state.event_store import EventStore


def test_state_store(tmp_path: Path):
    db = tmp_path / "events.db"
    store = EventStore(db)
    store.upsert_track(1, "now", "vehicle", 0.9, {"x": 1})
    store.add_event("now", 1, "test", 0.8, {})
    assert len(store.active_tracks()) == 1
    assert len(store.recent_events()) == 1
    store.close()
