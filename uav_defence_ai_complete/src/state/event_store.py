from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, db_path: str | Path = "outputs/state/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, track_id INTEGER, event TEXT, confidence REAL, details TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS tracks (track_id INTEGER PRIMARY KEY, last_seen TEXT, class_name TEXT, confidence REAL, metadata TEXT)"
        )
        self.conn.commit()

    def add_event(self, timestamp: str, track_id: int, event: str, confidence: float, details: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO events(timestamp, track_id, event, confidence, details) VALUES (?, ?, ?, ?, ?)",
            (timestamp, track_id, event, confidence, json.dumps(details)),
        )
        self.conn.commit()

    def upsert_track(self, track_id: int, last_seen: str, class_name: str, confidence: float, metadata: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO tracks(track_id,last_seen,class_name,confidence,metadata) VALUES(?,?,?,?,?) "
            "ON CONFLICT(track_id) DO UPDATE SET last_seen=excluded.last_seen,class_name=excluded.class_name,confidence=excluded.confidence,metadata=excluded.metadata",
            (track_id, last_seen, class_name, confidence, json.dumps(metadata)),
        )
        self.conn.commit()

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp, track_id, event, confidence, details FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"timestamp": r[0], "track_id": r[1], "event": r[2], "confidence": r[3], "details": json.loads(r[4])}
            for r in rows
        ]

    def active_tracks(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT track_id,last_seen,class_name,confidence,metadata FROM tracks ORDER BY track_id").fetchall()
        return [
            {"track_id": r[0], "last_seen": r[1], "class_name": r[2], "confidence": r[3], "metadata": json.loads(r[4])}
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()
