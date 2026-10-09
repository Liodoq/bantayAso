"""Event log: every Warning/Danger is stored in SQLite with a snapshot (all local)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
import threading
import time
from pathlib import Path

import cv2

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,            -- unix time
    level INTEGER NOT NULL,      -- 2 warning, 3 danger
    reason TEXT NOT NULL,
    action TEXT,
    near TEXT,                   -- comma-separated hazard names
    zone TEXT,
    dog TEXT,                    -- track id or name (Batch 6c)
    vlm TEXT,                    -- local VLM sentence (Batch 5)
    snapshot TEXT,               -- path to JPEG
    false_alarm INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class EventLog:
    def __init__(self, db_path: Path, snap_dir: Path):
        self.db_path = Path(db_path)
        self.snap_dir = Path(snap_dir)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=5)
        try:
            with c:
                yield c
        finally:
            c.close()

    def add(self, assessment, frame=None, dog_label: str | None = None) -> int:
        ts = time.time()
        snap = ""
        if frame is not None:
            snap = str(self.snap_dir / f"event_{time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))}_{assessment.level}.jpg")
            cv2.imwrite(snap, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO events (ts, level, reason, action, near, zone, dog, snapshot) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, assessment.level, assessment.reason, assessment.action,
                 ",".join(assessment.near or []), assessment.zone,
                 dog_label or str(assessment.track_id), snap))
            return cur.lastrowid

    def set_vlm(self, event_id: int, text: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE events SET vlm=? WHERE id=?", (text, event_id))

    def mark_false_alarm(self, event_id: int, value: bool = True) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE events SET false_alarm=? WHERE id=?", (int(value), event_id))

    def today(self, limit: int = 200) -> list[dict]:
        start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                             (start, limit)).fetchall()
        return [dict(r) for r in rows]

    def recent(self, dog=None, today=False, limit=100):
        conditions=['false_alarm = 0']; args=[]
        if dog:
            conditions.append('dog = ? COLLATE NOCASE'); args.append(dog)
        if today:
            conditions.append('ts >= ?')
            args.append(time.mktime(time.localtime()[:3]+(0,0,0,0,0,-1)))
        args.append(max(1,min(int(limit),100)))
        with self._conn() as c:
            c.row_factory=sqlite3.Row
            rows=c.execute('SELECT * FROM events WHERE '+' AND '.join(conditions)+' ORDER BY ts DESC, id DESC LIMIT ?',args).fetchall()
        return [dict(row) for row in rows]

    def for_dog_today(self, name: str) -> list[dict]:
        """Latest genuine alerts for one named dog, independent of other dogs' event volume."""
        start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM events WHERE ts >= ? AND dog = ? COLLATE NOCASE "
                             "AND false_alarm = 0 ORDER BY ts DESC LIMIT 3", (start, name)).fetchall()
        return [dict(row) for row in rows]

    def summary_today(self) -> dict:
        ev = self.today(1000)
        real = [e for e in ev if not e["false_alarm"]]
        zones = {}
        for e in real:
            if e["zone"]:
                zones[e["zone"]] = zones.get(e["zone"], 0) + 1
        by_hour = {}
        for e in real:
            hr = time.localtime(e["ts"]).tm_hour
            by_hour[hr] = by_hour.get(hr, 0) + 1
        return {"danger": sum(e["level"] == 3 for e in real),
                "warning": sum(e["level"] == 2 for e in real),
                "hotspot": max(zones, key=zones.get) if zones else None,
                "by_hour": by_hour, "total": len(real)}
