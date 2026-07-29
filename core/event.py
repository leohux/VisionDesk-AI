"""SQLite event memory — stable schema for UI / API / analytics / demo."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from core.types import DetectionResult

# Canonical columns (plus legacy aliases kept for older rows)
SCHEMA_COLS = [
    "id",
    "timestamp",
    "profile",
    "source",
    "object_class",
    "confidence",
    "trigger_reason",
    "ai_summary",
    "frame_path",
    "latency_ms",
    "backend",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",
    "meta",
]


class EventStore:
    def __init__(self, db_path: str | Path, snapshot_dir: str | Path | None = None) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else self.path.parent / "event_frames"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                profile TEXT,
                source TEXT,
                object_class TEXT,
                confidence REAL,
                trigger_reason TEXT,
                ai_summary TEXT,
                frame_path TEXT,
                latency_ms REAL,
                backend TEXT,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                track_id INTEGER,
                meta TEXT
            )
            """
        )
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)").fetchall()}

        # Migrate very old schema (ts/label/summary/snapshot)
        migrations = {
            "timestamp": "REAL",
            "profile": "TEXT",
            "source": "TEXT",
            "object_class": "TEXT",
            "confidence": "REAL",
            "trigger_reason": "TEXT",
            "ai_summary": "TEXT",
            "frame_path": "TEXT",
            "latency_ms": "REAL",
            "backend": "TEXT",
            "x1": "REAL",
            "y1": "REAL",
            "x2": "REAL",
            "y2": "REAL",
            "track_id": "INTEGER",
            "meta": "TEXT",
            # legacy kept if present
            "ts": "REAL",
            "label": "TEXT",
            "summary": "TEXT",
            "snapshot": "TEXT",
        }
        for name, typ in migrations.items():
            if name not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {typ}")
                except sqlite3.OperationalError:
                    pass

        # Backfill canonical columns from legacy names when empty
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)").fetchall()}
        if "ts" in cols and "timestamp" in cols:
            self._conn.execute(
                "UPDATE events SET timestamp = ts WHERE timestamp IS NULL AND ts IS NOT NULL"
            )
        if "label" in cols and "object_class" in cols:
            self._conn.execute(
                "UPDATE events SET object_class = label WHERE object_class IS NULL AND label IS NOT NULL"
            )
        if "summary" in cols and "ai_summary" in cols:
            self._conn.execute(
                "UPDATE events SET ai_summary = summary WHERE ai_summary IS NULL AND summary IS NOT NULL"
            )
        if "snapshot" in cols and "frame_path" in cols:
            self._conn.execute(
                "UPDATE events SET frame_path = snapshot WHERE frame_path IS NULL AND snapshot IS NOT NULL"
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def ping(self) -> bool:
        try:
            self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def total_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(row[0] if row else 0)

    def log_result(
        self,
        result: DetectionResult,
        summary: str | None = None,
        frame_bgr: np.ndarray | None = None,
        save_snapshot: bool = False,
        *,
        profile: str | None = None,
        source: str = "screen",
        trigger_reason: str | None = None,
        latency_ms: float | None = None,
    ) -> int | None:
        ts = time.time()
        text = summary if summary is not None else result.summary
        reason = trigger_reason or (result.extras or {}).get("reason")
        lat = latency_ms
        if lat is None:
            lat = (result.extras or {}).get("latency_ms", result.infer_ms)

        snap_path = None
        if save_snapshot and frame_bgr is not None and frame_bgr.size:
            name = time.strftime("%H%M%S", time.localtime(ts)) + f"_{int((ts % 1) * 1000):03d}.jpg"
            snap_file = self.snapshot_dir / name
            cv2.imwrite(str(snap_file), frame_bgr)
            snap_path = str(snap_file)

        meta = json.dumps(result.extras or {}, ensure_ascii=False)
        rows = []
        boxes = result.boxes or [None]
        for b in boxes:
            if b is None:
                rows.append(
                    (
                        ts,
                        profile,
                        source,
                        None,
                        None,
                        reason,
                        text,
                        snap_path,
                        lat,
                        result.backend,
                        None,
                        None,
                        None,
                        None,
                        None,
                        meta,
                        # legacy mirrors
                        ts,
                        None,
                        text,
                        snap_path,
                    )
                )
            else:
                x1, y1, x2, y2 = b.as_xyxy()
                rows.append(
                    (
                        ts,
                        profile,
                        source,
                        b.label,
                        b.confidence,
                        reason,
                        text,
                        snap_path,
                        lat,
                        result.backend,
                        x1,
                        y1,
                        x2,
                        y2,
                        b.track_id,
                        meta,
                        ts,
                        b.label,
                        text,
                        snap_path,
                    )
                )

        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)").fetchall()}
        # Prefer writing both canonical + legacy when legacy columns exist
        if {"ts", "label", "summary", "snapshot"}.issubset(cols):
            self._conn.executemany(
                """
                INSERT INTO events
                (timestamp, profile, source, object_class, confidence, trigger_reason,
                 ai_summary, frame_path, latency_ms, backend, x1, y1, x2, y2, track_id, meta,
                 ts, label, summary, snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        else:
            slim = [r[:16] for r in rows]
            self._conn.executemany(
                """
                INSERT INTO events
                (timestamp, profile, source, object_class, confidence, trigger_reason,
                 ai_summary, frame_path, latency_ms, backend, x1, y1, x2, y2, track_id, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                slim,
            )
        self._conn.commit()
        cur = self._conn.execute("SELECT last_insert_rowid()")
        return int(cur.fetchone()[0])

    def counts_since(self, since_ts: float) -> dict[str, int]:
        cur = self._conn.execute(
            """
            SELECT COALESCE(object_class, label), COUNT(*) FROM events
            WHERE COALESCE(timestamp, ts) >= ?
              AND COALESCE(object_class, label) IS NOT NULL
            GROUP BY COALESCE(object_class, label)
            ORDER BY COUNT(*) DESC
            """,
            (since_ts,),
        )
        return {str(k): int(v) for k, v in cur.fetchall()}

    def recent(self, limit: int = 20) -> list[dict]:
        cur = self._conn.execute(
            """
            SELECT
                id,
                COALESCE(timestamp, ts) AS timestamp,
                profile,
                source,
                COALESCE(object_class, label) AS object_class,
                confidence,
                trigger_reason,
                COALESCE(ai_summary, summary) AS ai_summary,
                COALESCE(frame_path, snapshot) AS frame_path,
                latency_ms,
                backend
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        keys = [
            "id",
            "ts",
            "profile",
            "source",
            "label",
            "confidence",
            "trigger_reason",
            "summary",
            "snapshot",
            "latency_ms",
            "backend",
        ]
        return [dict(zip(keys, row)) for row in cur.fetchall()]

    def get_snapshot(self, event_id: int) -> str | None:
        row = self._conn.execute(
            """
            SELECT COALESCE(frame_path, snapshot) FROM events WHERE id=?
            """,
            (event_id,),
        ).fetchone()
        if not row or not row[0]:
            return None
        p = Path(row[0])
        return str(p) if p.exists() else None
