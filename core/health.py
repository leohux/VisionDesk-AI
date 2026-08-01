"""Engine health metrics for cockpit stability monitoring."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EngineHealth:
    capture_ok: bool = True
    capture_ms: float = 0.0
    capture_fails: int = 0
    yolo_fps: float = 0.0
    yolo_ms: float = 0.0
    deep_avg_latency_s: float | None = None
    deep_last_latency_s: float | None = None
    deep_pending: int = 0
    deep_busy: bool = False
    deep_calls: int = 0
    deep_skipped: int = 0
    deep_superseded: int = 0
    memory_ok: bool = True
    memory_msg: str = "SQLite idle"
    events: int = 0
    gpu_used_gb: float | None = None
    gpu_peak_gb: float | None = None
    frames: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "capture_ok": self.capture_ok,
            "capture_ms": round(self.capture_ms, 1),
            "capture_fails": self.capture_fails,
            "yolo_fps": round(self.yolo_fps, 1),
            "yolo_ms": round(self.yolo_ms, 1),
            "deep_avg_latency_s": None
            if self.deep_avg_latency_s is None
            else round(self.deep_avg_latency_s, 2),
            "deep_last_latency_s": None
            if self.deep_last_latency_s is None
            else round(self.deep_last_latency_s, 2),
            "deep_pending": self.deep_pending,
            "deep_busy": self.deep_busy,
            "deep_calls": self.deep_calls,
            "deep_skipped": self.deep_skipped,
            "deep_superseded": self.deep_superseded,
            "memory_ok": self.memory_ok,
            "memory_msg": self.memory_msg,
            "events": self.events,
            "gpu_used_gb": self.gpu_used_gb,
            "gpu_peak_gb": self.gpu_peak_gb,
            "frames": self.frames,
            "uptime_s": round(max(0.0, time.time() - self.started_at), 1),
        }

    def format_panel(self) -> str:
        cap = "✓ Stable" if self.capture_ok else f"✗ fails={self.capture_fails}"
        yolo = f"✓ {self.yolo_fps:.1f} FPS" if self.yolo_fps > 0 else "— idle"
        if self.deep_avg_latency_s is not None:
            deep = f"✓ Avg latency {self.deep_avg_latency_s:.1f}s"
        elif self.deep_busy:
            deep = "… running"
        else:
            deep = "— idle"
        mem = f"✓ {self.memory_msg}" if self.memory_ok else f"✗ {self.memory_msg}"
        lines = [
            "Engine Health",
            "",
            f"Capture:\n{cap}  ({self.capture_ms:.0f}ms)",
            "",
            f"YOLO:\n{yolo}",
            "",
            f"Deep:\n{deep}",
            f"  calls={self.deep_calls}  skipped={self.deep_skipped}",
            f"  superseded={self.deep_superseded}",
            "",
            f"Memory:\n{mem}",
            f"  events={self.events}",
            "",
            f"Queue:\n{self.deep_pending} pending",
        ]
        if self.gpu_used_gb is not None:
            peak = self.gpu_peak_gb if self.gpu_peak_gb is not None else self.gpu_used_gb
            lines += ["", f"GPU:\n{self.gpu_used_gb}GB (peak {peak}GB)"]
        return "\n".join(lines)


class HealthTracker:
    """Lightweight rolling health state used by the live engine / replay."""

    def __init__(self) -> None:
        self.health = EngineHealth()
        self._cap_fail_streak = 0
        self._yolo_fps_ema = 0.0
        self._deep_latencies: list[float] = []

    def reset(self) -> None:
        self.health = EngineHealth()
        self._cap_fail_streak = 0
        self._yolo_fps_ema = 0.0
        self._deep_latencies.clear()

    def note_capture(self, ok: bool, ms: float) -> None:
        self.health.capture_ms = ms
        if ok:
            self._cap_fail_streak = 0
            self.health.capture_ok = True
        else:
            self._cap_fail_streak += 1
            self.health.capture_fails += 1
            self.health.capture_ok = self._cap_fail_streak < 3

    def note_yolo(self, ms: float) -> None:
        self.health.yolo_ms = ms
        fps = 1000.0 / ms if ms > 0 else 0.0
        if self._yolo_fps_ema <= 0:
            self._yolo_fps_ema = fps
        else:
            self._yolo_fps_ema = self._yolo_fps_ema * 0.85 + fps * 0.15
        self.health.yolo_fps = self._yolo_fps_ema
        self.health.frames += 1

    def note_deep_latency(self, ms: float) -> None:
        sec = ms / 1000.0
        self._deep_latencies.append(sec)
        if len(self._deep_latencies) > 50:
            self._deep_latencies = self._deep_latencies[-50:]
        self.health.deep_last_latency_s = sec
        self.health.deep_avg_latency_s = sum(self._deep_latencies) / len(self._deep_latencies)

    def note_gpu(self, used_gb: float | None) -> None:
        if used_gb is None:
            return
        self.health.gpu_used_gb = used_gb
        peak = self.health.gpu_peak_gb
        self.health.gpu_peak_gb = used_gb if peak is None else max(peak, used_gb)

    def note_memory(self, ok: bool, msg: str, events: int = 0) -> None:
        self.health.memory_ok = ok
        self.health.memory_msg = msg
        self.health.events = events

    def sync_brain(self, brain) -> None:
        if brain is None:
            self.health.deep_pending = 0
            self.health.deep_busy = False
            return
        self.health.deep_pending = int(getattr(brain, "pending_count", 0) or 0)
        self.health.deep_busy = bool(getattr(brain, "busy", False))
        self.health.deep_calls = int(getattr(brain, "deep_calls", 0) or 0)
        self.health.deep_skipped = int(getattr(brain, "skipped_calls", 0) or 0)
        self.health.deep_superseded = int(getattr(brain, "superseded", 0) or 0)
        avg = getattr(brain, "avg_latency_ms", None)
        last = getattr(brain, "last_latency_ms", None)
        if avg is not None:
            self.health.deep_avg_latency_s = float(avg) / 1000.0
        if last is not None:
            self.health.deep_last_latency_s = float(last) / 1000.0
