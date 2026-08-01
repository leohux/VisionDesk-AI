"""Dual-brain router: cheap YOLO scan + smart deep VLM deepen.

Stability: at most one pending deepen job; newer requests overwrite older ones.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from core.types import Box, DetectionResult


@dataclass
class DualBrainConfig:
    """Smart deepen: only call 3B when YOLO is uncertain / new / priority."""

    trigger_labels: set[str] = field(
        default_factory=lambda: {"person", "car", "truck", "gun", "rifle", "backpack"}
    )
    skip_above_conf: float = 0.90
    uncertain_below_conf: float = 0.65
    force_labels: set[str] = field(default_factory=lambda: {"gun", "rifle", "drone", "knife"})
    min_conf: float = 0.25
    cooldown_sec: float = 2.5
    max_roi_boxes: int = 3
    pad_ratio: float = 0.08
    on_new_label: bool = True
    # Prevent 3B pile-up / VRAM blowups — newest request wins
    max_pending_jobs: int = 1


class DualBrain:
    def __init__(self, deep_detector, config: DualBrainConfig | None = None) -> None:
        self.deep = deep_detector
        self.cfg = config or DualBrainConfig()
        self._lock = threading.Lock()
        self._busy = False
        self._last_trigger = 0.0
        self._last_label_sig: frozenset[str] = frozenset()
        self._pending: tuple | None = None  # (frame, seeds, yolo_summary, reason)
        self.last_deep: DetectionResult | None = None
        self.last_narrative: str = ""
        self.last_reason: str = ""
        self.deep_calls = 0
        self.skipped_calls = 0
        self.superseded = 0
        self.last_latency_ms: float | None = None
        self.avg_latency_ms: float | None = None
        self._latencies: list[float] = []
        self.reason_counts: dict[str, int] = {}

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def pending_count(self) -> int:
        with self._lock:
            return 1 if self._pending is not None else 0

    def maybe_trigger(self, frame_bgr: np.ndarray, yolo: DetectionResult) -> str | None:
        """
        Schedule deepen if needed.
        When busy: keep at most max_pending_jobs (default 1), overwrite older pending.
        """
        if self.deep is None:
            return None

        reason, seeds = self._select_seeds(yolo)
        if not seeds:
            self.skipped_calls += 1
            return None

        job = (frame_bgr.copy(), list(seeds), yolo.summary, reason)
        now = time.time()

        with self._lock:
            if self._busy:
                if self.cfg.max_pending_jobs <= 0:
                    self.skipped_calls += 1
                    return None
                if self._pending is not None:
                    self.superseded += 1
                self._pending = job
                return f"queued:{reason}"

            if now - self._last_trigger < self.cfg.cooldown_sec:
                self.skipped_calls += 1
                return None

            self._launch_unlocked(job)
            return reason

    def _launch_unlocked(self, job: tuple) -> None:
        self._busy = True
        self._pending = None
        self._last_trigger = time.time()
        self.last_reason = job[3]
        self._bump_reasons(job[3])
        threading.Thread(target=self._run_deep, args=job, daemon=True).start()

    def _bump_reasons(self, reason: str) -> None:
        for part in str(reason).split(","):
            kind = part.strip().split(":", 1)[0]
            if kind:
                self.reason_counts[kind] = self.reason_counts.get(kind, 0) + 1

    def reason_report(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self.reason_counts.items(), key=lambda kv: -kv[1]))

    def _select_seeds(self, yolo: DetectionResult) -> tuple[str, list[Box]]:
        cfg = self.cfg
        cands: list[tuple[str, Box]] = []
        labels_now = {b.label.lower() for b in yolo.boxes}

        for b in yolo.boxes:
            lab = b.label.lower()
            conf = float(b.confidence)
            if conf < cfg.min_conf:
                continue

            if lab in cfg.force_labels:
                cands.append((f"force:{lab}", b))
                continue

            if lab not in cfg.trigger_labels:
                continue

            if conf >= cfg.skip_above_conf:
                continue

            if conf <= cfg.uncertain_below_conf:
                cands.append((f"uncertain:{lab}@{conf:.2f}", b))
                continue

            if cfg.on_new_label and lab not in self._last_label_sig:
                cands.append((f"new:{lab}", b))

        if not cands and cfg.on_new_label:
            new_labs = labels_now & cfg.trigger_labels
            appeared = new_labs - self._last_label_sig
            if appeared:
                for b in yolo.boxes:
                    if b.label.lower() in appeared and b.confidence >= cfg.min_conf:
                        cands.append((f"appeared:{b.label}", b))

        if not cands:
            return "", []

        cands.sort(key=lambda x: x[1].area(), reverse=True)
        cands = cands[: cfg.max_roi_boxes]
        reason = ", ".join(dict.fromkeys(r for r, _ in cands))
        seeds = [b for _, b in cands]
        return reason, seeds

    def _run_deep(
        self, frame: np.ndarray, seeds: list[Box], yolo_summary: str, reason: str
    ) -> None:
        t0 = time.perf_counter()
        try:
            h, w = frame.shape[:2]
            x1 = min(b.x1 for b in seeds)
            y1 = min(b.y1 for b in seeds)
            x2 = max(b.x2 for b in seeds)
            y2 = max(b.y2 for b in seeds)
            pw, ph = (x2 - x1) * self.cfg.pad_ratio, (y2 - y1) * self.cfg.pad_ratio
            rx1 = int(max(0, x1 - pw))
            ry1 = int(max(0, y1 - ph))
            rx2 = int(min(w, x2 + pw))
            ry2 = int(min(h, y2 + ph))
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return
            # Optional context for narrators (e.g. Mage-VL); LocateAnything ignores it.
            if hasattr(self.deep, "set_context"):
                try:
                    self.deep.set_context(yolo_summary=yolo_summary, reason=reason)
                except Exception:
                    pass
            deep = self.deep.predict(roi)
            mapped = [
                Box(
                    x1=b.x1 + rx1,
                    y1=b.y1 + ry1,
                    x2=b.x2 + rx1,
                    y2=b.y2 + ry1,
                    label=b.label,
                    confidence=b.confidence,
                )
                for b in deep.boxes
            ]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            backend_tag = getattr(self.deep, "backend_name", None) or deep.backend or "deep"
            narrative = (
                f"[{backend_tag} trigger: {reason}] {yolo_summary} → "
                f"ROI ({rx1},{ry1})-({rx2},{ry2}): {deep.summary}"
            )
            result = DetectionResult(
                boxes=mapped,
                summary=narrative,
                backend=deep.backend,
                infer_ms=deep.infer_ms,
                extras={
                    "roi": [rx1, ry1, rx2, ry2],
                    "reason": reason,
                    "latency_ms": latency_ms,
                    "raw_preview": deep.extras.get("raw_preview", ""),
                },
            )
            with self._lock:
                self.last_deep = result
                self.last_narrative = narrative
                self.last_reason = reason
                self.deep_calls += 1
                self.last_latency_ms = latency_ms
                self._latencies.append(latency_ms)
                if len(self._latencies) > 50:
                    self._latencies = self._latencies[-50:]
                self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
                self._last_label_sig = frozenset(b.label.lower() for b in seeds)
        except Exception as e:
            with self._lock:
                self.last_narrative = f"Deep check failed: {e}"
        finally:
            with self._lock:
                nxt = self._pending
                self._pending = None
                if nxt is not None:
                    # Stay busy and immediately run the latest overwritten job
                    self._launch_unlocked(nxt)
                else:
                    self._busy = False

    def snapshot_deep(self) -> tuple[DetectionResult | None, str, str]:
        with self._lock:
            return self.last_deep, self.last_narrative, self.last_reason

    def wait_idle(self, timeout: float = 120.0) -> bool:
        """Block until no busy job and no pending (for replay/benchmark)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if not self._busy and self._pending is None:
                    return True
            time.sleep(0.05)
        return False
