"""Vision Engine controller — UI/API facade over capture + YOLO + dual-brain."""
from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.annotate import annotate
from core.capture import Region, ScreenCapture
from core.dual_brain import DualBrain
from core.event import EventStore
from core.factory import build_brain, build_deep, build_yolo
from core.health import HealthTracker
from core.tracker import Tracker
from core.types import DetectionResult
from models.locate3b import LocateAnythingDetector
from models.yolo import YoloDetector
from profiles import list_profiles, load_profile


def _gpu_stats() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    try:
        used = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        props = torch.cuda.get_device_properties(0)
        total = round(props.total_memory / (1024**3), 2)
        return used, total
    except Exception:
        return None, None


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _deep_key(cfg: dict) -> tuple:
    la = cfg.get("locate3b") or {}
    return (
        str(la.get("model_path", "")),
        tuple(la.get("classes") or []),
        str(la.get("generation_mode", "hybrid")),
        int(la.get("max_side", 960)),
        int(la.get("max_new_tokens", 768)),
    )


class VisionController:
    """
    Single engine process owned by the UI.

    Gradio / HTTP should only call these methods — never touch models directly.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.profile_name = "general"
        self.cfg: dict[str, Any] = {}
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_str = "cuda" if self.device != "cpu" else "cpu"

        self.cap: ScreenCapture | None = None
        self.region: Region | None = None
        self.yolo: YoloDetector | None = None
        self.deep: LocateAnythingDetector | None = None
        # 3B weights are ~7GB — keep one instance alive across restarts so a
        # profile switch never loads a second copy into VRAM.
        self._deep_cache: LocateAnythingDetector | None = None
        self._deep_cache_key: tuple | None = None
        self.brain: DualBrain | None = None
        self.tracker: Tracker | None = None
        self.store: EventStore | None = None
        self.health = HealthTracker()

        self.running = False
        self.deepen_on = True
        self.engine_ready = False
        self.load_error = ""

        self.fps = 0.0
        self.narrative = "Engine idle."
        self.last_trigger_reason = ""
        self.last_yolo: DetectionResult | None = None
        self.last_deep: DetectionResult | None = None
        self._frame_rgb: np.ndarray | None = None
        self._last_event_log = 0.0
        self._last_deep_logged_id = None
        self._last_deep_calls = 0
        self.yolo_ready = False
        self.deep_ready = False
        self.monitor_name = ""
        self.started_at = 0.0
        self.source = "screen"
        # UI preview is downscaled; a 4K frame per tick stalls the browser
        self.preview_max_w = 1100
        self.capture_max_w = 0
        self._last_health_poll = 0.0
        self._events_count = 0
        self.monitor_override: int | None = None
        self.region_override: tuple[int, int, int, int] | None = None
        self.capture_max_w_override: int | None = 1920

    # ---- lifecycle ----
    def start(self, profile: str = "general", load_deep: bool = True) -> dict:
        with self._lock:
            if self.running:
                return self.status()
            self._stop.clear()
            self.running = True
            self.load_error = ""
            self.engine_ready = False
            self.yolo_ready = False
            self.deep_ready = False
            self._last_deep_calls = 0
            self._last_deep_logged_id = None
            self.last_trigger_reason = ""
            self.health.reset()
            self._thread = threading.Thread(
                target=self._run_loop,
                kwargs={"profile": profile, "load_deep": load_deep},
                daemon=True,
            )
            self._thread.start()
        return {"ok": True, "message": f"starting profile={profile}"}

    def stop(self) -> dict:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=8)
        with self._lock:
            self.running = False
            self.engine_ready = False
            self._cleanup_unlocked()
            self.narrative = "Engine stopped."
        return {"ok": True, "message": "stopped"}

    def _cleanup_unlocked(self) -> None:
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass
            self.store = None
        if self.cap:
            try:
                self.cap.close()
            except Exception:
                pass
            self.cap = None
        self.yolo = None
        # drop engine reference only; cached 3B instance stays resident
        self.deep = None
        self.brain = None
        self.tracker = None
        self.yolo_ready = False
        self.deep_ready = False
        self.last_trigger_reason = ""
        _free_gpu()

    def _acquire_deep(self, cfg: dict) -> LocateAnythingDetector | None:
        key = _deep_key(cfg)
        if self._deep_cache is not None and self._deep_cache_key == key:
            return self._deep_cache
        if self._deep_cache is not None:
            self._deep_cache = None
            self._deep_cache_key = None
            _free_gpu()
        deep = build_deep(cfg, device_str=self.device_str)
        self._deep_cache = deep
        self._deep_cache_key = key if deep is not None else None
        return deep

    def unload_deep(self) -> dict:
        """Release the cached 3B model and its VRAM."""
        with self._lock:
            self.deep = None
            self.brain = None
            self._deep_cache = None
            self._deep_cache_key = None
            self.deep_ready = False
            self.deepen_on = False
        _free_gpu()
        return {"ok": True, "locate3b": False}

    # ---- control API ----
    def set_profile(self, profile: str) -> dict:
        if profile not in list_profiles():
            return {"ok": False, "error": f"unknown profile: {profile}"}
        with self._lock:
            was_running = self.running
            deepen = self.deepen_on
        if was_running:
            self.stop()
            return self.start(profile=profile, load_deep=deepen)
        self.profile_name = profile
        return {"ok": True, "profile": profile}

    def list_monitors(self) -> list[tuple[int, str, bool]]:
        try:
            with ScreenCapture() as cap:
                out = []
                for i, m in cap.list_monitors():
                    primary = bool(m.get("is_primary"))
                    tag = " (primary)" if primary else ""
                    out.append((i, f"{i}: {m['width']}x{m['height']}{tag}", primary))
                return out
        except Exception:
            return []

    def probe_capture(
        self, monitor: int | None, region: tuple | None, max_width: int | None = 1920
    ) -> tuple[np.ndarray | None, str]:
        """Grab one frame for the given monitor/region without starting the engine."""
        if self.running:
            return None, "Engine is running — Stop first to preview a region."
        try:
            with ScreenCapture() as cap:
                mon_i = int(monitor) if monitor else cap.primary_index()
                mon = cap.monitor_region(mon_i)
                if region and len(region) == 4:
                    x, y, w, h = (int(v) for v in region)
                    r = Region(mon.left + x, mon.top + y, w, h)
                else:
                    r = mon
                bgr = cap.grab(r)
                output = "native"
                if max_width and bgr.shape[1] > int(max_width):
                    scale = int(max_width) / float(bgr.shape[1])
                    bgr = cv2.resize(
                        bgr,
                        (int(max_width), max(1, int(bgr.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    output = f"{bgr.shape[1]}x{bgr.shape[0]}"
                info = (
                    f"monitor {mon_i} · capture {r.width}x{r.height} "
                    f"· processing {output}"
                )
                return cv2.cvtColor(self._preview(bgr), cv2.COLOR_BGR2RGB), info
        except Exception as e:
            return None, f"probe failed: {e}"

    def set_capture(
        self, monitor: int | None, region: tuple | None, max_width: int | None = 1920
    ) -> dict:
        with self._lock:
            self.monitor_override = int(monitor) if monitor else None
            self.region_override = tuple(int(v) for v in region) if region else None
            self.capture_max_w_override = (
                max(0, int(max_width)) if max_width is not None else None
            )
            return {
                "ok": True,
                "monitor": self.monitor_override,
                "region": self.region_override,
                "max_width": self.capture_max_w_override,
            }

    def set_deepen(self, enabled: bool) -> dict:
        """Attach/detach 3B without restarting the engine."""
        enabled = bool(enabled)
        if not enabled:
            with self._lock:
                self.deepen_on = False
                self.brain = None
                self.deep = None
                self.deep_ready = False
            _free_gpu()
            return {"ok": True, "locate3b": False}

        with self._lock:
            cfg = self.cfg
            running = self.running
        if not running or not cfg:
            with self._lock:
                self.deepen_on = True
            return {"ok": True, "locate3b": True, "note": "will load on start"}

        try:
            deep = self._acquire_deep(cfg)
        except Exception as e:
            with self._lock:
                self.load_error = f"3B load failed: {e}"
                self.deepen_on = False
            _free_gpu()
            return {"ok": False, "error": str(e)}

        with self._lock:
            self.deep = deep
            self.brain = build_brain(deep, cfg) if deep else None
            self.deep_ready = deep is not None
            self.deepen_on = deep is not None
            self._last_deep_calls = 0
            return {"ok": True, "locate3b": self.deepen_on}

    def status(self) -> dict:
        gpu_used, gpu_total = _gpu_stats()
        with self._lock:
            yolo_count = self.last_yolo.count if self.last_yolo else 0
            events_n = self._events_count if self.store else 0
            deep_calls = self.brain.deep_calls if self.brain else 0
            skipped = self.brain.skipped_calls if self.brain else 0
            pending = self.brain.pending_count if self.brain else 0
            state = (
                "running"
                if self.engine_ready and self.running
                else ("starting" if self.running else "stopped")
            )
            health = self.health.health.as_dict()
            return {
                "profile": self.profile_name,
                "running": self.running,
                "engine_ready": self.engine_ready,
                "state": state,
                "fps": round(self.fps, 1),
                "yolo": self.yolo is not None,
                "yolo_ready": self.yolo_ready,
                "locate3b": bool(self.deepen_on and self.deep is not None),
                "deep_ready": self.deep_ready,
                "deep_busy": bool(self.brain.busy) if self.brain else False,
                "deep_calls": deep_calls,
                "deep_skipped": skipped,
                "deep_pending": pending,
                "deep_superseded": self.brain.superseded if self.brain else 0,
                "deep_reasons": dict(self.brain.reason_counts) if self.brain else {},
                "deep_avg_latency_s": health.get("deep_avg_latency_s"),
                "last_trigger_reason": self.last_trigger_reason,
                "boxes": yolo_count,
                "events": events_n,
                "gpu_used_gb": gpu_used,
                "gpu_total_gb": gpu_total,
                "gpu_peak_gb": health.get("gpu_peak_gb"),
                "monitor": self.monitor_name,
                "processing_width": (
                    self.capture_max_w
                    if self.running
                    else (self.capture_max_w_override or 0)
                ),
                "source": self.source,
                "narrative": self.narrative,
                "load_error": self.load_error,
                "profiles": list_profiles(),
                "health": health,
                "health_panel": self.health.health.format_panel(),
            }

    def latest_events(self, limit: int = 20) -> list[dict]:
        with self._lock:
            if not self.store:
                return []
            rows = self.store.recent(limit=limit)
        out = []
        for r in rows:
            ts = r.get("ts") or 0
            conf = r.get("confidence")
            out.append(
                {
                    "id": r.get("id"),
                    "time": time.strftime("%H:%M:%S", time.localtime(ts)),
                    "backend": r.get("backend"),
                    "class": r.get("label"),
                    "confidence": None if conf is None else round(float(conf), 3),
                    "description": r.get("summary") or "",
                    "snapshot": r.get("snapshot"),
                    "trigger_reason": r.get("trigger_reason"),
                    "profile": r.get("profile"),
                    "source": r.get("source"),
                    "latency_ms": r.get("latency_ms"),
                }
            )
        return out

    def event_snapshot_image(self, event_id: int | None) -> np.ndarray | None:
        if event_id is None:
            return None
        with self._lock:
            if not self.store:
                return None
            path = self.store.get_snapshot(int(event_id))
        if not path:
            return None
        bgr = cv2.imread(path)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _preview(self, bgr: np.ndarray) -> np.ndarray:
        w = bgr.shape[1]
        if self.preview_max_w <= 0 or w <= self.preview_max_w:
            return bgr
        scale = self.preview_max_w / float(w)
        return cv2.resize(
            bgr,
            (self.preview_max_w, max(1, int(bgr.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def latest_frame_rgb(self) -> np.ndarray | None:
        with self._lock:
            if self._frame_rgb is None:
                return None
            return self._frame_rgb.copy()

    def latest_detections(self) -> list[dict]:
        with self._lock:
            src = self.last_yolo
            if not src:
                return []
            rows = []
            for b in src.boxes:
                rows.append(
                    {
                        "label": b.label,
                        "confidence": round(float(b.confidence), 3),
                        "track_id": b.track_id,
                        "box": [round(v, 1) for v in b.as_xyxy()],
                    }
                )
            return rows

    # ---- engine loop ----
    def _run_loop(self, profile: str, load_deep: bool) -> None:
        try:
            cfg = load_profile(profile)
            with self._lock:
                self.profile_name = profile
                self.cfg = cfg
                self.narrative = f"Loading profile '{profile}'..."

            self.cap = ScreenCapture()
            mon_i = int(self.monitor_override or (cfg.get("capture") or {}).get("monitor") or 0)
            if mon_i <= 0:
                mon_i = self.cap.primary_index()
            region_cfg = self.region_override or (cfg.get("capture") or {}).get("region")
            if region_cfg and len(region_cfg) == 4:
                mon = self.cap.monitor_region(mon_i)
                x, y, w, h = (int(v) for v in region_cfg)
                # region is relative to the selected monitor
                self.region = Region(mon.left + x, mon.top + y, w, h)
                self.source = "roi"
            else:
                self.region = self.cap.monitor_region(mon_i)
                self.source = "screen"

            cap_cfg = cfg.get("capture") or {}
            self.capture_max_w = (
                self.capture_max_w_override
                if self.capture_max_w_override is not None
                else int(cap_cfg.get("max_width") or 1920)
            )

            self.yolo = build_yolo(cfg, device=self.device)
            self.deep = self._acquire_deep(cfg) if load_deep else None
            self.brain = build_brain(self.deep, cfg) if self.deep else None

            self.tracker = Tracker(
                enabled=bool((cfg.get("tracker") or {}).get("enabled", True))
            )
            ev = cfg.get("events") or {}
            if ev.get("enabled", True):
                self.store = EventStore(ROOT / ev.get("db_path", "data/events.db"))
                self.health.note_memory(self.store.ping(), "SQLite OK", self.store.total_count())

            with self._lock:
                self.engine_ready = True
                self.yolo_ready = self.yolo is not None
                self.deep_ready = self.deep is not None
                self.deepen_on = bool(self.deep is not None)
                self.monitor_name = f"monitor {mon_i}"
                self.started_at = time.time()
                self.narrative = "Engine ready."

            fps_ema = 0.0
            while not self._stop.is_set():
                t0 = time.perf_counter()
                assert self.cap and self.region and self.yolo

                try:
                    t_cap = time.perf_counter()
                    frame = self.cap.grab(self.region)
                    if self.capture_max_w and frame.shape[1] > self.capture_max_w:
                        s = self.capture_max_w / float(frame.shape[1])
                        frame = cv2.resize(
                            frame,
                            (self.capture_max_w, max(1, int(frame.shape[0] * s))),
                            interpolation=cv2.INTER_AREA,
                        )
                    self.health.note_capture(True, (time.perf_counter() - t_cap) * 1000.0)
                except Exception:
                    self.health.note_capture(False, 0.0)
                    time.sleep(0.05)
                    continue

                t_y = time.perf_counter()
                yolo_res = self.yolo.predict(frame)
                if self.tracker:
                    yolo_res = self.tracker.update(yolo_res)
                self.health.note_yolo((time.perf_counter() - t_y) * 1000.0)

                narrative = yolo_res.summary
                deep_res = None
                trigger_reason = ""
                deep_just_finished = False
                if self.deepen_on and self.brain is not None:
                    self.brain.maybe_trigger(frame, yolo_res)
                    deep_res, deep_text, trigger_reason = self.brain.snapshot_deep()
                    if deep_text:
                        narrative = deep_text
                    if self.brain.deep_calls > self._last_deep_calls:
                        deep_just_finished = True
                        self._last_deep_calls = self.brain.deep_calls

                view = annotate(frame, yolo_res)
                if deep_res and deep_res.boxes:
                    view = annotate(view, deep_res)

                dt = time.perf_counter() - t0
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_ema = fps if fps_ema == 0 else fps_ema * 0.85 + fps * 0.15
                rgb = cv2.cvtColor(self._preview(view), cv2.COLOR_BGR2RGB)

                used, _ = _gpu_stats()
                self.health.note_gpu(used)
                self.health.sync_brain(self.brain)
                now = time.time()
                if self.store and now - self._last_health_poll > 2.0:
                    # COUNT(*) is O(rows); never run it per frame
                    self._events_count = self.store.total_count()
                    self.health.note_memory(self.store.ping(), "SQLite OK", self._events_count)
                    self._last_health_poll = now

                with self._lock:
                    self.fps = fps_ema
                    self.last_yolo = yolo_res
                    self.last_deep = deep_res
                    self.narrative = narrative
                    self.last_trigger_reason = trigger_reason
                    self._frame_rgb = rgb

                if self.store and deep_just_finished and deep_res is not None:
                    deep_id = id(deep_res)
                    if deep_id != self._last_deep_logged_id:
                        self.store.log_result(
                            deep_res,
                            summary=narrative,
                            frame_bgr=frame,
                            save_snapshot=True,
                            profile=profile,
                            source=self.source,
                            trigger_reason=trigger_reason,
                        )
                        self._last_deep_logged_id = deep_id
                        self._last_event_log = now
                elif self.store and now - self._last_event_log > 3.0 and yolo_res.count:
                    self.store.log_result(
                        yolo_res,
                        summary=narrative,
                        save_snapshot=False,
                        profile=profile,
                        source=self.source,
                    )
                    self._last_event_log = now

                time.sleep(0.001)

        except Exception as e:
            with self._lock:
                self.load_error = str(e)
                self.narrative = f"Engine error: {e}"
                self.running = False
                self.engine_ready = False
        finally:
            with self._lock:
                self.running = False
                self.engine_ready = False
                self._cleanup_unlocked()


# process-wide singleton for Gradio
CONTROLLER = VisionController()
