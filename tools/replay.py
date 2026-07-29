"""Replay video/image through VisionDesk engine and print benchmark stats.

Examples:
  python -m tools.replay path/to/traffic.mp4 --profile traffic
  python visiondesk.py replay traffic.mp4 --profile traffic --no-deep
  python visiondesk.py replay shot.jpg --profile general --stride 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.annotate import annotate
from core.event import EventStore
from core.factory import build_brain, build_deep, build_yolo
from core.health import HealthTracker
from core.tracker import Tracker
from profiles import load_profile


@dataclass
class BenchmarkReport:
    source: str
    profile: str
    total_frames: int = 0
    processed_frames: int = 0
    yolo_detections: int = 0
    deep_calls: int = 0
    skipped: int = 0
    superseded: int = 0
    average_fps: float = 0.0
    yolo_avg_ms: float = 0.0
    deep_avg_latency_s: float | None = None
    gpu_peak_gb: float | None = None
    elapsed_s: float = 0.0
    trigger_reasons: dict = field(default_factory=dict)
    reasoning_efficiency: float | None = None  # frames / 3B call
    notes: list[str] = field(default_factory=list)

    def format(self) -> str:
        skip_pct = 0.0
        denom = self.deep_calls + self.skipped
        if denom:
            skip_pct = 100.0 * self.skipped / denom
        eff = self.reasoning_efficiency
        eff_s = f"{eff:.1f} frames / reasoning call" if eff is not None else "n/a"
        lines = [
            "=== VisionDesk Replay Benchmark ===",
            f"Source:          {self.source}",
            f"Profile:         {self.profile}",
            f"Total frames:    {self.total_frames}",
            f"Processed:       {self.processed_frames}",
            f"YOLO detections: {self.yolo_detections}",
            f"3B calls:        {self.deep_calls}",
            f"Skipped:         {self.skipped}  ({skip_pct:.1f}%)",
            f"Superseded:      {self.superseded}",
            f"Avg FPS:         {self.average_fps:.1f}",
            f"YOLO avg ms:     {self.yolo_avg_ms:.1f}",
            f"3B avg latency:  {self.deep_avg_latency_s if self.deep_avg_latency_s is not None else 'n/a'}",
            f"Reasoning eff.:  {eff_s}",
            f"GPU peak:        {self.gpu_peak_gb if self.gpu_peak_gb is not None else 'n/a'} GB",
            f"Elapsed:         {self.elapsed_s:.1f}s",
        ]
        if self.trigger_reasons:
            lines.append("")
            lines.append("3B Trigger Report")
            for kind, n in self.trigger_reasons.items():
                lines.append(f"  {kind:<12} {n}")
        for n in self.notes:
            lines.append(f"Note:            {n}")
        return "\n".join(lines)


def _gpu_used_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    except Exception:
        return None


def _iter_frames(path: Path, stride: int, max_frames: int | None):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        yield 0, img
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                yield idx, frame
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def _count_video_frames(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return 1
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def run_replay(
    media: str | Path,
    profile: str = "traffic",
    *,
    stride: int = 1,
    max_frames: int | None = None,
    load_deep: bool = True,
    save_annotated: Path | None = None,
    log_events: bool = True,
    wait_deep: bool = True,
) -> BenchmarkReport:
    path = Path(media)
    if not path.exists():
        raise FileNotFoundError(path)

    cfg = load_profile(profile)
    device = 0 if torch.cuda.is_available() else "cpu"
    device_str = "cuda" if device != "cpu" else "cpu"

    yolo = build_yolo(cfg, device=device)
    deep = build_deep(cfg, device_str=device_str) if load_deep else None
    brain = build_brain(deep, cfg)
    tracker = Tracker(enabled=bool((cfg.get("tracker") or {}).get("enabled", True)))

    store = None
    if log_events and (cfg.get("events") or {}).get("enabled", True):
        db = ROOT / "data" / "replay_events.db"
        store = EventStore(db)

    health = HealthTracker()
    report = BenchmarkReport(source=str(path), profile=profile)
    report.total_frames = _count_video_frames(path)

    if save_annotated:
        save_annotated.mkdir(parents=True, exist_ok=True)

    yolo_ms_acc = 0.0
    det_boxes = 0
    t_wall0 = time.perf_counter()
    last_deep_calls = 0

    for frame_i, frame in _iter_frames(path, stride=stride, max_frames=max_frames):
        t_cap0 = time.perf_counter()
        health.note_capture(True, (time.perf_counter() - t_cap0) * 1000.0)

        t_y0 = time.perf_counter()
        yolo_res = yolo.predict(frame)
        yolo_res = tracker.update(yolo_res)
        yolo_ms = (time.perf_counter() - t_y0) * 1000.0
        health.note_yolo(yolo_ms)
        yolo_ms_acc += yolo_ms
        det_boxes += yolo_res.count

        narrative = yolo_res.summary
        deep_res = None
        reason = ""
        if brain is not None:
            brain.maybe_trigger(frame, yolo_res)
            deep_res, deep_text, reason = brain.snapshot_deep()
            if deep_text:
                narrative = deep_text
            if brain.deep_calls > last_deep_calls:
                last_deep_calls = brain.deep_calls
                if store is not None and deep_res is not None:
                    store.log_result(
                        deep_res,
                        summary=narrative,
                        frame_bgr=frame,
                        save_snapshot=True,
                        profile=profile,
                        source="replay",
                        trigger_reason=reason,
                    )

        view = annotate(frame, yolo_res)
        if deep_res and deep_res.boxes:
            view = annotate(view, deep_res)

        if save_annotated is not None:
            out = save_annotated / f"frame_{frame_i:06d}.jpg"
            cv2.imwrite(str(out), view)

        used = _gpu_used_gb()
        health.note_gpu(used)
        health.sync_brain(brain)
        if store is not None:
            health.note_memory(store.ping(), "SQLite OK", store.total_count())

        report.processed_frames += 1

    if brain is not None and wait_deep:
        ok = brain.wait_idle(timeout=180.0)
        if not ok:
            report.notes.append("timed out waiting for 3B queue to drain")
        # flush final deep result if any new calls
        deep_res, deep_text, reason = brain.snapshot_deep()
        if brain.deep_calls > last_deep_calls and store is not None and deep_res is not None:
            # no frame handy for final — log without snapshot
            store.log_result(
                deep_res,
                summary=deep_text,
                profile=profile,
                source="replay",
                trigger_reason=reason,
            )

    elapsed = time.perf_counter() - t_wall0
    report.elapsed_s = elapsed
    report.yolo_detections = det_boxes
    report.average_fps = report.processed_frames / elapsed if elapsed > 0 else 0.0
    report.yolo_avg_ms = yolo_ms_acc / report.processed_frames if report.processed_frames else 0.0
    if brain is not None:
        report.deep_calls = brain.deep_calls
        report.skipped = brain.skipped_calls
        report.superseded = brain.superseded
        report.trigger_reasons = brain.reason_report()
        if brain.avg_latency_ms is not None:
            report.deep_avg_latency_s = round(brain.avg_latency_ms / 1000.0, 2)
    if report.deep_calls > 0 and report.processed_frames > 0:
        report.reasoning_efficiency = round(report.processed_frames / report.deep_calls, 1)
    report.gpu_peak_gb = health.health.gpu_peak_gb

    if store is not None:
        store.close()
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VisionDesk replay / benchmark")
    p.add_argument("media", help="video or image path")
    p.add_argument("--profile", default="traffic")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-deep", action="store_true", help="YOLO-only benchmark")
    p.add_argument("--save-dir", type=str, default=None, help="save annotated frames")
    p.add_argument("--json", action="store_true", help="also print JSON report")
    p.add_argument("--no-events", action="store_true")
    args = p.parse_args(argv)

    report = run_replay(
        args.media,
        profile=args.profile,
        stride=args.stride,
        max_frames=args.max_frames,
        load_deep=not args.no_deep,
        save_annotated=Path(args.save_dir) if args.save_dir else None,
        log_events=not args.no_events,
    )
    print(report.format(), flush=True)
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
