"""VisionDesk AI — local desktop vision agent (YOLO realtime + LocateAnything-3B deepen).

Examples:
  python main.py --profile general
  python main.py --profile traffic --select-roi
  python main.py --profile person --no-deep

Hotkeys:
  q/ESC  quit
  1/2/3/4  switch profile general/traffic/person/security
  s      save snapshot
  r      re-select ROI on current monitor
  d      toggle dual-brain deepen
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.annotate import annotate
from core.capture import Region, ScreenCapture, select_roi_interactive
from core.event import EventStore
from core.factory import build_brain, build_deep, build_yolo
from core.tracker import Tracker
from profiles import list_profiles, load_profile

PROFILE_HOTKEYS = {
    ord("1"): "general",
    ord("2"): "traffic",
    ord("3"): "person",
    ord("4"): "security",
}


def resolve_weights(name: str) -> str:
    local = ROOT / name
    return str(local if local.exists() else name)


def region_from_profile(cap: ScreenCapture, cfg: dict, select_roi: bool) -> tuple[Region, int, int]:
    c = cfg.get("capture") or {}
    mon_i = int(c.get("monitor") or 0)
    ui_i = int(c.get("ui_monitor") or 0)
    if mon_i <= 0:
        mon_i = cap.primary_index()
    if ui_i <= 0:
        ui_i = cap.other_index(mon_i)

    region_cfg = c.get("region")
    if region_cfg and len(region_cfg) == 4:
        region = Region(*[int(v) for v in region_cfg])
    else:
        region = cap.monitor_region(mon_i)

    if select_roi:
        snap = cap.grab(region)
        picked = select_roi_interactive(snap)
        if picked:
            x, y, w, h = picked
            region = Region(region.left + x, region.top + y, w, h)
            print(f"ROI selected: {region}", flush=True)
    return region, mon_i, ui_i


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionDesk AI desktop vision agent")
    ap.add_argument("--profile", default="general", choices=list_profiles() or ["general"])
    ap.add_argument("--select-roi", action="store_true", help="Interactively pick ROI once at start")
    ap.add_argument("--no-deep", action="store_true", help="Disable LocateAnything-3B deepen")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"

    cfg = load_profile(args.profile)
    print(f"profile={cfg.get('_name')}  title={cfg.get('title')}", flush=True)

    yolo = build_yolo(cfg, device=device)
    deep = None if args.no_deep else build_deep(cfg, device_str=device_str)
    brain = build_brain(deep, cfg) if deep is not None else None

    tracker = Tracker(enabled=not args.no_track and bool((cfg.get("tracker") or {}).get("enabled", True)))
    ev_cfg = cfg.get("events") or {}
    store = EventStore(ROOT / ev_cfg.get("db_path", "data/events.db")) if ev_cfg.get("enabled", True) else None

    with ScreenCapture() as cap:
        print("displays:", flush=True)
        for i, m in cap.list_monitors():
            print(
                f"  [{i}] ({m['left']},{m['top']}) {m['width']}x{m['height']}"
                f"{' primary' if m.get('is_primary') else ''}",
                flush=True,
            )
        region, mon_i, ui_i = region_from_profile(cap, cfg, select_roi=args.select_roi)
        print(f"capture={mon_i} ui={ui_i} region={region}", flush=True)

        win = "VisionDesk AI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        preview_w = min(960, region.width)
        scale = preview_w / max(region.width, 1)
        preview_h = int(region.height * scale)
        cv2.resizeWindow(win, preview_w, preview_h)
        ui_mon = cap.monitor_region(ui_i)
        cv2.moveWindow(
            win,
            int(ui_mon.left + max(40, (ui_mon.width - preview_w) // 2)),
            int(ui_mon.top + max(40, (ui_mon.height - preview_h) // 2)),
        )

        fps_ema = 0.0
        last_log = 0.0
        narrative = "Dual-brain ready." if brain else "YOLO-only mode."
        deepen_on = brain is not None

        while True:
            t0 = time.perf_counter()
            frame = cap.grab(region)
            yolo_res = yolo.predict(frame)
            yolo_res = tracker.update(yolo_res)

            if deepen_on and brain is not None:
                brain.maybe_trigger(frame, yolo_res)
                deep_res, deep_text, _reason = brain.snapshot_deep()
                if deep_text:
                    narrative = deep_text
            else:
                deep_res = None
                narrative = yolo_res.summary

            # draw YOLO + latest deep boxes
            view = annotate(frame, yolo_res, title="")
            if deep_res and deep_res.boxes:
                # draw deep boxes thicker in another pass
                view = annotate(view, deep_res, title="")

            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema == 0 else fps_ema * 0.85 + fps * 0.15
            status = (
                f"VisionDesk | {cfg.get('_name')} | FPS {fps_ema:.1f} | "
                f"yolo {yolo_res.count} | deep {'ON' if deepen_on else 'OFF'}"
                f"{' BUSY' if brain and brain.busy else ''} | 1-4 profile q quit"
            )
            cv2.putText(view, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                view,
                narrative[:120],
                (12, view.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 255, 160),
                1,
                cv2.LINE_AA,
            )

            show = cv2.resize(view, (preview_w, preview_h)) if scale != 1.0 else view
            cv2.imshow(win, show)

            now = time.time()
            if store and now - last_log > 2.0:
                store.log_result(
                    yolo_res,
                    summary=narrative,
                    profile=cfg.get("_name"),
                    source="screen",
                )
                if deep_res:
                    store.log_result(
                        deep_res,
                        summary=narrative,
                        profile=cfg.get("_name"),
                        source="screen",
                        trigger_reason=_reason if deepen_on and brain else None,
                    )
                last_log = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = ROOT / "output" / "screen_live" / f"visiondesk_{int(time.time())}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
                print(f"saved {out}", flush=True)
            if key == ord("d") and brain is not None:
                deepen_on = not deepen_on
                print(f"deepen -> {deepen_on}", flush=True)
            if key == ord("r"):
                snap = cap.grab(cap.monitor_region(mon_i))
                picked = select_roi_interactive(snap)
                if picked:
                    mon = cap.monitor_region(mon_i)
                    x, y, w, h = picked
                    region = Region(mon.left + x, mon.top + y, w, h)
                    preview_w = min(960, region.width)
                    scale = preview_w / max(region.width, 1)
                    preview_h = int(region.height * scale)
                    print(f"ROI updated: {region}", flush=True)
            if key in PROFILE_HOTKEYS:
                name = PROFILE_HOTKEYS[key]
                print(f"switching profile -> {name} (restart process for full reload)", flush=True)
                # lightweight: only swap yolo keep ids / conf via reload profile yolo part
                try:
                    cfg = load_profile(name)
                    yolo = build_yolo(cfg, device=device)
                    print(f"YOLO reloaded for {name}", flush=True)
                except Exception as e:
                    print(f"profile switch failed: {e}", flush=True)

        cv2.destroyAllWindows()
        if store:
            day_ago = time.time() - 86400
            print("24h label counts:", store.counts_since(day_ago), flush=True)
            store.close()


if __name__ == "__main__":
    main()
