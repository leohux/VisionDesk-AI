"""VisionDesk AI Gradio cockpit — status bar + timeline + live view."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.controller import CONTROLLER
from profiles import list_profiles


_ICON = {
    "person": "👤",
    "car": "🚗",
    "truck": "🚚",
    "bus": "🚌",
    "motorcycle": "🏍️",
    "bicycle": "🚲",
    "dog": "🐕",
    "cat": "🐈",
    "gun": "🔫",
    "rifle": "🔫",
    "drone": "🛸",
}


def _icon_for(label: str | None, backend: str | None) -> str:
    if backend and "locate" in str(backend).lower():
        return "🧠"
    if not label:
        return "•"
    return _ICON.get(str(label).lower(), "📦")


def _fmt_status(st: dict) -> str:
    state = st.get("state") or ("running" if st.get("engine_ready") else "stopped")
    if state == "running":
        led = "🟢 Engine Running"
    elif state == "starting":
        led = "🟡 Engine Starting"
    else:
        led = "🔴 Engine Stopped"

    gpu_u, gpu_t = st.get("gpu_used_gb"), st.get("gpu_total_gb")
    peak = st.get("gpu_peak_gb")
    if gpu_u is not None and gpu_t is not None:
        gpu_line = f"{gpu_u}GB / {gpu_t}GB"
        if peak is not None:
            gpu_line += f"\npeak {peak}GB"
    elif gpu_u is not None:
        gpu_line = f"{gpu_u}GB"
    else:
        gpu_line = "n/a"

    yolo = "Loaded ✓" if st.get("yolo_ready") or st.get("yolo") else "—"
    deep_on = st.get("locate3b")
    if deep_on and st.get("deep_busy"):
        deep = "Busy…"
    elif deep_on and (st.get("deep_ready") or st.get("locate3b")):
        deep = "Ready ✓"
    elif deep_on:
        deep = "Loading…"
    else:
        deep = "Off"

    reason = st.get("last_trigger_reason") or "—"
    err = st.get("load_error") or ""
    health = st.get("health_panel") or ""
    reasons = st.get("deep_reasons") or {}
    reason_lines = (
        "\n".join(f"  {k:<10} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
        or "  (none yet)"
    )
    lines = [
        "VisionDesk AI",
        "",
        led,
        "",
        f"Profile:\n{st.get('profile') or '—'}",
        f"Source:\n{st.get('source') or 'screen'} · {st.get('monitor') or '—'}",
        f"Capture:\n{st.get('region') or '—'}",
        f"Processing:\n{str(pw) + 'px wide' if (pw := st.get('processing_width')) else 'native'}",
        "",
        f"FPS:\n{st.get('fps')}",
        "",
        f"YOLO:\n{yolo}  ({st.get('boxes', 0)} boxes)",
        "",
        f"LocateAnything-3B:\n{deep}",
        f"  calls={st.get('deep_calls', 0)}  skipped={st.get('deep_skipped', 0)}",
        f"  queue={st.get('deep_pending', 0)}  superseded={st.get('deep_superseded', 0)}",
        f"  last trigger:\n  {reason}",
        "",
        f"3B Trigger Report:\n{reason_lines}",
        "",
        f"GPU:\n{gpu_line}",
        "",
        f"Events:\n{st.get('events', 0)}",
        "",
        "────────────",
        health,
    ]
    if err:
        lines += ["", f"error:\n{err}"]
    return "\n".join(lines)


def _fmt_detections(rows: list[dict]) -> str:
    if not rows:
        return "(no detections)"
    lines = []
    for r in rows[:40]:
        tid = f"#{r['track_id']} " if r.get("track_id") is not None else ""
        icon = _icon_for(r.get("label"), None)
        lines.append(f"{icon} {tid}{r['label']:<12} {r['confidence']:.0%}")
    return "\n".join(lines)


def _fmt_timeline(rows: list[dict]) -> str:
    if not rows:
        return "Latest Events\n\n(no events yet)"
    lines = ["Latest Events", ""]
    for r in rows[:28]:
        icon = _icon_for(r.get("class"), r.get("backend"))
        cls = r.get("class") or "scene"
        conf = r.get("confidence")
        conf_s = f" {conf:.0%}" if isinstance(conf, (int, float)) else ""
        snap = " 🖼" if r.get("snapshot") else ""
        desc = (r.get("description") or "").strip()
        # prefer short AI line for 3B events
        if icon == "🧠" and desc:
            short = desc.replace("\n", " ")
            if len(short) > 90:
                short = short[:87] + "..."
            lines.append(f"{r.get('time', '')}")
            lines.append(f"{icon} AI:{snap}")
            lines.append(f'  "{short}"')
        else:
            lines.append(f"{r.get('time', '')}")
            lines.append(f"{icon} {cls} detected{conf_s}{snap}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _event_choices(rows: list[dict]) -> list[tuple[str, int]]:
    choices = []
    for r in rows:
        eid = r.get("id")
        if eid is None:
            continue
        icon = _icon_for(r.get("class"), r.get("backend"))
        cls = r.get("class") or "event"
        mark = " · snap" if r.get("snapshot") else ""
        choices.append((f"{r.get('time','')} {icon} {cls}{mark}", int(eid)))
    return choices


_REGION_PRESETS = {
    "Full monitor": None,
    "Center 1920x1080": "center-1920",
    "Center 1280x720": "center-1280",
    "Top-left 1920x1080": (0, 0, 1920, 1080),
    "Top-left 1280x720": (0, 0, 1280, 720),
    "Left half": "left-half",
}

_RESOLUTION_PRESETS = {
    "1080p · 1920x1080 (recommended)": 1920,
    "1440p · 2560x1440": 2560,
    "Native · highest detail": 0,
}


def _monitor_size(monitor_label: str) -> tuple[int, int]:
    try:
        wh = monitor_label.split(":")[1].strip().split(" ")[0]
        w, h = (int(v) for v in wh.split("x"))
        return w, h
    except Exception:
        return 1920, 1080


def _resolve_region(preset: str, monitor_label: str):
    spec = _REGION_PRESETS.get(preset)
    if spec is None:
        return None
    mw, mh = _monitor_size(monitor_label)
    if spec == "left-half":
        return (0, 0, mw // 2, mh)
    if isinstance(spec, str) and spec.startswith("center-"):
        rw = 1920 if spec.endswith("1920") else 1280
        rh = 1080 if rw == 1920 else 720
        rw, rh = min(rw, mw), min(rh, mh)
        return ((mw - rw) // 2, (mh - rh) // 2, rw, rh)
    return spec


def _monitor_index(monitor_label: str) -> int | None:
    try:
        return int(str(monitor_label).split(":")[0])
    except Exception:
        return None


def _processing_width(resolution_preset: str) -> int:
    return int(_RESOLUTION_PRESETS.get(resolution_preset, 1920))


def ui_start(
    profile: str,
    enable_3b: bool,
    monitor_label: str,
    region_preset: str,
    resolution_preset: str,
):
    CONTROLLER.set_capture(
        _monitor_index(monitor_label),
        _resolve_region(region_preset, monitor_label),
        _processing_width(resolution_preset),
    )
    CONTROLLER.deepen_on = bool(enable_3b)
    if CONTROLLER.running and CONTROLLER.profile_name != profile:
        CONTROLLER.stop()
    CONTROLLER.start(profile=profile, load_deep=bool(enable_3b))
    time.sleep(0.3)
    return refresh()


def ui_load(
    profile: str,
    enable_3b: bool,
    monitor_label: str,
    region_preset: str,
    resolution_preset: str,
):
    """Auto-start on first open, but never clobber an already-running engine."""
    if CONTROLLER.running:
        return refresh()
    return ui_start(profile, enable_3b, monitor_label, region_preset, resolution_preset)


def ui_apply_capture(monitor_label: str, region_preset: str, resolution_preset: str):
    """Capture dropdowns take effect right away, running or not."""
    CONTROLLER.set_capture(
        _monitor_index(monitor_label),
        _resolve_region(region_preset, monitor_label),
        _processing_width(resolution_preset),
    )
    time.sleep(0.2)
    return refresh()


def ui_stop():
    CONTROLLER.stop()
    time.sleep(0.2)
    return refresh()


def ui_set_profile(profile: str):
    CONTROLLER.set_profile(profile)
    time.sleep(0.4)
    return refresh()


def ui_toggle_3b(enabled: bool):
    CONTROLLER.set_deepen(enabled)
    return refresh()


def ui_unload_3b():
    CONTROLLER.unload_deep()
    return refresh()


def ui_probe(
    monitor_label: str, region_preset: str, resolution_preset: str
):
    img, info = CONTROLLER.probe_capture(
        _monitor_index(monitor_label),
        _resolve_region(region_preset, monitor_label),
        _processing_width(resolution_preset),
    )
    if img is None:
        return _BLANK, f"Region test: {info}"
    return img, f"Region test (static): {info}\nClick Start for realtime."


def ui_show_event(event_id):
    img = CONTROLLER.event_snapshot_image(event_id)
    if img is None:
        return np.zeros((240, 426, 3), dtype=np.uint8)
    return img


_BLANK = np.zeros((480, 854, 3), dtype=np.uint8)
_last_event_id: int | None = None


def refresh_live():
    """Fast path: video + status only. No DB reads, no component rebuilds."""
    st = CONTROLLER.status()
    frame = CONTROLLER.latest_frame_rgb()
    dets = _fmt_detections(CONTROLLER.latest_detections())
    if frame is None:
        # engine idle: keep whatever is shown (e.g. a region preview) instead of blanking
        return gr.update(), _fmt_status(st), dets, gr.update()
    return frame, _fmt_status(st), dets, st.get("narrative") or ""


def refresh_events():
    """Slow path: timeline + event picker. Dropdown only rebuilt when it changed."""
    global _last_event_id
    rows = CONTROLLER.latest_events(30)
    timeline = _fmt_timeline(rows)
    top_id = rows[0].get("id") if rows else None
    if top_id == _last_event_id:
        return timeline, gr.update()
    _last_event_id = top_id
    return timeline, gr.update(choices=_event_choices(rows))


def refresh():
    """Full refresh used by button handlers."""
    live, status, dets, narrative = refresh_live()
    timeline, pick = refresh_events()
    return live, status, dets, narrative, timeline, pick


def build_ui() -> gr.Blocks:
    profiles = list_profiles() or ["general"]
    css = """
    #status-panel textarea { font-family: ui-monospace, Consolas, monospace; font-size: 13px; line-height: 1.35; }
    #timeline-panel textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; line-height: 1.4; }
    .gradio-container { max-width: 1600px !important; }
    footer { display: none !important; }
    """
    with gr.Blocks(title="VisionDesk AI", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
# VisionDesk AI
Local-first desktop vision agent · **YOLO realtime** + **smart LocateAnything-3B**
"""
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                live = gr.Image(label="Live screen + AI boxes", type="numpy", height=560)
                with gr.Row():
                    dets = gr.Textbox(label="Current detections", lines=9)
                    narrative = gr.Textbox(label="AI understanding", lines=9)
            with gr.Column(scale=1, min_width=330):
                with gr.Row():
                    btn_start = gr.Button("Start", variant="primary")
                    btn_stop = gr.Button("Stop")
                profile = gr.Dropdown(profiles, value="traffic", label="Profile")
                enable_3b = gr.Checkbox(
                    value=False,
                    label="LocateAnything-3B deepen (loads ~7GB on first enable)",
                )
                with gr.Accordion("Capture settings", open=False):
                    mons = CONTROLLER.list_monitors()
                    mon_choices = [lbl for _, lbl, _ in mons] or ["1: unknown"]
                    default_mon = next(
                        (lbl for _, lbl, is_primary in mons if is_primary), mon_choices[0]
                    )
                    monitor = gr.Dropdown(
                        mon_choices, value=default_mon, label="Capture monitor"
                    )
                    region_preset = gr.Dropdown(
                        list(_REGION_PRESETS.keys()),
                        value="Full monitor",
                        label="Capture area",
                    )
                    resolution_preset = gr.Dropdown(
                        list(_RESOLUTION_PRESETS.keys()),
                        value="1080p · 1920x1080 (recommended)",
                        label="Processing resolution",
                    )
                    with gr.Row():
                        btn_probe = gr.Button("Test region (Stop first)")
                        btn_apply = gr.Button("Apply profile")
                    btn_unload = gr.Button("Unload 3B")
                status = gr.Textbox(
                    label="Status / Health", lines=26, max_lines=26, elem_id="status-panel"
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                events = gr.Textbox(
                    label="Event timeline", lines=17, max_lines=17, elem_id="timeline-panel"
                )
            with gr.Column(scale=2):
                event_pick = gr.Dropdown(label="Replay event snapshot", choices=[], value=None)
                event_snap = gr.Image(label="Event frame", type="numpy", height=330)

        outs = [live, status, dets, narrative, events, event_pick]
        live_outs = [live, status, dets, narrative]
        event_outs = [events, event_pick]
        start_inputs = [profile, enable_3b, monitor, region_preset, resolution_preset]

        btn_start.click(ui_start, inputs=start_inputs, outputs=outs)
        btn_stop.click(ui_stop, outputs=outs)
        btn_apply.click(ui_set_profile, inputs=[profile], outputs=outs)
        btn_unload.click(ui_unload_3b, outputs=outs)
        btn_probe.click(
            ui_probe,
            inputs=[monitor, region_preset, resolution_preset],
            outputs=[live, narrative],
        )
        capture_inputs = [monitor, region_preset, resolution_preset]
        for ctrl in (monitor, region_preset, resolution_preset):
            ctrl.change(ui_apply_capture, inputs=capture_inputs, outputs=outs)
        enable_3b.change(ui_toggle_3b, inputs=[enable_3b], outputs=outs)
        event_pick.change(ui_show_event, inputs=[event_pick], outputs=[event_snap])

        gr.Timer(0.25).tick(refresh_live, outputs=live_outs)
        gr.Timer(2.0).tick(refresh_events, outputs=event_outs)
        # open page → start YOLO realtime immediately (3B stays off until checked)
        demo.load(ui_load, inputs=start_inputs, outputs=outs)

    return demo


def main() -> None:
    demo = build_ui()
    print("VisionDesk UI → http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
