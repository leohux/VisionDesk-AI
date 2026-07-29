"""Load and normalize YAML profiles for VisionDesk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROFILES_DIR = Path(__file__).resolve().parent


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict[str, Any]:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = normalize_profile(data)
    data["_name"] = name
    return data


def normalize_profile(data: dict[str, Any]) -> dict[str, Any]:
    """
    Support both legacy keys and the clearer product schema:

      detection.classes / trigger.* / reasoning.prompt

    Merged into yolo / locate3b / dual_brain so the engine stays simple.
    """
    out = dict(data)

    # --- detection ---
    detection = dict(out.get("detection") or {})
    yolo = dict(out.get("yolo") or {})
    locate = dict(out.get("locate3b") or {})

    if detection.get("classes"):
        classes = list(detection["classes"])
        # COCO keep list stays in yolo.keep_class_ids if already set;
        # World / LA class lists get detection.classes
        if not yolo.get("classes") and yolo.get("backend") in (None, "world", "yolo-world"):
            yolo["classes"] = classes
        if not locate.get("classes"):
            locate["classes"] = classes
    if detection.get("conf") is not None and "conf" not in yolo:
        yolo["conf"] = detection["conf"]
    if detection.get("weights") and "weights" not in yolo:
        yolo["weights"] = detection["weights"]
    if detection.get("backend") and "backend" not in yolo:
        yolo["backend"] = detection["backend"]

    out["yolo"] = yolo
    out["locate3b"] = locate

    # --- trigger → dual_brain ---
    trigger = dict(out.get("trigger") or {})
    dual = dict(out.get("dual_brain") or {})
    mapping = {
        "skip_above_conf": "skip_above_conf",
        "uncertain_below_conf": "uncertain_below_conf",
        "min_conf": "min_conf",
        "cooldown_sec": "cooldown_sec",
        "max_roi_boxes": "max_roi_boxes",
        "on_new_label": "on_new_label",
        "max_pending_jobs": "max_pending_jobs",
        "force_labels": "force_labels",
        "labels": "trigger_labels",
        "trigger_labels": "trigger_labels",
    }
    for src, dst in mapping.items():
        if src in trigger and dst not in dual:
            dual[dst] = trigger[src]
    if detection.get("classes") and "trigger_labels" not in dual:
        dual["trigger_labels"] = list(detection["classes"])
    if "max_pending_jobs" not in dual:
        dual["max_pending_jobs"] = 1
    out["dual_brain"] = dual

    # --- reasoning ---
    reasoning = dict(out.get("reasoning") or {})
    if reasoning.get("prompt") and "prompt" not in locate:
        locate["prompt"] = reasoning["prompt"]
        out["locate3b"] = locate
    if reasoning.get("enabled") is not None and "enabled" not in dual:
        dual["enabled"] = bool(reasoning["enabled"])
        out["dual_brain"] = dual

    # convenience top-level name
    if "name" not in out and out.get("profile"):
        out["name"] = out["profile"]

    return out
