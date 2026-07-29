"""Shared builders for YOLO / LocateAnything / DualBrain from a profile dict."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.dual_brain import DualBrain, DualBrainConfig
from models.locate3b import LocateAnythingDetector
from models.yolo import YoloDetector

ROOT = Path(__file__).resolve().parents[1]


def resolve_weights(name: str) -> str:
    local = ROOT / name
    return str(local if local.exists() else name)


def resolve_deep_path(raw: str | None = None) -> str:
    """Prefer LOCATE_ANYTHING_PATH, then profile path, then ./LocateAnything-3B."""
    for candidate in (
        os.environ.get("LOCATE_ANYTHING_PATH"),
        raw,
        "LocateAnything-3B",
    ):
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.exists():
            return str(p.resolve())
        local = ROOT / candidate
        if local.exists():
            return str(local.resolve())
    return str(Path(raw or "LocateAnything-3B").expanduser())


def build_yolo(cfg: dict[str, Any], device) -> YoloDetector:
    y = cfg.get("yolo") or {}
    keep = y.get("keep_class_ids")
    return YoloDetector(
        weights=resolve_weights(y.get("weights", "yolov8m.pt")),
        classes=y.get("classes") or [],
        backend=y.get("backend", "coco"),
        conf=float(y.get("conf", 0.25)),
        imgsz=int(y.get("imgsz", 1280)),
        device=device,
        max_det=int(y.get("max_det", 300)),
        keep_class_ids=set(keep) if keep else None,
    )


def build_deep(cfg: dict[str, Any], device_str: str) -> LocateAnythingDetector | None:
    la = cfg.get("locate3b") or {}
    if not la.get("enabled", False):
        return None
    return LocateAnythingDetector(
        model_path=resolve_deep_path(la.get("model_path")),
        classes=la.get("classes") or ["person", "car"],
        device=device_str,
        generation_mode=la.get("generation_mode", "hybrid"),
        max_side=int(la.get("max_side", 960)),
        max_new_tokens=int(la.get("max_new_tokens", 768)),
    )


def build_brain(deep, cfg: dict[str, Any]) -> DualBrain | None:
    dual_raw = cfg.get("dual_brain") or {}
    if deep is None or not dual_raw.get("enabled", True):
        return None
    return DualBrain(
        deep,
        DualBrainConfig(
            trigger_labels={str(x).lower() for x in dual_raw.get("trigger_labels", ["person", "car"])},
            force_labels={str(x).lower() for x in dual_raw.get("force_labels", [])},
            min_conf=float(dual_raw.get("min_conf", 0.35)),
            skip_above_conf=float(dual_raw.get("skip_above_conf", 0.90)),
            uncertain_below_conf=float(dual_raw.get("uncertain_below_conf", 0.65)),
            cooldown_sec=float(dual_raw.get("cooldown_sec", 2.5)),
            max_roi_boxes=int(dual_raw.get("max_roi_boxes", 3)),
            on_new_label=bool(dual_raw.get("on_new_label", True)),
            max_pending_jobs=int(dual_raw.get("max_pending_jobs", 1)),
        ),
    )
