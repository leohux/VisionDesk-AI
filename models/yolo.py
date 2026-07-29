"""Ultralytics YOLO / YOLO-World detector adapters."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from core.types import Box, DetectionResult


class YoloDetector:
    def __init__(
        self,
        weights: str,
        classes: list[str] | None = None,
        backend: str = "coco",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | int = 0,
        max_det: int = 300,
        keep_class_ids: set[int] | None = None,
    ) -> None:
        path = Path(weights)
        self.weights = str(path if path.exists() else weights)
        self.backend = backend
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.keep_class_ids = keep_class_ids
        self.classes = classes or []
        self.model = YOLO(self.weights)
        if backend == "world" and self.classes:
            self.model.set_classes(self.classes)

    def predict(self, frame_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        result = self.model.predict(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        names = result.names
        boxes: list[Box] = []
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(boxes=[], backend=f"yolo-{self.backend}", infer_ms=infer_ms)

        for xyxy, cls_id, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
        ):
            cid = int(cls_id)
            if self.keep_class_ids is not None and cid not in self.keep_class_ids:
                continue
            if self.backend == "world":
                label = self.classes[cid] if cid < len(self.classes) else str(cid)
            else:
                label = str(names.get(cid, cid))
            boxes.append(
                Box(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    label=label,
                    confidence=float(conf),
                )
            )
        summary = _summarize(boxes, prefix="YOLO")
        return DetectionResult(
            boxes=boxes,
            summary=summary,
            backend=f"yolo-{self.backend}",
            infer_ms=infer_ms,
        )


def _summarize(boxes: list[Box], prefix: str) -> str:
    if not boxes:
        return f"{prefix}: no objects."
    counts: dict[str, int] = {}
    for b in boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
    parts = [f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return f"{prefix} saw " + ", ".join(parts) + "."
