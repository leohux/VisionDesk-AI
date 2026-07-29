"""Drawing helpers using Supervision."""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from core.types import DetectionResult


def annotate(frame_bgr: np.ndarray, result: DetectionResult, title: str = "") -> np.ndarray:
    out = frame_bgr.copy()
    if result.boxes:
        xyxy = np.array([b.as_xyxy() for b in result.boxes], dtype=np.float32)
        conf = np.array([b.confidence for b in result.boxes], dtype=np.float32)
        class_id = np.zeros(len(result.boxes), dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        labels = []
        for b in result.boxes:
            tid = f"#{b.track_id} " if b.track_id is not None else ""
            labels.append(f"{tid}{b.label} {b.confidence:.0%}")
        out = sv.BoxAnnotator(thickness=2).annotate(scene=out, detections=det)
        out = sv.LabelAnnotator(text_thickness=1, text_scale=0.5).annotate(
            scene=out, detections=det, labels=labels
        )
    if title:
        cv2.putText(
            out,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if result.summary:
        # wrap summary at bottom
        y = out.shape[0] - 16
        cv2.putText(
            out,
            result.summary[:110],
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
    return out
