"""Lightweight tracker wrapper around Supervision ByteTrack."""
from __future__ import annotations

import numpy as np
import supervision as sv

from core.types import Box, DetectionResult


class Tracker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._bt = sv.ByteTrack() if enabled else None

    def update(self, result: DetectionResult) -> DetectionResult:
        if not self.enabled or self._bt is None or not result.boxes:
            return result
        xyxy = np.array([b.as_xyxy() for b in result.boxes], dtype=np.float32)
        conf = np.array([b.confidence for b in result.boxes], dtype=np.float32)
        # map labels to temporary class ids
        label_to_id: dict[str, int] = {}
        class_ids = []
        for b in result.boxes:
            if b.label not in label_to_id:
                label_to_id[b.label] = len(label_to_id)
            class_ids.append(label_to_id[b.label])
        det = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.array(class_ids, dtype=int),
        )
        det = self._bt.update_with_detections(det)
        id_to_label = {v: k for k, v in label_to_id.items()}
        boxes: list[Box] = []
        tids = det.tracker_id if det.tracker_id is not None else [None] * len(det)
        for xy, cid, cf, tid in zip(det.xyxy, det.class_id, det.confidence, tids):
            boxes.append(
                Box(
                    x1=float(xy[0]),
                    y1=float(xy[1]),
                    x2=float(xy[2]),
                    y2=float(xy[3]),
                    label=id_to_label.get(int(cid), str(cid)),
                    confidence=float(cf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return DetectionResult(
            boxes=boxes,
            summary=result.summary,
            backend=result.backend,
            infer_ms=result.infer_ms,
            extras=result.extras,
        )
