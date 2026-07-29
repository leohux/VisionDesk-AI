"""Screen / ROI capture helpers."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from mss import MSS


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_monitor(cls, monitor: dict) -> "Region":
        return cls(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = MSS()

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def list_monitors(self) -> list[tuple[int, dict]]:
        return list(enumerate(self._sct.monitors[1:], start=1))

    def monitor_region(self, index: int) -> Region:
        return Region.from_monitor(self._sct.monitors[index])

    def primary_index(self) -> int:
        for i, m in self.list_monitors():
            if m.get("is_primary"):
                return i
        return 1

    def other_index(self, capture_idx: int) -> int:
        others = [i for i, _ in self.list_monitors() if i != capture_idx]
        return others[0] if others else capture_idx

    def grab(self, region: Region) -> np.ndarray:
        shot = np.asarray(self._sct.grab(region.as_mss()))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def select_roi_interactive(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """OpenCV drag-select ROI on a snapshot. Returns x,y,w,h in frame coords."""
    r = cv2.selectROI("Select ROI (ENTER confirm, ESC cancel)", frame_bgr, showCrosshair=True)
    cv2.destroyWindow("Select ROI (ENTER confirm, ESC cancel)")
    x, y, w, h = [int(v) for v in r]
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h
