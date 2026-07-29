"""Shared types for VisionDesk screen agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float = 1.0
    track_id: int | None = None

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def area(self) -> float:
        x1, y1, x2, y2 = self.as_xyxy()
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionResult:
    boxes: list[Box] = field(default_factory=list)
    summary: str = ""
    backend: str = ""
    infer_ms: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.boxes)

    def labels(self) -> list[str]:
        return [b.label for b in self.boxes]
