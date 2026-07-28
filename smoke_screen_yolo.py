"""One-shot smoke test for YOLO-World screen capture."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mss
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

OUT = Path(__file__).resolve().parent / "output" / "screen_live"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = [
    "vape",
    "e-cigarette",
    "disposable vape",
    "electronic cigarette",
    "phone",
    "person",
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    print("loading yolov8s-worldv2.pt ...", flush=True)
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)
    print("model ready", flush=True)

    with mss.mss() as sct:
        mon = sct.monitors[1]
        w, h = 960, 720
        region = {
            "left": mon["left"] + max(0, (mon["width"] - w) // 2),
            "top": mon["top"] + max(0, (mon["height"] - h) // 2),
            "width": min(w, mon["width"]),
            "height": min(h, mon["height"]),
        }
        frame = cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)

    t0 = time.perf_counter()
    results = model.predict(frame, conf=0.2, imgsz=640, device=device, verbose=False)
    dt = time.perf_counter() - t0
    det = sv.Detections.from_ultralytics(results[0])
    labels = [f"{CLASSES[int(cid)]} {float(cf):.0%}" for cid, cf in zip(det.class_id, det.confidence)]
    ann = sv.BoxAnnotator(thickness=2).annotate(scene=frame.copy(), detections=det)
    if len(det):
        ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.5).annotate(
            scene=ann, detections=det, labels=labels
        )
    path = OUT / "smoke_test.jpg"
    cv2.imwrite(str(path), ann)
    print(f"boxes={len(det)} infer={dt:.3f}s fps~{1/dt:.1f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
