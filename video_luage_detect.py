"""Video dynamic detect: LocateAnything-3B (sampled frames) + Supervision annotate."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = Path(__import__("os").environ.get("LOCATE_ANYTHING_PATH", "LocateAnything-3B"))
VIDEO_IN = Path(r"D:\luage\public\hero-alibarbar.mp4")
OUT_DIR = Path(r"D:\supervision\output\luage_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT = OUT_DIR / "hero-alibarbar_detect.mp4"

# sample ~1 frame per second for speed (LocateAnything is not realtime)
SAMPLE_EVERY_SEC = 1.0
QUERY_CATS = ["vape", "e-cigarette", "disposable vape", "electronic cigarette"]
GENERATION_MODE = "fast"  # faster than hybrid for video
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.1
MAX_SIDE = 960

if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))


class Worker:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        print("Loading model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        print("Model ready.", flush=True)

    @torch.no_grad()
    def detect(self, image: Image.Image, categories: list[str]) -> str:
        cats = "</c>".join(categories)
        question = f"Locate all the instances that matches the following description: {cats}."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            generation_mode=GENERATION_MODE,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        return response[0] if isinstance(response, tuple) else response


def parse_boxes(answer: str, w: int, h: int) -> list[dict]:
    items = []
    for m in re.finditer(
        r"(?:<ref>(.*?)</ref>)?\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer or "",
        flags=re.DOTALL,
    ):
        label, x1, y1, x2, y2 = m.groups()
        items.append({
            "label": (label or "vape").strip() or "vape",
            "x1": int(x1) / 1000 * w,
            "y1": int(y1) / 1000 * h,
            "x2": int(x2) / 1000 * w,
            "y2": int(y2) / 1000 * h,
        })
    return items


def to_detections(items: list[dict]) -> sv.Detections:
    if not items:
        return sv.Detections.empty()
    xyxy = np.array([[it["x1"], it["y1"], it["x2"], it["y2"]] for it in items], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(len(items), dtype=np.float32),
        class_id=np.zeros(len(items), dtype=int),
    )


def annotate(frame_bgr: np.ndarray, items: list[dict]) -> np.ndarray:
    detections = to_detections(items)
    labels = [it["label"] for it in items]
    out = frame_bgr.copy()
    if len(detections) == 0:
        return out
    out = sv.BoxAnnotator(thickness=3).annotate(scene=out, detections=detections)
    out = sv.LabelAnnotator(text_thickness=2, text_scale=0.6).annotate(
        scene=out, detections=detections, labels=labels
    )
    return out


def resize_for_model(frame_bgr: np.ndarray) -> tuple[Image.Image, float]:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), scale


def main() -> None:
    if not VIDEO_IN.exists():
        raise SystemExit(f"视频不存�? {VIDEO_IN}")

    info = sv.VideoInfo.from_video_path(str(VIDEO_IN))
    sample_every = max(1, int(round(info.fps * SAMPLE_EVERY_SEC)))
    print(
        f"video={VIDEO_IN.name}  {info.width}x{info.height}  fps={info.fps:.1f}  "
        f"frames={info.total_frames}  sample_every={sample_every}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker = Worker(str(MODEL), device=device)

    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

    last_items: list[dict] = []
    events = []
    t0 = time.time()

    with sv.VideoSink(str(VIDEO_OUT), video_info=info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(str(VIDEO_IN))):
            need_detect = (frame_idx % sample_every == 0)
            if need_detect:
                pil, _ = resize_for_model(frame)
                t_det = time.time()
                answer = worker.detect(pil, QUERY_CATS)
                # parse in original frame coords: model saw resized image,
                # boxes are normalized 0-1000 so map to current pil size then to original
                items_small = parse_boxes(answer, pil.width, pil.height)
                # scale boxes back to original frame
                sx = frame.shape[1] / pil.width
                sy = frame.shape[0] / pil.height
                last_items = [
                    {
                        "label": it["label"],
                        "x1": it["x1"] * sx,
                        "y1": it["y1"] * sy,
                        "x2": it["x2"] * sx,
                        "y2": it["y2"] * sy,
                    }
                    for it in items_small
                ]
                dt = time.time() - t_det
                print(
                    f"frame {frame_idx}/{info.total_frames} detect boxes={len(last_items)} {dt:.2f}s",
                    flush=True,
                )
                events.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / max(info.fps, 1e-6), 2),
                    "boxes": len(last_items),
                    "elapsed_sec": round(dt, 2),
                })

            detections = to_detections(last_items)
            labels = [it["label"] for it in last_items]
            annotated = frame.copy()
            if len(detections):
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )
            # overlay status
            cv2.putText(
                annotated,
                f"frame {frame_idx} | boxes {len(last_items)} | sample {SAMPLE_EVERY_SEC}s",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            sink.write_frame(annotated)

    report = {
        "source": str(VIDEO_IN),
        "output": str(VIDEO_OUT),
        "sample_every_sec": SAMPLE_EVERY_SEC,
        "query": QUERY_CATS,
        "events": events,
        "total_elapsed_sec": round(time.time() - t0, 2),
    }
    report_path = OUT_DIR / "video_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE -> {VIDEO_OUT}", flush=True)
    print(f"report -> {report_path}", flush=True)
    print(f"elapsed {report['total_elapsed_sec']}s, detect calls {len(events)}", flush=True)


if __name__ == "__main__":
    main()
