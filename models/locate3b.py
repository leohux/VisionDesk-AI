"""LocateAnything-3B open-world grounding adapter."""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from core.types import Box, DetectionResult

DEFAULT_MODEL = Path(os.environ.get("LOCATE_ANYTHING_PATH", "LocateAnything-3B"))


class LocateAnythingDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        classes: list[str] | None = None,
        device: str | None = None,
        generation_mode: str = "hybrid",
        max_side: int = 960,
        max_new_tokens: int = 768,
    ) -> None:
        self.model_path = Path(model_path)
        if str(self.model_path) not in sys.path:
            sys.path.insert(0, str(self.model_path))
        self.classes = classes or ["person", "car", "dog", "cat", "phone", "cup", "chair"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_mode = generation_mode
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading LocateAnything-3B on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                str(self.model_path), torch_dtype=self.dtype, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        print("LocateAnything-3B ready.", flush=True)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> DetectionResult:
        pil, sx, sy = self._prepare(frame_bgr)
        cats = "</c>".join(self.classes)
        question = (
            f"Locate all the instances that matches the following description: {cats}. "
            f"For each instance, output <ref>category</ref><box><x1><y1><x2><y2></box> "
            f"where category must be one of: {', '.join(self.classes)}."
        )
        t0 = time.perf_counter()
        answer = self._generate(pil, question)
        infer_ms = (time.perf_counter() - t0) * 1000
        boxes = parse_boxes(answer, pil.width, pil.height, self.classes)
        # scale back to original frame
        scaled = [
            Box(
                x1=b.x1 * sx,
                y1=b.y1 * sy,
                x2=b.x2 * sx,
                y2=b.y2 * sy,
                label=b.label,
                confidence=b.confidence,
            )
            for b in boxes
        ]
        summary = narrate(scaled, answer)
        return DetectionResult(
            boxes=scaled,
            summary=summary,
            backend="locate-anything-3b",
            infer_ms=infer_ms,
            extras={"raw_preview": (answer or "")[:400]},
        )

    def _prepare(self, frame_bgr: np.ndarray) -> tuple[Image.Image, float, float]:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return pil, w / pil.width, h / pil.height

    def _generate(self, image: Image.Image, question: str) -> str:
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
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            generation_mode=self.generation_mode,
            temperature=0.1,
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        return response[0] if isinstance(response, tuple) else response


def parse_boxes(answer: str, w: int, h: int, categories: list[str]) -> list[Box]:
    fallback = categories[0] if len(categories) == 1 else "object"
    cat_set = {c.lower() for c in categories}
    items: list[Box] = []
    for m in re.finditer(
        r"(?:<ref>(.*?)</ref>)?\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer or "",
        flags=re.DOTALL,
    ):
        label, a, b, c, d = m.groups()
        label = (label or "").strip()
        if not label or label.lower() in {"object", "box", "instance"}:
            head = (answer or "")[max(0, m.start() - 40) : m.start()].lower()
            label = next((x for x in categories if x.lower() in head), fallback)
        elif cat_set and label.lower() not in cat_set:
            label = next(
                (x for x in categories if x.lower() in label.lower() or label.lower() in x.lower()),
                label,
            )
        x1, y1, x2, y2 = sorted_box(
            int(a) / 1000 * w,
            int(b) / 1000 * h,
            int(c) / 1000 * w,
            int(d) / 1000 * h,
        )
        items.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=0.9))
    return items


def sorted_box(x1, y1, x2, y2):
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def narrate(boxes: list[Box], raw: str) -> str:
    if not boxes:
        return "LocateAnything-3B: no matching objects in the ROI."
    counts: dict[str, int] = {}
    for b in boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
    parts = [f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    # simple spatial hint using largest box
    main = max(boxes, key=lambda b: b.area())
    cx = (main.x1 + main.x2) / 2
    # relative position unknown without frame width; keep generic
    return (
        f"LocateAnything-3B grounded {', '.join(parts)}. "
        f"Primary focus appears to be '{main.label}'."
    )
