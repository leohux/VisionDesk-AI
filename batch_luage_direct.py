"""Direct batch: LocateAnything-3B detect + Supervision annotate (no Gradio)."""
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

MODEL = Path(r"D:\locate any\LocateAnything-3B")
OUT_DIR = Path(r"D:\supervision\output\luage_detect")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

QUERY_CATS = ["vape", "e-cigarette", "disposable vape", "electronic cigarette"]
GENERATION_MODE = "hybrid"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

IMAGES = [
    Path(r"D:\luage\dist\assets\hero-device-C7GwudRi.png"),
    Path(r"D:\luage\dist\assets\custom-5-pack-BE4VxD9x.png"),
    Path(r"D:\luage\dist\assets\custom-20-pack-BaAWLn2l.png"),
    Path(r"D:\luage\dist\assets\flavor-mango-CxuRB-M_.png"),
    Path(r"D:\luage\dist\assets\flavor-grape-jupuAUm7.png"),
    Path(r"D:\luage\public\authenticity\packaging-reference.png"),
    Path(r"D:\luage\public\reviews\review-01.jpg"),
    Path(r"D:\luage\public\reviews\review-02.jpg"),
    Path(r"D:\luage\public\reviews\review-04.jpg"),
    Path(r"D:\luage\public\reviews\review-07.jpg"),
]


class Worker:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        print("Loading tokenizer/processor...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading model onto {device}...", flush=True)
        self.model = (
            AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        print("Model ready.", flush=True)

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str) -> str:
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

    def detect(self, image: Image.Image, categories: list[str]) -> str:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt)


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


def annotate_bgr(bgr: np.ndarray, items: list[dict]) -> np.ndarray:
    if not items:
        return bgr.copy()
    xyxy = np.array([[it["x1"], it["y1"], it["x2"], it["y2"]] for it in items], dtype=np.float32)
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(len(items), dtype=np.float32),
        class_id=np.zeros(len(items), dtype=int),
    )
    labels = [it["label"] for it in items]
    out = sv.BoxAnnotator(thickness=3).annotate(scene=bgr.copy(), detections=detections)
    out = sv.LabelAnnotator(text_thickness=2, text_scale=0.6).annotate(
        scene=out, detections=detections, labels=labels
    )
    return out


def prepare_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    max_side = 1280
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def main() -> None:
    images = [p for p in IMAGES if p.exists()]
    if not images:
        raise SystemExit("没有找到图片")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker = Worker(str(MODEL), device=device)
    report = []

    for i, path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {path.name}", flush=True)
        t0 = time.time()
        try:
            pil = prepare_image(path)
            answer = worker.detect(pil, QUERY_CATS)
            items = parse_boxes(answer, pil.width, pil.height)
            # draw on prepared size, then optionally keep that size
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            annotated = annotate_bgr(bgr, items)
            out_img = OUT_DIR / f"{path.stem}_sv.jpg"
            out_json = OUT_DIR / f"{path.stem}_sv.json"
            cv2.imwrite(str(out_img), annotated)
            payload = {
                "source": str(path),
                "query": QUERY_CATS,
                "boxes": items,
                "elapsed_sec": round(time.time() - t0, 2),
                "raw_preview": str(answer)[:800],
            }
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  boxes={len(items)}  {payload['elapsed_sec']}s  -> {out_img.name}", flush=True)
            report.append({"image": str(path), "ok": True, "boxes": len(items), "out": str(out_img), "elapsed_sec": payload["elapsed_sec"]})
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)
            report.append({"image": str(path), "ok": False, "error": str(e), "boxes": 0})
            # clear CUDA fragmentation if any
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = OUT_DIR / "batch_report.json"
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in report if r.get("ok"))
    boxes = sum(r.get("boxes", 0) for r in report)
    print(f"\nDONE {ok}/{len(report)} ok, {boxes} boxes total", flush=True)
    print(f"report: {summary}", flush=True)


if __name__ == "__main__":
    main()
