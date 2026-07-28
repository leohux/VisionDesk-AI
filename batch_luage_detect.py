"""Batch detect e-cigarettes in D:\\luage images via LocateAnything-3B + Supervision."""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from gradio_client import Client, handle_file
from PIL import Image

# --- config ---
GRADIO_URL = "http://127.0.0.1:7860"
QUERY = "vape, e-cigarette, disposable vape, electronic cigarette"
TASK = "Object Detection"
MODE = "hybrid"
TEMPERATURE = 0.2
MAX_TOKENS = 512

OUT_DIR = Path(__file__).resolve().parent / "output" / "luage_detect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer product / review photos (skip tiny icons)
IMAGE_CANDIDATES = [
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


def parse_boxes(answer: str, w: int, h: int) -> list[dict]:
    items = []
    for m in re.finditer(
        r"(?:<ref>(.*?)</ref>)?\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer or "",
        flags=re.DOTALL,
    ):
        label, x1, y1, x2, y2 = m.groups()
        items.append(
            {
                "label": (label or "vape").strip() or "vape",
                "x1": int(x1) / 1000 * w,
                "y1": int(y1) / 1000 * h,
                "x2": int(x2) / 1000 * w,
                "y2": int(y2) / 1000 * h,
            }
        )
    return items


def to_supervision(items: list[dict]) -> sv.Detections:
    if not items:
        return sv.Detections.empty()
    xyxy = np.array([[it["x1"], it["y1"], it["x2"], it["y2"]] for it in items], dtype=np.float32)
    class_id = np.zeros(len(items), dtype=int)
    confidence = np.ones(len(items), dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def annotate(bgr: np.ndarray, items: list[dict]) -> np.ndarray:
    detections = to_supervision(items)
    labels = [it["label"] for it in items]
    out = bgr.copy()
    if len(detections) == 0:
        return out
    out = sv.BoxAnnotator(thickness=3).annotate(scene=out, detections=detections)
    out = sv.LabelAnnotator(text_thickness=2, text_scale=0.6).annotate(
        scene=out, detections=detections, labels=labels
    )
    return out


def load_bgr(path: Path) -> np.ndarray:
    # Pillow handles webp; OpenCV may not on all builds
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def as_upload_jpg(path: Path, tmp_dir: Path) -> Path:
    """Convert any input image to a moderate-size JPG for Gradio upload."""
    img = Image.open(path).convert("RGB")
    # Cap long edge to keep inference stable
    max_side = 1280
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    out = tmp_dir / f"{path.stem}_upload.jpg"
    img.save(out, format="JPEG", quality=92)
    return out


def main() -> None:
    images = [p for p in IMAGE_CANDIDATES if p.exists()]
    if not images:
        raise SystemExit("没有找到可检测的图片")

    tmp_dir = OUT_DIR / "_tmp_upload"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"连接 LocateAnything: {GRADIO_URL}", flush=True)
    client = Client(GRADIO_URL)
    print(f"共 {len(images)} 张图，查询: {QUERY}", flush=True)
    print(f"输出目录: {OUT_DIR}", flush=True)

    report = []
    for i, path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {path.name}", flush=True)
        t0 = time.time()
        try:
            upload = as_upload_jpg(path, tmp_dir)
            # outputs: image_out, output_stream, summary, prompt_used, raw
            result = client.predict(
                handle_file(str(upload)),
                TASK,
                QUERY,
                MODE,
                TEMPERATURE,
                float(MAX_TOKENS),
                api_name="/run_stream",
            )
            raw = result[4] if isinstance(result, (list, tuple)) and len(result) >= 5 else ""
            summary = result[2] if isinstance(result, (list, tuple)) and len(result) >= 3 else ""
        except Exception as e:
            print(f"  失败: {e}", flush=True)
            report.append({"image": str(path), "ok": False, "error": str(e), "boxes": 0})
            continue

        bgr = load_bgr(path)
        h, w = bgr.shape[:2]
        items = parse_boxes(str(raw), w, h)
        annotated = annotate(bgr, items)

        stem = path.stem
        out_img = OUT_DIR / f"{stem}_sv.jpg"
        out_json = OUT_DIR / f"{stem}_sv.json"
        cv2.imwrite(str(out_img), annotated)
        payload = {
            "source": str(path),
            "query": QUERY,
            "boxes": items,
            "elapsed_sec": round(time.time() - t0, 2),
            "raw_preview": str(raw)[:500],
            "summary_preview": str(summary)[:500],
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  检测到 {len(items)} 个框  耗时 {payload['elapsed_sec']}s  -> {out_img.name}", flush=True)
        report.append(
            {
                "image": str(path),
                "ok": True,
                "boxes": len(items),
                "out": str(out_img),
                "elapsed_sec": payload["elapsed_sec"],
            }
        )

    summary_path = OUT_DIR / "batch_report.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in report if r.get("ok"))
    boxes = sum(r.get("boxes", 0) for r in report)
    print(f"\n完成: {ok}/{len(report)} 张成功, 共 {boxes} 个框", flush=True)
    print(f"报告: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
