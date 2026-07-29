"""屏幕实时检测：LocateAnything-3B（万物定位）+ Supervision

�?YOLO 慢，所以默认每�?interval 秒检一次，中间帧复用上一批框�?

推荐�?
  & "D:\\locate any\\.venv\\Scripts\\python.exe" D:\\supervision\\screen_locate_anything.py

按键�?
  q / ESC  退�?
  s        保存
  c        清空右侧输出
  1/2/3    抽检间隔 0.8 / 1.5 / 3.0 �?
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from mss import MSS
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = Path(__import__("os").environ.get("LOCATE_ANYTHING_PATH", "LocateAnything-3B"))
OUT_DIR = Path(__file__).resolve().parent / "output" / "screen_live"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

DEFAULT_CATS = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "bird",
    "horse",
    "phone",
    "laptop",
    "keyboard",
    "mouse",
    "tv",
    "cup",
    "bottle",
    "bowl",
    "chair",
    "couch",
    "bed",
    "dining table",
    "bag",
    "backpack",
    "handbag",
    "umbrella",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "traffic light",
    "stop sign",
    "bench",
    "drone",
    "gun",
    "rifle",
]

STREAM_BG = (0, 0, 0)
HEADER_BG = (17, 17, 17)
BORDER = (42, 42, 42)
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (170, 170, 170)
TOKEN_MTP = (124, 255, 107)
TOKEN_AR = (255, 209, 102)
ACCENT = (59, 130, 246)


class Worker:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        print("Loading LocateAnything-3B...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        print("Model ready.", flush=True)

    @torch.no_grad()
    def detect(self, image: Image.Image, categories: list[str], generation_mode: str = "fast") -> str:
        cats = "</c>".join(categories)
        # 明确要求 <ref>类别</ref>，否则模型常只吐�?<box>，界面就全是 object
        question = (
            f"Locate all the instances that matches the following description: {cats}. "
            f"For each instance, output <ref>category</ref><box><x1><y1><x2><y2></box> "
            f"where category must be one of: {', '.join(categories)}."
        )
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
            max_new_tokens=1024,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=0.1,
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        return response[0] if isinstance(response, tuple) else response


def _norm_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def parse_boxes(answer: str, w: int, h: int, categories: list[str] | None = None) -> list[dict]:
    """解析 LocateAnything 输出；尽量保�?<ref> 类别，缺省时回退�?""
    categories = categories or []
    fallback = categories[0] if len(categories) == 1 else "object"
    cat_set = {c.lower() for c in categories}
    items: list[dict] = []

    # 1) 标准�?ref>label</ref><box><a><b><c><d></box>
    for m in re.finditer(
        r"(?:<ref>(.*?)</ref>)?\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer or "",
        flags=re.DOTALL,
    ):
        label, a, b, c, d = m.groups()
        label = (label or "").strip()
        if not label or label.lower() in {"object", "box", "instance"}:
            # �?box 前方近邻文本里再找类别名
            start = max(0, m.start() - 40)
            head = (answer or "")[start:m.start()].lower()
            found = next((c for c in categories if c.lower() in head), "")
            label = found or fallback
        elif cat_set and label.lower() not in cat_set:
            # 模糊：包含关�?
            hit = next((c for c in categories if c.lower() in label.lower() or label.lower() in c.lower()), label)
            label = hit
        x1, y1, x2, y2 = _norm_box(
            int(a) / 1000 * w,
            int(b) / 1000 * h,
            int(c) / 1000 * w,
            int(d) / 1000 * h,
        )
        items.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    if items:
        return items

    # 2) 兜底：只有坐�?token �?
    nums = [int(x) for x in re.findall(r"<(\d+)>", answer or "")]
    for i in range(0, len(nums) - 3, 4):
        x1, y1, x2, y2 = _norm_box(
            nums[i] / 1000 * w,
            nums[i + 1] / 1000 * h,
            nums[i + 2] / 1000 * w,
            nums[i + 3] / 1000 * h,
        )
        items.append({"label": fallback, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
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


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("consola.ttf", "CascadiaMono.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class OutputStreamPanel:
    def __init__(self, width: int = 460, height: int = 720, max_lines: int = 100):
        self.width = width
        self.height = height
        self.lines: deque[tuple[str, tuple[int, int, int]]] = deque(maxlen=max_lines)
        self.font_header = load_font(15)
        self.font_body = load_font(14)
        self.font_small = load_font(12)
        self.frame_count = 0

    def clear(self) -> None:
        self.lines.clear()
        self.add("Output stream cleared.", TEXT_DIM)

    def add(self, text: str, color: tuple[int, int, int] = TEXT_MAIN) -> None:
        for raw in text.splitlines() or [""]:
            self.lines.append((raw, color))

    def push(self, items: list[dict], infer_ms: float, interval: float) -> None:
        self.frame_count += 1
        self.add(
            f"[detect {self.frame_count}] infer={infer_ms:.0f}ms interval={interval:.1f}s boxes={len(items)}",
            TEXT_DIM,
        )
        if not items:
            self.add("  <none>", TEXT_DIM)
            return
        for i, it in enumerate(items):
            color = TOKEN_MTP if i % 2 == 0 else TOKEN_AR
            self.add(
                f"  <det> {it['label']} "
                f"box=({it['x1']:.0f},{it['y1']:.0f})-({it['x2']:.0f},{it['y2']:.0f})</det>",
                color,
            )

    def render(self) -> np.ndarray:
        img = Image.new("RGB", (self.width, self.height), STREAM_BG)
        draw = ImageDraw.Draw(img)
        header_h = 40
        draw.rectangle([0, 0, self.width, header_h], fill=HEADER_BG)
        draw.line([0, header_h, self.width, header_h], fill=BORDER, width=1)
        draw.text((14, 11), "OUTPUT STREAM  ·  LocateAnything-3B", fill=TEXT_MAIN, font=self.font_header)
        x, y = 14, header_h + 12
        line_h = 20
        max_visible = max(1, (self.height - header_h - 70) // line_h)
        for text, color in list(self.lines)[-max_visible:]:
            shown = text if len(text) <= 62 else text[:59] + "..."
            draw.text((x, y), shown, fill=color, font=self.font_body)
            y += line_h
        footer_top = self.height - 52
        draw.line([0, footer_top, self.width, footer_top], fill=BORDER, width=1)
        draw.rectangle([0, footer_top, self.width, self.height], fill=(10, 10, 10))
        draw.text((14, footer_top + 10), "q quit | s save | c clear | 1/2/3 interval", fill=TEXT_DIM, font=self.font_small)
        draw.text((14, footer_top + 28), "LocateAnything-3B + Supervision  ·  sampled live", fill=ACCENT, font=self.font_small)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def compose(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    if left_bgr.shape[0] != h:
        s = h / left_bgr.shape[0]
        left_bgr = cv2.resize(left_bgr, (int(left_bgr.shape[1] * s), h))
    if right_bgr.shape[0] != h:
        s = h / right_bgr.shape[0]
        right_bgr = cv2.resize(right_bgr, (int(right_bgr.shape[1] * s), h))
    gap = np.full((h, 8, 3), (42, 42, 42), dtype=np.uint8)
    return np.hstack([left_bgr, gap, right_bgr])


def grab_bgr(sct: MSS, region: dict) -> np.ndarray:
    return cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def prepare_pil(frame_bgr: np.ndarray, max_side: int) -> tuple[Image.Image, float, float]:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), (w / frame_bgr.shape[1]), (h / frame_bgr.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=0, help="截屏屏；0=主屏")
    ap.add_argument("--ui-monitor", type=int, default=0, help="预览屏；0=另一�?)
    ap.add_argument("--classes", default=",".join(DEFAULT_CATS))
    ap.add_argument("--interval", type=float, default=1.5, help="抽检间隔�?)
    ap.add_argument("--mode", default="fast", choices=["fast", "hybrid", "slow"])
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--stream-width", type=int, default=460)
    args = ap.parse_args()

    cats = [c.strip() for c in args.classes.replace("�?, ",").split(",") if c.strip()]
    interval = float(args.interval)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker = Worker(str(MODEL), device=device)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    with MSS() as sct:
        physical = list(enumerate(sct.monitors[1:], start=1))
        print("displays:", flush=True)
        for i, m in physical:
            print(f"  [{i}] ({m['left']},{m['top']}) {m['width']}x{m['height']}"
                  f"{' primary' if m.get('is_primary') else ''}", flush=True)

        def pick_capture() -> int:
            if args.monitor > 0:
                return args.monitor
            for i, m in physical:
                if m.get("is_primary"):
                    return i
            return physical[0][0]

        def pick_ui(cap: int) -> int:
            if args.ui_monitor > 0:
                return args.ui_monitor
            others = [i for i, _ in physical if i != cap]
            return others[0] if others else cap

        cap_idx = pick_capture()
        ui_idx = pick_ui(cap_idx)
        mon = sct.monitors[cap_idx]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        print(f"capture={cap_idx} ui={ui_idx} cats={cats}", flush=True)

        preview_w = min(720, region["width"])
        scale = preview_w / max(region["width"], 1)
        preview_h = int(region["height"] * scale)
        stream = OutputStreamPanel(width=args.stream_width, height=max(preview_h, 640))
        stream.add("LocateAnything-3B ready. Sampling screen...", TEXT_DIM)
        stream.add(f"classes: {', '.join(cats)}", TEXT_DIM)

        win = "LocateAnything-3B + Output Stream"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        win_w = preview_w + 8 + args.stream_width
        win_h = preview_h
        cv2.resizeWindow(win, win_w, win_h)
        ui_mon = sct.monitors[ui_idx]
        cv2.moveWindow(
            win,
            int(ui_mon["left"] + max(40, (ui_mon["width"] - win_w) // 2)),
            int(ui_mon["top"] + max(40, (ui_mon["height"] - win_h) // 2)),
        )

        last_items: list[dict] = []
        last_detect_t = 0.0
        busy = False
        lock = threading.Lock()

        def run_detect(frame_snap: np.ndarray) -> None:
            nonlocal last_items, last_detect_t, busy
            try:
                pil, sx, sy = prepare_pil(frame_snap, args.max_side)
                t0 = time.perf_counter()
                answer = worker.detect(pil, cats, generation_mode=args.mode)
                dt = (time.perf_counter() - t0) * 1000
                small = parse_boxes(answer, pil.width, pil.height, categories=cats)
                items = [
                    {
                        "label": it["label"],
                        "x1": it["x1"] * sx,
                        "y1": it["y1"] * sy,
                        "x2": it["x2"] * sx,
                        "y2": it["y2"] * sy,
                    }
                    for it in small
                ]
                with lock:
                    last_items = items
                    last_detect_t = time.time()
                    stream.push(items, infer_ms=dt, interval=interval)
                print(f"detect boxes={len(items)} {dt:.0f}ms", flush=True)
            except Exception as e:
                with lock:
                    stream.add(f"ERROR: {e}", (255, 100, 100))
                print("ERROR", e, flush=True)
            finally:
                busy = False

        while True:
            frame = grab_bgr(sct, region)
            now = time.time()

            if (not busy) and (now - last_detect_t >= interval):
                busy = True
                threading.Thread(target=run_detect, args=(frame.copy(),), daemon=True).start()

            with lock:
                items_view = list(last_items)
                detect_t = last_detect_t

            detections = to_detections(items_view)
            labels = [it["label"] for it in items_view]
            annotated = frame.copy()
            if len(detections):
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )
            age = time.time() - detect_t if detect_t else 0
            cv2.putText(
                annotated,
                f"LA-3B | boxes {len(items_view)} | every {interval:.1f}s | age {age:.1f}s | q=quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            left = cv2.resize(annotated, (preview_w, preview_h)) if scale != 1.0 else annotated
            if stream.height != left.shape[0]:
                stream.height = left.shape[0]
            composed = compose(left, stream.render())
            cv2.imshow(win, composed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = OUT_DIR / f"la3b_{int(time.time())}.jpg"
                cv2.imwrite(str(path), composed)
                with lock:
                    stream.add(f"saved {path.name}", ACCENT)
            if key == ord("c"):
                with lock:
                    stream.clear()
            if key == ord("1"):
                interval = 0.8
                with lock:
                    stream.add(f"interval -> {interval}s", ACCENT)
            if key == ord("2"):
                interval = 1.5
                with lock:
                    stream.add(f"interval -> {interval}s", ACCENT)
            if key == ord("3"):
                interval = 3.0
                with lock:
                    stream.add(f"interval -> {interval}s", ACCENT)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
