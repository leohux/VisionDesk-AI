"""本机屏幕区域实时检测：YOLO-World + Supervision

左侧：实时画面画框
右侧：类似 LocateAnything 的 Output Stream 代码输出面板

推荐：
  & "D:\\locate any\\.venv\\Scripts\\python.exe" D:\\supervision\\screen_yolo_world.py --device 0

按键：
  q / ESC  退出
  s        保存当前合成画面
  c        清空右侧输出流
  1/2/3    置信度 0.15 / 0.25 / 0.40
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mss
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

OUT_DIR = Path(__file__).resolve().parent / "output" / "screen_live"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CLASSES = [
    # people / pets
    "person",
    "dog",
    "cat",
    "bird",
    "fish",
    "rabbit",
    "hamster",
    # common objects
    "phone",
    "laptop",
    "cup",
    "bottle",
    "chair",
    "book",
    "bag",
    "keyboard",
    "mouse",
    "tv",
    "car",
    "bicycle",
    # aerial / weapons
    "drone",
    "UAV",
    "gun",
    "rifle",
    "pistol",
    "firearm",
]

# LocateAnything 风格配色 (BGR for OpenCV panels / RGB for PIL)
STREAM_BG = (0, 0, 0)
HEADER_BG = (17, 17, 17)
BORDER = (42, 42, 42)
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (170, 170, 170)
TOKEN_MTP = (124, 255, 107)  # green-ish like LA mtp tokens
TOKEN_AR = (255, 209, 102)  # yellow-ish like LA ar tokens
ACCENT = (59, 130, 246)


def parse_region(text: str | None) -> dict | None:
    if not text:
        return None
    parts = [int(x.strip()) for x in text.replace("，", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("--region 格式应为 x,y,width,height")
    x, y, w, h = parts
    return {"left": x, "top": y, "width": w, "height": h}


def grab_bgr(sct: mss.MSS, region: dict) -> np.ndarray:
    shot = np.asarray(sct.grab(region))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def build_model(weights: str, classes: list[str]) -> YOLO:
    model = YOLO(weights)
    model.set_classes(classes)
    return model


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("consola.ttf", "CascadiaMono.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class OutputStreamPanel:
    """右侧代码流面板，风格接近 LocateAnything Output Stream。"""

    def __init__(self, width: int = 480, height: int = 720, max_lines: int = 80):
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

    def push_detections(
        self,
        detections: sv.Detections,
        classes: list[str],
        fps: float,
        conf_th: float,
        infer_ms: float,
    ) -> None:
        self.frame_count += 1
        n = len(detections)
        self.add(
            f"[frame {self.frame_count}] fps={fps:.1f} infer={infer_ms:.0f}ms conf>={conf_th:.2f} boxes={n}",
            TEXT_DIM,
        )
        if n == 0:
            self.add("  <none>", TEXT_DIM)
            return

        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else [None] * n
        )
        for i, (xyxy, cid, conf_i, tid) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence, tracker_ids)
        ):
            name = classes[int(cid)] if cid is not None and int(cid) < len(classes) else str(cid)
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            tid_s = f"id={int(tid)} " if tid is not None else ""
            # 交替颜色，类似 LA 的 mtp/ar token
            color = TOKEN_MTP if i % 2 == 0 else TOKEN_AR
            self.add(
                f"  <det> {tid_s}{name} {float(conf_i):.0%} "
                f"box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})</det>",
                color,
            )

    def render(self) -> np.ndarray:
        img = Image.new("RGB", (self.width, self.height), STREAM_BG)
        draw = ImageDraw.Draw(img)

        header_h = 40
        draw.rectangle([0, 0, self.width, header_h], fill=HEADER_BG)
        draw.line([0, header_h, self.width, header_h], fill=BORDER, width=1)
        draw.text((14, 11), "OUTPUT STREAM", fill=TEXT_MAIN, font=self.font_header)

        # 正文区
        x = 14
        y = header_h + 12
        line_h = 20
        usable = self.height - header_h - 70
        max_visible = max(1, usable // line_h)
        visible = list(self.lines)[-max_visible:]
        for text, color in visible:
            # 简单截断过长行
            shown = text if len(text) <= 64 else text[:61] + "..."
            draw.text((x, y), shown, fill=color, font=self.font_body)
            y += line_h

        # 底部状态条
        footer_top = self.height - 52
        draw.line([0, footer_top, self.width, footer_top], fill=BORDER, width=1)
        draw.rectangle([0, footer_top, self.width, self.height], fill=(10, 10, 10))
        draw.text(
            (14, footer_top + 10),
            "q quit | s save | c clear | 1/2/3 conf",
            fill=TEXT_DIM,
            font=self.font_small,
        )
        draw.text(
            (14, footer_top + 28),
            "YOLO-World + Supervision  ·  live screen",
            fill=ACCENT,
            font=self.font_small,
        )

        # PIL RGB -> OpenCV BGR
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def compose(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    if left_bgr.shape[0] != h:
        scale = h / left_bgr.shape[0]
        left_bgr = cv2.resize(left_bgr, (int(left_bgr.shape[1] * scale), h))
    if right_bgr.shape[0] != h:
        scale = h / right_bgr.shape[0]
        right_bgr = cv2.resize(right_bgr, (int(right_bgr.shape[1] * scale), h))
    gap = np.full((h, 8, 3), BORDER[::-1] if False else (42, 42, 42), dtype=np.uint8)
    # BORDER was RGB; use gray in BGR
    gap[:] = (42, 42, 42)
    return np.hstack([left_bgr, gap, right_bgr])


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen detect with LocateAnything-style output stream")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--region", default="", help="x,y,width,height；空=按 --monitor 截屏")
    ap.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="截屏显示器编号（mss）；0=自动选主屏/内容屏。不要和预览窗在同一块屏",
    )
    ap.add_argument(
        "--ui-monitor",
        type=int,
        default=0,
        help="预览窗口放在哪块屏；0=自动选另一块屏（避免截到自己导致画面假死）",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    ap.add_argument("--stream-width", type=int, default=460, help="右侧输出面板宽度")
    ap.add_argument("--log-every", type=int, default=3, help="每 N 帧写一次右侧流，避免刷太快")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.replace("，", ",").split(",") if c.strip()]
    if not classes:
        raise SystemExit("类别列表为空")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conf = float(args.conf)

    # 权重优先用 supervision 目录下已下载的文件
    weights = args.weights
    local_w = Path(__file__).resolve().parent / weights
    if local_w.exists():
        weights = str(local_w)

    print(f"device={device}  weights={weights}", flush=True)
    print(f"classes={classes}", flush=True)
    model = build_model(weights, classes)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    tracker = sv.ByteTrack()

    with mss.MSS() as sct:
        # monitors[0] = 全部虚拟屏；1..N = 各物理屏
        physical = list(enumerate(sct.monitors[1:], start=1))
        print("displays:", flush=True)
        for i, m in physical:
            flag = " primary" if m.get("is_primary") else ""
            print(
                f"  [{i}] ({m['left']},{m['top']}) {m['width']}x{m['height']}{flag}",
                flush=True,
            )

        def pick_capture_idx() -> int:
            if args.monitor > 0:
                return args.monitor
            for i, m in physical:
                if m.get("is_primary"):
                    return i
            return physical[0][0]

        def pick_ui_idx(capture_idx: int) -> int:
            if args.ui_monitor > 0:
                return args.ui_monitor
            others = [i for i, _ in physical if i != capture_idx]
            return others[0] if others else capture_idx

        capture_idx = pick_capture_idx()
        ui_idx = pick_ui_idx(capture_idx)
        if ui_idx == capture_idx and len(physical) > 1:
            print(
                "WARN: 预览窗和截屏在同一块屏，画面容易看起来不刷新；建议双屏分离",
                flush=True,
            )

        region = parse_region(args.region) if args.region else None
        if region is None:
            mon = sct.monitors[capture_idx]
            region = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        print(
            f"capture monitor={capture_idx} region=({region['left']},{region['top']}) "
            f"{region['width']}x{region['height']}",
            flush=True,
        )
        print(f"ui monitor={ui_idx}", flush=True)

        preview_w = min(720, region["width"])
        scale = preview_w / max(region["width"], 1)
        preview_h = int(region["height"] * scale)
        stream = OutputStreamPanel(width=args.stream_width, height=max(preview_h, 640))
        stream.add("Ready. Waiting for detections...", TEXT_DIM)
        stream.add(f"classes: {', '.join(classes)}", TEXT_DIM)
        stream.add(f"capture=monitor {capture_idx}  ui=monitor {ui_idx}", ACCENT)

        win = "YOLO-World + Output Stream"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        win_w = preview_w + 8 + args.stream_width
        win_h = preview_h
        cv2.resizeWindow(win, win_w, win_h)
        # 把预览窗放到另一块屏，避免截到自己导致“画面不更新”
        ui_mon = sct.monitors[ui_idx]
        ui_x = int(ui_mon["left"] + max(40, (ui_mon["width"] - win_w) // 2))
        ui_y = int(ui_mon["top"] + max(40, (ui_mon["height"] - win_h) // 2))
        cv2.moveWindow(win, ui_x, ui_y)

        fps_ema = 0.0
        tick = 0
        while True:
            t0 = time.perf_counter()
            frame = grab_bgr(sct, region)

            results = model.predict(
                frame,
                conf=conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)

            labels = []
            tracker_ids = (
                detections.tracker_id
                if detections.tracker_id is not None
                else [None] * len(detections)
            )
            for cid, conf_i, tid in zip(detections.class_id, detections.confidence, tracker_ids):
                name = classes[int(cid)] if cid is not None and int(cid) < len(classes) else str(cid)
                tid_s = f"#{int(tid)} " if tid is not None else ""
                labels.append(f"{tid_s}{name} {float(conf_i):.0%}")

            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            if len(detections):
                annotated = label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )

            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema == 0 else (fps_ema * 0.85 + fps * 0.15)
            cv2.putText(
                annotated,
                f"FPS {fps_ema:.1f} | boxes {len(detections)} | conf {conf:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            left = (
                cv2.resize(annotated, (preview_w, preview_h))
                if scale != 1.0
                else annotated
            )
            # 右侧高度跟随左侧
            if stream.height != left.shape[0]:
                stream.height = left.shape[0]

            tick += 1
            if tick % max(1, args.log_every) == 0:
                stream.push_detections(
                    detections,
                    classes,
                    fps=fps_ema,
                    conf_th=conf,
                    infer_ms=dt * 1000,
                )

            composed = compose(left, stream.render())
            cv2.imshow(win, composed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = OUT_DIR / f"capture_{int(time.time())}.jpg"
                cv2.imwrite(str(path), composed)
                print(f"saved {path}", flush=True)
                stream.add(f"saved -> {path.name}", ACCENT)
            if key == ord("c"):
                stream.clear()
            if key == ord("1"):
                conf = 0.15
                stream.add(f"conf -> {conf}", ACCENT)
            if key == ord("2"):
                conf = 0.25
                stream.add(f"conf -> {conf}", ACCENT)
            if key == ord("3"):
                conf = 0.40
                stream.add(f"conf -> {conf}", ACCENT)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
