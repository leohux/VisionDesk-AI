# VisionDesk AI

**Local-first desktop vision agent** — real-time screen perception, adaptive YOLO detection, and LocateAnything-3B reasoning only when it matters.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)](__version__.py)

> Privacy-first: runs on **your GPU**. No cloud upload required for the core loop.

```text
Screen / ROI
    ↓
YOLO realtime (≈34–38 FPS)
    ↓
Smart Trigger Router
    ↓
LocateAnything-3B  ← only when uncertain / new / forced
    ↓
AI Summary → SQLite Event Memory → Timeline Replay
```

## Why this exists

Most “AI vision” demos either:

- burn the GPU on every frame with a heavy VLM, or
- stay at YOLO-only and never explain *what is going on*.

VisionDesk sits in between: **fast eyes + selective brain**.

| Capability | Status |
|---|---|
| Real-time screen / ROI capture | ✅ |
| YOLO adaptive detection | ✅ |
| Adaptive Vision Routing (3B on demand) | ✅ |
| Event memory + snapshot replay | ✅ |
| Gradio cockpit + OpenCV desktop | ✅ |
| Offline replay / benchmark CLI | ✅ |
| Scene profiles (YAML) | ✅ |
| VLM Q&A over event history | 🚧 v0.2 |

## Performance (v0.1.0)

Measured on `demo/traffic.mp4` (377 frames, `traffic` profile, ~16GB CUDA):

| Mode | FPS | 3B calls | Reasoning efficiency | GPU |
|------|-----|----------|----------------------|-----|
| YOLO only | **37 FPS** | 0 | — | YOLO-only |
| YOLO + smart 3B | **23 FPS** | **6 / 377** | **62.8 frames / call** | ~8–10 GB |

≈ **98.3%** of deepen candidates skipped · queue pending ≤ 1

Reproduce:

```bash
python visiondesk.py replay demo/traffic.mp4 --profile traffic --json
python visiondesk.py replay demo/traffic.mp4 --profile traffic --no-deep --json
```

Numbers live in [`demo/benchmark/SUMMARY.json`](demo/benchmark/SUMMARY.json).

## Quick start

### 1. Install

```bash
git clone https://github.com/hkhk792/VisionDesk-AI.git
cd VisionDesk-AI

# Install CUDA PyTorch for your GPU first: https://pytorch.org
pip install -r requirements.txt

# Optional: place yolov8m.pt in the project root (Ultralytics can also auto-download)
```

### 2. Launch the Gradio cockpit (recommended)

```bash
python visiondesk.py ui
# or
python app_ui.py
```

Open **http://127.0.0.1:7860**

Suggested first run:

1. Profile `traffic`, **LocateAnything-3B OFF** → confirm live FPS
2. Open **Capture settings** → pick monitor / 1080p processing
3. Enable 3B when ready (~7GB VRAM on first load)
4. Watch **Event Timeline** → click an event to replay its snapshot

### 3. OpenCV desktop window

```bash
python main.py --profile general
python main.py --profile traffic --select-roi
python main.py --profile person --no-deep
```

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `1`–`4` | Switch profile |
| `d` | Toggle 3B deepen |
| `r` | Re-select ROI |
| `s` | Save snapshot |

### 4. Replay / benchmark any video or image

```bash
python visiondesk.py replay path/to/video.mp4 --profile traffic --json
python visiondesk.py replay shot.jpg --profile general --no-deep
```

## LocateAnything-3B setup

3B is optional. YOLO-only mode works out of the box.

1. Download / clone [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) locally.
2. Point VisionDesk at it **one** of these ways:

```bash
# environment variable (preferred)
set LOCATE_ANYTHING_PATH=C:\path\to\LocateAnything-3B   # Windows
export LOCATE_ANYTHING_PATH=/path/to/LocateAnything-3B  # Linux/macOS
```

```yaml
# or in profiles/*.yaml
locate3b:
  enabled: true
  model_path: LocateAnything-3B   # relative or absolute
```

3. Use the same Python env that can import that model’s dependencies, or install them into this venv.

## Architecture

```text
Gradio UI / CLI
       ↓
api/controller.py     ← engine facade, 3B singleton cache, health
       ↓
Capture → YOLO → DualBrain Router → LocateAnything-3B
       ↓
SQLite events + optional frame snapshots
```

```text
VisionDesk-AI/
├── visiondesk.py          # CLI: ui | replay
├── app_ui.py              # Gradio entry (crash-safe)
├── main.py                # OpenCV desktop entry
├── api/controller.py      # Vision Engine
├── core/                  # capture, dual_brain, health, events, factory
├── models/                # yolo, locate3b wrappers
├── profiles/              # YAML scenes
├── ui/gradio_app.py       # Cockpit
├── tools/replay.py        # Offline benchmark
└── demo/                  # sample media + measured benchmarks
```

## Profiles

| Profile | Best for |
|---------|----------|
| `general` | Everyday objects |
| `traffic` | Vehicles + people |
| `person` | People-focused |
| `security` | Higher-sensitivity / force labels |
| `gaming` | Example custom profile |

Edit `profiles/*.yaml` — user-facing sections: `detection`, `trigger`, `reasoning`, `capture`.

Default processing width is **1920px (1080p)** for stable FPS on 4K monitors; raise to 1440p / native in the UI when you need more detail.

## Requirements

- Python **3.10+**
- CUDA GPU recommended (16GB+ comfortable for YOLO + 3B)
- `yolov8m.pt` (or change weights in profile)
- LocateAnything-3B only if you enable deepen

## Roadmap

| Version | Focus |
|---------|--------|
| **v0.1.0** | Perception + adaptive 3B + memory + cockpit + benchmark |
| v0.2.0 | VLM Q&A over event history (“why did this alert fire?”) |
| later | Multi-monitor layouts, export packs, plugin profiles |

## Contributing

Issues and PRs welcome. Please keep changes focused:

- no drive-by refactors
- include a short note if you change trigger / profile defaults
- for performance claims, attach `visiondesk.py replay … --json` output

## License

[MIT](LICENSE) © 2026 [hkhk792](https://github.com/hkhk792)

YOLO / Ultralytics, Supervision, and LocateAnything-3B remain under their respective licenses.
