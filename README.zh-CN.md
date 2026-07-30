# VisionDesk AI

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <strong>本地优先的桌面视觉智能体</strong><br />
  YOLO 实时感知 + 按需 LocateAnything-3B 推理 + 事件记忆
</p>

<p align="center">
  <a href="https://github.com/leohux/VisionDesk-AI/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/leohux/VisionDesk-AI/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/leohux/VisionDesk-AI/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/leohux/VisionDesk-AI?display_name=tag&sort=semver" /></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" />
</p>

> 隐私优先：核心链路运行在**你自己的 GPU** 上，不需要把屏幕画面上传云端。

## 为什么做 VisionDesk？

多数桌面视觉 Demo 要么每一帧都调用重型 VLM、浪费 GPU，要么只停留在 YOLO 检测，无法解释画面里发生了什么。

VisionDesk 采用“**快眼睛 + 选择性大脑**”：

```text
屏幕 / ROI
    ↓
YOLO 实时检测（约 34–38 FPS）
    ↓
智能触发路由
    ↓
LocateAnything-3B（仅在不确定 / 新事件 / 强制时调用）
    ↓
AI 摘要 → SQLite 事件记忆 → 时间线回放
```

## 能力

| 能力 | 状态 |
|------|------|
| 实时屏幕 / ROI 捕获 | ✅ |
| YOLO 自适应检测 | ✅ |
| 按需 3B 视觉推理 | ✅ |
| 事件记忆与快照回放 | ✅ |
| Gradio 控制台 + OpenCV 桌面窗口 | ✅ |
| 离线回放 / Benchmark CLI | ✅ |
| YAML 场景配置 | ✅ |
| 基于历史事件的 VLM 问答 | 🚧 v0.2 |

## 性能（v0.1.0）

基于 `demo/traffic.mp4`（377 帧、`traffic` profile、约 16GB CUDA）：

| 模式 | FPS | 3B 调用 | 推理效率 | GPU |
|------|-----|---------|----------|-----|
| 仅 YOLO | **37 FPS** | 0 | — | YOLO-only |
| YOLO + 智能 3B | **23 FPS** | **6 / 377** | **62.8 帧 / 次** | 约 8–10 GB |

约 **98.3%** 的 deepen 候选被跳过，队列待处理 ≤ 1。原始数据见 [`demo/benchmark/SUMMARY.json`](demo/benchmark/SUMMARY.json)。

## 快速开始

```bash
git clone https://github.com/leohux/VisionDesk-AI.git
cd VisionDesk-AI

# 先按你的 GPU 安装 CUDA 版 PyTorch：https://pytorch.org
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

启动 Gradio 控制台：

```bash
python visiondesk.py ui
# 或
python app_ui.py
```

打开 `http://127.0.0.1:7860`。

推荐首次运行：

1. 选择 `traffic` profile，先关闭 LocateAnything-3B，确认实时 FPS。
2. 在 Capture settings 选择显示器与处理分辨率。
3. 准备好约 7GB 额外显存后再开启 3B。
4. 在 Event Timeline 中回放事件快照。

OpenCV 桌面窗口：

```bash
python main.py --profile general
python main.py --profile traffic --select-roi
python main.py --profile person --no-deep
```

离线回放与基准测试：

```bash
python visiondesk.py replay path/to/video.mp4 --profile traffic --json
python visiondesk.py replay shot.jpg --profile general --no-deep
```

## LocateAnything-3B 配置

3B 是可选项；YOLO-only 模式可独立运行。

1. 从 [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) 下载模型。
2. 用环境变量指定本地路径：

```bash
set LOCATE_ANYTHING_PATH=C:\path\to\LocateAnything-3B   # Windows
export LOCATE_ANYTHING_PATH=/path/to/LocateAnything-3B  # Linux/macOS
```

也可以在 `profiles/*.yaml` 中设置 `locate3b.model_path`。

## 目录结构

```text
VisionDesk-AI/
├── visiondesk.py          # CLI：ui | replay
├── app_ui.py              # Gradio 入口
├── main.py                # OpenCV 桌面入口
├── api/controller.py      # 视觉引擎 facade
├── core/                  # capture、dual_brain、health、events
├── models/                # YOLO、LocateAnything-3B 包装
├── profiles/              # YAML 场景配置
├── ui/gradio_app.py       # 控制台
├── tools/replay.py        # 离线 benchmark
└── demo/                  # 示例与实测数据
```

## 参与贡献

欢迎 Issues 和 PR。提交前请阅读：

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`SECURITY.md`](SECURITY.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- [`CHANGELOG.md`](CHANGELOG.md)

不要上传私人屏幕截图、事件数据库、模型权重、凭据或本地敏感路径。

## 许可证

[MIT](LICENSE) © 2026 [Leo Hu](https://github.com/leohux)

YOLO / Ultralytics、Supervision 和 LocateAnything-3B 分别遵循其自身许可证。
