# Demo 录制清单（30 秒）

## 准备
- [ ] 只保留一个 VisionDesk 进程（`nvidia-smi` 显存干净）
- [ ] 打开 UI：http://127.0.0.1:7860
- [ ] Profile 选择：`traffic`
- [ ] 开始时 3B 复选框：关闭（OFF）
- [ ] 内容窗口：交通 / CCTV 视频（不要录 Gradio 窗口本身）

## 分镜脚本
| 时间 | 展示内容 |
|------|----------|
| 0–5 秒 | 状态栏：Engine Running · YOLO Loaded · 3B Ready（Start 之后再打开 3B） |
| 5–15 秒 | 实时检测框：car / truck / person 及置信度 |
| 15–25 秒 | 状态栏：3B 触发原因 + AI Understanding 文字 |
| 25–30 秒 | 事件时间线 → 点击事件 → 回放截图 |

## 导出
保存为 `demo/traffic.gif`（或 mp4），用于 README 第一屏。
