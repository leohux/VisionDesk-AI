# Demo 录制清单（可选 GIF）

仓库 README 已用静态截图（`docs/screenshots/`）。若还想补一段循环 GIF，可按下面录。

## 准备
- [ ] 只保留一个 VisionDesk 进程
- [ ] 打开 UI：http://127.0.0.1:7860
- [ ] Profile：`traffic`；开始时 3B 关闭
- [ ] 内容窗口放车流/行人视频（不要截 Gradio 自己）

## 分镜（约 30 秒）
| 时间 | 展示内容 |
|------|----------|
| 0–5 秒 | Status：Engine Running · YOLO · FPS |
| 5–15 秒 | 实时检测框 |
| 15–25 秒 | 打开 3B → AI Understanding |
| 25–30 秒 | Event Timeline → 点事件回放 |

## 导出
可选：`demo/traffic.gif`（README 目前不依赖此文件）。
