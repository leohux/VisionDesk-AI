# Screen Live Detect (YOLO-World + Supervision)

本机屏幕区域实时检测：左侧画面画框，右侧类似 LocateAnything 的 Output Stream 输出。

## Features

- YOLO-World 开集检测（用文字指定类别，无需为每个新物体重新训练）
- Supervision 画框 + ByteTrack 跟踪
- 双屏自动分离：截内容屏、预览窗放到另一块屏（避免截到自己导致画面假死）
- 默认类别含人、宠物、常见物体、无人机、枪械

## Requirements

- Windows
- Python 3.10+
- NVIDIA GPU 推荐（CUDA）
- 依赖：`ultralytics`、`supervision`、`mss`、`opencv-python`、`pillow`、`torch`

## Quick start

```powershell
# 建议使用已装好 CUDA 的 Python 环境
python screen_yolo_world.py --device 0 --conf 0.25
```

常用参数：

```powershell
# 指定截屏显示器（mss 编号，可用脚本启动日志查看）
python screen_yolo_world.py --monitor 2 --ui-monitor 1

# 只截一块区域 x,y,width,height
python screen_yolo_world.py --region 100,50,720,1280

# 自定义类别
python screen_yolo_world.py --classes "person,dog,cat,drone,gun,phone"
```

按键：

- `q` / `ESC`：退出
- `s`：保存当前合成画面到 `output/screen_live/`
- `c`：清空右侧输出流
- `1` / `2` / `3`：置信度 0.15 / 0.25 / 0.40

## Other scripts

| Script | Description |
|--------|-------------|
| `demo_annotate.py` | Supervision 最小画框示例 |
| `batch_luage_direct.py` | LocateAnything-3B 批量图片检测 + Supervision 画框 |
| `video_luage_detect.py` | 视频抽帧检测示例 |
| `smoke_screen_yolo.py` | 屏幕截取单帧冒烟测试 |

## Notes

- YOLO-World 不是“识别一切自动分类”，每次只检测你通过 `--classes` / 默认列表指定的类别。
- 权重文件（如 `yolov8s-worldv2.pt`）首次运行会自动下载，不纳入本仓库。
- 枪械/无人机检测仅用于内容安全与研究演示，请遵守当地法律法规与平台规则。

## License

MIT
