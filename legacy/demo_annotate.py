"""Supervision 最小示例：创建检测框并画到图片上"""
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# 1) 造一张简单背景图（蓝天 + 灰地）
h, w = 480, 640
image = np.zeros((h, w, 3), dtype=np.uint8)
image[:300, :] = (210, 180, 140)  # 天空（BGR）
image[300:, :] = (80, 80, 80)  # 地面

# 2) 假检测结果：两个人（框 + 置信度 + 类别）
# xyxy = [左, 上, 右, 下]
detections = sv.Detections(
    xyxy=np.array(
        [
            [120, 180, 220, 420],  # 左边的人
            [380, 200, 500, 430],  # 右边的人
        ],
        dtype=np.float32,
    ),
    confidence=np.array([0.92, 0.81], dtype=np.float32),
    class_id=np.array([0, 0], dtype=int),
)

labels = [
    f"person {conf:.0%}"
    for conf in detections.confidence
]

# 3) 画框 + 文字
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)

annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(
    scene=annotated, detections=detections, labels=labels
)

# 4) 保存结果
out_path = OUT_DIR / "demo_result.jpg"
cv2.imwrite(str(out_path), annotated)

print(f"supervision 版本: {sv.__version__}")
print(f"检测到目标数: {len(detections)}")
print(f"结果已保存: {out_path.resolve()}")
