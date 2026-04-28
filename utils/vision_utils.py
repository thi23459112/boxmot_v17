# coding=utf-8
"""
vision_utils.py
影像幾何/工具函式：縮放點位、裁切 bbox、取類別名稱
"""

import numpy as np

def scale_points(points: np.ndarray, w: int, h: int, base_w: int, base_h: int) -> np.ndarray:
    # 點位是用 base_w/base_h 畫的，先算縮放倍率
    sx = float(w) / float(base_w)
    sy = float(h) / float(base_h)

    # 用 float 計算後再轉 int，避免整數除法誤差
    pts = points.astype(np.float32).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts.astype(np.int32)


def clamp_crop(frame, x1, y1, x2, y2):
    # OpenCV frame shape = (H, W, C)
    H, W = frame.shape[:2]

    # bbox 邊界夾限，避免裁切越界
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))

    # bbox 無效就回 None（避免後面送空圖到模型）
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def class_name(names, cls_id: int) -> str:
    # Ultralytics names 多數是 dict，但也可能是 list
    if isinstance(names, dict):
        return names.get(cls_id, f"class_{cls_id}")
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return f"class_{cls_id}"
