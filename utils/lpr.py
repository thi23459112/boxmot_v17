# coding=utf-8
"""
lpr.py
只負責：車牌影像 -> 車牌字串（字元偵測 + NMS + 排序組字）
"""

import cv2
import numpy as np
from . import config

def process_plate_number(num_model, plate_img, conf_thresh: float) -> str:
    # 空圖或模型不存在，直接回空字串
    if plate_img is None or plate_img.size == 0 or num_model is None:
        return ""

    try:
        # 依你的要求，不做 predict_with_device/predict_safe wrapper，也不硬塞 device=
        res = num_model.predict(plate_img, conf=conf_thresh, imgsz=640, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return ""

        # boxes.data = (N,6) => x1,y1,x2,y2,conf,cls
        det = res[0].boxes.data.cpu().numpy()
        if det.shape[0] == 0:
            return ""

        x1, y1, x2, y2 = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        scores = det[:, 4]
        cls_ids = det[:, 5]

        # NMSBoxes 需要 [x,y,w,h]
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()

        # NMS 去重，避免同一字元出現多個重疊框
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), 0.0, config.CHAR_NMS_IOU)
        if idxs is None or len(idxs) == 0:
            return ""

        idxs = idxs.flatten()
        fx1 = x1[idxs]
        fx2 = x2[idxs]
        fcls = cls_ids[idxs]

        # 依 X 中心點排序（左 -> 右）
        order = np.argsort((fx1 + fx2) / 2.0)

        # names 可能是 dict 或 list，這裡統一支援
        names = getattr(num_model, "names", {})
        if isinstance(names, dict):
            return "".join(names.get(int(fcls[i]), "") for i in order)

        out = []
        for i in order:
            ci = int(fcls[i])
            if 0 <= ci < len(names):
                out.append(names[ci])
        return "".join(out)

    except Exception:
        return ""
