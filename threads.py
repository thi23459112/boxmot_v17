import time
import queue
import threading
import numpy as np
import cv2
import torch
from dataclasses import dataclass
from utils import config

# =========================================================================
# Config / 常數
# =========================================================================

@dataclass
class Config:
    """用於在執行緒間傳遞共用設定的資料類別"""
    source: str = ''            # 影片來源 (檔案路徑或串流 URL)
    fps: float = 0.0            # 幀率 (從影片讀取或手動指定)
    current_vsec: float = 0.0   # 當前影格在影片中的時間位置 (秒)

FRAME_END = object()            # 哨兵物件，用於通知佇列「影片結束」

# =========================================================================
# Queue 小工具
# =========================================================================

# def safe_put(q: queue.Queue, item, stop_event: threading.Event, timeout: float = 1.0) -> bool:
#     """
#     安全地將項目放入佇列，若佇列已滿則等待 timeout 秒。
#     若在等待期間 stop_event 被設定，則放棄放入並回傳 False。
#     回傳 True 表示成功放入，False 表示因停止事件而放棄。
#     """
#     try:
#         q.put(item, timeout=timeout)
#         return True
#     except queue.Full:
#         # 佇列已滿且超時，檢查是否收到停止訊號
#         return not stop_event.is_set()
def safe_put(q: queue.Queue, item, stop_event: threading.Event, timeout: float = 1.0) -> bool:
    """
    安全地將項目放入佇列。
    若佇列滿了，就持續等待直到有空位，除非收到 stop_event 停止訊號才放棄，確保絕不掉幀。
    """
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True  # 成功放入
        except queue.Full:
            # 佇列已滿且超時，不回傳，繼續下一輪迴圈繼續等
            continue
            
    return False # 收到停止訊號，放棄放入
# =========================================================================
# Thread-1: frame_reader
# =========================================================================

def frame_reader(cfg: Config, frame_q: queue.Queue, stop_event: threading.Event):
    """
    讀取執行緒：從影片來源讀取影格，進行縮放，並放入 frame_q 供處理執行緒使用。
    :param cfg: Config 物件，會更新其中的 fps 和 source (輸入用)
    :param frame_q: 存放 (fid, frame, vsec) 的佇列
    :param stop_event: 停止事件，被設定時執行緒應結束
    """
    while not stop_event.is_set():
        # 嘗試開啟影片來源 (可能是檔案或串流)
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            print(f"[ERROR] 無法開啟來源：{cfg.source}")
            time.sleep(0.5)          # 等待後重試
            continue

        # 設定緩衝區大小為 1，減少延遲 (特別對串流重要)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 判斷是否為網路串流 (RTSP/HTTP)
        is_stream = cfg.source.lower().startswith(("rtsp://", "http://", "https://"))
        
        # 若全域設定中有手動指定串流 FPS，則強制使用該值
        forced_stream_fps = getattr(config, "STREAM_FPS", None)

        if is_stream and forced_stream_fps:
            # 只有串流模式才強制覆蓋 FPS
            cfg.fps = float(forced_stream_fps)
            print(f"[INFO] 串流模式：強制使用手動 FPS={cfg.fps:.2f}")
        else:
            # 影片檔案：使用 OpenCV 讀到的原生 FPS
            cfg.fps = float(cap.get(cv2.CAP_PROP_FPS))
            if np.isnan(cfg.fps) or cfg.fps <= 0:
                cfg.fps = 15.0
            print(f"[INFO] 使用影片原生 FPS={cfg.fps:.2f}")

        # 若是檔案影片，取得總幀數 (用於判斷結束)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
        print(f"[INFO] 開始讀取, FPS: {cfg.fps:.2f}")

        fid = 0                          # 幀數索引 (從 0 開始)
        error_count = 0                  # 連續錯誤計數
        max_consecutive_errors = 10      # 最大連續錯誤次數，超過則放棄

        # 目標輸出尺寸 (從全域設定讀取)
        target_w = int(getattr(config, "BASE_W", 1920))
        target_h = int(getattr(config, "BASE_H", 1080))

        # 主讀取迴圈
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    # 讀取失敗：增加錯誤計數
                    error_count += 1
                    if not is_stream:
                        # 檔案模式：若已讀到最後一幀，正常結束
                        if total_frames > 0 and fid >= total_frames - 1:
                            safe_put(frame_q, FRAME_END, stop_event)
                            cap.release()
                            return
                        # 連續錯誤過多，視為異常結束
                        if error_count >= max_consecutive_errors:
                            safe_put(frame_q, FRAME_END, stop_event)
                            cap.release()
                            return
                    else:
                        # 串流模式：若連續錯誤過多則跳出內層迴圈，嘗試重新連線
                        if error_count >= max_consecutive_errors:
                            break
                        time.sleep(0.05)   # 短暫等待後重試
                    continue

                # 讀取成功，重設錯誤計數
                error_count = 0
                # 防呆：若 frame 為空，跳過
                if frame is None or frame.size == 0:
                    fid += 1
                    continue

                # 統一縮放到目標尺寸 (若尺寸不符)
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    frame = cv2.resize(frame, (target_w, target_h))

                # 優先讀取影片內建的真實時間戳 (微秒)
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec > 0:
                    vsec = msec / 1000.0  # 將毫秒轉換為秒
                else:
                    # 如果該影片格式不支援讀取 msec (讀出 0)，才退回用推算的
                    vsec = fid / cfg.fps if (cfg.fps and cfg.fps > 0) else 0.0                

                # 將幀放入佇列，若佇列已滿且收到停止訊號則跳出
                if not safe_put(frame_q, (fid, frame, vsec), stop_event):
                    break

                fid += 1

            except Exception as e:
                # 發生例外時同樣增加錯誤計數
                error_count += 1
                if error_count >= max_consecutive_errors:
                    if not is_stream:
                        safe_put(frame_q, FRAME_END, stop_event)
                    break
                time.sleep(0.05)
                continue

        # 釋放影片擷取物件
        cap.release()
        time.sleep(0.5)   # 等待後可能重新連線 (若為串流)

# =========================================================================
# Thread-2: frame_processor
# =========================================================================

def frame_processor(cfg: Config, frame_q: queue.Queue, processed_q: queue.Queue,
                    stop_event: threading.Event, yolo_model, tracker,
                    draw_func, show_trajectories: bool = True,
                    yolo_conf: float = 0.5, yolo_classes=None,
                    mask_points=None, mask_base_size=(1920, 1080),
                    mask_point_mode="bottom_center"):
    """
    處理執行緒：從 frame_q 取得影格，執行偵測、追蹤、繪圖，再將結果放入 processed_q。
    :param cfg: Config 物件 (會更新 current_vsec)
    :param frame_q: 輸入佇列，元素為 (fid, frame, vsec)
    :param processed_q: 輸出佇列，元素為 (fid, processed_frame)
    :param stop_event: 停止事件
    :param yolo_model: YOLO 模型物件
    :param tracker: BoXMOT 追蹤器物件
    :param draw_func: 繪圖回呼函式，簽名為 draw_func(frame, tracks, show_trajectories, tracker, yolo_model)
    :param show_trajectories: 是否繪製軌跡
    :param yolo_conf: YOLO 偵測信心門檻
    :param yolo_classes: 指定要偵測的類別 (None 表示全部)
    :param mask_points: 遮罩多邊形點座標 (以 base_size 為基準)
    :param mask_base_size: 遮罩點座標的基準解析度 (預設 1920x1080)
    :param mask_point_mode: 判斷點是否在遮罩內的模式："bottom_center" (底部中心) 或 "center" (中心)
    """
    print("[INFO] 處理執行緒啟動 (裁切優化模式)")

    frame_count = 0                      # 已處理的幀數計數
    mask_ready = False                   # 是否已初始化遮罩相關變數
    crop_rect = None                     # 裁切區域 (x, y, w, h)
    local_mask = None                    # 在裁切區域內的二值化遮罩
    mask_points_scaled = None            # 根據目前畫面縮放後的遮罩多邊形點座標

    # 確保 point_mode 為有效值
    if mask_point_mode not in ("bottom_center", "center"):
        mask_point_mode = "bottom_center"

    # 使用 torch 推論模式 (減少記憶體開銷)
    with torch.inference_mode():
        while not stop_event.is_set():
            try:
                # 從輸入佇列取得影格 (超時 0.5 秒)
                item = frame_q.get(timeout=0.5)
                if item is FRAME_END:
                    # 收到結束訊號，轉傳給輸出佇列後退出
                    safe_put(processed_q, FRAME_END, stop_event)
                    break

                fid, frame, vsec = item
                cfg.current_vsec = float(vsec)   # 更新當前時間，供 draw_func 使用
                frame_count += 1

                # (0) 初始化遮罩相關變數 (只在第一次執行)
                if (not mask_ready) and (mask_points is not None):
                    h, w = frame.shape[:2]
                    base_w, base_h = mask_base_size
                    # 計算縮放比例
                    sx, sy = float(w)/base_w, float(h)/base_h

                    # 將原始遮罩點座標縮放到目前畫面尺寸
                    pts = np.array(mask_points, dtype=np.float32).copy()
                    pts[:, 0] *= sx
                    pts[:, 1] *= sy
                    mask_points_scaled = pts.astype(np.int32)

                    # 計算遮罩多邊形的邊界矩形 (裁切區域)
                    cx, cy, cw, ch = cv2.boundingRect(mask_points_scaled)
                    # 確保裁切區域不超出畫面邊界
                    cx, cy = max(0, cx), max(0, cy)
                    cw, ch = min(w - cx, cw), min(h - cy, ch)
                    crop_rect = (cx, cy, cw, ch)

                    # 建立裁切區域內的遮罩 (大小為 ch x cw)
                    local_mask = np.zeros((ch, cw), dtype=np.uint8)
                    # 將多邊形點座標轉換到裁切區域內
                    local_pts = mask_points_scaled.copy()
                    local_pts[:, 0] -= cx
                    local_pts[:, 1] -= cy
                    cv2.fillPoly(local_mask, [local_pts], 255)
                    mask_ready = True

                # (1) 根據遮罩裁切畫面 (若有遮罩)
                if mask_ready and crop_rect is not None:
                    cx, cy, cw, ch = crop_rect
                    cropped_frame = frame[cy:cy+ch, cx:cx+cw]
                    
                    # 確保 local_mask 與裁切畫面尺寸一致 (可能因整數運算有微小差異)
                    if local_mask.shape[:2] != cropped_frame.shape[:2]:
                        local_mask = cv2.resize(local_mask, (cropped_frame.shape[1], cropped_frame.shape[0]))
                    
                    # 套用遮罩：只保留遮罩內區域 (其他塗黑)
                    det_input = cv2.bitwise_and(cropped_frame, cropped_frame, mask=local_mask)
                else:
                    # 無遮罩：直接使用整張畫面
                    det_input = frame
                    cx, cy = 0, 0   # 裁切偏移量為 0

                # (2) YOLO 偵測 (只對 det_input 進行)
                results = yolo_model(det_input, conf=yolo_conf, verbose=False, classes=yolo_classes)

                # (3) 將偵測到的邊框座標還原至原始畫面座標 (若曾裁切)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    if mask_ready:
                        # 還原：加上裁切區域的左上角偏移
                        boxes[:, 0] += cx
                        boxes[:, 1] += cy
                        boxes[:, 2] += cx
                        boxes[:, 3] += cy

                    # 組合為 [x1, y1, x2, y2, score, class] 陣列
                    detections = np.column_stack((boxes, scores, classes)).astype(np.float32)
                else:
                    detections = np.empty((0, 6), dtype=np.float32)

                # (4) 二次過濾：根據遮罩多邊形剔除不在遮罩內的偵測
                if mask_ready and mask_points_scaled is not None and len(detections) > 0:
                    keep_idx = []
                    for i in range(len(detections)):
                        x1, y1, x2, y2, sc, cl = detections[i]
                        # 決定用哪個點測試：預設底部中心 (x2, y2) 或中心點
                        pt_x, pt_y = (x1 + x2) / 2.0, y2 if mask_point_mode != "center" else (y1 + y2) / 2.0
                        # 若點在多邊形內部 (>=0) 則保留
                        if cv2.pointPolygonTest(mask_points_scaled, (float(pt_x), float(pt_y)), False) >= 0:
                            keep_idx.append(i)
                    detections = detections[keep_idx] if keep_idx else np.empty((0, 6), dtype=np.float32)

                # (5) 追蹤器更新
                tracks = tracker.update(detections, frame)   # tracks 為追蹤結果陣列

                # (6) 繪圖 (呼叫外部傳入的 draw_func)
                frame = draw_func(frame, tracks, show_trajectories, tracker, yolo_model)

                # 將處理後的影格放入輸出佇列
                safe_put(processed_q, (fid, frame), stop_event)

                # 每 1000 幀輸出除錯訊息
                if frame_count % 1000 == 0:
                    print(f"[DEBUG] 已處理 {frame_count} 幀")

            except queue.Empty:
                # 佇列空閒，繼續迴圈 (檢查 stop_event)
                continue
            except Exception as e:
                print(f"[ERROR] 處理執行緒異常: {e}")
                import traceback
                traceback.print_exc()
                continue