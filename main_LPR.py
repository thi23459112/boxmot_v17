import cv2
import time
import queue
import threading
import datetime
from pathlib import Path
import torch
import numpy as np
import multiprocessing as mp
import gc
import yaml
from ultralytics import YOLO
from boxmot import create_tracker, get_tracker_config
from boxmot.utils.torch_utils import select_device

# 自定義模組
from threads import frame_reader, frame_processor, FRAME_END, Config
from colors import COLOR_MAP, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS, LABEL_TEXT_COLOR
from utils import config
from utils.vision_utils import scale_points, clamp_crop, class_name
from utils.lpr import process_plate_number
from utils.state_sqlite import init_state, consider_cleanup_and_finalize, flush_all_sessions_now


def run_single(setting_path: str, global_stop_event):
    """
    單一影片處理流程 (由主行程或子行程呼叫)
    :param setting_path: 設定檔路徑 (.yaml)
    :param global_stop_event: 跨行程停止事件 (multiprocessing.Event)
    """
    # ============================================================
    # 1) 載入設定檔並套用至全域 config 模組
    # ============================================================
    with open(setting_path, "r", encoding="utf-8") as f:
        setting_dict = yaml.safe_load(f) or {}
    config.apply_setting(setting_dict)

    print(f"[INFO] 使用設定檔: {setting_path} (source={config.VIDEO_SOURCE})")

    # ============================================================
    # 2) 時間與串流初始化
    # ============================================================
    # 判斷是否為網路串流 (包含 "://" 字串)
    is_stream = "://" in config.VIDEO_SOURCE
    
    # 若設定檔有提供影片起始時間字串，則嘗試解析為 datetime 物件
    video_start_dt = None
    if config.VIDEO_START_TIME_STR:
        try:
            video_start_dt = datetime.datetime.strptime(str(config.VIDEO_START_TIME_STR), "%Y-%m-%d %H:%M:%S")
            if not is_stream:
                print(f"[INFO] 影片模式：設定起始時間為 {video_start_dt}")
        except ValueError:
            print(f"[WARNING] 時間格式錯誤，將使用系統時間")
            video_start_dt = None

    start_time = time.time()              # 記錄此行程開始時間 (用於最後顯示執行時間)
    cfg = Config()                        # 用於傳遞影片資訊 (fps, 當前秒數等) 給執行緒
    cfg.source = config.VIDEO_SOURCE

    # ============================================================
    # 3) 建立佇列 (記憶體優化關鍵)
    # ============================================================
    frame_q = queue.Queue(maxsize=30)      # 存放原始影格 (由 reader 放入)
    processed_q = queue.Queue(maxsize=30)  # 存放處理後影格 (由 processor 放入)
    stop_event = threading.Event()         # 通知執行緒結束

    # ============================================================
    # 4) 載入模型
    # ============================================================
    device = select_device("")             # 自動選擇 GPU/CPU
    print(f"[INFO] 使用裝置: {device}")

    yolo_model = YOLO(config.YOLO_MODEL_PATH)                                                         # 車輛偵測模型
    plate_model = YOLO(config.PLATE_MODEL_PATH) if Path(config.PLATE_MODEL_PATH).exists() else None   # 車牌偵測模型
    num_model = YOLO(config.NUM_MODEL_PATH) if Path(config.NUM_MODEL_PATH).exists() else None         # 車牌號碼辨識模型

    if plate_model is None: print(f"[WARNING] 找不到 {config.PLATE_MODEL_PATH}")
    if num_model is None: print(f"[WARNING] 找不到 {config.NUM_MODEL_PATH}")

    # ============================================================
    # 5) 建立追蹤器
    # ============================================================
    # 根據追蹤器類型決定是否需要外觀 Re-ID 模型
    needs_reid = config.TRACKER_TYPE in ["botsort", "deepocsort", "strongsort", "boosttrack", "imprassoc", "hybridsort"]
    tracker = create_tracker(
        tracker_type=config.TRACKER_TYPE,
        tracker_config=get_tracker_config(config.TRACKER_TYPE),
        reid_weights=Path(config.REID_MODEL_PATH) if needs_reid else None,
        device=device,
        half=False,
        per_class=False
    )

    # 狀態變數
    states = {}                # 每個追蹤 ID 的狀態字典 (由 SQLite 模組初始化)
    stats_list = []            # 保留變數以相容舊模組 (實際上未被使用)
    frame_counter = 0          # 處理的影格計數器

    # 幾何遮罩相關變數 (在第一次繪圖時初始化)
    geom_ready = False         # 是否已完成幾何初始化 (遮罩、ROI 縮放)
    mask_scaled = None         # 根據當前畫面縮放的 MASK 區域 (車輛塗黑區)
    roi_scaled = None          # 根據當前畫面縮放的 ROI 區域 (觸發車牌辨識區)
    mask_img = None            # 整張畫面的二值化遮罩影像

    # ============================================================
    # Draw Func (FrameProcessor 執行緒呼叫)
    # ============================================================
    def draw_func(frame, tracks, show_traj, tracker, yolo):
        """
        在影格上繪製追蹤結果、車牌、軌跡，並更新狀態
        :param frame: 原始影像 (numpy array, BGR)
        :param tracks: 追蹤器輸出的目標陣列，每列格式 [x1, y1, x2, y2, track_id, conf, class_id, -1]
        :param show_traj: 是否繪製軌跡
        :param tracker: BoXMOT 追蹤器實體 (用於取得歷史軌跡)
        :param yolo: YOLO 模型 (用於類別名稱對照)
        :return: 繪製完成的影像
        """
        nonlocal geom_ready, mask_scaled, roi_scaled, mask_img, frame_counter
        frame_counter += 1

        # (1) 幾何初始化 (僅在第一次呼叫時執行)
        if not geom_ready:
            h, w = frame.shape[:2]
            # 將設定檔中的原始座標 (以 BASE_W, BASE_H 為基準) 縮放到當前解析度
            mask_scaled = scale_points(config.MASK_POINTS, w, h, config.BASE_W, config.BASE_H)
            roi_scaled = scale_points(config.REGION_POINTS, w, h, config.BASE_W, config.BASE_H)
            # 建立二值化遮罩影像 (用於車牌辨識前將非車身區域塗黑)
            mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_img, [mask_scaled], 255)
            geom_ready = True

        # (2) 時間計算：取得當前影格的實際世界時間 (用於狀態記錄)
        vsec = float(getattr(cfg, "current_vsec", 0.0))          # 影片內經過的秒數 (由 frame_reader 更新)
        current_real_time = datetime.datetime.now()
        if not is_stream and video_start_dt is not None:
            # 影片模式且指定起始時間：真實時間 = 起始時間 + 影片秒數
            current_real_time = video_start_dt + datetime.timedelta(seconds=vsec)

        seen_ids = set()          # 當前影格中出現的所有追蹤 ID
        in_roi_ids = set()        # 位於 ROI 內的 ID 集合
        plates_draw = {}          # 暫存要繪製的車牌文字，格式 {tid: [(txt, x1, y1, x2, y2), ...]}

        # (3) 處理每個追蹤目標
        for tr in tracks:
            x1, y1, x2, y2, tid, conf, cls, det_ind = tr
            tid, cls = int(tid), int(cls)
            seen_ids.add(tid)

            # 判斷目標是否在 ROI 內 (以底部中心點為準)
            center_pt = ((x1 + x2) / 2.0, y2)
            in_roi = cv2.pointPolygonTest(roi_scaled, center_pt, False) >= 0
            if in_roi:
                in_roi_ids.add(tid)

            # 初始化或取得該 ID 的狀態字典
            st = states.setdefault(tid, init_state())
            st["missed_frames"] = 0            # 重設遺失幀數
            st["last_in_roi"] = bool(in_roi)   # 記錄當前是否在 ROI 內
            st["classes"][cls] += 1            # 累計該類別出現次數

            # 車牌辨識 (僅在目標位於 ROI 內且車牌模型存在時執行)
            if in_roi and plate_model is not None:
                # 裁切出車輛區域
                veh = clamp_crop(frame, x1, y1, x2, y2)
                if veh is None: continue

                # 套用 MASK 遮罩將非車身區域塗黑 (降低車牌誤檢)
                if mask_img is not None:
                    veh_mask = clamp_crop(mask_img, x1, y1, x2, y2)
                    if veh_mask is not None and veh_mask.shape[:2] == veh.shape[:2]:
                        veh = cv2.bitwise_and(veh, veh, mask=veh_mask)

                try:
                    # 車牌偵測
                    pres = plate_model.predict(veh, conf=config.PLATE_CONF, verbose=False)
                    if pres and len(pres[0].boxes):
                        for b in pres[0].boxes:
                            px1, py1, px2, py2 = map(int, b.xyxy[0].cpu().numpy())

                            # 擴大車牌框 10%，以包含更多上下文 (利於數字辨識)
                            pw, ph = px2 - px1, py2 - py1
                            px1, py1 = max(0, px1 - int(pw * 0.1)), max(0, py1 - int(ph * 0.1))
                            px2, py2 = min(veh.shape[1], px2 + int(pw * 0.1)), min(veh.shape[0], py2 + int(ph * 0.1))

                            plate_sub = veh[py1:py2, px1:px2]          # 裁切車牌子圖
                            txt = process_plate_number(num_model, plate_sub, config.NUM_CONF)  # 車牌號碼辨識

                            if txt and len(txt) >= 2:                  # 有效車牌至少 2 字元
                                st["plates"][txt] += 1                 # 累計該車牌出現次數
                                area = int((px2 - px1) * (py2 - py1))
                                best = st["plate_best"].get(txt)

                                # 若該車牌號碼尚未儲存最佳影像，或目前面積更大則更新
                                if best is None or area > best["area"]:
                                    # ⭐⭐ [優化] 儲存車輛影像時若寬度 > 640 則縮小，避免佔用過多記憶體
                                    save_veh = veh.copy()
                                    if save_veh.shape[1] > 640:
                                        scale = 640 / save_veh.shape[1]
                                        save_veh = cv2.resize(save_veh, None, fx=scale, fy=scale)

                                    st["plate_best"][txt] = {
                                        "veh_img": save_veh,
                                        "plate_crop": plate_sub.copy(),
                                        "video_time": float(vsec),
                                        "area": area
                                    }

                                # 記錄車牌繪圖資訊 (座標需轉回原始影像座標)
                                plates_draw.setdefault(tid, []).append(
                                    (txt, int(x1) + px1, int(y1) + py1, int(x1) + px2, int(y1) + py2)
                                )
                except Exception:
                    pass

        # (4) 狀態維護與資料庫寫入 (委託 SQLite 模組處理)
        class_names = getattr(yolo, "names", {})    # 類別編號對應的名稱字典
        to_remove = []                              # 待移除的 ID 列表

        for tid, st in list(states.items()):
            seen = tid in seen_ids
            in_roi = tid in in_roi_ids

            # 決定是否應結束此 ID 的狀態 (車輛離開 ROI 逾時等)
            should_remove = consider_cleanup_and_finalize(
                tid=tid, st=st, seen=seen, in_roi=in_roi, class_names=class_names,
                current_world_time=current_real_time, stats_list=stats_list,
                current_frame_id=frame_counter
            )
            if should_remove:
                to_remove.append(tid)

        # 移除已結束的 ID
        for tid in to_remove:
            states.pop(tid, None)

        # ⭐⭐ [優化] 激進的孤兒狀態清理 (當狀態數量過多時)
        MAX_STATES_SIZE = 500                        # 狀態字典大小上限
        if frame_counter % 100 == 0 and len(states) > MAX_STATES_SIZE:
            orphan_candidates = []                   # 收集尚未計數 (counted=False) 的孤兒狀態
            for tid, st in states.items():
                if not st["counted"]:
                    orphan_candidates.append((tid, st["missed_frames"]))

            # 依 miss 幀數由高到低排序
            orphan_candidates.sort(key=lambda x: x[1], reverse=True)
            # 移除前 100 個最久未見的孤兒 (前提是當前影格未出現)
            for tid, _ in orphan_candidates[:100]:
                if tid not in seen_ids:
                    states.pop(tid, None)

        # (5) 繪製車輛邊框與 ID 標籤
        for tr in tracks:
            x1, y1, x2, y2, tid, conf, cls, det_ind = tr
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            tid, cls = int(tid), int(cls)

            clr = COLOR_MAP.get(cls, (128, 128, 128))               # 依類別選定顏色
            cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)

            lbl = f"ID:{tid} {class_name(class_names, cls)}"        # 標籤文字
            (tw, th), base = cv2.getTextSize(lbl, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            # 標籤背景
            cv2.rectangle(frame, (x1, y2 - th - base), (x1 + tw, y2), clr, -1)
            cv2.putText(frame, lbl, (x1, y2 - base), LABEL_FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_THICKNESS)

        # (6) 繪製車牌框與號碼
        for tid, items in plates_draw.items():
            for (txt, ax1, ay1, ax2, ay2) in items:
                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)          # 綠色框
                (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 黑色背景
                cv2.rectangle(frame, (ax1, max(0, ay1 - th - base - 6)), (ax1 + tw + 10, ay1), (0, 0, 0), -1)
                # 白色文字
                cv2.putText(frame, txt, (ax1 + 5, max(0, ay1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # (7) 繪製軌跡 (若啟用)
        if show_traj and hasattr(tracker, "active_tracks"):
            for at in tracker.active_tracks:
                if len(at.history_observations) >= 3:
                    at_cls = int(getattr(at, "cls", -1))
                    trk_color = COLOR_MAP.get(at_cls, (128, 128, 128))
                    # 取最近 30 個軌跡點繪製
                    for box in list(at.history_observations)[-30:]:
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        cv2.circle(frame, (cx, cy), 2, trk_color, -1)

        # 繪製遮罩輪廓 (藍色) 與 ROI 輪廓 (黃色)
        cv2.polylines(frame, [mask_scaled], True, (255, 0, 0), 2)
        cv2.polylines(frame, [roi_scaled], True, (0, 255, 255), 2)
        return frame

    # ============================================================
    # 啟動讀取與處理執行緒 (設為 daemon 確保主執行緒結束時自動回收)
    # ============================================================
    t_reader = threading.Thread(
        target=frame_reader, args=(cfg, frame_q, stop_event),
        name="FrameReader", daemon=True
    )
    t_proc = threading.Thread(
        target=frame_processor,
        args=(cfg, frame_q, processed_q, stop_event, yolo_model, tracker, draw_func,
              config.SHOW_TRAJECTORIES, config.YOLO_CONF, config.YOLO_CLASSES),
        kwargs={"mask_points": config.MASK_POINTS, "mask_base_size": (config.BASE_W, config.BASE_H),
                "mask_point_mode": "bottom_center"},
        name="FrameProcessor", daemon=True
    )

    print("[INFO] 啟動執行緒...")
    t_reader.start()
    t_proc.start()

    # ============================================================
    # 主迴圈 (負責顯示、錄影、停止訊號監聽)
    # ============================================================
    if config.SAVE_OUTPUT_VIDEO:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = None                # VideoWriter 物件，用於輸出影片
    prev = time.time()        # 上一幀的時間戳 (用於計算 FPS)
    cnt = 0                   # 已顯示的影格計數

    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            try:
                item = processed_q.get(timeout=1.0)    # 等待處理完的影格，避免 busy loop
            except queue.Empty:
                continue

            if item is FRAME_END:                      # 收到結束訊號
                break

            _, frame = item
            cnt += 1

            # 計算並顯示即時 FPS
            now = time.time()
            fps = 1.0 / (now - prev) if cnt > 1 else 0.0
            prev = now
            cv2.putText(frame, f"FPS:{fps:.1f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 儲存輸出影片 (若啟用)
            if config.SAVE_OUTPUT_VIDEO:
                if out is None:
                    h, w = frame.shape[:2]
                    fps_out = float(cfg.fps) if getattr(cfg, "fps", 0) > 0 else config.FALLBACK_FPS
                    out = cv2.VideoWriter(str(config.OUTPUT_VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
                if out is not None:
                    out.write(frame)

            # 顯示視窗 (按 'q' 停止所有)
            cv2.imshow("BoXMOT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                flush_all_sessions_now("q")            # 立即寫出所有未完成的 session
                stop_event.set()
                global_stop_event.set()
                break

            # ⭐⭐ [優化] 定期手動垃圾回收，防止記憶體累積
            if cnt % 500 == 0:
                gc.collect()

    finally:
        # 清理資源
        stop_event.set()
        t_reader.join(timeout=5)
        t_proc.join(timeout=5)

        cv2.destroyAllWindows()
        if out is not None:
            out.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        flush_all_sessions_now("exit")                  # 結束前強制寫入所有資料
        print(f"[Done] 結束: {time.time() - start_time:.2f}s")


def main():
    """
    主函式：掃描 LPR_setting 目錄下的所有設定檔，並為每個設定檔啟動一個獨立行程 (Process)
    """
    script_dir = Path(__file__).resolve().parent
    setting_dir = script_dir / "LPR_setting"
    # 收集所有 .yaml 或 .yml 設定檔
    setting_files = sorted(list(setting_dir.glob("*.yaml")) + list(setting_dir.glob("*.yml")))

    if not setting_files:
        return

    # 使用 spawn 方式啟動行程 (避免 CUDA 繼承問題)
    ctx = mp.get_context("spawn")
    global_stop_event = ctx.Event()       # 跨行程停止事件，可供所有子行程監聽
    procs = []                            # 存放所有子行程物件

    for sp in setting_files:
        p = ctx.Process(target=run_single, args=(str(sp), global_stop_event), daemon=False)
        p.start()
        procs.append(p)

    try:
        while True:
            # 計算目前存活的子行程數量
            alive_count = sum(1 for p in procs if p.is_alive())
            if global_stop_event.is_set() or alive_count == 0:
                break
            time.sleep(1.0)
            gc.collect()                   # 主行程定期回收記憶體

    except KeyboardInterrupt:
        global_stop_event.set()            # 使用者按 Ctrl+C 停止所有
    finally:
        global_stop_event.set()
        for p in procs:
            if p.is_alive():
                p.terminate()              # 強制終止
                p.join()


if __name__ == "__main__":
    mp.freeze_support()   # 必須呼叫，以支援 Windows 下多行程打包
    main()