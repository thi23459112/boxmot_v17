import cv2
import time
import queue
import threading
import datetime
from pathlib import Path
import torch
import numpy as np

# ======== [功能] 多組 setting/*.yaml 同步啟動 ========
import multiprocessing as mp

# ======== [功能] 讀取 setting/*.yaml 用 ========
try:
    import yaml
except Exception as e:
    raise RuntimeError("缺少 PyYAML，請先安裝：pip install pyyaml") from e

# 外部依賴庫
from ultralytics import YOLO
from boxmot import create_tracker, get_tracker_config
from boxmot.utils.torch_utils import select_device

# 自定義模組
from threads import frame_reader, frame_processor, FRAME_END, Config
from colors import COLOR_MAP, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS, LABEL_TEXT_COLOR

from utils import config
from utils.vision_utils import scale_points, clamp_crop, class_name

# ======== [修改點] 改為引用車流專用的狀態管理模組 ========
from utils.state_excel_track import init_state, consider_cleanup_and_finalize, flush_all_sessions_now

def run_single(setting_path: str, global_stop_event):
    """
    旁注：
    - 單一路（單份 YAML）跑一套 pipeline
    - 由主程序用 multiprocessing 開多個 process 各自呼叫 run_single()
    """
    # ============================================================
    # 1) [輸入] 載入 YAML 設定
    # ============================================================
    with open(setting_path, "r", encoding="utf-8") as f:
        setting_dict = yaml.safe_load(f) or {}
    config.apply_setting(setting_dict)

    print(f"[INFO] 啟動車流監控: {setting_path} (source_id={config.SOURCE_ID}, source={config.VIDEO_SOURCE})")

    # ============================================================
    # 2) [判斷] 設定載入後，判斷串流與時間
    # ============================================================
    # [邏輯] 判斷是否為串流（RTSP/RTMP等）
    is_stream = "://" in config.VIDEO_SOURCE

    # [時間] 解析 YAML 設定的起始時間
    video_start_dt = None
    if config.VIDEO_START_TIME_STR:
        try:
            # [格式] 解析時間格式: 2025-05-20 10:44:13
            video_start_dt = datetime.datetime.strptime(str(config.VIDEO_START_TIME_STR), "%Y-%m-%d %H:%M:%S")
            if not is_stream:
                print(f"[INFO] 影片模式：設定起始時間為 {video_start_dt}")
        except ValueError:
            print(f"[WARNING] 時間格式錯誤 ({config.VIDEO_START_TIME_STR})，將使用系統時間")
            video_start_dt = None

    # ============================================================
    # 3) [統計] 效能統計基準時間點
    # ============================================================
    start_time = time.time()  # [效能] 記錄程式開始執行時間

    # ============================================================
    # 4) [設定] 設定檔載入 (連動 threads.py)
    # ============================================================
    cfg = Config()
    cfg.source = config.VIDEO_SOURCE  # [輸入] 指定影片來源路徑

    # ============================================================
    # 5) [管道] 建立 Pipeline 佇列 (Queue)
    # ============================================================
    # 流程：Reader --(frame_q)--> Processor --(processed_q)--> Main
    frame_q = queue.Queue(maxsize=30)              # [緩衝] 原始影像佇列
    processed_q = queue.Queue(maxsize=30)          # [緩衝] 處理後(畫好圖)的影像佇列
    stop_event = threading.Event()                 # [控制] 單一流程停止訊號

    # ============================================================
    # 6) [硬體] 硬體裝置選擇
    # ============================================================
    device = select_device("")                      # [硬體] 自動選擇 GPU (若有) 或 CPU
    print(f"[INFO] 使用裝置: {device}")

    # ============================================================
    # 7) [模型] 載入 AI 模型 (純偵測車輛)
    # ============================================================
    print("[INFO] 載入車輛偵測模型...")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)       # [核心] 用於偵測車輛 (Car, Truck, Bus...)

    # ============================================================
    # 8) [追蹤] 建立追蹤器 (BoXMOT)
    # ============================================================
    print("[INFO] 建立追蹤器...")
    # [邏輯] 部分追蹤器需要 ReID 模型來處理遮擋重連
    needs_reid = config.TRACKER_TYPE in ["botsort", "deepocsort", "strongsort", "boosttrack", "imprassoc", "hybridsort"]

    tracker = create_tracker(
        tracker_type=config.TRACKER_TYPE,
        tracker_config=get_tracker_config(config.TRACKER_TYPE),
        reid_weights=Path(config.REID_MODEL_PATH) if needs_reid else None,
        device=device,
        half=False,
        per_class=False
    )

    # ============================================================
    # 9) [狀態] 全域狀態容器
    # ============================================================
    states = {}      # [核心狀態] Key=TID, Value=State Dict (存放車流狀態)
    stats_list = []  # [結算清單] 為相容性保留，但 state_excel_track 不再使用

    # ⭐ [新增] 幀計數器，用於孤兒清理機制
    frame_counter = 0

    # ============================================================
    # 10) [幾何] 幾何座標快取
    # ============================================================
    geom_ready = False      # [標記] 座標縮放是否已完成
    mask_scaled = None      # [幾何] 縮放後的遮罩多邊形座標
    roi_scaled = None       # [幾何] 縮放後的 ROI 多邊形座標

    # ============================================================
    # draw_func 定義：FrameProcessor 的核心邏輯
    # 注意：此函數在 FrameProcessor 執行緒中運行，而非主執行緒
    # ============================================================
    def draw_func(frame, tracks, show_traj, tracker, yolo):
        # ⭐ [修改] 加入 frame_counter 到 nonlocal
        nonlocal geom_ready, mask_scaled, roi_scaled, frame_counter
        
        # ⭐ [新增] 幀計數器遞增
        frame_counter += 1

        # ------------------------------------------------------------
        # (1) [初始化] 座標縮放初始化 (僅執行一次)
        # ------------------------------------------------------------
        if not geom_ready:
            h, w = frame.shape[:2]
            # [幾何] 將 config 定義的相對座標點位轉換為當前影片解析度的絕對座標
            mask_scaled = scale_points(config.MASK_POINTS, w, h, config.BASE_W, config.BASE_H)  # 視覺遮罩
            roi_scaled = scale_points(config.REGION_POINTS, w, h, config.BASE_W, config.BASE_H) # 觸發區域
            geom_ready = True

        # ------------------------------------------------------------
        # (2) [時間] 取得當前影片時間
        # ------------------------------------------------------------
        vsec = float(getattr(cfg, "current_vsec", 0.0))  # [時間] 用於記錄車輛出現的秒數
        current_real_time = datetime.datetime.now()      # 預設用系統時間

        if not is_stream and video_start_dt is not None:
            # 如果是「影片檔」且有設定「起始時間」
            # 真實時間 = 起始時間 + 影片經過的秒數
            current_real_time = video_start_dt + datetime.timedelta(seconds=vsec)

        # ------------------------------------------------------------
        # (3) [集合] 本幀狀態集合初始化
        # ------------------------------------------------------------
        seen_ids = set()    # [集合] 本幀追蹤器有輸出的 TID (代表沒被遮擋/遺失)
        in_roi_ids = set()  # [集合] 本幀位於 ROI 區域內的 TID

        # ------------------------------------------------------------
        # (4) [遍歷] 遍歷所有追蹤物件 (Tracks Loop)
        # ------------------------------------------------------------
        for tr in tracks:
            x1, y1, x2, y2, tid, conf, cls, det_ind = tr
            tid = int(tid)          # 轉 int 確保作為 dict key 安全
            cls = int(cls)
            seen_ids.add(tid)       # 標記此 ID 本幀存在

            # [邏輯] ROI 判定：使用 BBox 底部中心點
            center_pt = ((x1 + x2) / 2.0, y2)
            in_roi = cv2.pointPolygonTest(roi_scaled, center_pt, False) >= 0
            if in_roi:
                in_roi_ids.add(tid) # 標記此 ID 在 ROI 內

            # [狀態] 取得或初始化該 ID 的 state (由 state_excel_track 提供)
            st = states.setdefault(tid, init_state())

            # ⭐⭐⭐ [車流專用] 更新 Y 軸位移數據 ⭐⭐⭐
            if in_roi:
                st["y_last"] = y2                     # [位置] 記錄最後 Y 座標
                st["roi_recent"].append(cls)          # [統計] 記錄最近出現的車種
                if st["y_first"] is None:             # [首次] 如果是第一次進入 ROI
                    st["y_first"] = y2                # [位置] 記錄首次 Y 座標
                    st["first_vsec"] = vsec           # [時間] 記錄首次進入時間
            
            st["last_in_roi"] = bool(in_roi)          # [狀態] 更新最後位置是否在 ROI
            st["classes"][cls] += 1                   # [統計] 車種投票 (累積每幀的分類結果)

        # ------------------------------------------------------------
        # (5) [管理] 狀態管理與結算 (呼叫 state_excel_track 模組)
        # ------------------------------------------------------------
        class_names = getattr(yolo, "names", {})      # 取得 Class ID 對應名稱
        to_remove = []                                # 待刪除列表 (避免遍歷時修改 Dict)

        for tid, st in list(states.items()):
            # [核心呼叫] 將複雜的生命週期邏輯交給 state_excel_track.py
            # ⭐ [修改] 加入 current_frame_id 參數，用於孤兒清理
            should_remove = consider_cleanup_and_finalize(
                tid=tid,
                st=st,
                seen=(tid in seen_ids),               # 影響: missed_frames 累積
                in_roi=(tid in in_roi_ids),           # 影響: frames_since_left 重置
                class_names=class_names,              # 輸出: 用於解析最終車種
                current_world_time=current_real_time, # [時間] 傳入計算好的真實時間
                stats_list=stats_list,                # 相容性參數，不再使用
                current_frame_id=frame_counter        # ⭐ [新增] 傳入幀號，用於孤兒清理
            )
            if should_remove:
                to_remove.append(tid)

        # [清理] 執行狀態清理
        for tid in to_remove:
            states.pop(tid, None)

        # ⭐⭐⭐ [關鍵新增] 定期強制清理孤兒 states，避免記憶體無限增長 ⭐⭐⭐
        # 每 100 幀檢查一次，如果 states 過大，強制清理最老的未結算項目
        MAX_STATES_SIZE = 2000  # 最多保留 2000 個活躍追蹤
        if frame_counter % 100 == 0 and len(states) > MAX_STATES_SIZE:
            print(f"[WARNING] states 過大 ({len(states)} > {MAX_STATES_SIZE})，強制清理...")
            
            # 找出未結算且消失最久的項目
            orphan_candidates = []
            for tid, st in states.items():
                if not st["counted"]:
                    orphan_candidates.append((tid, st["missed_frames"], st.get("first_seen_frame", 0)))
            
            # 按 missed_frames 排序，清理最老的 500 個
            orphan_candidates.sort(key=lambda x: x[1], reverse=True)
            cleanup_count = 0
            for tid, missed, _ in orphan_candidates[:500]:
                if tid not in seen_ids:  # 確保不是當前活躍的
                    states.pop(tid, None)
                    cleanup_count += 1
            
            print(f"[INFO] 強制清理完成，移除 {cleanup_count} 個孤兒，剩餘 {len(states)}")

        # ------------------------------------------------------------
        # (6) [繪圖] 視覺化繪製：車輛框與標籤
        # ------------------------------------------------------------
        for tr in tracks:
            x1, y1, x2, y2, tid, conf, cls, det_ind = tr
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # [繪圖] 取得對應車種顏色
            clr = COLOR_MAP.get(int(cls), (128, 128, 128))

            # [繪圖] 畫車身 BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
            
            # [繪圖] 製作標籤文字 (ID + Class)
            lbl = f"ID:{int(tid)} {class_name(class_names, int(cls))}"
            (tw, th), base = cv2.getTextSize(lbl, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)

            # [繪圖] 畫標籤背景 (實心矩形，位於框的底部)
            # 算法：y2-th-base 確保文字底端貼齊 BBox 底線
            cv2.rectangle(frame, (x1, y2 - th - base), (x1 + tw, y2), clr, -1)

            # [繪圖] 畫標籤文字
            cv2.putText(frame, lbl, (x1, y2 - base),
                        LABEL_FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_THICKNESS)
        # ------------------------------------------------------------
        # (7) 視覺化繪製：移動軌跡 (Optional)
        # ------------------------------------------------------------
        if show_traj and hasattr(tracker, "active_tracks") and getattr(tracker, "active_tracks", None):
            for at in tracker.active_tracks:
                if hasattr(at, "history_observations") and len(at.history_observations) >= 3:
                    # at.cls 是 tracker 內保存的類別（通常存在），用它去取顏色
                    at_cls = int(getattr(at, "cls", -1))  # 取不到就 -1
                    trk_color = COLOR_MAP.get(at_cls, (128, 128, 128))  # 與 bbox 顏色相同來源
                    # 取最後 30 個點畫軌跡
                    for box in list(at.history_observations)[-30:]:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        cv2.circle(frame, (cx, cy), 2, trk_color, -1)

        # ------------------------------------------------------------
        # (8) [繪圖] 視覺化繪製：區域線與遮罩線
        # ------------------------------------------------------------
        cv2.polylines(frame, [mask_scaled], True, (255, 0, 0), 2)   # 藍線: MASK
        cv2.polylines(frame, [roi_scaled], True, (0, 255, 255), 2)  # 黃線: ROI

        return frame

    # ============================================================
    # 啟動多執行緒 Pipeline
    # ============================================================
    # 執行緒 1: FrameReader (負責解碼影片，放入 frame_q)
    t_reader = threading.Thread(
        target=frame_reader,
        args=(cfg, frame_q, stop_event),
        name="FrameReader"
    )

    # 執行緒 2: FrameProcessor (負責 YOLO 推論與呼叫 draw_func，放入 processed_q)
    t_proc = threading.Thread(
        target=frame_processor,
        args=(
            cfg, frame_q, processed_q, stop_event,
            yolo_model, tracker, draw_func,
            config.SHOW_TRAJECTORIES, config.YOLO_CONF, config.YOLO_CLASSES
        ),
        kwargs={
            "mask_points": config.MASK_POINTS,
            "mask_base_size": (config.BASE_W, config.BASE_H)
        },
        name="FrameProcessor"
    )

    print("[INFO] 啟動執行緒...")
    t_reader.start()
    t_proc.start()

    # ============================================================
    # 主執行緒迴圈：顯示畫面與寫入影片
    # ============================================================
    # ✅ [輸出] 只有在 SAVE_OUTPUT_VIDEO=True 時才建立輸出資料夾
    if config.SAVE_OUTPUT_VIDEO:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = None  # [錄影] VideoWriter 物件初始化
    prev = time.time()  # ⭐ [補回] 用於計算 FPS 的前一時間點
    cnt = 0             # ⭐ [補回] 影格計數
    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            try:
                # [取圖] 從處理完的佇列拿取 (包含影像與標註)
                item = processed_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is FRAME_END:  # 收到結束訊號
                break

            _, frame = item
            cnt += 1

            # [FPS] 計算並顯示即時 FPS
            now = time.time()
            fps = 1.0 / (now - prev) if cnt > 1 else 0.0
            prev = now
            cv2.putText(frame, f"FPS:{fps:.1f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # ✅ [輸出] 只有在 SAVE_OUTPUT_VIDEO=True 才初始化/寫入影片
            if config.SAVE_OUTPUT_VIDEO:
                # [錄影] 初始化 VideoWriter (僅第一次執行)
                if out is None:
                    h, w = frame.shape[:2]
                    # 優先使用 cfg 中的 FPS，失敗則用 Fallback
                    fps_out = float(cfg.fps) if getattr(cfg, "fps", 0) > 0 else config.FALLBACK_FPS
                    out = cv2.VideoWriter(str(config.OUTPUT_VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
                
                if out is not None:
                    out.write(frame)

            # [顯示] OpenCV 視窗
            cv2.imshow("Vehicle Tracking Only", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                flush_all_sessions_now("q")  # ✅ 按 q 立刻強制寫入 (不等 10 分鐘)
                stop_event.set()             # 觸發停止事件，通知所有執行緒結束
                global_stop_event.set()      # [同步] 觸發全域停止
                break


    finally:
        # ============================================================
        # 程式收尾 (Cleanup)
        # ============================================================
        # 1. 立即停止所有執行緒
        stop_event.set()
        
        # 2. 快速回收執行緒，設定較短的 timeout
        t_reader.join(timeout=1.0)
        t_proc.join(timeout=1.0)

        cv2.destroyAllWindows()
        if out is not None:
            out.release()
        
        # ⭐ [修改] 呼叫 flush_all_sessions_now 確保所有資料寫入，而非直接使用 pandas
        # state_excel_track 使用批次寫入機制，不再需要最後的 pandas 寫入
        flush_all_sessions_now("exit")
        
        # ⭐ [刪除] 移除舊的 pandas 寫入邏輯，改由 state_excel_track 處理
        # if stats_list:
        #     import pandas as pd
        #     df = pd.DataFrame(stats_list)
        #     df.to_excel(config.RESULT_EXCEL_PATH, index=False)
        #     print(f"[Done] {config.SOURCE_ID} 報表產出至: {config.RESULT_EXCEL_PATH}")

def main():
    # ============================================================
    # [入口] 掃描 setting 資料夾下所有 yaml 檔案，同步啟動
    # ============================================================
    script_dir = Path(__file__).resolve().parent
    # ⭐ 確保這裡讀的是你想要的資料夾 (track_setting)
    setting_dir = script_dir / "track_setting"

    # 支援 .yaml 檔案
    setting_files = sorted(list(setting_dir.glob("*.yaml")))

    if not setting_files:
        print(f"[ERROR] 找不到設定檔：{setting_dir} ...")
        return

    print(f"[INFO] 偵測到 {len(setting_files)} 份設定檔，將同步啟動車流監控：")
    for p in setting_files:
        print(f"  - {p.name}")

    # ============================================================
    # [多進程] 每份 setting 開一個 Process (使用 spawn 避免 CUDA 衝突)
    # ============================================================
    # 【關鍵】CUDA + 多進程在 Linux 預設 fork 會出錯，必須使用 spawn
    ctx = mp.get_context("spawn")
    
    # ==========================================
    # [同步] 建立一個跨 Process 的事件鎖
    # ==========================================
    global_stop_event = ctx.Event()

    procs = []
    for sp in setting_files:
        # ==========================================
        # [啟動] 將 global_stop_event 傳入 args
        # ==========================================
        p = ctx.Process(target=run_single, args=(str(sp), global_stop_event), daemon=False)
        p.start()
        procs.append(p)

    # ==========================================
    # [監控] 主迴圈監聽：只要有人喊停，就全部停
    # ==========================================
    try:
        while True:
            # [檢查] 統計存活子程序數量
            alive_count = sum(1 for p in procs if p.is_alive())
            
            # [條件] 如果收到全域停止訊號，或者所有子程序都結束
            if global_stop_event.is_set() or alive_count == 0:
                break
            
            time.sleep(0.5)  # [控制] 避免主迴圈吃滿 CPU
            
    except KeyboardInterrupt:
        print("\n[Main] 收到 Ctrl+C，強制結束所有程序...")
        global_stop_event.set()

    finally:
        print("[Main] 正在關閉所有子程序...")
        # 先設全域停止
        global_stop_event.set()

        # 等待子程序優雅結束
        for p in procs:
            if p.is_alive():
                p.join(timeout=10.0) # 縮短等待時間，給 10 秒寫 Excel 夠了
        
        # 如果 10 秒後還活著，代表卡住了，直接殺掉
        for p in procs:
            if p.is_alive():
                print(f"[Main] 程序 {p.pid} 逾時未響應，強制終止。")
                p.terminate()
                p.join() # 確保資源釋放
        
        print("[Main] 全部結束")

if __name__ == "__main__":
    # [安全] Windows / Linux 多進程啟動保護
    mp.freeze_support()
    main()