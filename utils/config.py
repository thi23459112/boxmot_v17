# coding=utf-8
"""
config.py - 參數集中管理模組 (集中化設定管理)

採用「預設值 + apply_setting()」設計模式：
1. 主程式讀取 YAML 設定檔 (例如 setting/B1.yaml)
2. 呼叫 config.apply_setting(setting_dict) 載入設定

設計目的：
- 單一程式碼庫可透過不同 YAML profile (B1/B2/B3...) 動態切換來源、ROI、遮罩、參數與輸出路徑
- 支援多路監控 (multi-process)，每路獨立設定互不干擾
- 提供清晰的預設值設定，確保未載入 YAML 時程式仍可正常執行
"""

from pathlib import Path
import numpy as np

# ============================================================
# 0) 預設值設定 (未套用 YAML 時的基礎設定，確保程式可獨立執行)
# ============================================================

# ---- 來源標識符 (用於區分不同攝影機/來源，DB欄位與檔案分類) ----
SOURCE_ID = "B1"  # 預設值，通常由 YAML 設定檔覆寫

# ========= 影片來源與模型路徑 =========
VIDEO_SOURCE = 'video/output_part1.mp4'                           # 影片檔案路徑或 RTSP 串流網址
YOLO_MODEL_PATH = 'weight/car.engine'                             # 車輛偵測模型 (TensorRT engine)
PLATE_MODEL_PATH = 'weight/plate.engine'                          # 車牌偵測模型 (TensorRT engine)
NUM_MODEL_PATH = 'weight/num.engine'                              # 車牌字元辨識模型 (TensorRT engine)
REID_MODEL_PATH = 'weight/mobilenetv2_x1_0_market1501.engine'     # ReID 模型 (部分追蹤器需用於外觀特徵提取)

# ⭐ [新增] 影片起始時間戳記 (手動校正時間軸用)
VIDEO_START_TIME_STR = None                                       # 格式: "2023-01-01 08:00:00"

# ========= 串流 (RTSP) 手動 FPS 設定 =========
STREAM_FPS = 15.0                                                 # None: 自動偵測串流 FPS；設定數值則強制以此 FPS 計算時間軸 (video_seconds)

# ========= 輸出：影片檔案設定 =========
SAVE_OUTPUT_VIDEO = True                                          # 是否儲存處理後的輸出影片
OUTPUT_DIR = Path("output_video")                                 # 輸出影片存放目錄
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "output.mp4"                     # 輸出影片完整路徑
FALLBACK_FPS = 15.0                                               # 當無法取得影片 FPS 時的預設幀率

# ========= 輸出：CSV/Excel 檔案 (兼容舊版 state_edit.py) =========
RESULT_EXCEL_DIR = Path("output_xlsx")                            # Excel 檔案輸出目錄
RESULT_EXCEL_PATH = RESULT_EXCEL_DIR / "output.xlsx"              # Excel 完整路徑
RESULT_CSV_PATH = RESULT_EXCEL_PATH                               # state_edit.py 優先讀取此變數 (向下兼容)

# ========= 輸出：SQLite 資料庫 (state_db / writer 模組使用) =========
RESULT_DB_DIR = Path("output_db")                                 # SQLite 資料庫輸出目錄
RESULT_DB_PATH = RESULT_DB_DIR / "output.db"                      # 資料庫完整路徑

# ========= 車輛截圖根目錄 (多路監控建議: screenshot/<SOURCE_ID>/...) =========
SCREENSHOT_DIR = Path("screenshot")                               # 截圖儲存根目錄

# ========= 偵測相關參數 =========
YOLO_CONF = 0.4                                                   # 車輛偵測信心度閾值
YOLO_CLASSES = [0, 1, 2, 3, 4, 5, 6]                              # 車輛類別過濾清單 (對應模型中的 class IDs)
PLATE_CONF = 0.5                                                  # 車牌偵測信心度閾值
NUM_CONF = 0.5                                                    # 字元辨識信心度閾值

# ========= 追蹤器參數 =========
TRACKER_TYPE = 'imprassoc'                                        # 追蹤器類型，可選: 'botsort','deepocsort','ocsort','strongsort','boosttrack','bytetrack','imprassoc','hybridsort'
SHOW_TRAJECTORIES = True                                          # 是否顯示車輛軌跡線

# ========= 原始影片解析度 (用於座標縮放計算) =========
BASE_W, BASE_H = 1920, 1080                                       # 基準解析度，ROI/遮罩座標以此為參考

# ========= 偵測遮罩 (Mask) - 忽略區域設定 =========
MASK_POINTS = np.array([
    (0, 250), (1400, 250), (1400, 1080), (0, 1080)
], dtype=np.int32)

# ========= ROI 區域 (計數觸發線) =========
REGION_POINTS = np.array([
    (755, 330), (1235, 330), (1370, 1080), (0, 1080)
], dtype=np.int32)

# ========= 統計邏輯參數 (state_edit/state_db 共用) =========
LEAVE_ROI_FRAMES_TO_COUNT = 15                                    # 車輛離開 ROI 後需等待多少幀數才進行結算 (防抖動)
PLATE_MIN_VOTES = 20                                              # 車牌辨識結果至少需出現 N 次才採信 (投票機制)
CLEANUP_FRAMES = 60                                               # 結算後多少幀數清除記憶體中暫存資料 (記憶體管理)
MISS_GRACE_FRAMES = 45                                            # 消失容忍幀數 (防止短暫遮擋導致 ID 斷開或誤判離開)

# ========= Session 合併與落盤參數 =========
PLATE_MERGE_GAP_SECONDS = 10 * 60                                 # 相同車牌在 N 秒內視為同一筆資料 (跨 ID 合併)
SESSION_FLUSH_INTERVAL_SECONDS = 30                               # 每隔 N 秒掃描逾時 session 並寫入資料庫 (節流機制)

# ========= LPR 字元 NMS (非極大值抑制) =========
CHAR_NMS_IOU = 0.5                                                # 字元偵測框的 IOU 閾值，用於合併重複偵測

# ⭐ [新增] 車流統計專用參數 (Track Only)
MOVEMENT_THRESHOLD_PX = 70                                        # 判定 IN/OUT 的位移門檻
MIN_ROI_HITS = 5                                                  # 進入結算的最少 ROI 命中次數

# ============================================================
# 1) YAML 設定覆寫入口函式
# ============================================================
def apply_setting(s: dict):
    """
    讀取 YAML 設定字典並覆寫本模組的全域變數。
    
    設計原則：
    - 保持向下兼容：既有程式碼仍使用 config.XXX 存取參數
    - 錯誤容忍：只覆寫存在鍵值，缺失鍵保持預設值
    - 多路安全：每路監控建議使用獨立 process，避免設定互相汙染
    - 類型安全：確保覆寫後變數保持正確類型
    
    參數：
        s (dict): 從 YAML 檔案載入的設定字典，應包含以下結構區塊：
            - source_id: 來源標識符
            - source: 影片/串流來源
            - models: 模型路徑設定
            - output: 輸出相關設定
            - detect: 偵測參數
            - tracker: 追蹤器參數
            - geometry: 幾何設定
            - session: 統計邏輯參數
    """
    # 宣告全域變數以允許函式內部修改
    global SOURCE_ID, VIDEO_SOURCE, STREAM_FPS, VIDEO_START_TIME_STR
    global YOLO_MODEL_PATH, PLATE_MODEL_PATH, NUM_MODEL_PATH, REID_MODEL_PATH
    global SAVE_OUTPUT_VIDEO, OUTPUT_DIR, OUTPUT_VIDEO_PATH, FALLBACK_FPS
    global RESULT_EXCEL_DIR, RESULT_EXCEL_PATH, RESULT_CSV_PATH
    global RESULT_DB_DIR, RESULT_DB_PATH
    global SCREENSHOT_DIR
    global YOLO_CONF, YOLO_CLASSES, PLATE_CONF, NUM_CONF, CHAR_NMS_IOU
    global TRACKER_TYPE, SHOW_TRAJECTORIES
    global BASE_W, BASE_H, MASK_POINTS, REGION_POINTS
    global LEAVE_ROI_FRAMES_TO_COUNT, PLATE_MIN_VOTES, CLEANUP_FRAMES, MISS_GRACE_FRAMES
    global PLATE_MERGE_GAP_SECONDS, SESSION_FLUSH_INTERVAL_SECONDS
    # ⭐ [新增] 宣告車流專用全域變數
    global MOVEMENT_THRESHOLD_PX, MIN_ROI_HITS

    # ==============================
    # 1) 基本設定 (source 區塊)
    # ==============================
    SOURCE_ID = str(s.get("source_id", SOURCE_ID))        # B1/B2/B3... 標識符
    VIDEO_SOURCE = str(s.get("source", VIDEO_SOURCE))     # 影片檔案或 RTSP URL
    STREAM_FPS = s.get("stream_fps", STREAM_FPS)          # RTSP 手動 FPS，None 為自動
    VIDEO_START_TIME_STR = s.get("start_time", None)      # ⭐ 讀取影片起始時間 (時間軸校正)

    # ==============================
    # 2) 模型路徑設定 (models 區塊)
    # ==============================
    m = s.get("models", {}) or {}
    YOLO_MODEL_PATH = str(m.get("vehicle", YOLO_MODEL_PATH))
    PLATE_MODEL_PATH = str(m.get("plate", PLATE_MODEL_PATH))
    NUM_MODEL_PATH = str(m.get("num", NUM_MODEL_PATH))
    REID_MODEL_PATH = str(m.get("reid", REID_MODEL_PATH))

    # ==============================
    # 3) 輸出設定 (output 區塊)
    # ==============================
    out = s.get("output", {}) or {}
    
    # 3.1) 輸出影片開關
    SAVE_OUTPUT_VIDEO = bool(out.get("save_output_video", SAVE_OUTPUT_VIDEO))
    
    # 3.2) 統一提取「來源主檔名」供 Excel 與截圖資料夾使用
    try:
        # 邏輯：影片檔案取檔名 (如 test.mp4 → test)
        #      RTSP 串流取 source_id (如 B1)
        if "://" not in VIDEO_SOURCE:
            source_stem = Path(VIDEO_SOURCE).stem        # 檔案模式：取得不含副檔名的檔名
        else:
            source_stem = SOURCE_ID                      # 串流模式：使用 YAML 設定的 source_id
    except Exception:
        source_stem = "output"                           # 異常處理：解析失敗時使用預設名稱

    # 3.3) 輸出影片命名規則 (優先順序)
    #   1. YAML 手動指定 → 最高優先
    #   2. RTSP 串流 → output_SOURCEID.mp4
    #   3. 影片檔案 → output_原檔名.mp4
    out_video_dir = out.get("output_video_dir", str(OUTPUT_DIR))
    out_video_name_yaml = out.get("output_video_name", None)
    is_stream = "://" in VIDEO_SOURCE                     # 判斷來源類型

    if out_video_name_yaml:
        out_video_name = out_video_name_yaml              # ⭐ 優先級 1: YAML 手動指定
    else:
        if is_stream:
            out_video_name = f"output_{SOURCE_ID}.mp4"    # ⭐ 優先級 2: RTSP 串流模式
        else:
            out_video_name = f"output_{source_stem}.mp4"  # ⭐ 優先級 3: 影片檔案模式

    # 確保副檔名為 .mp4
    if not out_video_name.endswith(".mp4"):
        out_video_name = Path(out_video_name).stem + ".mp4"

    OUTPUT_DIR = Path(out_video_dir)
    OUTPUT_VIDEO_PATH = OUTPUT_DIR / out_video_name

    # 3.4) 設定截圖目錄 (基於 source_stem)
    # 邏輯：強制設定為 "screenshot/來源主檔名/"
    # 效果：不同來源的截圖自動分流至不同資料夾
    SCREENSHOT_DIR = Path("screenshot") / source_stem

    # 3.5) 設定 Excel 輸出路徑 (基於 source_stem)
    # 邏輯：YAML 設定優先，無設定則使用「來源主檔名.xlsx」
    csv_dir = out.get("csv_dir", str(RESULT_EXCEL_DIR))
    csv_name = out.get("csv_name", None)

    if not csv_name:
        csv_name = f"{source_stem}.xlsx"                   # 自動生成 Excel 檔名
    
    # 確保副檔名為 .xlsx (避免 YAML 誤設為 .csv)
    if not csv_name.endswith(".xlsx"):
        csv_name = Path(csv_name).stem + ".xlsx"

    RESULT_EXCEL_DIR = Path(csv_dir)
    RESULT_EXCEL_PATH = RESULT_EXCEL_DIR / str(csv_name)
    RESULT_CSV_PATH = RESULT_EXCEL_PATH                    # 兼容舊版變數名稱

    # ==============================
    # 4) 偵測參數 (detect 區塊)
    # ==============================
    det = s.get("detect", {}) or {}
    YOLO_CONF = float(det.get("yolo_conf", YOLO_CONF))
    YOLO_CLASSES = det.get("yolo_classes", YOLO_CLASSES)
    PLATE_CONF = float(det.get("plate_conf", PLATE_CONF))
    NUM_CONF = float(det.get("num_conf", NUM_CONF))
    CHAR_NMS_IOU = float(det.get("char_nms_iou", CHAR_NMS_IOU))

    # 確保 YOLO_CLASSES 為整數列表
    try:
        YOLO_CLASSES = [int(x) for x in YOLO_CLASSES]
    except Exception:
        # 格式異常時保持原值，避免程式崩潰
        pass

    # ==============================
    # 5) 追蹤器參數 (tracker 區塊)
    # ==============================
    trk = s.get("tracker", {}) or {}
    TRACKER_TYPE = str(trk.get("tracker_type", TRACKER_TYPE))
    SHOW_TRAJECTORIES = bool(trk.get("show_trajectories", SHOW_TRAJECTORIES))

    # ==============================
    # 6) 幾何設定 (geometry 區塊)
    # ==============================
    geo = s.get("geometry", {}) or {}
    BASE_W = int(geo.get("base_w", BASE_W))
    BASE_H = int(geo.get("base_h", BASE_H))

    # 遮罩點座標 (從 YAML 列表轉換為 numpy array)
    mp = geo.get("mask_points", None)
    if mp is not None:
        MASK_POINTS = np.array(mp, dtype=np.int32)

    # ROI 點座標 (從 YAML 列表轉換為 numpy array)
    rp = geo.get("region_points", None)
    if rp is not None:
        REGION_POINTS = np.array(rp, dtype=np.int32)

    # ==============================
    # 7) Session 管理參數 (session 區塊)
    # ==============================
    ses = s.get("session", {}) or {}
    LEAVE_ROI_FRAMES_TO_COUNT = int(ses.get("leave_roi_frames_to_count", LEAVE_ROI_FRAMES_TO_COUNT))
    PLATE_MIN_VOTES = int(ses.get("plate_min_votes", PLATE_MIN_VOTES))
    CLEANUP_FRAMES = int(ses.get("cleanup_frames", CLEANUP_FRAMES))
    MISS_GRACE_FRAMES = int(ses.get("miss_grace_frames", MISS_GRACE_FRAMES))
    PLATE_MERGE_GAP_SECONDS = int(ses.get("plate_merge_gap_seconds", PLATE_MERGE_GAP_SECONDS))
    SESSION_FLUSH_INTERVAL_SECONDS = int(ses.get("flush_interval_seconds", SESSION_FLUSH_INTERVAL_SECONDS))
    
    # ==============================
    # 8) 車流統計專用參數 (從 YAML 中的 track_logic 區塊讀取)
    # ==============================
    track_cfg = s.get("track_logic", {}) or {}
    MOVEMENT_THRESHOLD_PX = int(track_cfg.get("movement_threshold", MOVEMENT_THRESHOLD_PX))
    MIN_ROI_HITS = int(track_cfg.get("min_roi_hits", MIN_ROI_HITS))