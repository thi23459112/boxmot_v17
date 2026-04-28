# coding=utf-8
"""
state_sqlite.py
===============
[記憶體優化 + 自動清理版]
功能：
1. 將車輛資料寫入 SQLite 資料庫 (取代記憶體內堆積的 List)。
2. 只存圖片路徑 (Path)，不存圖片物件 (Bytes)，極大化降低 RAM 佔用。
3. [新功能] 程式啟動時，自動檢查並「刪除」同名的舊 DB 檔，確保資料不重複。
"""

import sqlite3
import threading
import queue
import time
from pathlib import Path
from collections import Counter
import cv2
import datetime
from . import config
from .vision_utils import class_name

# ==========================================
# 全域變數控制
# ==========================================
_write_queue = queue.Queue()  # 寫入佇列 (Producer-Consumer 模式)
_running = True               # 控制寫入執行緒是否繼續執行
_db_thread = None             # 背景寫入執行緒物件

def _get_db_path():
    """
    [路徑邏輯] 決定 DB 的存放路徑與檔名
    策略：
    1. 優先抓取影片檔名 (例如 video/car_test.mp4 -> car_test.db)
    2. 確保每一支影片都有獨立的 .db 檔，避免多線程同時寫入同一個檔案導致鎖死 (Database Locked)
    """
    db_dir = Path("output_db")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # 嘗試從 config.VIDEO_SOURCE 解析檔名
    src = getattr(config, "VIDEO_SOURCE", "")
    name = None
    
    if src and "://" not in src: # 如果是檔案路徑 (非 RTSP 串流)
        try:
            name = Path(src).stem # 取主檔名 (去除路徑與副檔名)
        except:
            pass
            
    # 如果解析失敗 (例如是串流)，則退回使用 source_id (例如 B1, B6)
    if not name:
        name = getattr(config, "SOURCE_ID", "unknown_task")
        
    return db_dir / f"{name}.db"

def _init_db_file_clean(db_path):
    """
    [關鍵功能] 初始化 DB 檔案前，先執行清理
    - 如果 db_path 已經存在，代表是上次跑過的舊資料。
    - 為了避免資料混淆 (Append)，這裡執行強制刪除。
    """
    if db_path.exists():
        try:
            db_path.unlink() # 刪除檔案
            print(f"[SQLite] 發現舊資料庫 {db_path.name}，已刪除 (確保資料乾淨)")
        except Exception as e:
            print(f"[SQLite] 警告：無法刪除舊資料庫 {db_path}: {e}")

def _create_table(db_path):
    """建立資料表結構"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 建立 vehicle_records 表，欄位包含辨識所需的所有資訊
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            class_name TEXT,
            plate_text TEXT,
            plate_count INTEGER,
            video_time TEXT,
            real_time TEXT,
            veh_img_path TEXT,
            plate_img_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def _db_writer_worker():
    """
    [背景執行緒] 負責從 Queue 取資料並寫入 SQLite
    """
    # 1. 取得 DB 路徑
    db_path = _get_db_path()
    
    # 2. [關鍵] 啟動時先刪除舊檔 (Reset DB)
    _init_db_file_clean(db_path)
    
    # 3. 建立新檔與資料表
    _create_table(db_path)
    
    print(f"[INFO] SQLite 寫入服務啟動: {db_path.name}")
    
    # 4. 建立連線 (SQLite 連線是 Thread-local 的，必須在執行緒內建立)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    while _running or not _write_queue.empty():
        try:
            # 從佇列取資料，timeout 0.5秒避免死鎖
            data = _write_queue.get(timeout=0.5)
            
            if data is None: # 收到結束訊號
                break
            
            # 執行 Insert 語句
            cursor.execute('''
                INSERT INTO vehicle_records 
                (track_id, class_name, plate_text, plate_count, video_time, real_time, veh_img_path, plate_img_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data["track_id"],
                data["class_name"],
                data["plate_text"],
                data["plate_count"],
                data["video_time"],
                data["real_time"],
                data["veh_img_path"],
                data["plate_img_path"]
            ))
            
            # 立即 Commit，確保資料不遺失 (SQLite 寫入速度極快，頻繁 commit 影響不大)
            conn.commit()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] DB 寫入失敗: {e}")
    
    # 結束時關閉連線
    conn.close()
    print(f"[INFO] SQLite 寫入服務結束 ({db_path.name})")

def _start_service():
    """啟動背景寫入執行緒 (如果尚未啟動)"""
    global _db_thread, _running
    if _db_thread is None:
        _running = True
        # daemon=True 代表主程式結束時，這個執行緒也會跟著結束，不會卡住 Process
        _db_thread = threading.Thread(target=_db_writer_worker, daemon=True)
        _db_thread.start()

def flush_all_sessions_now(reason="exit"):
    """
    [結束處理] 通知寫入執行緒結束並等待
    reason: 呼叫原因 (例如 "exit" 或 "manual")
    """
    global _running
    _running = False
    if _db_thread:
        # 等待執行緒把 Queue 裡剩下的東西寫完 (最多等 5 秒)
        _db_thread.join(timeout=5)

# ============================================================
# State 邏輯 (車輛狀態管理)
# ============================================================

def init_state():
    """初始化單一車輛的狀態字典"""
    return {
        "plates": Counter(),          # 車牌投票箱
        "classes": Counter(),         # 車種投票箱
        "frames_since_left": 0,       # 離開 ROI 的幀數計數
        "missed_frames": 0,           # 連續消失幀數 (用於遮擋容忍)
        "counted": False,             # 是否已結算過
        "last_in_roi": False,         # 上一幀是否在 ROI 內
        "plate_best": {},             # 暫存最佳車牌截圖 (key=plate_str)
        "enter_dt": None,             # 進入 ROI 的真實時間
        "first_seen_frame": 0,        # 第一次出現的幀號
    }

def consider_cleanup_and_finalize(
    tid: int, st: dict, seen: bool, in_roi: bool, class_names,
    current_world_time: datetime.datetime, stats_list: list = None,
    current_frame_id: int = 0
):
    """
    核心邏輯：判斷是否該結算、是否該刪除狀態
    """
    # 確保寫入服務已啟動
    _start_service()

    now_dt = current_world_time
    if st.get("first_seen_frame") == 0:
        st["first_seen_frame"] = current_frame_id

    # 1. 更新進出狀態與時間
    if seen and in_roi:
        if st["enter_dt"] is None: st["enter_dt"] = now_dt
        st["last_in_roi"] = True
    elif not in_roi:
        st["last_in_roi"] = False

    # 2. 更新 Missed 計數 (用於遮擋判斷)
    if not seen: st["missed_frames"] += 1
    else: st["missed_frames"] = 0

    # 3. 更新離開 ROI 計數
    if in_roi: 
        st["frames_since_left"] = 0
    elif seen: 
        st["frames_since_left"] += 1
    else:
        # 如果最後一次在 ROI 內且消失不久，視為遮擋，不算離開
        if st["last_in_roi"] and st["missed_frames"] <= config.MISS_GRACE_FRAMES:
            st["frames_since_left"] = 0
        else:
            st["frames_since_left"] += 1

    # 4. 結算觸發 (離開夠久 且 未結算過)
    if st["frames_since_left"] > config.LEAVE_ROI_FRAMES_TO_COUNT and not st["counted"]:
        most_p = st["plates"].most_common(1)
        most_c = st["classes"].most_common(1)

        # 必須滿足最小投票數
        if most_p and most_c and most_p[0][1] >= config.PLATE_MIN_VOTES:
            plate_str, plate_cnt = most_p[0]
            cls_id = int(most_c[0][0])
            cls_txt = class_name(class_names, cls_id)
            best = st["plate_best"].get(plate_str)

            if best is not None:
                # 準備時間資料
                video_sec = best.get("video_time", 0.0)
                time_axis = time.strftime("%H:%M:%S", time.gmtime(video_sec))
                real_time_str = st.get("enter_dt", now_dt).strftime("%Y-%m-%d %H:%M:%S")
                file_ts = st.get("enter_dt", now_dt).strftime("%m%d_%H%M%S")

                # 準備截圖路徑 (依車種分類)
                v_dir = config.SCREENSHOT_DIR / str(cls_txt)
                p_dir = config.SCREENSHOT_DIR / "LPR"
                v_dir.mkdir(parents=True, exist_ok=True)
                p_dir.mkdir(parents=True, exist_ok=True)

                v_path = v_dir / f"{tid}_{file_ts}.jpg"
                p_path = p_dir / f"{tid}_{plate_str}_{file_ts}.jpg"

                # [IO 操作] 直接存圖到硬碟 (釋放 RAM)
                # 不做 Resize，讓後續轉檔工具決定，這裡保留原圖品質
                cv2.imwrite(str(v_path), best["veh_img"])
                cv2.imwrite(str(p_path), best["plate_crop"])

                # 將資料打包放入寫入佇列
                record = {
                    "track_id": tid,
                    "class_name": cls_txt,
                    "plate_text": plate_str,
                    "plate_count": plate_cnt,
                    "video_time": time_axis,
                    "real_time": real_time_str,
                    "veh_img_path": str(v_path),
                    "plate_img_path": str(p_path)
                }
                _write_queue.put(record)
                
                # 印出你習慣的統計格式
                print(f"[統計] ID={tid}, 車種={cls_txt}, 車號={plate_str}, 次數={plate_cnt}, 影片時間軸={time_axis}")

        # 標記為已結算
        st["counted"] = True
        # [關鍵] 清空圖片快取，釋放記憶體
        st["plate_best"].clear()

    # 5. 判斷是否刪除狀態 (State Cleanup)
    should_remove = False
    
    # 情況A: 已結算，且離開很久了 -> 刪除
    if st["counted"] and st["frames_since_left"] > config.CLEANUP_FRAMES:
        should_remove = True
    
    # 情況B: 孤兒清理 (從未結算，但消失太久，例如誤偵測) -> 刪除
    ORPHAN_MAX_AGE = 300
    if not st["counted"] and st["missed_frames"] > ORPHAN_MAX_AGE:
        should_remove = True

    return should_remove