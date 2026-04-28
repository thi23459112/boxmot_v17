# coding=utf-8
"""
state_db.py
===========
【用途】
- 記錄每個 tid 的起訖時間（時間點_進/時間點_出）
- 以「車牌」為主鍵做跨 tid 合併：
    * 若同 plate_str 的新事件「進場時間」與前一筆「出場時間」間隔 <= 10 分鐘
    * 視為同一台車（遮擋/斷ID），合併成同一筆（保留最早 tid）
- 改用 SQLite 逐筆落盤：
    * DB 只寫截圖路徑（veh_path / lpr_path）
    * 避免 RTSP 長時間跑造成 stats_list 無限增長爆記憶體
    * 新增等待時間欄位：duration = 時間點_出 - 時間點_進

【最小改動策略】
- 保持函式介面與你原本 state.py / state_edit.py 一樣：
    init_state()
    consider_cleanup_and_finalize(tid, st, seen, in_roi, class_names, start_datetime, stats_list)
  讓主程式不用改中間邏輯（最多改 import 指到 state_db）

【時間來源說明（RTSP 建議）】
- 本檔預設用「datetime.now()」當作時間點（最適合 RTSP 連續串流）
- 如果你想用「start_datetime + vsec」當時間點，也可以，但 RTSP 重連時 vsec 容易跳回去

【強制寫入（你已在 state_edit 用過）】
- atexit / SIGINT / SIGTERM：程式中斷或提前結束時，強制把未落盤的 session 寫入 DB
- flush_all_sessions_now("q")：主程式按 Q 可主動呼叫，立刻寫入
"""

from collections import Counter
from pathlib import Path
import datetime
import time
import cv2

# ======== DB 依賴（新增）========
# SQLite 為內嵌資料庫，不需要額外服務，適合 RTSP 長時間 append
import sqlite3
import threading

# ======== 強制寫入用（新增）========
import atexit
import signal

from .vision_utils import class_name  # 用於 cls_id -> 名稱（names dict/list 皆可）
from . import config                  # 讀取 MERGE 分鐘、輸出資料夾等設定（若沒有就用預設）


# ============================================================
# 可調參數（若 config.py 沒設，也不會壞）
# ============================================================

# 同車牌合併的最大間隔（秒）—預設 10 分鐘
MERGE_GAP_SECONDS = int(getattr(config, "PLATE_MERGE_GAP_SECONDS", 10 * 60))

# 多久掃一次「逾時 session」並落盤（秒）—避免每幀掃描造成負擔
FLUSH_INTERVAL_SECONDS = int(getattr(config, "SESSION_FLUSH_INTERVAL_SECONDS", 30))


# ============================================================
# DB 路徑
# ============================================================
def _get_db_path() -> Path:
    """
    決定 DB 輸出路徑（最小改動：盡量沿用你現有 config 的 output_xlsx 目錄）
    - 若 config.RESULT_DB_PATH 已定義 → 直接用
    - 否則若 config.RESULT_DB_DIR 已定義 → 用該資料夾輸出 .db
    - 否則若 config.RESULT_EXCEL_DIR 已定義 → 用同資料夾輸出 .db
    - 否則退回 output_xlsx/車輛統計_完整版.db
    """
    p = getattr(config, "RESULT_DB_PATH", None)
    if p:
        return Path(p)

    d2 = getattr(config, "RESULT_DB_DIR", None)
    if d2:
        return Path(d2) / "車輛統計_完整版.db"

    d = getattr(config, "RESULT_EXCEL_DIR", None)
    if d:
        return Path(d) / "車輛統計_完整版.db"

    return Path("output_xlsx") / "車輛統計_完整版.db"


# ============================================================
# datetime / duration 工具（新增等待時間欄位用）
# ============================================================
_DT_FMT = "%Y-%m-%d %H:%M:%S"  # 統一時間字串格式，便於 DB 查詢與 Excel 顯示

def _dt_str(dt: datetime.datetime | None) -> str:
    """datetime -> 字串（給 DB 用）"""
    if dt is None:
        return ""
    return dt.strftime(_DT_FMT)

def _parse_dt(s: str) -> datetime.datetime | None:
    """字串 -> datetime（計算 duration 用）"""
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, _DT_FMT)
    except Exception:
        return None

def _sec_to_hms(seconds: int) -> str:
    """秒數 -> HH:MM:SS（顯示用）"""
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _compute_duration(enter_str: str, exit_str: str) -> tuple[int, str]:
    """
    計算等待時間（停留時間）= 時間點_出 - 時間點_進
    - 回傳 (duration_sec, duration_hms)
    - 若時間缺失/解析失敗則回 (0, "00:00:00")
    """
    enter_dt = _parse_dt(enter_str)
    exit_dt = _parse_dt(exit_str)
    if (enter_dt is None) or (exit_dt is None):
        return 0, "00:00:00"

    sec = int((exit_dt - enter_dt).total_seconds())
    if sec < 0:
        sec = 0
    return sec, _sec_to_hms(sec)


# ============================================================
# SQLite 初始化 / 寫入（新增）
# ============================================================

# DB lock，避免多執行緒（含 signal/atexit）同時寫入造成資料庫鎖衝突
_db_lock = threading.Lock()

# 避免重複初始化
_db_inited = False

def _db_init_if_needed():
    """
    初始化 DB（建表/索引），只做一次
    - WAL 模式適合長時間 append
    - 本表採 append-only：每次落盤新增一列，不做 update（最穩且簡單）
    """
    global _db_inited
    if _db_inited:
        return

    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _db_lock:
        if _db_inited:
            return

        conn = sqlite3.connect(str(db_path))
        try:
            # WAL + NORMAL，長時間寫入更穩定且較快
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            # 不需要 flush_reason（依你要求），但要新增 duration 欄位
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,         -- ✅ 新增：來源代號（B1/B2/B3...）
                    tid_first INTEGER,      -- 保留最早的 tid（你要的 id1=ID8 -> 保留 id1）
                    cls_name TEXT,          -- 車種名稱（例如 car/bus/truck...）
                    plate_str TEXT,         -- 車牌字串
                    enter_dt TEXT,          -- 時間點_進（YYYY-MM-DD HH:MM:SS）
                    exit_dt TEXT,           -- 時間點_出（YYYY-MM-DD HH:MM:SS）
                    duration_sec INTEGER,   -- 等待/停留秒數（可用於統計）
                    duration_hms TEXT,      -- HH:MM:SS（方便直接看）
                    votes_total INTEGER,    -- 票數累積（代表辨識累積，不一定等同實際出現次數）
                    veh_path TEXT,          -- 車身截圖路徑
                    lpr_path TEXT,          -- 車牌截圖路徑
                    created_at TEXT         -- 寫入 DB 的時間
                );
                """
            )

            # ✅ 自動補欄位（舊 DB 沒有 source_id 時也能直接升級，不用刪 DB）
            cols = [r[1] for r in conn.execute("PRAGMA table_info(plate_sessions);").fetchall()]
            if "source_id" not in cols:
                conn.execute("ALTER TABLE plate_sessions ADD COLUMN source_id TEXT;")

            # 常用索引（依需求可再加）
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plate ON plate_sessions(plate_str);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exit_dt ON plate_sessions(exit_dt);")

            conn.commit()
            _db_inited = True
        finally:
            conn.close()

def _insert_db_row(row: dict):
    """
    將一筆結果寫入 SQLite（append）
    - 每次 insert 都 commit（避免中斷資料丟失）
    - 會自動補上 duration_sec/duration_hms
    - ✅ 新增：source_id（B1/B2...）
    """
    _db_init_if_needed()

    db_path = _get_db_path()
    created_at = _dt_str(datetime.datetime.now())

    # ✅ 新增：來源代號，直接用目前載入 setting 的 config.SOURCE_ID
    source_id = str(getattr(config, "SOURCE_ID", "") or "")

    # 欄位沿用你 CSV 的 key 命名（減少主程式改動）
    tid_first = row.get("ID", "")
    cls_name_str = row.get("車種", "")
    plate_str = row.get("車號", "")
    enter_str = row.get("時間點_進", "")
    exit_str = row.get("時間點_出", "")
    votes_total = row.get("次數", 0)
    veh_path = row.get("veh_path", "")
    lpr_path = row.get("lpr_path", "")

    # 補等待時間欄位（你要求）
    dur_sec, dur_hms = _compute_duration(enter_str, exit_str)

    with _db_lock:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            conn.execute(
                """
                INSERT INTO plate_sessions
                (source_id, tid_first, cls_name, plate_str, enter_dt, exit_dt, duration_sec, duration_hms,
                 votes_total, veh_path, lpr_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    int(tid_first) if str(tid_first).isdigit() else None,
                    cls_name_str,
                    plate_str,
                    enter_str,
                    exit_str,
                    int(dur_sec),
                    dur_hms,
                    int(votes_total) if votes_total is not None else 0,
                    veh_path,
                    lpr_path,
                    created_at
                )
            )
            conn.commit()
        finally:
            conn.close()


# ============================================================
# plate session 聚合器（跨 tid 合併用）
# ============================================================
#
# - key = plate_str
# - value = session dict（目前尚未「真正關閉」的一筆）
_plate_sessions = {}

# 降低每幀掃描成本，用節流方式定期 flush
_last_flush_ts = 0.0

# 避免重複 flush 寫入（最小更動）
_flush_all_done = False


def _flush_stale_sessions(now_dt: datetime.datetime):
    """
    將「超過 MERGE_GAP_SECONDS 沒更新」的 session 落盤 DB，並從記憶體移除
    - 這是 RTSP 長時間跑不爆炸的關鍵
    """
    global _plate_sessions

    to_del = []
    for plate_str, sess in _plate_sessions.items():
        exit_dt = sess.get("exit_dt")
        if exit_dt is None:
            continue

        # 距離最後出場時間已經超過合併窗口，代表不會再被合併了 -> 可以落盤
        idle_sec = (now_dt - exit_dt).total_seconds()
        if idle_sec > MERGE_GAP_SECONDS:
            _insert_db_row({
                "ID": sess.get("tid_first", ""),
                "車種": sess.get("cls_name", ""),
                "車號": plate_str,
                "時間點_進": _dt_str(sess.get("enter_dt")),
                "時間點_出": _dt_str(sess.get("exit_dt")),
                "次數": sess.get("votes_total", 0),
                "veh_path": sess.get("veh_path", ""),
                "lpr_path": sess.get("lpr_path", ""),
            })

            # 印出「最終落盤」訊息（這裡表示確定結束且不再合併）
            print(
                f"[SESSION] ID={sess.get('tid_first')}, 車號={plate_str}, "
                f"結算_進={_dt_str(sess.get('enter_dt'))}, 結算_出={_dt_str(sess.get('exit_dt'))}, "
                f"次數={sess.get('votes_total', 0)}"
            )
            to_del.append(plate_str)

    for k in to_del:
        _plate_sessions.pop(k, None)


def _merge_or_start_session(
    tid: int,
    cls_name_str: str,
    plate_str: str,
    enter_dt: datetime.datetime,
    exit_dt: datetime.datetime,
    plate_cnt: int,
    veh_path: str,
    lpr_path: str,
    best_area: int
):
    """
    將「tid 結算出的事件」合併到 plate session（或開新 session）
    規則：
    - 若同 plate_str 已有未關閉 session，且 new_enter - old_exit <= 10分鐘 → 合併延長 exit
    - 否則先把舊 session 落盤，再開新 session
    """
    global _plate_sessions

    sess = _plate_sessions.get(plate_str)

    if sess is None:
        # 第一次看到這個 plate_str，直接開新 session
        _plate_sessions[plate_str] = {
            "tid_first": tid,             # 保留最早 tid（你要的 id1=ID8 -> 保留 id1）
            "cls_name": cls_name_str,
            "enter_dt": enter_dt,
            "exit_dt": exit_dt,
            "votes_total": int(plate_cnt),
            "veh_path": veh_path,
            "lpr_path": lpr_path,
            "best_area": int(best_area),
        }
        return

    # 判斷是否能合併（新進場距離舊出場 <= 10分鐘）
    gap_sec = (enter_dt - sess["exit_dt"]).total_seconds()
    if gap_sec <= MERGE_GAP_SECONDS:
        # ✅ 合併：延長 exit_dt（取較晚者）
        if exit_dt > sess["exit_dt"]:
            sess["exit_dt"] = exit_dt

        # 次數可做累加（不一定是真正車牌出現次數，但能代表辨識累積）
        sess["votes_total"] = int(sess.get("votes_total", 0)) + int(plate_cnt)

        # 若新樣本更清楚（area 大），就用新截圖路徑取代（舊檔可留著不刪）
        if int(best_area) > int(sess.get("best_area", 0)):
            sess["best_area"] = int(best_area)
            sess["veh_path"] = veh_path
            sess["lpr_path"] = lpr_path

        # 車種字串若之前是空/unknown，可用新的補上
        if (not sess.get("cls_name")) or sess.get("cls_name") == "unknown":
            sess["cls_name"] = cls_name_str

        return

    # ❌ 不能合併：先落盤舊 session，再開新 session
    _insert_db_row({
        "ID": sess.get("tid_first", ""),
        "車種": sess.get("cls_name", ""),
        "車號": plate_str,
        "時間點_進": _dt_str(sess.get("enter_dt")),
        "時間點_出": _dt_str(sess.get("exit_dt")),
        "次數": sess.get("votes_total", 0),
        "veh_path": sess.get("veh_path", ""),
        "lpr_path": sess.get("lpr_path", ""),
    })
    print(
        f"[SESSION] ID={sess.get('tid_first')}, 車號={plate_str}, "
        f"結算_進={_dt_str(sess.get('enter_dt'))}, 結算_出={_dt_str(sess.get('exit_dt'))}, "
        f"次數={sess.get('votes_total', 0)}"
    )

    _plate_sessions[plate_str] = {
        "tid_first": tid,
        "cls_name": cls_name_str,
        "enter_dt": enter_dt,
        "exit_dt": exit_dt,
        "votes_total": int(plate_cnt),
        "veh_path": veh_path,
        "lpr_path": lpr_path,
        "best_area": int(best_area),
    }


# ============================================================
# 強制寫入：不必等 10 分鐘（最小更動）
# ============================================================
def _flush_all_sessions(reason: str = "exit"):
    """
    強制把目前所有尚未落盤的 plate session 寫入 DB（不等 10 分鐘）
    - 用於：程式中斷、異常退出、按 Q 結束想立刻寫入等情境
    """
    global _flush_all_done, _plate_sessions
    if _flush_all_done:
        return
    _flush_all_done = True

    now_dt = datetime.datetime.now()

    # 把目前所有 session 都寫出去，避免中斷造成資料遺失
    for plate_str, sess in list(_plate_sessions.items()):
        enter_dt = sess.get("enter_dt") or now_dt
        exit_dt = sess.get("exit_dt") or now_dt

        _insert_db_row({
            "ID": sess.get("tid_first", ""),
            "車種": sess.get("cls_name", ""),
            "車號": plate_str,
            "時間點_進": _dt_str(enter_dt),
            "時間點_出": _dt_str(exit_dt),
            "次數": sess.get("votes_total", 0),
            "veh_path": sess.get("veh_path", ""),
            "lpr_path": sess.get("lpr_path", ""),
        })

        print(
            f"[FLUSH-{reason}] ID={sess.get('tid_first')}, 車號={plate_str}, "
            f"結算_進={_dt_str(enter_dt)}, 結算_出={_dt_str(exit_dt)}, 次數={sess.get('votes_total', 0)}"
        )

    # 清空記憶體（避免重複寫入）
    _plate_sessions.clear()


def flush_all_sessions_now(reason: str = "manual"):
    """
    提供給主程式呼叫的「手動強制寫入」入口。
    - 若你想要「按 Q 當下立刻寫入」，主程式按 Q 時呼叫：
        flush_all_sessions_now("q")
    - 即使主程式不呼叫，本檔也會在 atexit / SIGINT / SIGTERM 時自動寫入
    """
    _flush_all_sessions(reason)


# ============================================================
# 原本的 state 介面：init_state / consider_cleanup_and_finalize
# ============================================================
def init_state():
    """
    建立每個 tid 的狀態（最小改動：保留你原本欄位 + 補上進出時間）
    """
    return {
        "plates": Counter(),          # plate_str -> votes（投票）
        "classes": Counter(),         # cls_id -> votes（車種投票）
        "frames_since_left": 0,       # 離開 ROI 幀數（>15 才結算）
        "missed_frames": 0,           # 連續沒 seen 幀數（遮擋容忍）
        "counted": False,             # 同 tid 只結算一次
        "last_in_roi": False,         # 最後一次 seen 是否在 ROI（給 MISS_GRACE）
        "plate_best": {},             # plate_str -> best sample（最大面積那張）
        # ===== 新增（最小改動）=====
        "enter_dt": None,             # 此 tid 第一次「進入 ROI」的時間點（datetime）
        "last_in_roi_dt": None,       # 此 tid 最後一次仍在 ROI 的時間點（當作時間點_出）
    }


def consider_cleanup_and_finalize(
    tid: int,
    st: dict,
    seen: bool,
    in_roi: bool,
    class_names,
    start_datetime: datetime.datetime,
    stats_list: list,  # 為了保持介面不變仍保留，但在 DB 模式不再累積，避免 RTSP 爆掉
):
    """
    這個函式每幀都會被呼叫（對每個 tid 一次）
    - 更新 missed_frames / frames_since_left（含 MISS_GRACE）
    - 判斷是否結算（離開 > 15 幀且 not counted）
    - 結算後：不再 append stats_list，而是交給 plate session 合併，最後落盤 DB
    - 判斷是否該清理 tid（counted 且離開太久）
    """

    # ============================================================
    # 0) 取得「現實時間」作為時間點（RTSP 最準）
    # ============================================================
    now_dt = datetime.datetime.now()

    # 節流 flush（避免每幀掃所有 session）
    global _last_flush_ts
    if time.time() - _last_flush_ts > FLUSH_INTERVAL_SECONDS:
        _last_flush_ts = time.time()
        _flush_stale_sessions(now_dt)

    # ============================================================
    # 1) 更新進出時間
    # ============================================================
    if seen and in_roi:
        # 第一次進 ROI 就記 enter_dt（只記一次，保持最早）
        if st.get("enter_dt") is None:
            st["enter_dt"] = now_dt

        # 只要在 ROI 內且有 seen，就持續更新「最後在 ROI 的時間」
        st["last_in_roi_dt"] = now_dt

    # ============================================================
    # 2) missing 計數（本幀沒看到 tid）
    # ============================================================
    if not seen:
        st["missed_frames"] += 1

    # ============================================================
    # 3) frames_since_left 核心邏輯（保持你原本行為）
    # ============================================================
    if in_roi:
        st["frames_since_left"] = 0
    elif seen:
        st["frames_since_left"] += 1
    else:
        if st["last_in_roi"] and st["missed_frames"] <= config.MISS_GRACE_FRAMES:
            st["frames_since_left"] = 0
        else:
            st["frames_since_left"] += 1

    # ============================================================
    # 4) 結算（離開 ROI > 15 且未 counted）
    # ============================================================
    if st["frames_since_left"] > config.LEAVE_ROI_FRAMES_TO_COUNT and not st["counted"]:
        most_p = st["plates"].most_common(1)
        most_c = st["classes"].most_common(1)

        # 票數不足視為雜訊，不建立事件（但依規則仍 counted=True，tid 不再重試）
        if most_p and most_c and most_p[0][1] >= config.PLATE_MIN_VOTES:
            plate_str, plate_cnt = most_p[0]
            cls_id = int(most_c[0][0])
            cls_name_str = class_name(class_names, cls_id)

            # 取該車牌字串的最佳樣本（最大面積那張），用來存圖
            best = st["plate_best"].get(plate_str)

            if best is not None:
                # 時間點_進 / 時間點_出（優先用 tid 狀態記錄；如果沒記到就用 now）
                enter_dt = st.get("enter_dt") or now_dt
                exit_dt = st.get("last_in_roi_dt") or now_dt

                # 準備截圖存檔路徑（保持你原本 screenshot 結構）
                real_ts = exit_dt.strftime("%m%d_%H%M%S")

                v_dir = config.SCREENSHOT_DIR / str(cls_name_str)
                p_dir = config.SCREENSHOT_DIR / "LPR"
                v_dir.mkdir(parents=True, exist_ok=True)
                p_dir.mkdir(parents=True, exist_ok=True)

                # 車牌圖也加時間戳，避免同車牌被覆寫
                veh_path = str(v_dir / f"{tid}_{real_ts}.jpg")
                lpr_path = str(p_dir / f"{tid}_{plate_str}_{real_ts}.jpg")

                # 存圖（best 內是 numpy image）
                cv2.imwrite(veh_path, best["veh_img"])
                cv2.imwrite(lpr_path, best["plate_crop"])

                # 同車牌跨 tid 合併（10 分鐘內合併成同一筆）
                _merge_or_start_session(
                    tid=tid,
                    cls_name_str=cls_name_str,
                    plate_str=plate_str,
                    enter_dt=enter_dt,
                    exit_dt=exit_dt,
                    plate_cnt=int(plate_cnt),
                    veh_path=veh_path,
                    lpr_path=lpr_path,
                    best_area=int(best.get("area", 0)),
                )

                # 你之前要的「確定離開就印」
                print(
                    f"[統計] ID={tid}, 車種={cls_name_str}, 車號={plate_str}, "
                    f"次數={plate_cnt}, 時間點_進={_dt_str(enter_dt)}, 時間點_出={_dt_str(exit_dt)}"
                )

        # 依你原本規則——不論成功與否，同 tid 只結算一次
        st["counted"] = True

        # 清掉 plate_best 釋放記憶體（RTSP 長時間跑很重要）
        st["plate_best"].clear()

    # ============================================================
    # 5) 清理 tid（counted 且離開太久）
    # ============================================================
    if st["counted"] and st["frames_since_left"] > config.CLEANUP_FRAMES:
        return True

    return False


# ============================================================
# 程式中斷/提前結束時：強制寫入（最小更動）
# ============================================================
def _atexit_flush():
    # 正常退出、例外退出，多半會走到這裡
    _flush_all_sessions("atexit")

atexit.register(_atexit_flush)


def _signal_handler(signum, frame):
    # Ctrl+C(SIGINT) / SIGTERM 收到時先寫入，再結束
    _flush_all_sessions(f"signal-{signum}")
    raise SystemExit(0)

try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    # 某些環境可能無法註冊 signal，忽略不影響主流程
    pass
