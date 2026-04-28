# coding=utf-8
"""
state.py
管理每個 track_id 的狀態 + 結算邏輯
"""

from collections import Counter
from pathlib import Path
import datetime
import time
import cv2
from . import config
from .vision_utils import class_name

def init_state():
    # 每台車（每個 tid）在 dict 內的欄位定義
    return {
        "plates": Counter(),          # plate_str -> votes（投票用）
        "classes": Counter(),         # cls_id -> votes（決定車種用）
        "frames_since_left": 0,       # 離開 ROI 幀數（>15 才結算）
        "missed_frames": 0,           # 連續沒 seen 幀數（遮擋容忍用）
        "counted": False,             # 是否已結算（同 tid 只算一次）
        "last_in_roi": False,         # 最後一次 seen 是否在 ROI（搭配 missed 判斷遮擋）
        "plate_best": {},             # plate_str -> best sample（最大面積那張）
    }


def consider_cleanup_and_finalize(
    tid: int,
    st: dict,
    seen: bool,
    in_roi: bool,
    class_names,
    start_datetime: datetime.datetime,
    stats_list: list,
):
    """
    這個函式只做三件事（邏輯固定）：
    1) 更新 missed_frames / frames_since_left（含 MISS_GRACE）
    2) 如果符合結算條件（離開 ROI > 15 且 not counted），就結算一次並 counted=True
    3) 如果 counted 且離開太久，就回傳 True 代表要刪除狀態
    """

    # ========= 1) missing 計數 =========
    if not seen:
        st["missed_frames"] += 1  # 本幀 tracker 沒吐出 tid，視為消失

    # ========= 2) 更新 frames_since_left（核心規則）=========
    if in_roi:
        st["frames_since_left"] = 0  # 只要在 ROI 內，離開計數永遠歸零
    elif seen:
        st["frames_since_left"] += 1 # seen 且不在 ROI，確定離開，開始累加
    else:
        # 沒 seen（消失）
        if st["last_in_roi"] and st["missed_frames"] <= config.MISS_GRACE_FRAMES:
            st["frames_since_left"] = 0  # 遮擋容忍期間，視為仍在 ROI，不算離開
        else:
            st["frames_since_left"] += 1 # 消失太久或最後一次不在 ROI，視為離開/結束

    # ========= 3) 結算條件 =========
    if st["frames_since_left"] > config.LEAVE_ROI_FRAMES_TO_COUNT and not st["counted"]:
        # 取票數最多的 plate_str
        most_p = st["plates"].most_common(1)
        # 取票數最多的車種 cls_id
        most_c = st["classes"].most_common(1)

        if most_p and most_c and most_p[0][1] >= config.PLATE_MIN_VOTES:
            plate_str, plate_cnt = most_p[0]
            cls_id = int(most_c[0][0])
            cls_txt = class_name(class_names, cls_id)
            cls_name_str = class_name(class_names, cls_id)  # ← 車種字串（你想印的）

            # 取該 plate_str 的最佳樣本（最大面積那張）
            best = st["plate_best"].get(plate_str)

            if best is not None:
                # 影片秒數 -> 真實時間
                real_time = start_datetime + datetime.timedelta(seconds=best["video_time"])
                time_axis = time.strftime("%H:%M:%S", time.gmtime(best["video_time"]))
                ts = real_time.strftime("%m%d_%H%M%S")

                # 建立輸出資料夾
                v_dir = Path("screenshot") / str(cls_txt)
                p_dir = Path("screenshot") / "LPR"
                v_dir.mkdir(parents=True, exist_ok=True)
                p_dir.mkdir(parents=True, exist_ok=True)

                # 車牌圖也加時間戳，避免同 tid 覆寫
                v_path = v_dir / f"{tid}_{ts}.jpg"
                p_path = p_dir / f"{tid}_{plate_str}_{ts}.jpg"

                # 寫檔
                cv2.imwrite(str(v_path), best["veh_img"])
                cv2.imwrite(str(p_path), best["plate_crop"])

                # 寫入 Excel 統計列表
                stats_list.append({
                    "ID": tid,
                    "veh_path": str(v_path),
                    "lpr_path": str(p_path),
                    "車種": cls_txt,
                    "車號": plate_str,
                    "次數": plate_cnt,
                    "時間軸": time_axis,
                    "時間點": real_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "電腦執行時間": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                
                print(f"[統計] ID={tid}, 車種={cls_name_str}, 車號={plate_str}, 次數={plate_cnt}, 影片時間軸={time_axis}")

        # 不論成功與否，同 tid 只結算一次（你指定的行為）
        st["counted"] = True

        # 釋放影像暫存（省 RAM，不影響 counted/cleanup）
        st["plate_best"].clear()

    # ========= 4) 清除條件 =========
    if st["counted"] and st["frames_since_left"] > config.CLEANUP_FRAMES:
        return True  # 告訴外層可以 del states[tid]

    return False
