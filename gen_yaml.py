import os
import re
from pathlib import Path

# ================= 設定區域 =================
INPUT_DIR = "test"        # ✅ 影片來源資料夾
OUTPUT_DIR = "generated_configs" # 生成的 yaml 存放資料夾
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= YAML 模板 =================
# 使用 f-string 模板，保留你的註解與格式
def get_yaml_content(filename_stem, full_video_path, start_time_str):
    return f"""# {filename_stem}.yaml
# ------------------------------------------------------------
# 說明：
# - 這份檔案是自動生成的設定（來源：{filename_stem}）
# ------------------------------------------------------------

source_id: "{filename_stem}"            

# ========= 輸入來源（影片/串流） =========
source: "test/{filename_stem}.mp4"     # ✅ 影片路徑
stream_fps: 15.0                                    # RTSP 手動 FPS（null 表示不強制）
# 影片的真實起始時間 (格式：YYYY-MM-DD HH:MM:SS)
start_time: "{start_time_str}"

# ========= 模型路徑 =========
models:
  vehicle: "weight/car.engine"
  plate: "weight/plate.engine"
  num: "weight/num.engine"
  reid: "weight/mobilenetv2_x1_0_market1501.engine"

# ========= 輸出設定 =========
output:
  save_output_video: True
  output_video_dir: "output_video" # ✅ 輸出影片路徑

  # ❌ only for 礁溪轉運站，其他專案不影響 
  db_dir: "output_db"
  db_name: "J219_BusStation.db"

  # 截圖根目錄（建議每路加子資料夾避免互相覆蓋）
  screenshot_dir: "screenshot/{filename_stem}"

# ========= 偵測參數 =========
detect:
  yolo_conf: 0.25                        # 車輛信心值
  yolo_classes: [0, 1, 2, 3, 4, 5, 6]    # bicycle, bus, car, motorbike, smalltruck, trailer, truck 
  plate_conf: 0.25                       # 車牌信心值
  num_conf: 0.30                         # 車號信心值
  char_nms_iou: 0.50                     # 字元框 NMS IoU

# ========= 追蹤器參數 =========
tracker:
  # 追蹤器類型，可改為 'botsort','deepocsort','ocsort','strongsort','boosttrack','bytetrack','imprassoc','hybridsort','fasttracker','sfsort','cbiou'
  tracker_type: "cbiou"              # ✅ 如果 cbiou 追蹤不好，改 imprassoc      
  show_trajectories: false           # 是否顯示軌跡

# ========= 座標基準與 ROI / MASK =========
geometry:                                   # ✅ 影片/串流解析度
  base_w: 1920
  base_h: 1080

  # 偵測遮罩（Mask）：遮罩外不偵測
  mask_points:                              # ✅ 藍色遮罩(順時針)
    - [0, 0]
    - [1920, 0]
    - [1920, 1080]
    - [0, 1080]

  # 計數/辨識 ROI：只有 ROI 內才做 LPR/累積
  region_points:                            # ✅ 黃色 ROI (順時針)
    - [0, 0]
    - [1920, 0]
    - [1920, 1080]
    - [0, 1080]

# ========= 統計/合併規則 =========
session:
  leave_roi_frames_to_count: 15     # ✅ 離開 ROI > N 幀才結算
  plate_min_votes: 2                # ✅ 車牌至少出現 N 次才採信
  cleanup_frames: 60                # 結算後離開太久就清除 tid 狀態
  miss_grace_frames: 45             # 遮擋/漏偵測容忍幀數
  plate_merge_gap_seconds: 600      # 同車牌 10 分鐘內視為同一筆（跨 ID 合併）
  flush_interval_seconds: 30        # 每隔 N 秒掃一次逾時 session 並落盤

# ========= 容忍統計規則 =========
track_logic:
  movement_threshold: 30             # 位移像素門檻（用於判定進出方向）
  min_roi_hits: 5                    # 命中 ROI 的最小幀數（用於統計結算）
"""

# ================= 主程式邏輯 =================

def parse_timestamp(filename):
    """
    從檔名中解析影片起始時間

    支援的檔名時間格式：

    1. YYYYMMDD_HHMMSS
       範例：xxx-20250522_002357.mp4

    2. YYYYMMDDHHMMSS
       範例：xxx-20250522002357.mp4

    3. YYYY_MMDD_HHMMSS / YYYY-MMDD-HHMMSS
       範例：xxx-2025_1217_000000.mp4

    回傳格式：
        YYYY-MM-DD HH:MM:SS

    若無法解析則回傳 None
    """

    # --------------------------------------------------
    # 格式 1：YYYYMMDD_HHMMSS 或 YYYYMMDD-HHMMSS
    # 例：20250522_002357
    # --------------------------------------------------
    match = re.search(r'(\d{8})[-_](\d{6})', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS

        return (
            f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} "
            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )

    # --------------------------------------------------
    # 格式 2：YYYYMMDDHHMMSS（14 碼連續數字）
    # 例：20250522002357
    # --------------------------------------------------
    match = re.search(r'(\d{14})', filename)
    if match:
        dt = match.group(1)

        return (
            f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]} "
            f"{dt[8:10]}:{dt[10:12]}:{dt[12:14]}"
        )

    # --------------------------------------------------
    # 格式 3：YYYY_MMDD_HHMMSS / YYYY-MMDD-HHMMSS
    # 例：2025_1217_000000
    # --------------------------------------------------
    match = re.search(r'(\d{4})[-_](\d{4})[-_](\d{6})', filename)
    if match:
        year, md, time_str = match.groups()

        month = md[:2]
        day = md[2:4]

        return (
            f"{year}-{month}-{day} "
            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )

    # --------------------------------------------------
    # 無法解析時間
    # --------------------------------------------------
    return None

def main():
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists():
        print(f"錯誤: 找不到資料夾 '{INPUT_DIR}'")
        return

    print(f"正在掃描資料夾: {INPUT_DIR} ...")
    count = 0

    files = sorted(input_path.iterdir())
    
    for file in files:
        if file.is_file() and file.suffix.lower() in VIDEO_EXTS:
            # 1. 取得檔名 (不含副檔名) 作為 ID
            file_stem = file.stem 
            
            # 2. 解析時間
            start_time = parse_timestamp(file.name)
            if not start_time:
                print(f"[跳過] 無法解析時間: {file.name}")
                continue

            # 3. 組合 source 路徑 (使用 forward slash 避免 Windows 路徑問題)
            # 你的範例是 video/xxx，但如果你是從 merge_video 讀取，路徑應該是 merge_video/xxx
            # 這裡設定為: merge_video/檔名.mp4
            video_path = f"{INPUT_DIR}/{file.name}"

            # 4. 生成內容
            yaml_content = get_yaml_content(file_stem, video_path, start_time)

            # 5. 寫入檔案
            output_filename = f"{file_stem}.yaml"
            output_path = Path(OUTPUT_DIR) / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            print(f"[生成] {output_filename} \t(時間: {start_time})")
            count += 1

    print(f"\n✅ 完成！共生成 {count} 個 YAML 檔案，位於 '{OUTPUT_DIR}' 資料夾。")

if __name__ == "__main__":
    main()