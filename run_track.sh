#!/bin/bash

# 設定配置資料夾路徑
SETTING_DIR="./track_setting"
# 設定同時執行的最大任務數
MAX_JOBS=2

# 取得所有 YAML 檔案清單
FILES=($SETTING_DIR/*.yaml)

echo "[INFO] 偵測到 ${#FILES[@]} 個設定檔，準備開始批次處理..."
echo "[INFO] 同時執行上限：$MAX_JOBS"

# ===== 記錄開始時間 =====
START_TIME=$(date +%s)

# 使用 xargs 達成自動排隊邏輯
printf "%s\n" "${FILES[@]}" | xargs -I {} -P $MAX_JOBS python3 run_job_track.py {}

# ===== 記錄結束時間 =====
END_TIME=$(date +%s)

# 計算總秒數
ELAPSED=$((END_TIME - START_TIME))

# 轉換成 HH:MM:SS
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

printf "[DONE] 共處理 %d 個任務，總耗時 %02d:%02d:%02d\n" \
"${#FILES[@]}" "$HOURS" "$MINUTES" "$SECONDS"
