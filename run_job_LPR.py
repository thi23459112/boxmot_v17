import sys
import time
import multiprocessing as mp
from main_LPR import run_single

def wrapper(yaml_path):
    """
    用一個 wrapper 函數來建立獨立的 Multiprocessing Context
    確保 run_single 跑在一個乾淨的環境
    """
    # 建立一個模擬的全域停止事件 (給 run_single 用)
    # 注意：在單一任務模式下，這個 event 其實只影響自己
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    
    try:
        run_single(yaml_path, stop_event)
    except Exception as e:
        print(f"[ERROR] 任務發生錯誤 {yaml_path}: {e}")
    finally:
        # 確保在 wrapper 結束時設定停止訊號
        stop_event.set()

if __name__ == "__main__":
    # Windows / Linux 多進程保護
    mp.freeze_support()

    if len(sys.argv) < 2:
        print("Usage: python run_job_LPR.py <path_to_yaml>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    
    print(f"[JOB START] 啟動任務: {yaml_path}")
    
    # 使用 spawn 模式啟動子進程
    # 這是最關鍵的一步：讓 run_single 在獨立的 Process 中執行
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=wrapper, args=(yaml_path,))
    
    p.start()
    
    try:
        # 等待子進程結束
        p.join()
        
    except KeyboardInterrupt:
        print(f"[JOB INTERRUPT] 使用者中斷任務: {yaml_path}")
        p.terminate() # 強制殺死
        p.join()
    
    # 雙重保險：確保 Process 真的死了
    if p.is_alive():
        print(f"[JOB KILL] 強制終止殘留進程: {yaml_path}")
        p.terminate()
        p.join()
        
    print(f"[JOB DONE] 任務結束，記憶體應已釋放: {yaml_path}")