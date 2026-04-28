import sys
import multiprocessing as mp
import threading
from main_track import run_single # 車流專用

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_job_track.py <path_to_yaml>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    
    # 建立一個模擬的全域停止事件
    ctx = mp.get_context("spawn")
    global_stop_event = ctx.Event()
    
    # 執行單一任務
    try:
        run_single(yaml_path, global_stop_event)
    except KeyboardInterrupt:
        global_stop_event.set()
        print(f"User interrupted {yaml_path}")