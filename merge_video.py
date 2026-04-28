import os
import subprocess
import re
import signal
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

# ================= 配置區域 =================
OUTPUT_PREFIX = "忠C-"
FOLDERS = ['A', 'B', 'C', 'D', 'E']
OUTPUT_DIR_NAME = "merge_video"

# 根據你的硬體調整 (SSD建議 3-5, HDD建議 1-2)
MAX_WORKERS = 5
# ===========================================

console = Console()
shutdown_event = threading.Event()
active_pids = set()
pid_lock = threading.Lock()

# 預編譯 Regex
RE_FRAME = re.compile(r'frame=\s*(\d+)')
RE_FPS = re.compile(r'fps=\s*([\d.]+)')
RE_TIME = re.compile(r'time=(\d+:\d+:\d+\.\d+)')
RE_SPEED = re.compile(r'speed=\s*([\d.eE+-]+x)')
RE_TIMESTAMP_1 = re.compile(r'(\d{8})[-_](\d{6})')
RE_TIMESTAMP_2 = re.compile(r'(\d{14})')

# 全域監控數據
monitor_data = {}
data_lock = threading.Lock()

def init_monitor_data():
    for folder in FOLDERS:
        monitor_data[folder] = {
            'status': '等待中...',
            'fps': '0.0',
            'speed': '0.00x',
            'frames': 0,
            'time': '00:00:00',
            'color': 'dim'
        }

def update_monitor(folder, **kwargs):
    with data_lock:
        if folder in monitor_data:
            monitor_data[folder].update(kwargs)

def signal_handler(signum, frame):
    shutdown_event.set()
    with pid_lock:
        for pid in list(active_pids):
            try:
                if os.name == 'nt':
                    subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except:
                pass
    os._exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def register_pid(pid):
    with pid_lock:
        active_pids.add(pid)

def unregister_pid(pid):
    with pid_lock:
        if pid in active_pids:
            active_pids.remove(pid)

def format_speed(speed_str):
    try:
        val = float(speed_str.replace('x', ''))
        return f"{val:.2f}x"
    except:
        return speed_str

def extract_timestamp(filename):
    match = RE_TIMESTAMP_1.search(filename)
    if match: return (match.group(1) + match.group(2), f"{match.group(1)}_{match.group(2)}")
    match = RE_TIMESTAMP_2.search(filename)
    if match: 
        t = match.group(1)
        return (t, f"{t[:8]}_{t[8:]}")
    return (filename, filename)

def get_video_files(folder_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    folder = Path(folder_path)
    if not folder.exists(): return []
    video_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    video_files.sort(key=lambda x: extract_timestamp(x.name)[0])
    return video_files

# --- 新增功能: 獲取精確時長 ---
def get_final_duration(file_path):
    """使用 ffprobe 讀取檔案的精確時長"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(file_path)
        ]
        # 執行並獲取輸出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        seconds = float(result.stdout.strip())
        
        # 轉換為 HH:MM:SS 格式
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        return None  # 如果失敗，回傳 None (保持原本顯示)
# ---------------------------

def merge_single_folder(folder_name, prefix, output_dir_path):
    filelist_path = f"filelist_{folder_name}.txt"
    start_time = time.time()
    
    try:
        if shutdown_event.is_set(): 
            update_monitor(folder_name, status="已取消", color="red")
            return
            
        video_files = get_video_files(folder_name)
        if not video_files:
            update_monitor(folder_name, status="無影片", color="yellow")
            return
        
        first_timestamp = extract_timestamp(video_files[0].name)[1]
        output_filename = f"{prefix}{first_timestamp}.mp4"
        output_full_path = output_dir_path / output_filename
        
        update_monitor(folder_name, status=f"準備合併 {len(video_files)} 個檔", color="blue")
        
        with open(filelist_path, 'w', encoding='utf-8', buffering=65536) as f:
            for video in video_files:
                f.write(f"file '{str(video.absolute()).replace(os.sep, '/')}'\n")
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', filelist_path,
            '-c:v', 'copy', '-an', '-y',
            str(output_full_path)
        ]
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='ignore', bufsize=-1
        )
        
        register_pid(process.pid)
        update_monitor(folder_name, status="🚀 處理中", color="green")
        
        last_update = time.time()
        
        for line in process.stdout:
            if shutdown_event.is_set():
                process.terminate()
                break
            
            if line.startswith('frame=') and (time.time() - last_update > 0.2):
                updates = {}
                frame_match = RE_FRAME.search(line)
                if frame_match: updates['frames'] = int(frame_match.group(1))
                fps_match = RE_FPS.search(line)
                if fps_match: updates['fps'] = float(fps_match.group(1))
                speed_match = RE_SPEED.search(line)
                if speed_match: updates['speed'] = format_speed(speed_match.group(1))
                time_match = RE_TIME.search(line)
                if time_match: updates['time'] = time_match.group(1) # 這裡是過程中的時間
                
                if updates:
                    update_monitor(folder_name, **updates)
                    last_update = time.time()
        
        process.wait()
        unregister_pid(process.pid)
        
        if os.path.exists(filelist_path):
            try: os.remove(filelist_path)
            except: pass
        
        if process.returncode == 0:
            # === 修正點：任務成功後，讀取最終真實時間 ===
            final_time = get_final_duration(output_full_path)
            
            update_data = {
                "status": "✅ 完成", 
                "color": "bold green", 
                "speed": "-", 
                "fps": 0.0
            }
            
            # 如果成功讀取到最終時間，就覆蓋掉原本的過程時間
            if final_time:
                update_data["time"] = final_time
            
            update_monitor(folder_name, **update_data)
            # ========================================
        else:
            update_monitor(folder_name, status="❌ 失敗", color="bold red")
            
    except Exception as e:
        update_monitor(folder_name, status="❌ 錯誤", color="bold red")

def create_dashboard():
    current_time = datetime.now().strftime('%H:%M:%S')
    
    table = Table(
        title=f"[{current_time}] 影片合併 (操作：按 Ctrl+C 安全結束)",
        box=box.ROUNDED,
        header_style="bold cyan",
        expand=True,
        caption=f"多工處理核心數: {MAX_WORKERS}"
    )

    table.add_column("資料夾", justify="center", style="bold cyan", no_wrap=True)
    table.add_column("狀態", justify="left", no_wrap=True)
    table.add_column("時間", justify="center", style="white")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("倍率", justify="right", style="magenta")
    table.add_column("總幀數", justify="right", style="yellow")

    existing_folders = [f for f in FOLDERS if f in monitor_data]
    
    for folder in existing_folders:
        data = monitor_data[folder]
        status_color = data['color']
        
        fps_display = f"{data['fps']:.1f}" if isinstance(data['fps'], float) else str(data['fps'])
        frames_display = f"{data['frames']:,}"
        
        table.add_row(
            f"{folder}",
            f"[{status_color}]{data['status']}[/]",
            f"{data['time']}",
            fps_display,
            f"{data['speed']}",
            frames_display
        )
        
    return table

def main():
    global_start_time = time.time()
    output_dir = Path(OUTPUT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_folders = [f for f in FOLDERS if Path(f).exists()]
    if not existing_folders:
        console.print("[red]找不到資料夾[/red]")
        return

    init_monitor_data()
    
    with Live(create_dashboard(), refresh_per_second=2, console=console) as live:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for folder in existing_folders:
                futures.append(executor.submit(merge_single_folder, folder, OUTPUT_PREFIX, output_dir))
            
            while any(not f.done() for f in futures):
                if shutdown_event.is_set(): break
                live.update(create_dashboard())
                time.sleep(0.25)
            
            live.update(create_dashboard())

    total_time = time.time() - global_start_time
    console.print(f"\n[bold green]全部完成！總耗時: {timedelta(seconds=int(total_time))}[/bold green]")

if __name__ == "__main__":
    main()