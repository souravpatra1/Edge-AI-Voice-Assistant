# memory_logger.py
import os
import datetime
from utils import config

os.makedirs(config.LOG_DIR, exist_ok=True)

def get_today_log_path() -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return os.path.join(config.LOG_DIR, f"{today}.log")

def log_speech(speaker: str, text: str):
    """Append timestamped speech lines to today's log file."""
    if not text.strip():
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {speaker}: {text}\n"
    
    with open(get_today_log_path(), "a", encoding="utf-8") as f:
        f.write(entry)

def get_today_transcript() -> str:
    """Fetch today's entire log content."""
    path = get_today_log_path()
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
