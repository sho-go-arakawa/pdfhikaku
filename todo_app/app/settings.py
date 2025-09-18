import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / ".data"
DATABASE_PATH = DATA_DIR / "todo.duckdb"

DATA_DIR.mkdir(exist_ok=True)

PRIORITY_LEVELS = {
    0: "Low",
    1: "Medium", 
    2: "High"
}

TASK_STATUSES = ["todo", "doing", "done"]

RECURRENCE_RULES = ["daily", "weekly", "monthly"]

DEFAULT_PAGE_SIZE = 50

NOTIFICATION_HOURS_BEFORE = [24, 1]  # 24時間前と1時間前に通知

APP_TITLE = "ToDoアプリ"
APP_VERSION = "v0.3"