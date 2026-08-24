#!/usr/bin/env python3
"""Auto loop: download missing histories, then retrain models, repeat.

Usage: python scripts/auto_download_and_retrain.py
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "auto_retrain.log"

PY = sys.executable

def log(msg: str):
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def run_command(cmd: list, cwd: Path = ROOT, timeout: int = None) -> int:
    log(f"RUN: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout)
        log(f"EXIT {r.returncode} for: {' '.join(cmd)}")
        return r.returncode
    except Exception as e:
        log(f"ERROR running {' '.join(cmd)}: {e}")
        return 1

def main():
    log("auto_download_and_retrain starting")
    iteration = 0
    while True:
        iteration += 1
        log(f"iteration={iteration}: starting download pass")
        run_command([PY, "scripts/download_missing_histories.py"])

        log(f"iteration={iteration}: starting training pass")
        run_command([PY, "train_models.py"])

        # brief pause to avoid hammering MT5
        log(f"iteration={iteration}: sleeping 60s before next loop")
        time.sleep(60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log("auto_download_and_retrain stopped by user")
