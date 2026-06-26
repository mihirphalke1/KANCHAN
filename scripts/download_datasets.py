#!/usr/bin/env python3
"""
Download all 6 public datasets for KANCHAN-AI.

Requirements:
  - DS-2, DS-3: Kaggle API key at ~/.kaggle/kaggle.json
  - DS-4: Roboflow public API (no key needed)
  - DS-1, DS-5: git clone (no auth)
  - DS-6: wget/requests (no auth)

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --only DS-1 DS-5   # specific datasets
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

BASE = Path("data/raw")
LOG  = Path("AGENT_LOG.md")


def log(msg: str):
    with open(LOG, "a") as f:
        f.write(f"\n---\n[{datetime.now().isoformat()}] {msg}\n---\n")
    print(f"[LOG] {msg[:80]}")


def run_cmd(cmd: str, cwd=None) -> bool:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:300]}")
    return result.returncode == 0


def dl_ds1():
    target = BASE / "counterfeit_gold"
    if (target / ".git").exists():
        print("DS-1 already cloned.")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(f"git clone https://github.com/ysaidcan/counterfeit_gold_detection {target}")
    log(f"Phase 1 — DS-1 (counterfeit_gold)\nSTATUS: {'DONE' if ok else 'FAILED'}\nACTION: git clone")
    return ok


def dl_ds2():
    target = BASE / "neu_defect"
    target.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(f"kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database --unzip -p {target}")
    log(f"Phase 1 — DS-2 (NEU defect)\nSTATUS: {'DONE' if ok else 'FAILED'}\nACTION: kaggle download")
    return ok


def dl_ds3():
    target = BASE / "tanishq"
    target.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(f"kaggle datasets download -d sapnilpatel/tanishq-jewellery-dataset --unzip -p {target}")
    log(f"Phase 1 — DS-3 (Tanishq)\nSTATUS: {'DONE' if ok else 'FAILED'}\nACTION: kaggle download")
    return ok


def dl_ds4():
    target = BASE / "roboflow_jewelry"
    target.mkdir(parents=True, exist_ok=True)
    url = "https://universe.roboflow.com/valuable-object-detection/jewelry-dkgqg/dataset/1/download/yolov5"
    print(f"  DS-4 requires manual Roboflow download from:")
    print(f"  {url}")
    print("  Download, extract, and place images in data/raw/roboflow_jewelry/images/")
    log("Phase 1 — DS-4 (Roboflow jewelry)\nSTATUS: SKIPPED\nACTION: Manual download required")
    return False


def dl_ds5():
    target = BASE / "esc50"
    if (target / ".git").exists():
        print("DS-5 already cloned.")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(f"git clone https://github.com/karoldvl/ESC-50 {target}")
    log(f"Phase 1 — DS-5 (ESC-50)\nSTATUS: {'DONE' if ok else 'FAILED'}\nACTION: git clone")
    return ok


def dl_ds6():
    target = BASE / "banknote"
    target.mkdir(parents=True, exist_ok=True)
    dest = target / "data_banknote_authentication.txt"
    if dest.exists():
        print("DS-6 already downloaded.")
        return True
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    print(f"  Downloading {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        log("Phase 1 — DS-6 (UCI Banknote)\nSTATUS: DONE\nACTION: HTTP download")
        return True
    except Exception as e:
        log(f"Phase 1 — DS-6 (UCI Banknote)\nSTATUS: FAILED\nDETAIL: {e}")
        return False


DATASETS = {
    "DS-1": dl_ds1,
    "DS-2": dl_ds2,
    "DS-3": dl_ds3,
    "DS-4": dl_ds4,
    "DS-5": dl_ds5,
    "DS-6": dl_ds6,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=DATASETS.keys(), default=None)
    args = parser.parse_args()

    targets = args.only or list(DATASETS.keys())
    results = {}
    for ds in targets:
        print(f"\n{'='*40}\nDownloading {ds}…\n{'='*40}")
        results[ds] = DATASETS[ds]()

    print("\n\n── Download Summary ──")
    for ds, ok in results.items():
        status = "✓ DONE   " if ok else "✗ FAILED  "
        if ds == "DS-4":
            status = "↷ MANUAL  "
        print(f"  {status} {ds}")

    failed = [ds for ds, ok in results.items() if not ok and ds != "DS-4"]
    if failed:
        print(f"\nFailed datasets: {failed}")
        print("Run validate_datasets.py to check what's missing.")
        sys.exit(1)
