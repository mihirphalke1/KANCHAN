#!/usr/bin/env python3
"""
Download DS-4: Roboflow Jewelry Dataset (324 mixed jewelry images).
Requires ROBOFLOW_API_KEY in .env.

Run: python scripts/download_ds4.py
Then re-run: python scripts/build_and_train.py  (embeddings cache will be updated)
"""
import os, sys, zipfile, shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
if not API_KEY or API_KEY == "your_roboflow_key_here":
    print("ERROR: ROBOFLOW_API_KEY not set in .env")
    print("  1. Go to https://roboflow.com → Settings → API")
    print("  2. Copy your API key")
    print("  3. Add ROBOFLOW_API_KEY=<your_key> to .env")
    sys.exit(1)

import requests

OUT = ROOT / "data" / "raw" / "roboflow_jewelry"
OUT.mkdir(parents=True, exist_ok=True)

print("Fetching Roboflow dataset info...")
url = (
    "https://api.roboflow.com/valuable-object-detection/jewelry-dkgqg/1"
    f"/yolov5pytorch?api_key={API_KEY}"
)
resp = requests.get(url, timeout=30)
if resp.status_code != 200:
    print(f"ERROR: Roboflow API returned {resp.status_code}: {resp.text[:200]}")
    sys.exit(1)

data = resp.json()
download_url = data.get("export", {}).get("link")
if not download_url:
    print("ERROR: Could not get download link from Roboflow API")
    print(f"  Response: {data}")
    sys.exit(1)

print(f"Downloading dataset...")
zip_path = OUT / "jewelry.zip"
with requests.get(download_url, stream=True, timeout=120) as r:
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done*100//total}%", end="\r", flush=True)

print(f"\nExtracting...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(OUT)
zip_path.unlink()

# Flatten: collect all images into genuine/ and fake/ subfolders
# Roboflow exports YOLOv5 format: train/images/, valid/images/
# Label "jewelry" = genuine proxy, "fashion" or non-labelled = fake proxy
img_count = 0
for split in ("train", "valid", "test"):
    img_dir = OUT / split / "images"
    if img_dir.exists():
        for img in img_dir.glob("*.jpg"):
            dest = OUT / "images" / img.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy(img, dest)
            img_count += 1

print(f"Extracted {img_count} images → {OUT}/images/")
print("\nNext step: run python scripts/build_and_train.py to include DS-4 in training")
