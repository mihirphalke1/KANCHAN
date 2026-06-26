#!/usr/bin/env python3
"""
DS-7 self-collected data collection CLI.

Usage:
  python scripts/collect_data.py --type genuine --case-id G001
  python scripts/collect_data.py --type fake    --case-id F001
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect DS-7 self-collected data")
    parser.add_argument("--type",    choices=["genuine", "fake"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--karat",   type=int, choices=[14, 18, 22, 24], default=22)
    parser.add_argument("--item-type", default="ring")
    args = parser.parse_args()

    base = Path(f"data/raw/self_collected/{args.type}")
    for sub in ["images", "audio", "streak"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    print(f"\n── Collecting {args.type} item: {args.case_id} ──\n")
    print("1. Take 4–6 photos from different angles.")
    print(f"   Save them to: {base}/images/{args.case_id}_1.jpg, _2.jpg, ...\n")
    print("2. Record a tap test (WAV, ~5 seconds).")
    print(f"   Save to: {base}/audio/{args.case_id}.wav\n")
    print("3. Take a touchstone streak photo under fluorescent light.")
    print(f"   Save to: {base}/streak/{args.case_id}.jpg\n")

    weight_dry = input("4. Enter dry weight (g): ").strip()
    weight_sub = input("5. Enter submerged weight (g): ").strip()

    meta_path = base / "metadata.csv"
    header_needed = not meta_path.exists()
    with open(meta_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id","karat","weight_dry","weight_submerged","item_type","timestamp"])
        if header_needed:
            writer.writeheader()
        writer.writerow({
            "case_id":          args.case_id,
            "karat":            args.karat,
            "weight_dry":       weight_dry,
            "weight_submerged": weight_sub,
            "item_type":        args.item_type,
            "timestamp":        datetime.utcnow().isoformat(),
        })

    print(f"\n[OK] Metadata saved to {meta_path}")
    print("Remember to copy your image/audio/streak files to the paths above.")


if __name__ == "__main__":
    main()
