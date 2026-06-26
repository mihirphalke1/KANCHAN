#!/usr/bin/env python3
"""
Run all training scripts in order.

Pre-requisites:
  1. Run scripts/download_datasets.py
  2. Run scripts/validate_datasets.py (must pass)
  3. Collect DS-7 (run scripts/collect_data.py)

Then run: python scripts/train_all.py
"""
import subprocess
import sys

STEPS = [
    ("Validate datasets",     ["python", "scripts/validate_datasets.py"]),
    ("Train acoustic model",  ["python", "scripts/train_acoustic.py"]),
    ("Train fusion model",    ["python", "scripts/train_fusion.py"]),
]

for name, cmd in STEPS:
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFAILED: {name}")
        sys.exit(1)

print("\n\nAll training complete. Run scripts/seed_demo_data.py then start the server.")
