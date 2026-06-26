#!/usr/bin/env python3
"""
Validate all downloaded datasets before starting model training.

Run after scripts/download_datasets.py. Logs pass/fail to AGENT_LOG.md.
Aborts with exit code 1 if any required dataset fails.
"""
from datetime import datetime
from pathlib import Path

BASE = Path("data/raw")
LOG  = Path("AGENT_LOG.md")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"\n---\n[{datetime.now().isoformat()}] {msg}\n---\n")


def count_files(path: Path, ext: str) -> int:
    return len(list(path.rglob(f"*.{ext}"))) if path.exists() else 0


def validate_ds1():
    base = BASE / "counterfeit_gold"
    imgs = count_files(base, "jpg") + count_files(base, "png")
    wavs = count_files(base, "wav")
    ok   = imgs >= 60 and wavs >= 20
    return ok, f"images={imgs} (need≥60), wav={wavs} (need≥20)"


def validate_ds2():
    base = BASE / "neu_defect"
    imgs = count_files(base, "bmp") + count_files(base, "jpg") + count_files(base, "png")
    ok   = imgs >= 1800
    return ok, f"images={imgs} (need 1800)"


def validate_ds3():
    base = BASE / "tanishq"
    imgs = count_files(base, "jpg") + count_files(base, "png") + count_files(base, "webp")
    ok   = imgs >= 100
    return ok, f"images={imgs} (need≥100)"


def validate_ds5():
    base  = BASE / "esc50"
    meta  = (base / "meta" / "esc50.csv").exists()
    wavs  = count_files(base / "audio", "wav") + count_files(base, "wav")
    ok    = meta and wavs >= 2000
    return ok, f"meta={meta}, wav={wavs} (need≥2000)"


def validate_ds6():
    f  = BASE / "banknote" / "data_banknote_authentication.txt"
    if not f.exists():
        return False, "file not found"
    rows = len(f.read_text().strip().splitlines())
    ok   = rows == 1372
    return ok, f"rows={rows} (need 1372)"


VALIDATORS = {
    "DS-1 counterfeit_gold":  validate_ds1,
    "DS-2 NEU defect":        validate_ds2,
    "DS-3 Tanishq":           validate_ds3,
    "DS-5 ESC-50":            validate_ds5,
    "DS-6 UCI Banknote":      validate_ds6,
}

if __name__ == "__main__":
    print("\n══ KANCHAN-AI Dataset Validation ══\n")
    all_pass = True

    for name, fn in VALIDATORS.items():
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, str(e)

        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {name:<28} {detail}")
        log(f"Phase 1 — Validate {name}\nSTATUS: {'DONE' if ok else 'FAILED'}\nDETAIL: {detail}")

        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("All datasets validated. Ready for Phase 2 model training.")
        log("Phase 1 — Dataset validation complete\nSTATUS: DONE\nNEXT: Phase 2 model training")
    else:
        print("Some datasets failed validation. Run download_datasets.py again for failed ones.")
        log("Phase 1 — Dataset validation\nSTATUS: FAILED\nNEXT: Re-run download_datasets.py")
        import sys; sys.exit(1)
