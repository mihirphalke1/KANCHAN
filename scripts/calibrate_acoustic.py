#!/usr/bin/env python3
"""
Calibrate the genuine ring-frequency band from known-genuine tap recordings.

Protocol (run before demo day / per branch):
  1. Collect 5+ tap recordings of KNOWN-GENUINE items of one class
     (e.g. bangles) into a folder.
  2. python3 scripts/calibrate_acoustic.py <folder> [item_type]
  3. The measured band (min*0.97 .. max*1.03) is written to
     data/acoustic_calibration.json under that item_type.

The decision layer only ever flags a stiff core relative to THIS measured
band — no theoretical constant decides anything on its own.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.acoustic_physics import CALIBRATION_PATH, extract_ring_frequency


def main(folder: str, item_type: str = "default") -> None:
    wavs = sorted(Path(folder).glob("*.wav"))
    if len(wavs) < 3:
        raise SystemExit(f"Need ≥3 recordings, found {len(wavs)} in {folder}")

    freqs = []
    for w in wavs:
        r = extract_ring_frequency(w.read_bytes())
        status = f"{r['dominant_freq_hz']} Hz" if r["quality"] == "usable" else r["quality"]
        print(f"  {w.name}: {status} (SNR {r['snr_db']} dB)")
        if r["quality"] == "usable":
            freqs.append(r["dominant_freq_hz"])

    if len(freqs) < 3:
        raise SystemExit("Fewer than 3 usable recordings — re-record in quieter conditions")

    band = {
        "f_low":  round(min(freqs) * 0.97, 1),
        "f_high": round(max(freqs) * 1.03, 1),
        "n_samples": len(freqs),
        "calibrated_on": str(date.today()),
        "source_folder": str(folder),
    }
    cal = {}
    if CALIBRATION_PATH.exists():
        cal = json.loads(CALIBRATION_PATH.read_text())
    cal[item_type] = band
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    print(f"\nCalibrated '{item_type}': {band['f_low']}–{band['f_high']} Hz "
          f"({band['n_samples']} recordings) -> {CALIBRATION_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: calibrate_acoustic.py <folder-of-genuine-wavs> [item_type]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "default")
