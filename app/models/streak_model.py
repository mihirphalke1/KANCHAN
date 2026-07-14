"""
Touchstone streak modality — HSV hue-band physics (no ML).

Genuine gold leaves a warm yellow streak on the touchstone; the exact hue,
saturation and brightness shift with karat — higher purity reads warmer and
more saturated, because the alloy has less silver/copper diluting the colour.
Bands are calibrated per karat in data/streak_calibration.json (same pattern
as data/acoustic_calibration.json), with documented fallback bands below when
no calibration file exists.

This replaces a previously orphaned LogisticRegression model: no training
script for it existed anywhere in the repository or git history, so it could
not be retrained, recalibrated, or audited for what data it ever saw. An
unauditable black box is a worse fit for a bank decision system than a
calibrated, hand-checkable physics rule — the same trade-off already made for
density and the acoustic ring-frequency cross-check.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path("data/streak_calibration.json")

# Fallback bands in standard 0-360° hue (converted from OpenCV's 0-179 at
# extraction time). Higher karat -> narrower, warmer, more saturated band;
# the alloy is more silver/copper at lower karats, which cools and desaturates
# the streak. "0" (karat not declared) uses the widest union band, purely as
# a sanity check that the streak looks like SOME gold karat at all.
DEFAULT_BANDS = {
    "24": {"hue_low": 32, "hue_high": 50, "sat_min": 0.55, "val_min": 0.55},
    "22": {"hue_low": 28, "hue_high": 50, "sat_min": 0.48, "val_min": 0.50},
    "18": {"hue_low": 22, "hue_high": 52, "sat_min": 0.40, "val_min": 0.45},
    "14": {"hue_low": 16, "hue_high": 55, "sat_min": 0.32, "val_min": 0.40},
    "0":  {"hue_low": 16, "hue_high": 55, "sat_min": 0.32, "val_min": 0.40},
}


def load_bands() -> dict:
    """Calibrated bands when available; documented fallbacks otherwise."""
    if CALIBRATION_PATH.exists():
        try:
            cal = json.loads(CALIBRATION_PATH.read_text())
            return {**DEFAULT_BANDS, **cal}
        except Exception as e:
            logger.warning("Could not read streak calibration: %s", e)
    return DEFAULT_BANDS


def _hsv_stats(streak_bytes: bytes) -> dict:
    from app.utils.preprocess import load_image_bytes, extract_hsv_stats
    rgb = load_image_bytes(streak_bytes)
    stats = extract_hsv_stats(rgb)
    # OpenCV hue is 0-179 (degrees/2) -> standard 0-360 degrees.
    return {
        "hue_deg": stats["hue_mean"] * 2.0,
        "sat":     stats["sat_mean"] / 255.0,
        "val":     stats["val_mean"] / 255.0,
    }


def analyze_streak(streak_bytes: bytes | None, declared_karat: int = 0) -> dict:
    if streak_bytes is None:
        return {"risk_score": 0.5, "confidence": "low", "mode": "no_streak"}

    try:
        stats = _hsv_stats(streak_bytes)
    except Exception as e:
        logger.warning("Streak HSV extraction failed (%s)", e)
        return {"risk_score": 0.5, "confidence": "low", "mode": "hsv_error"}

    bands = load_bands()
    band = bands.get(str(declared_karat)) or bands["0"]
    hue, sat, val = stats["hue_deg"], stats["sat"], stats["val"]

    risk = 0.0
    signals = []

    if hue < band["hue_low"] or hue > band["hue_high"]:
        dist = (band["hue_low"] - hue) if hue < band["hue_low"] else (hue - band["hue_high"])
        risk += min(0.45, 0.20 + 0.0125 * dist)   # saturates ~20° past the band
        signals.append(
            f"Streak hue {hue:.0f}° is outside the "
            f"{declared_karat or 'reference'}K genuine band "
            f"({band['hue_low']:.0f}–{band['hue_high']:.0f}°)"
        )
    if sat < band["sat_min"]:
        risk += 0.30 * min(1.0, (band["sat_min"] - sat) / band["sat_min"])
        signals.append(f"Streak saturation {sat:.2f} is below the expected {band['sat_min']:.2f} — pale/washed streak")
    if val < band["val_min"]:
        risk += 0.20 * min(1.0, (band["val_min"] - val) / band["val_min"])
        signals.append(f"Streak brightness {val:.2f} is below the expected {band['val_min']:.2f} — dull streak")

    risk = min(risk, 1.0)
    if not signals:
        signals.append(
            f"Streak colour (hue {hue:.0f}°, sat {sat:.2f}, val {val:.2f}) matches the "
            f"{declared_karat or 'reference'}K genuine band"
        )

    return {
        "risk_score": round(risk, 4),
        "confidence": "medium" if abs(risk - 0.5) > 0.25 else "low",
        "mode":       "hsv_bands",
        "signals":    signals,
        "features": {
            "hue_deg": round(hue, 1), "sat": round(sat, 3), "val": round(val, 3),
            "band":    band,
        },
    }
