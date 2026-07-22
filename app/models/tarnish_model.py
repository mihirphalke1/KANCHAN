"""
Discoloration / rust / tarnish detection, and a UV-pass fluorescence
heuristic — both extend the same HSV toolkit already used for the streak
and DSIP modalities.

Rust reads as a hue shift toward orange-brown with a localized saturation
spike; tarnish reads as a value (brightness) drop with desaturation. Both
are LOCALIZED — a genuine item can carry a small tarnished patch without
being fake — so this reports an anomalous-area fraction, not a single
whole-photo verdict, and is fused as a low-weight advisory signal.
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Reference bands (OpenCV HSV units) for a clean, uniformly-coloured gold
# surface at each karat — same source pattern as data/streak_calibration.json.
REFERENCE_BANDS = {
    24: {"hue_low": 16, "hue_high": 28, "sat_min": 90, "val_min": 120},
    22: {"hue_low": 14, "hue_high": 28, "sat_min": 80, "val_min": 110},
    18: {"hue_low": 11, "hue_high": 30, "sat_min": 65, "val_min": 95},
    14: {"hue_low": 8,  "hue_high": 32, "sat_min": 50, "val_min": 85},
}
RUST_HUE_LOW, RUST_HUE_HIGH = 0, 14   # orange-brown, OpenCV 0-179 units
ANOMALY_AREA_FLAG_PCT = 6.0


def _classify_discoloration(hsv: np.ndarray, mask: np.ndarray, total: int) -> dict:
    """Distinguish an INTENTIONAL antique patina (a legitimate finish) from
    CONTAMINATION tarnish/verdigris (a genuine anomaly).

    Antique patina is applied deliberately and reads as a broad, uniform, warm
    (orange/brown/red) darkening across most of the piece. Contamination is
    localised and/or shifts toward green (verdigris) — the corrosion tell. The
    returned risk_multiplier lets analyze_tarnish DOWN-weight a patina finish
    instead of flagging it like corrosion."""
    n = int(mask.sum())
    if n == 0 or total == 0:
        return {"type": "none", "risk_multiplier": 1.0, "coverage_frac": 0.0}
    frac = n / float(total)
    hh = hsv[..., 0][mask].astype(np.float32)
    hue_mean = float(np.mean(hh))
    hue_std  = float(np.std(hh))

    warm     = (hue_mean <= 35) or (hue_mean >= 160)   # orange/brown/red (OpenCV hue 0-180)
    greenish = 40 <= hue_mean <= 100                    # verdigris / cool contamination
    broad    = frac >= 0.25
    uniform  = hue_std <= 20

    if broad and warm and uniform and not greenish:
        return {"type": "intentional_patina", "risk_multiplier": 0.3,
                "coverage_frac": round(frac, 3), "hue_mean": round(hue_mean, 1)}
    return {"type": "contamination_tarnish", "risk_multiplier": 1.0,
            "coverage_frac": round(frac, 3), "hue_mean": round(hue_mean, 1)}


def analyze_tarnish(image_bytes: bytes, declared_karat: int = 22) -> dict:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        logger.warning("Tarnish image decode failed: %s", e)
        return {"risk_score": 0.5, "confidence": "low", "mode": "tarnish_error"}

    band = REFERENCE_BANDS.get(declared_karat, REFERENCE_BANDS[22])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)

    rust_mask    = (h >= RUST_HUE_LOW) & (h <= RUST_HUE_HIGH) & (s > band["sat_min"])
    tarnish_mask = (v < band["val_min"] * 0.6) & (s < band["sat_min"] * 0.5)

    total_px    = h.size
    rust_pct    = float(rust_mask.sum()) / total_px * 100
    tarnish_pct = float(tarnish_mask.sum()) / total_px * 100

    risk, signals = 0.0, []
    if rust_pct > ANOMALY_AREA_FLAG_PCT:
        risk += min(0.5, rust_pct / 40.0)
        signals.append(f"{rust_pct:.1f}% of the frame shows a rust-like orange-brown hue with elevated saturation")
    if tarnish_pct > ANOMALY_AREA_FLAG_PCT:
        risk += min(0.5, tarnish_pct / 40.0)
        signals.append(f"{tarnish_pct:.1f}% of the frame is dark and desaturated — consistent with tarnish/oxidation")
    risk = min(1.0, risk)

    # Patina vs contamination: an intentional antique patina is a legitimate
    # finish, not corrosion — classify the discoloured region and down-weight
    # the risk when it reads as a broad, uniform, warm patina.
    disc_mask = rust_mask | tarnish_mask
    discoloration = _classify_discoloration(hsv, disc_mask, total_px)
    if discoloration["type"] == "intentional_patina" and risk > 0:
        risk = round(min(risk, risk * discoloration["risk_multiplier"]), 4)
        signals.append("Discolouration reads as a broad, uniform warm patina — treated as an intentional antique finish, not corrosion")

    if not signals:
        signals.append("No significant rust or tarnish regions detected")

    return {
        "risk_score": round(risk, 4),
        "confidence": "medium" if (rust_pct + tarnish_pct) > 2 else "low",
        "mode":       "tarnish_hsv",
        "signals":    signals,
        "features":   {"rust_area_pct": round(rust_pct, 2), "tarnish_area_pct": round(tarnish_pct, 2),
                       "discoloration_type": discoloration["type"]},
    }


def analyze_uv_pass(image_bytes: bytes) -> dict:
    """
    UV-pass ('blacklight') photo heuristic. Solid genuine gold should not
    visibly fluoresce; rhodium plating and many synthetic/paste stones DO
    fluoresce distinctly under UV-A. Flags large, unusually bright uniform
    regions as a plating/coating tell. Deliberately conservative and framed
    as assistive, not conclusive — fluorescence strength varies with the UV
    source and ambient light, which this cannot control for.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        logger.warning("UV image decode failed: %s", e)
        return {"risk_score": 0.5, "confidence": "low", "mode": "uv_error"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    bright_pct = float((v > 200).mean()) * 100

    risk = min(1.0, max(0.0, (bright_pct - 15.0) / 50.0)) if bright_pct > 15.0 else 0.0
    signals = (
        [f"{bright_pct:.1f}% of the UV-pass frame fluoresces brightly — check for plating/coating or "
         "synthetic stones; genuine solid gold should not glow this strongly"]
        if risk > 0 else
        ["No widespread fluorescence detected — consistent with unplated solid metal"]
    )
    return {
        "risk_score": round(risk, 4),
        "confidence": "low",
        "mode":       "uv_heuristic",
        "signals":    signals,
        "features":   {"bright_area_pct": round(bright_pct, 2)},
    }
