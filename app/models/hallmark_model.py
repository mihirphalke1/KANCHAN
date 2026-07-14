"""
Hallmark engraving-authenticity check — classical CV on a macro photo of the
BIS hallmark/HUID engraving.

The objection this answers: a HUID is just a handful of characters, trivial
to copy onto fake metal. Looking the code up (mock BIS registry — see
app/utils/bis_registry.py) proves nothing on its own, since a scammer can
copy a genuine code verbatim. What's harder to fake is the physical
ENGRAVING ITSELF: genuine BIS hallmarking is laser-engraved, with consistent
stroke width and character spacing; a hand-punched or engraving-tool
forgery shows irregular stroke width and uneven spacing. This is one weak
signal, fused at low weight — never a standalone pass/fail, and always
paired with the karat the PHYSICS measured (density test), not the karat
printed on the mark.
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STROKE_VAR_GENUINE_MAX  = 0.35   # normalised stroke-width coefficient of variation
SPACING_VAR_GENUINE_MAX = 0.40   # normalised character-gap coefficient of variation


def _binarize_text(bgr: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
    )
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def analyze_hallmark_engraving(image_bytes: bytes) -> dict:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode hallmark image")
    except Exception as e:
        logger.warning("Hallmark image decode failed: %s", e)
        return {"risk_score": 0.5, "confidence": "low", "mode": "hallmark_error"}

    binary = _binarize_text(bgr)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < 12 or h < 4 or w < 2:
            continue
        glyphs.append((x, y, w, h))

    if len(glyphs) < 3:
        return {
            "risk_score": 0.5, "confidence": "low", "mode": "hallmark_insufficient",
            "signals": ["Fewer than 3 engraved characters detected — retake the macro photo, closer and better lit"],
        }

    glyphs.sort(key=lambda g: g[0])
    strokes = []
    for (x, y, w, h) in glyphs:
        roi = binary[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
        if dist.max() > 0:
            strokes.append(float(dist.max()) * 2.0)
    stroke_cv = (np.std(strokes) / np.mean(strokes)) if strokes and np.mean(strokes) > 0 else 1.0

    gaps = []
    for i in range(1, len(glyphs)):
        prev_x, _, prev_w, _ = glyphs[i - 1]
        cur_x, _, _, _ = glyphs[i]
        gap = cur_x - (prev_x + prev_w)
        if gap > 0:
            gaps.append(gap)
    spacing_cv = (np.std(gaps) / np.mean(gaps)) if gaps and np.mean(gaps) > 0 else 0.0

    stroke_risk  = min(1.0, stroke_cv / STROKE_VAR_GENUINE_MAX * 0.5)
    spacing_risk = min(1.0, spacing_cv / SPACING_VAR_GENUINE_MAX * 0.5)
    risk = round(min(1.0, 0.6 * stroke_risk + 0.4 * spacing_risk), 4)

    signals = [f"{len(glyphs)} engraved character(s) detected for analysis"]
    if stroke_risk > 0.5:
        signals.append(f"Stroke width is irregular (CV {stroke_cv:.2f}) — inconsistent with a single laser pass")
    if spacing_risk > 0.5:
        signals.append(f"Character spacing is uneven (CV {spacing_cv:.2f}) — inconsistent with machine engraving")
    if stroke_risk <= 0.5 and spacing_risk <= 0.5:
        signals.append("Stroke width and character spacing are consistent with laser engraving")

    return {
        "risk_score": risk,
        "confidence": "medium" if len(glyphs) >= 5 else "low",
        "mode":       "hallmark_engraving",
        "signals":    signals,
        "features": {
            "n_glyphs": len(glyphs), "stroke_cv": round(stroke_cv, 3), "spacing_cv": round(spacing_cv, 3),
        },
    }
