"""
Touchstone streak modality.
Genuine gold leaves a golden-yellow streak (hue 20-55°, high sat, mid-high val).
Plated base metals often show reddish or dull streaks.

Model: LogReg trained on DS-7 self-collected streak images.
Fallback: HSV hue check on the streak region.
"""
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STREAK_MODEL_PATH = Path("models/streak_logreg.pkl")

STREAK_HUE_LOW  = 18.0
STREAK_HUE_HIGH = 55.0
STREAK_SAT_MIN  = 70.0
STREAK_VAL_MIN  = 80.0


def _heuristic_risk(streak_bytes: bytes) -> float:
    from app.utils.preprocess import load_image_bytes, extract_hsv_stats
    rgb   = load_image_bytes(streak_bytes)
    stats = extract_hsv_stats(rgb)

    hue = stats["hue_mean"]
    sat = stats["sat_mean"]
    val = stats["val_mean"]

    risk = 0.0
    if not (STREAK_HUE_LOW <= hue <= STREAK_HUE_HIGH):
        risk += 0.40
    if sat < STREAK_SAT_MIN:
        risk += 0.30
    if val < STREAK_VAL_MIN:
        risk += 0.20

    return min(risk, 1.0)


def analyze_streak(streak_bytes: bytes | None) -> dict:
    if streak_bytes is None:
        return {"risk_score": 0.5, "confidence": "low", "mode": "no_streak"}

    if STREAK_MODEL_PATH.exists():
        try:
            import colorsys
            from PIL import Image as PILImage
            with open(STREAK_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            img = PILImage.open(__import__("io").BytesIO(streak_bytes)).convert("RGB").resize((128, 128))
            arr = np.array(img) / 255.0
            hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in arr.reshape(-1, 3)])
            h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
            gold_hue = ((h >= 0.05) & (h <= 0.17))
            feats = np.array([
                h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std(),
                gold_hue.mean(), (s > 0.4).mean(), (v > 0.5).mean(),
                np.percentile(h, 25), np.percentile(h, 75),
                np.percentile(s, 25), np.percentile(s, 75),
            ], dtype=np.float32).reshape(1, -1)
            prob = model.predict_proba(feats)[0][1]
            confidence = "high" if abs(prob - 0.5) > 0.2 else "medium"
            return {"risk_score": round(float(prob), 4), "confidence": confidence, "mode": "logreg"}
        except Exception as e:
            logger.warning("Streak model failed (%s), using heuristic", e)

    try:
        risk = _heuristic_risk(streak_bytes)
    except Exception as e:
        logger.warning("Streak heuristic failed (%s)", e)
        risk = 0.5

    return {"risk_score": round(risk, 4), "confidence": "low", "mode": "heuristic"}
