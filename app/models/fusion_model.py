"""
Evidence fusion across modalities.

DEFAULT MODE — "logodds": transparent tempered-naive-Bayes combination.
Each performed test's risk p converts to log-odds L = ln(p/(1-p)); the
combined log-odds is the weighted sum, mapped back to a probability.
Every verdict is recomputable by hand. Properties:

  * A missing test (p = 0.5) has log-odds exactly 0 — provably neutral.
  * RELIABILITY weights temper each test by its evidence quality.
  * GENUINE_FLOOR bounds each test's evidence TOWARD genuine at the
    probability its known blind spot implies: a passing density test can
    never certify genuineness below p=0.25, because tungsten-core fakes
    pass it; plating passes the visual tests. Evidence toward fake is
    uncapped (capped only at 0.98 to keep log-odds finite).

ALTERNATIVE MODE — "xgboost": the trained meta-classifier
(models/fusion_xgb.pkl), kept as a comparison baseline.
Select with FUSION_MODE env var.
"""
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np

from app.models.contradiction import compute_contradiction_features

logger = logging.getLogger(__name__)

FUSION_MODEL_PATH = Path("models/fusion_xgb.pkl")
FUSION_MODE = os.getenv("FUSION_MODE", "logodds")

# Evidence-quality weights (rationale documented per modality):
#   density  1.0  — calibrated physics with propagated uncertainty
#   acoustic 0.6  — real tap recordings, but small training sample
#   image    0.5  — DSIP structural features, hand-calibrated
#   streak   0.2  — proxy-trained heuristic pending DS-7
RELIABILITY = {
    "image":    float(os.getenv("W_VISUAL",   "0.5")),
    "density":  float(os.getenv("W_DENSITY",  "1.0")),
    "acoustic": float(os.getenv("W_ACOUSTIC", "0.6")),
    "streak":   float(os.getenv("W_STREAK",   "0.2")),
}

# Blind-spot floors: minimum risk a PASSING test can assert, because known
# fake classes pass it. Derivation: tungsten passes density; gold plating
# passes visual and streak; solid-composite ring damping is usually audible.
GENUINE_FLOOR = {
    "image":    0.30,
    "density":  0.25,
    "acoustic": 0.15,
    "streak":   0.35,
}

FAKE_CAP = 0.98   # keeps log-odds finite; evidence toward fake otherwise uncapped

MODALITY_WEIGHTS = {
    "density":  0.35,
    "acoustic": 0.30,
    "image":    0.25,
    "streak":   0.10,
}


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def logodds_fusion(
    image_risk: float, density_risk: float, acoustic_risk: float, streak_risk: float,
) -> dict:
    """Transparent evidence combination. Returns per-test contributions so
    the UI can show exactly what each test added — no attribution model
    needed, the contributions ARE the computation."""
    risks = {"image": image_risk, "density": density_risk,
             "acoustic": acoustic_risk, "streak": streak_risk}
    contributions = {}
    total = 0.0   # neutral prior (bank may set from historical base rate)
    for name, p in risks.items():
        bounded = min(max(p, GENUINE_FLOOR[name]), FAKE_CAP)
        c = RELIABILITY[name] * _logit(bounded)
        contributions[f"{name}_risk"] = round(c, 4)
        total += c
    prob = 1.0 / (1.0 + math.exp(-total))
    return {
        "risk_score":  round(prob, 4),
        "mode":        "logodds",
        "log_odds":    round(total, 4),
        "shap_values": contributions,   # same shape the UI already renders
    }


def build_feature_vector(
    image_risk:    float,
    density_risk:  float,
    acoustic_risk: float,
    streak_risk:   float,
) -> np.ndarray:
    """Return 10-dim feature vector: [4 modality scores | 6 contradiction features]."""
    scores = {
        "image":    image_risk,
        "density":  density_risk,
        "acoustic": acoustic_risk,
        "streak":   streak_risk,
    }
    modality_feats    = np.array([image_risk, density_risk, acoustic_risk, streak_risk], dtype=np.float32)
    contradiction_feats = compute_contradiction_features(scores)
    return np.concatenate([modality_feats, contradiction_feats])


def _weighted_heuristic(image_risk, density_risk, acoustic_risk, streak_risk) -> float:
    total_w = sum(MODALITY_WEIGHTS.values())
    risk = (
        MODALITY_WEIGHTS["density"]  * density_risk
        + MODALITY_WEIGHTS["acoustic"] * acoustic_risk
        + MODALITY_WEIGHTS["image"]    * image_risk
        + MODALITY_WEIGHTS["streak"]   * streak_risk
    ) / total_w
    return float(risk)


def analyze_fusion(
    image_risk:    float,
    density_risk:  float,
    acoustic_risk: float,
    streak_risk:   float,
) -> dict:
    """
    Run evidence fusion. Default: transparent log-odds combination.
    Set FUSION_MODE=xgboost for the trained meta-classifier baseline.
    """
    if FUSION_MODE == "logodds":
        return logodds_fusion(image_risk, density_risk, acoustic_risk, streak_risk)

    mode = "heuristic"

    if FUSION_MODEL_PATH.exists():
        try:
            import shap
            with open(FUSION_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            feats = build_feature_vector(image_risk, density_risk, acoustic_risk, streak_risk).reshape(1, -1)
            prob  = model.predict_proba(feats)[0][1]

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feats)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            FEATURE_NAMES = [
                "image_risk", "density_risk", "acoustic_risk", "streak_risk",
                "contra_density_acoustic", "contra_density_image",
                "contra_image_streak", "contra_acoustic_image",
                "contra_density_streak", "contra_acoustic_streak",
            ]
            shap_dict = {
                name: round(float(val), 4)
                for name, val in zip(FEATURE_NAMES, shap_values[0])
            }
            return {
                "risk_score":  round(float(prob), 4),
                "mode":        "xgboost",
                "shap_values": shap_dict,
            }
        except Exception as e:
            logger.warning("XGBoost fusion failed (%s), using heuristic", e)

    risk = _weighted_heuristic(image_risk, density_risk, acoustic_risk, streak_risk)
    return {
        "risk_score":  round(risk, 4),
        "mode":        mode,
        "shap_values": None,
    }
