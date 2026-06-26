"""
XGBoost meta-classifier fusing all modality scores + contradiction features.
Heuristic weighted-average fallback when pkl absent.
"""
import logging
import pickle
from pathlib import Path

import numpy as np

from app.models.contradiction import compute_contradiction_features

logger = logging.getLogger(__name__)

FUSION_MODEL_PATH = Path("models/fusion_xgb.pkl")

MODALITY_WEIGHTS = {
    "density":  0.35,
    "acoustic": 0.30,
    "image":    0.25,
    "streak":   0.10,
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
    Run meta-classifier fusion.
    Returns final risk score + SHAP values if available.
    """
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
