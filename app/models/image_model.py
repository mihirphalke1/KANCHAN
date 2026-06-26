"""
Image modality — EfficientNet-B3 embeddings + linear probe.
Fallback: HSV color analysis (genuine gold: hue 20-55°, high saturation).
"""
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL_PATH = Path("models/image_model.pth")
PROBE_PATH       = Path("models/image_probe.pkl")

GOLD_HUE_LOW  = 18.0
GOLD_HUE_HIGH = 58.0
GOLD_SAT_MIN  = 60.0


def _load_efficientnet():
    import torch
    import timm
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    if EMBED_MODEL_PATH.exists():
        state = torch.load(EMBED_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def extract_image_embedding(image_bytes: bytes) -> np.ndarray:
    """Return 1536-dim EfficientNet-B3 embedding."""
    import torch
    import torchvision.transforms as T
    from app.utils.preprocess import load_image_bytes

    rgb = load_image_bytes(image_bytes, size=(300, 300))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = transform(tensor)

    model = _load_efficientnet()
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy()
    return emb.astype(np.float32)


def _heuristic_risk(image_bytes_list: list[bytes]) -> tuple[float, dict]:
    """
    Heuristic: analyse gold-hue coverage across all submitted images.
    Genuine gold should occupy hue 18-58° with high saturation.
    """
    from app.utils.preprocess import load_image_bytes, extract_hsv_stats

    scores = []
    for raw in image_bytes_list:
        rgb   = load_image_bytes(raw)
        stats = extract_hsv_stats(rgb)
        hue   = stats["hue_mean"]
        sat   = stats["sat_mean"]

        in_gold_hue = GOLD_HUE_LOW <= hue <= GOLD_HUE_HIGH
        sufficient_sat = sat >= GOLD_SAT_MIN

        score = 0.0
        if not in_gold_hue:
            score += 0.40
        if not sufficient_sat:
            score += 0.30
        scores.append(min(score, 1.0))

    risk  = float(np.mean(scores)) if scores else 0.5
    stats_out = {"images_analysed": len(scores), "mean_heuristic_risk": round(risk, 4)}
    return risk, stats_out


def analyze_image(image_bytes_list: list[bytes]) -> dict:
    """
    Analyse one or more images of the gold item.
    Returns modality_score dict.
    """
    if not image_bytes_list:
        return {"risk_score": 0.5, "confidence": "low", "mode": "no_images"}

    mode = "heuristic"

    if PROBE_PATH.exists():
        try:
            with open(PROBE_PATH, "rb") as f:
                probe = pickle.load(f)
            embeddings = [extract_image_embedding(b) for b in image_bytes_list]
            mean_emb   = np.mean(embeddings, axis=0).reshape(1, -1)
            prob       = probe.predict_proba(mean_emb)[0][1]
            confidence = "high" if abs(prob - 0.5) > 0.2 else "medium"
            return {
                "risk_score": round(float(prob), 4),
                "confidence": confidence,
                "mode":       "efficientnet",
            }
        except Exception as e:
            logger.warning("Image model failed (%s), using heuristic", e)

    try:
        risk, extra = _heuristic_risk(image_bytes_list)
    except Exception as e:
        logger.warning("Image heuristic failed (%s)", e)
        risk, extra = 0.5, {}

    return {
        "risk_score": round(risk, 4),
        "confidence": "low",
        "mode":       mode,
        **extra,
    }
