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


def _classify_gold_alloy_hue(hue: float, sat: float, val: float) -> tuple[str, float]:
    """Classify a dominant colour into a known gold-alloy band and a risk.

    Plain yellow-gold hue-gating alone wrongly flags legitimate alloys —
    rose/pink gold (copper-rich, redder hue) and white gold / rhodium-plated
    gold (near-neutral, so very low saturation). Each recognised alloy gets a
    low risk; only colours matching NO gold alloy (e.g. brassy green, blue
    casts) get a high risk. Returns (alloy_name, risk_0_1)."""
    # White gold / rhodium plating: near-neutral (very low saturation) and
    # bright — hue is meaningless at this saturation, so gate on sat/val first.
    if sat < 40 and val >= 120:
        return "white_gold", 0.20
    # Rose / pink gold: copper-rich, hue pulled toward red (below yellow band).
    if 4 <= hue <= 17 and sat >= 40:
        return "rose_gold", 0.25
    # Yellow gold: the classic band.
    if GOLD_HUE_LOW <= hue <= GOLD_HUE_HIGH and sat >= 40:
        return "yellow_gold", 0.15
    # Antique/deep gold can read slightly warm-desaturated — a mild band.
    if 8 <= hue <= 60 and 25 <= sat < 40 and val >= 90:
        return "antique_gold", 0.35
    return "unmatched", 0.60


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
        val   = stats.get("val_mean", 150.0)

        # Alloy-aware: rose/white/antique gold are legitimate and must not be
        # flagged just for sitting outside the yellow-gold hue band.
        alloy, alloy_risk = _classify_gold_alloy_hue(hue, sat, val)
        scores.append(min(alloy_risk, 1.0))

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
