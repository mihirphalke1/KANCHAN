"""
Split the item photo from the calibration photo, and use the card ONLY for
calibration — real-world scale and colour quality — never as image content.

The officer's SOP is two photos: one clean shot of the ornament, and one shot
that includes the printed calibration card for scale/tamper-evidence. Every CV
stage (material scan, stone detection, gold-vs-gems, tarnish, 3D) must run on
the CARD-FREE photo so the card is never counted as part of the item. The card
photo contributes two things and nothing else:

  * scale    — px-per-mm from the card's known physical size (see fiducial.py)
  * colour   — a white-balance reference sampled from the card's neutral body,
               applied to the item photo so gold/gem colours read true.

All best-effort: any miss falls back to the existing single-photo behaviour.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _decode(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def classify_photos(image_bytes_list: list[bytes]) -> dict:
    """Decide which uploaded photo is the item (card-free) and which is the
    calibration photo (has a card).

    Returns indices into image_bytes_list:
      item_index  — first card-free photo (falls back to 0 if all have cards)
      card_index  — first photo containing a calibration card (or None)
      other_index — remaining photos (candidate stone close-ups)
    """
    from app.utils.fiducial import locate_card

    has_card = []
    for b in image_bytes_list:
        try:
            bgr = _decode(b)
            has_card.append(bgr is not None and locate_card(bgr) is not None)
        except Exception:
            has_card.append(False)

    card_index = next((i for i, c in enumerate(has_card) if c), None)
    item_index = next((i for i, c in enumerate(has_card) if not c), None)
    if item_index is None:
        # Every photo has a card — keep photo 0 as the item frame (the card
        # region is still excluded from material stats downstream).
        item_index = 0
    other_index = [i for i in range(len(image_bytes_list))
                   if i not in (item_index, card_index)]
    return {
        "item_index": item_index,
        "card_index": card_index,
        "other_index": other_index,
        "has_card": has_card,
    }


def white_balance_gains(card_bytes: bytes) -> Optional[np.ndarray]:
    """Per-channel BGR gains that neutralise the colour cast, sampled from the
    calibration card's white/grey body (its most neutral, brightest, least
    saturated pixels). None when no usable neutral reference is found."""
    from app.utils.fiducial import locate_card

    try:
        bgr = _decode(card_bytes)
        if bgr is None:
            return None
        card = locate_card(bgr)
        if not card or not card.get("bbox"):
            return None
        x0, y0, x1, y1 = [int(v) for v in card["bbox"]]
        H, W = bgr.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Neutral card body: bright and low-saturation (skip the black finder
        # squares, the checksum strip, and any shadowed edges).
        neutral = (hsv[..., 1] < 45) & (hsv[..., 2] > 130)
        if int(neutral.sum()) < 300:
            return None
        means = roi[neutral].reshape(-1, 3).mean(axis=0)  # BGR
        gray = float(means.mean())
        if gray < 1e-3:
            return None
        gains = gray / (means + 1e-6)
        # Clamp so a slightly-off reference can't wildly recolour the item.
        return np.clip(gains, 0.7, 1.5).astype(np.float32)
    except Exception as e:
        logger.info("white-balance sampling failed: %s", e)
        return None


def apply_white_balance(image_bytes: bytes, gains: np.ndarray) -> bytes:
    """Apply BGR gains to an item photo and re-encode. Returns the original
    bytes unchanged on any failure."""
    try:
        bgr = _decode(image_bytes)
        if bgr is None:
            return image_bytes
        out = bgr.astype(np.float32) * gains.reshape(1, 1, 3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", out)
        return buf.tobytes() if ok else image_bytes
    except Exception as e:
        logger.info("white-balance apply failed: %s", e)
        return image_bytes
