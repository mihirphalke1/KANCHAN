"""
Daily fiducial calibration card — tamper-evidence + real-world scale
reference, printed fresh at branch open each morning.

Design: a square card with four solid black "finder" squares at the
corners (the same idea QR codes use for orientation-independent detection)
and an 8-bit checksum strip along the bottom. The checksum byte is derived
from SHA-256(date | branch_id | server-side salt) — deterministic for a
given branch and day, unpredictable without the salt, and different every
day. Two things follow from a photo containing a valid, TODAY-dated card:

  1. Tamper-evidence: a photo from yesterday (or last month, re-used from
     another case) cannot contain today's card, because the evaluator
     printed it only this morning.
  2. Scale calibration: the card's real-world size is known exactly, so its
     measured pixel size in the photo gives a px-per-mm conversion — used to
     turn detected stone pixel-areas into real millimetres (see
     app/utils/gem_grid.py) without any extra hardware.

Detection is classical CV (contour analysis for the finder squares, a
perspective warp, then sampling the checksum strip) — no ML, consistent
with the rest of the visual pipeline. It is deliberately best-effort: on any
failure it returns detected=False with a reason, and callers treat that as
"no scale reference available", never as a hard block.
"""
import hashlib
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Real-world printed card size. Nominal 60 mm, but any print-scaling error
# (a printer "fit to page", the wrong paper size) silently corrupts every
# stone size and gem weight derived from px_per_mm. Measure a real printed card
# edge-to-edge with a ruler once and set KANCHAN_CARD_SIDE_MM to the true value
# (P1-7). Defaults to the nominal size when unset.
CARD_SIDE_MM   = float(os.getenv("KANCHAN_CARD_SIDE_MM", "60.0"))
FINDER_FRAC    = 0.12     # each corner finder square, as a fraction of card side
MARGIN_FRAC    = 0.04     # inset of each finder square from the card edge
CHECKSUM_BITS  = 8

# Server-side salt (never printed). The whole tamper-evidence guarantee rests
# on this being unpredictable to anyone who could re-use an old photo, so it
# must NOT live in source in production — read it from the environment and
# rotate it per deployment. The literal below is a dev/demo fallback only; if
# it is ever in use we log a loud warning so it can't silently ship.
_DEV_FALLBACK_SALT = "KANCHAN_FIDUCIAL_SALT_V1"


def _load_salt() -> str:
    salt = os.getenv("KANCHAN_FIDUCIAL_SALT", "").strip()
    if salt:
        return salt
    logger.warning(
        "KANCHAN_FIDUCIAL_SALT is not set — using the built-in dev fallback salt. "
        "Anyone with repo access can forge a branch's calibration card. "
        "Set KANCHAN_FIDUCIAL_SALT to a secret value (and rotate it) before any real deployment."
    )
    return _DEV_FALLBACK_SALT


FIDUCIAL_SALT = _load_salt()


def _checksum_byte(day: date, branch_id: str) -> int:
    digest = hashlib.sha256(f"{day.isoformat()}|{branch_id}|{FIDUCIAL_SALT}".encode()).digest()
    return digest[0]


def generate_marker_png(branch_id: str, on_date: Optional[date] = None, px_per_mm: int = 8) -> bytes:
    """Render the printable calibration card as PNG bytes."""
    d = on_date or date.today()
    side_px = int(CARD_SIDE_MM * px_per_mm)
    finder_px = int(side_px * FINDER_FRAC)
    margin = int(side_px * MARGIN_FRAC)

    img = Image.new("RGB", (side_px, side_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Four corner finder squares
    corners = [
        (margin, margin),
        (side_px - margin - finder_px, margin),
        (margin, side_px - margin - finder_px),
        (side_px - margin - finder_px, side_px - margin - finder_px),
    ]
    for (x, y) in corners:
        draw.rectangle([x, y, x + finder_px, y + finder_px], fill=(0, 0, 0))
        inset = finder_px // 3
        draw.rectangle([x + inset, y + inset, x + finder_px - inset, y + finder_px - inset], fill=(255, 255, 255))

    # Checksum strip: 8 cells between the bottom two finder squares
    byte = _checksum_byte(d, branch_id)
    strip_y0 = side_px - margin - finder_px
    strip_y1 = side_px - margin
    strip_x0 = margin + finder_px + margin
    strip_x1 = side_px - margin - finder_px - margin
    cell_w = (strip_x1 - strip_x0) / CHECKSUM_BITS
    for i in range(CHECKSUM_BITS):
        bit = (byte >> (CHECKSUM_BITS - 1 - i)) & 1
        x0 = strip_x0 + i * cell_w
        x1 = x0 + cell_w
        draw.rectangle([x0, strip_y0, x1, strip_y1], fill=(0, 0, 0) if bit else (255, 255, 255),
                        outline=(160, 160, 160))

    # Human-readable label (also lets an officer eyeball "is this today's card")
    try:
        font = ImageFont.load_default(size=max(12, side_px // 34))
    except TypeError:
        font = ImageFont.load_default()
    label = f"KANCHAN CALIBRATION CARD  ·  {branch_id}  ·  {d.isoformat()}"
    draw.text((margin, side_px // 2 - 8), label, fill=(20, 24, 36), font=font)
    draw.text((margin, side_px // 2 + 14), "Print fresh each morning · place flat beside the item",
               fill=(90, 95, 110), font=font)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _find_finder_squares(bgr: np.ndarray) -> list[tuple[float, float, float]]:
    """Locate solid dark, roughly-square blobs with a light interior notch —
    candidate finder squares. Returns (cx, cy, side_px) for each."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(grey, 90, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # RETR_EXTERNAL only, not RETR_LIST — each finder square is a black
    # frame with a white notch cut into it, which is a single ring-shaped
    # blob. RETR_LIST would also return the notch's inner boundary as a
    # second, smaller "square" candidate at the same centre, contaminating
    # the size-consistency match below.
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if not (0.75 <= aspect <= 1.33):
            continue
        rect_fill = area / max(w * h, 1)
        if rect_fill < 0.55:      # a solid square, not a thin outline/text glyph
            continue
        candidates.append((x + w / 2.0, y + h / 2.0, (w + h) / 2.0))
    return candidates


def locate_card(bgr: np.ndarray) -> Optional[dict]:
    """Shared geometry step: find the 4 finder squares, verify they're one
    consistent card, and return its scale + full bounding box. Returns None
    if no plausible card is present — every caller (detect_marker for
    tamper-evidence, xray.py to exclude the card from material analysis)
    treats that as "no card in frame", never an error."""
    try:
        candidates = _find_finder_squares(bgr)
    except Exception:
        return None
    if len(candidates) < 4:
        return None

    # Pick the 4 candidates whose mutual sizes are most consistent (finder
    # squares are all the same physical size; noise blobs vary).
    candidates.sort(key=lambda c: c[2])
    best_set, best_spread = None, float("inf")
    for i in range(len(candidates) - 3):
        window = candidates[i:i + 4]
        sizes = [c[2] for c in window]
        spread = (max(sizes) - min(sizes)) / max(np.mean(sizes), 1e-6)
        if spread < best_spread:
            best_spread, best_set = spread, window
    if best_set is None or best_spread > 0.35:
        return None

    xs = [c[0] for c in best_set]
    ys = [c[1] for c in best_set]
    card_w_px = max(xs) - min(xs)
    card_h_px = max(ys) - min(ys)
    centre_span_px = (card_w_px + card_h_px) / 2.0
    finder_side_px = float(np.mean([c[2] for c in best_set]))
    # Each finder centre sits inset from its card edge by (margin + finder/2)
    # — both defined as fractions of the (unknown) card side — so
    # centre_span = card_side * (1 - 2*MARGIN_FRAC - FINDER_FRAC).
    card_side_px = centre_span_px / max(1e-6, (1.0 - 2 * MARGIN_FRAC - FINDER_FRAC))
    if card_side_px <= 0:
        return None

    # Full card bounding box: extend the finder-centre bbox outward by the
    # same inset each centre sits at from its edge, plus 20% slack for
    # detection imprecision — better to exclude a little extra background
    # around the card than to leave a sliver of it inside the item mask.
    pad = card_side_px * (MARGIN_FRAC + FINDER_FRAC / 2.0) * 1.2
    h, w = bgr.shape[:2]
    bbox = [
        max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
        min(w, int(max(xs) + pad)), min(h, int(max(ys) + pad)),
    ]

    return {
        "px_per_mm":     card_side_px / CARD_SIDE_MM,
        "card_side_px":  card_side_px,
        "finder_side_px": finder_side_px,
        "best_set":      best_set,
        "bbox":          bbox,
    }


def card_bbox(bgr: np.ndarray) -> Optional[list[int]]:
    """Convenience wrapper for callers (xray.py) that only need the region
    to exclude from material/stone analysis, not the full geometry."""
    card = locate_card(bgr)
    return card["bbox"] if card else None


def sample_white_balance(bgr: np.ndarray, card: Optional[dict] = None) -> dict:
    """White-balance a photo using the calibration card's white field as a
    known-neutral reference (P1-5).

    The card prints on a white background; its bright, low-saturation pixels are
    physically neutral, so any colour cast in the photo (warm incandescent,
    cool shade) shows up as a per-channel imbalance there. We estimate per-
    channel gains that restore those pixels to grey and apply them to the whole
    frame — correcting the cast for downstream metal-colour/alloy analysis.

    Best-effort: if no card is present, returns applied=False and the untouched
    image, so callers can carry on with the self-referential metal-chromaticity
    fit that already handles the no-card case. Returns
    {applied, gains_bgr, corrected_image, reason}."""
    if card is None:
        card = locate_card(bgr)
    if not card or "bbox" not in card:
        return {"applied": False, "gains_bgr": [1.0, 1.0, 1.0],
                "corrected_image": bgr,
                "reason": "No calibration card in frame — cannot white-balance from a neutral reference"}

    x0, y0, x1, y1 = card["bbox"]
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return {"applied": False, "gains_bgr": [1.0, 1.0, 1.0],
                "corrected_image": bgr, "reason": "Card region empty after clipping"}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    # Bright, low-saturation pixels = the card's white field (not the black
    # finder squares, not the printed label).
    white = (v >= np.percentile(v, 70)) & (s <= 60)
    if int(white.sum()) < 20:
        white = v >= np.percentile(v, 85)
    px = roi.reshape(-1, 3)[white.reshape(-1)].astype(np.float32)
    if px.shape[0] < 10:
        return {"applied": False, "gains_bgr": [1.0, 1.0, 1.0],
                "corrected_image": bgr, "reason": "Too few neutral pixels on the card to estimate white balance"}

    means = px.mean(axis=0)                 # B, G, R
    gray = float(means.mean())
    gains = gray / np.maximum(means, 1e-3)
    corrected = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return {"applied": True,
            "gains_bgr": [round(float(g), 4) for g in gains],
            "corrected_image": corrected,
            "reason": "White-balanced from the calibration card's white field"}


_CANON_PX = 480          # canonical warped card size for perspective-safe reads
_FINDER_INSET_FRAC = MARGIN_FRAC + FINDER_FRAC / 2.0   # finder-centre inset from edge


def _read_checksum_perspective(bgr: np.ndarray, best_set: list) -> Optional[int]:
    """Read the 8-bit checksum after a homography warp that removes perspective
    (P1-6). The four finder-square centres are mapped to their known canonical
    positions, so an off-angle photo is rectified before the strip is sampled —
    instead of the head-on sampler silently reading the wrong cells. Returns the
    byte, or None if the warp/read isn't reliable (caller falls back)."""
    try:
        pts = [(c[0], c[1]) for c in best_set]
        pts_sorted = sorted(pts, key=lambda p: p[1])          # by image-y (upright assumption)
        top = sorted(pts_sorted[:2], key=lambda p: p[0])       # TL, TR
        bot = sorted(pts_sorted[2:], key=lambda p: p[0])       # BL, BR
        src = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)  # TL,TR,BR,BL

        lo = _FINDER_INSET_FRAC * _CANON_PX
        hi = (1.0 - _FINDER_INSET_FRAC) * _CANON_PX
        dst = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(bgr, H, (_CANON_PX, _CANON_PX))
        grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Canonical strip geometry (matches generate_marker_png): the 8 cells run
        # x∈[0.20,0.80]·side between the two bottom finder squares, y∈[0.84,0.96]·side.
        strip_x0 = (MARGIN_FRAC + FINDER_FRAC + MARGIN_FRAC) * _CANON_PX
        strip_x1 = (1.0 - MARGIN_FRAC - FINDER_FRAC - MARGIN_FRAC) * _CANON_PX
        strip_yc = (1.0 - MARGIN_FRAC - FINDER_FRAC / 2.0) * _CANON_PX
        band = max(2, int(0.02 * _CANON_PX))
        y0b, y1b = int(strip_yc - band), int(strip_yc + band)
        cell_w = (strip_x1 - strip_x0) / CHECKSUM_BITS
        byte = 0
        for i in range(CHECKSUM_BITS):
            centre = strip_x0 + (i + 0.5) * cell_w
            half = cell_w * 0.30
            cell = grey[y0b:y1b, int(centre - half):max(int(centre + half), int(centre - half) + 1)]
            if cell.size == 0:
                return None
            byte = (byte << 1) | (1 if float(cell.mean()) < 128 else 0)
        return byte
    except Exception as e:
        logger.warning("Perspective checksum read failed: %s", e)
        return None


def detect_marker(bgr: np.ndarray, branch_id: str, grace_days: int = 1) -> dict:
    """Best-effort marker detection. Returns:
      detected          - a plausible 4-finder-square card was found
      px_per_mm         - scale reference, only when detected
      checksum_valid    - matches today's (or grace_days-ago) expected byte
      reason            - human-readable outcome, always present
    """
    card = locate_card(bgr)
    if card is None:
        return {"detected": False, "px_per_mm": None, "checksum_valid": None,
                "reason": "Card not visible in frame, or corner markers too inconsistent to be one card"}

    best_set       = card["best_set"]
    card_side_px   = card["card_side_px"]
    finder_side_px = card["finder_side_px"]
    px_per_mm      = card["px_per_mm"]

    # Checksum read: sample the 8-cell strip between the two BOTTOM finder
    # squares (by y, since image y grows downward) at their shared mid-line.
    # This assumes a roughly head-on shot (no perspective correction) — good
    # enough for an MVP tamper-evidence check; a skewed photo just yields
    # checksum_valid=None rather than a false result. Geometry (px_per_mm)
    # is trusted independently of the checksum outcome either way.
    checksum_valid, read_byte = None, None
    read_method = None
    # Primary: perspective-corrected read (handles off-angle photos). Falls
    # back to the head-on mid-line read below if the warp can't produce a byte.
    today = date.today()
    expected = {_checksum_byte(today - timedelta(days=g), branch_id) for g in range(grace_days + 1)}
    warped_byte = _read_checksum_perspective(bgr, best_set)
    if warped_byte is not None:
        read_byte = warped_byte
        checksum_valid = warped_byte in expected
        read_method = "perspective_warp"
    try:
        if checksum_valid:
            raise StopIteration        # perspective read already verified — skip fallback
        bottom_two = sorted(best_set, key=lambda c: -c[1])[:2]
        bottom_two.sort(key=lambda c: c[0])
        (lx, ly, lsz), (rx, ry, rsz) = bottom_two
        strip_y = int((ly + ry) / 2.0)
        x0, x1 = lx + lsz / 2.0, rx - rsz / 2.0
        if x1 > x0 and 0 <= strip_y < bgr.shape[0]:
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            band = max(2, int(finder_side_px * 0.15))
            y0b, y1b = max(0, strip_y - band), min(grey.shape[0], strip_y + band)
            cell_w = (x1 - x0) / CHECKSUM_BITS
            bits = []
            for i in range(CHECKSUM_BITS):
                # Sample only the middle third of each cell — a few percent
                # of geometric mis-estimate (finder squares aren't detected
                # with sub-pixel precision) otherwise bleeds one cell's edge
                # into its neighbour and flips a bit.
                centre = x0 + (i + 0.5) * cell_w
                half = cell_w * 0.30
                cx0, cx1 = int(centre - half), int(centre + half)
                cell = grey[y0b:y1b, cx0:max(cx1, cx0 + 1)]
                if cell.size == 0:
                    bits = None
                    break
                bits.append(1 if float(cell.mean()) < 128 else 0)
            if bits is not None:
                fallback_byte = 0
                for b in bits:
                    fallback_byte = (fallback_byte << 1) | b
                # Only let the head-on read override if the perspective read
                # didn't already verify; prefer whichever produces a valid byte.
                if checksum_valid is not True:
                    read_byte = fallback_byte
                    checksum_valid = fallback_byte in expected
                    read_method = "head_on"
    except StopIteration:
        pass
    except Exception as e:
        logger.warning("Fiducial checksum read failed: %s", e)

    return {
        "detected":       True,
        "px_per_mm":      round(px_per_mm, 3),
        "card_side_px":   round(card_side_px, 1),
        "checksum_valid": checksum_valid,
        "read_byte":      read_byte,
        "checksum_read_method": read_method,
        "reason": (
            "Calibration card detected and today's checksum verified" if checksum_valid
            else "Calibration card detected but checksum did not match today's (or yesterday's) expected value — possible re-used photo"
            if checksum_valid is False
            else "Calibration card detected — scale reference available (checksum unreadable, likely off-angle)"
        ),
    }
