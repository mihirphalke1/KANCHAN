"""
DSIP pseudo X-ray pipeline — material segmentation of jewellery photographs.

Stages 1-7 are classical image processing (no ML), stage 8 is a
classical/ML hybrid:
  1. Illumination normalisation (white-balance + local contrast correction)
  2. BT.709 luminance greyscale
  3. Intensity inversion (radiographic negative)
  4. Multi-level thresholding into 4 material classes (Otsu-anchored)
  5. Sobel gradient magnitude (material boundaries)
  6. False-colour material map (threshold classes + edge overlay)
  7. Jet heatmap of raw luminance
  8. Stone detection — the human-eye rule: measure the item's OWN gold
     colour (median CIE Lab over the item pixels), then flag whatever
     differs clearly from it (lightness-deemphasised Lab distance, so
     shadows/highlights on the SAME gold don't trigger it but a genuinely
     different-coloured stone does, however muted or dimly lit). Those
     colour-threshold candidates seed a pretrained MobileSAM segmentation
     pass (app/models/ml_stone_detection.py) that supplies the actual region
     boundaries (fixes touching-stone undercounting and facet-reflection
     overcounting a colour threshold alone produces), falling back to the
     classical candidate pool if the model is unavailable. Each region is
     then scored on four independent confidence signals (boundary contrast,
     local colour contrast, shape regularity, size consistency) for
     detection confidence, and separately identified by nearest-match
     against a reference stone-colour table (ruby, emerald, sapphire,
     diamond, ...) — two different questions, two different, both
     inspectable, non-black-box answers.
"""
import base64
import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models import ml_stone_detection
from app.utils.fiducial import card_bbox

logger = logging.getLogger(__name__)

MAX_SIDE = 640
EDGE_THRESHOLD = 70          # on the 0-255 normalised Sobel magnitude
MIN_GEM_AREA_FRAC = 0.002    # ≥0.2% of item area — beadwork stones are small
# A single large centre/statement stone (common in cocktail rings and
# pendants) can legitimately dominate the item's visible area — the old
# 0.35 ceiling was excluding exactly this case, not just glare sheets.
MAX_STONE_AREA_FRAC = 0.60
# Colourless (desaturated) candidates are the hardest class — a specular
# highlight, a polished white-gold/rhodium bead, and a real diamond/pearl
# all look the same in hue. A single colourless region this large is far
# more likely a glare sheet or a metal segment than one stone — UNLESS it
# also contrasts strongly against the surrounding metal (MAX_COLOURLESS_
# AREA_FRAC_HIGH_CONTRAST), which a glare bleed on a metal facet does not:
# glare fades gradually into its own metal, a real large stone doesn't.
MAX_COLOURLESS_AREA_FRAC = 0.08
MAX_COLOURLESS_AREA_FRAC_HIGH_CONTRAST = 0.45
HIGH_CONTRAST_FLOOR = 0.5

# Background subtraction: the backdrop colour is estimated from the frame
# border (SOP: photograph on a plain fixed-colour backdrop — the light box
# enforces this, not evaluator memory). Weighted HSV distance above this
# threshold = item pixel. Calibration parameter, env-overridable.
BACKDROP_DIST_THRESHOLD = float(os.getenv("BACKDROP_DIST_THRESHOLD", "60"))
# A border-touching component this small relative to the frame is almost
# always a finger/prop intrusion; larger ones are the item itself running
# off the edge of a tightly-cropped phone photo, and are kept.
BORDER_INTRUSION_MAX_FRAC = 0.08

# ── Stone detection: the human-eye rule ─────────────────────────────────
# "A gem looks like a gem because its colour is clearly different from the
# gold around it." No hard-coded hue bands: measure the item's OWN gold
# colour from the photo (metal_lab, below), then flag any region whose
# colour clearly differs from THAT — in CIE Lab space, which separates true
# colour (a*/b*) from lighting (L*). A shadow or a specular highlight on the
# SAME gold moves L* a lot but barely touches a*/b*; a genuinely different-
# coloured stone (including a muted, dimly-lit one) moves a*/b* regardless
# of how it's lit. De-emphasising L* is what lets a pale stone still
# register while a lighting gradient across curved polished metal — which
# fooled a plain HSV-distance threshold — does not.
LAB_LIGHTNESS_WEIGHT = float(os.getenv("STONE_LAB_L_WEIGHT", "0.25"))
# Delta-E (weighted) above this = candidate. Calibrated so ordinary shading/
# specular variation on a SINGLE gold surface (dominated by L*, discounted
# above) stays under it, while a genuinely different-coloured stone —
# including muted/opaque cabochons — clears it comfortably.
STONE_DELTA_E_MIN = float(os.getenv("STONE_DELTA_E_MIN", "28"))
# A real stone is uniformly coloured; a residual lighting/compression
# artefact that still clears the distance threshold isn't — connected
# components whose own delta-E values are this spread out (coefficient of
# variation) are rejected as noise, not a stone.
STONE_CV_MAX = float(os.getenv("STONE_CV_MAX", "0.10"))
# Only regions at least this fraction of the item's area are checked for
# colour consistency at all — see the comment in _candidate_mask.
STONE_CV_GATE_FRAC = float(os.getenv("STONE_CV_GATE_FRAC", "0.08"))

# A cut stone has an "identity signature" a specular highlight never has:
# either dense internal facet sparkle (micro-edge texture) or a genuinely
# round/oval smooth outline (pearl/cabochon). A curved polished metal
# surface produces a smooth, elongated, irregularly-shaped bright patch that
# matches neither — this is what actually separates "real stone" from
# "glare on the band" once colour alone can't (a diamond is colourless too).
CLS_LAPLACE_EDGE     = 40     # |Laplacian| above this = micro-edge (facet) pixel
CLS_FACET_FRAC_REF   = 0.15   # micro-edge fraction at/above this = clearly faceted
CIRCULARITY_FLOOR    = 0.40   # below this, roundness contributes nothing
CIRCULARITY_CONFIDENT = 0.72  # at/above this, roundness is fully credited

INCLUSION_OVERLAP_EXPLAINED = 0.25   # ≥25% overlap with a kept stone = explained

# Per-candidate confidence = weighted blend of five independent signals.
# Env-overridable, same pattern as BACKDROP_DIST_THRESHOLD, so this can be
# recalibrated against a labelled photo panel later. Weighted so that no
# single signal (e.g. a round, smooth highlight scoring well on "identity")
# can push a candidate to CONFIDENT on its own — CONFIDENT_THRESHOLD sits
# well above any one signal's own weight.
W_EDGE     = float(os.getenv("STONE_W_EDGE", "0.28"))       # boundary contrast
W_CONTRAST = float(os.getenv("STONE_W_CONTRAST", "0.22"))   # local colour contrast
W_SHAPE    = float(os.getenv("STONE_W_SHAPE", "0.15"))      # convexity / aspect
W_IDENTITY = float(os.getenv("STONE_W_IDENTITY", "0.20"))   # facet sparkle or roundness
W_SIZE     = float(os.getenv("STONE_W_SIZE", "0.15"))       # size consistency
# Deliberately conservative: a false "confirmed" silently asserts something
# wrong, while a false "uncertain" still routes to officer review — so the
# bar to CONFIDENT is set high, and ties go to "uncertain", not the other
# way round.
CONFIDENT_THRESHOLD = float(os.getenv("STONE_CONFIDENT_THRESHOLD", "0.72"))
UNCERTAIN_THRESHOLD = float(os.getenv("STONE_UNCERTAIN_THRESHOLD", "0.40"))

ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ELLIPSE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
ELLIPSE_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Material class colours (RGB) — kept in sync with XRayView.jsx legend
CLASS_COLOURS_RGB = {
    0: (220, 38, 38),    # darkest  → gemstone body (red)
    1: (15, 118, 110),   # low-mid  → solder / joints / shadowed metal (teal)
    2: (30, 58, 95),     # mid-high → metal body (navy)
    3: (217, 119, 6),    # brightest→ specular facets (gold)
}
CLASS_NAMES = {0: "gemstone", 1: "joint", 2: "metal", 3: "facet"}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _load_bgr(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _border_patch(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    b = max(4, int(0.05 * min(h, w)))
    return np.concatenate([
        img[:b].reshape(-1, img.shape[2]), img[-b:].reshape(-1, img.shape[2]),
        img[:, :b].reshape(-1, img.shape[2]), img[:, -b:].reshape(-1, img.shape[2]),
    ])


def _normalize_illumination(bgr: np.ndarray) -> np.ndarray:
    """
    Correct white-balance drift and even out local lighting BEFORE any
    thresholding — every downstream stage uses fixed absolute colour cut-offs,
    and those only stay valid across different shots if pixel values are
    normalised first. Two steps:
      1. Gray-world white balance, referenced to the frame border (the
         backdrop is meant to be a plain neutral colour; any tint sitting in
         the border sample is a colour cast from the light source, not the
         item, so we scale it back toward neutral grey).
      2. CLAHE (contrast-limited adaptive histogram equalisation) on the LAB
         L-channel, which evens out angle-dependent shadow/highlight
         gradients without blowing out global contrast the way a plain
         histogram-equalise would.
    """
    border = _border_patch(bgr).astype(np.float32)
    bg_mean = border.mean(axis=0) + 1e-6
    target = float(bg_mean.mean())
    gain = np.clip(target / bg_mean, 0.6, 1.8)
    balanced = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _bt709_grey(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    grey = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(grey, 0, 255).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (e.g. gems whose colour matches the backdrop)."""
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def _refine_with_grabcut(bgr: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """
    Snap a coarse colour-distance mask to real colour/texture edges with
    GrabCut, seeded from the seed mask itself: an eroded core is 'sure
    foreground', a dilated ring is 'sure background', everything else is
    'probable'. Falls back to the seed mask on any failure (small item,
    degenerate seed, etc.) so this can never make segmentation worse.
    """
    try:
        h, w = seed_mask.shape
        if (seed_mask > 0).sum() < 200:
            return seed_mask
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc_mask[seed_mask > 0] = cv2.GC_PR_FGD
        core = cv2.erode(seed_mask, ELLIPSE_9)
        gc_mask[core > 0] = cv2.GC_FGD
        outside = cv2.dilate(seed_mask, np.ones((15, 15), np.uint8))
        gc_mask[outside == 0] = cv2.GC_BGD
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return refined if refined.any() else seed_mask
    except Exception as e:
        logger.warning("GrabCut refinement failed, using seed mask: %s", e)
        return seed_mask


def _item_mask(bgr: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """
    Separate the item from a plain backdrop. The backdrop colour is the
    median HSV of the frame border; item pixels are those whose weighted
    HSV distance from it exceeds BACKDROP_DIST_THRESHOLD. The coarse mask is
    then snapped to real edges with GrabCut. Falls back to the full frame
    when segmentation is implausible (item <3% or >97% of the frame).

    Returns (item mask, background_removed, stone_eligible mask). The third
    mask excludes large backdrop-coloured interior regions that only became
    "item" via hole-filling/GrabCut recovery, not because they actually
    differ from the backdrop — e.g. a ring's own opening shows the same
    plain backdrop through it, and without this it reads as one giant
    bright, low-saturation "stone" the size of the hole. A SMALL such region
    stays eligible (a genuine gem that happens to match the backdrop colour
    is exactly the case hole-filling exists for).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, w = hsv.shape[:2]
    bg = np.median(_border_patch(hsv), axis=0)

    dh = np.abs(hsv[..., 0] - bg[0])
    dh = np.minimum(dh, 180 - dh)                    # hue is circular
    dist = dh * 2.0 + np.abs(hsv[..., 1] - bg[1]) * 1.0 + np.abs(hsv[..., 2] - bg[2]) * 0.5

    raw_mask = (dist > BACKDROP_DIST_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, ELLIPSE_7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ELLIPSE_7)

    # keep components ≥1% of frame AND ≥15% of the largest one — true
    # multi-piece items (earring pairs) are comparable in size, while stray
    # props (a flower, a coin) are not. A border-touching component is only
    # dropped when it's small (a finger/prop intrusion); a larger one is the
    # item itself photographed edge-to-edge, and is kept. Fill holes after.
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    if n > 1:
        largest = max(cc_stats[i, cv2.CC_STAT_AREA] for i in range(1, n))
        for i in range(1, n):
            area = cc_stats[i, cv2.CC_STAT_AREA]
            x, y = cc_stats[i, cv2.CC_STAT_LEFT], cc_stats[i, cv2.CC_STAT_TOP]
            bw, bh = cc_stats[i, cv2.CC_STAT_WIDTH], cc_stats[i, cv2.CC_STAT_HEIGHT]
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            is_intrusion = touches_border and area < BORDER_INTRUSION_MAX_FRAC * mask.size
            if area >= 0.01 * mask.size and area >= 0.15 * largest and not is_intrusion:
                keep[labels == i] = 255
        if not keep.any():
            big = max(range(1, n), key=lambda i: cc_stats[i, cv2.CC_STAT_AREA])
            keep[labels == big] = 255
    keep = _fill_holes(keep)
    keep = _refine_with_grabcut(bgr, keep)
    keep = _fill_holes(keep)

    frac = float((keep > 0).mean())
    if frac < 0.03 or frac > 0.97:
        full = np.full(mask.shape, 255, np.uint8)
        return full, False, full

    backdrop_like = ((keep > 0) & (raw_mask == 0)).astype(np.uint8) * 255
    n2, labels2, cc2, _ = cv2.connectedComponentsWithStats(backdrop_like, connectivity=8)
    large_hole = np.zeros_like(keep)
    for i in range(1, n2):
        if cc2[i, cv2.CC_STAT_AREA] > 0.03 * keep.size:
            large_hole[labels2 == i] = 255
    eligible = keep.copy()
    eligible[large_hole > 0] = 0
    return keep, True, eligible


# ── Reference stone colours ──────────────────────────────────────────────
# Approximate sRGB, illustrative — the same "documented fallback, calibrate
# against a real labelled panel later" status as BACKDROP_DIST_THRESHOLD and
# every other constant in this module. `bucket` is the legacy 4-way class
# (red/green/blue/other/colourless) that composition.py's stone-density
# lookup and the report/UI colour-coding already key off; `name` is the
# richer identification this rewrite adds on top.
STONE_COLOR_REFERENCES = [
    # name              bucket        sRGB (R, G, B)
    ("diamond",         "colourless", (245, 245, 250)),
    ("pearl",           "colourless", (245, 240, 225)),
    ("white_sapphire",  "colourless", (235, 238, 236)),
    ("ruby",            "red",        (155, 17,  30)),
    ("garnet",          "red",        (120, 20,  25)),
    ("coral",           "red",        (230, 110, 80)),
    ("emerald",         "green",      (0,   130, 90)),
    ("jade",            "green",      (80,  150, 110)),
    ("peridot",         "green",      (150, 190, 60)),
    ("sapphire",        "blue",       (15,  60,  120)),
    ("aquamarine",      "blue",       (140, 210, 210)),
    ("turquoise",       "blue",       (60,  180, 175)),
    ("amethyst",        "other",      (110, 60,  140)),
    ("topaz",           "other",      (230, 180, 60)),
    ("citrine",         "other",      (220, 150, 40)),
    ("onyx",            "other",      (25,  25,  28)),
]
# Nearest-match distance beyond which a stone is "unidentified" rather than
# forced into the closest reference regardless of fit.
STONE_MATCH_MAX_DIST = float(os.getenv("STONE_MATCH_MAX_DIST", "45"))


def _lab_of_rgb(rgb: tuple[int, int, int]) -> np.ndarray:
    bgr = np.array([[rgb[::-1]]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


_STONE_REF_LAB = [(name, bucket, _lab_of_rgb(rgb)) for name, bucket, rgb in STONE_COLOR_REFERENCES]


def _classify_stone_color(bgr_pixels: np.ndarray) -> tuple[str, str, float]:
    """
    bgr_pixels: Nx3 BGR uint8 array of a detected region's own pixels.
    Nearest-neighbour match (Euclidean distance in Lab) against the
    reference table, using the MEDIAN Lab colour of the region — robust to
    a minority of outlier pixels (e.g. a faceted colourless stone flashing
    spectral colour off a few facets under certain lighting doesn't drag a
    median the way it would a mean).
    Returns (stone_name, legacy_bucket, match_confidence 0-1).
    """
    lab_pixels = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    median_lab = np.median(lab_pixels, axis=0)
    best_name, best_bucket, best_dist = "unidentified", "other", float("inf")
    for name, bucket, ref_lab in _STONE_REF_LAB:
        d = float(np.linalg.norm(median_lab - ref_lab))
        if d < best_dist:
            best_dist, best_name, best_bucket = d, name, bucket
    confidence = max(0.0, 1.0 - best_dist / STONE_MATCH_MAX_DIST)
    if confidence <= 0.05:
        best_name = "unidentified"
    return best_name, best_bucket, round(confidence, 3)


GEM_OUTLINE_BGR = {
    "red":   (60, 60, 230),
    "green": (80, 200, 80),
    "blue":  (230, 140, 40),
    "other": (200, 200, 200),
}
CONFIRMED_COLOURLESS_BGR = (170, 140, 20)   # teal — confirmed, not a hue-coloured gem
UNCERTAIN_OUTLINE_BGR    = (0, 165, 255)    # amber — officer confirmation needed


def _candidate_mask(bgr: np.ndarray, item: np.ndarray, metal_lab: np.ndarray) -> np.ndarray:
    """
    The human-eye rule, in code: a stone is whatever differs clearly in
    COLOUR from the item's own measured gold — see the module-level comment
    by STONE_DELTA_E_MIN for why this uses lightness-deemphasised CIE Lab
    distance rather than a hard-coded hue/brightness band. One threshold,
    one candidate pool — no separate "coloured" vs "pale" buckets to fall
    between.
    """
    item_area = max(1, int((item > 0).sum()))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = lab[..., 0] - metal_lab[0]
    da = lab[..., 1] - metal_lab[1]
    db = lab[..., 2] - metal_lab[2]
    delta_e = np.sqrt(LAB_LIGHTNESS_WEIGHT * dL ** 2 + da ** 2 + db ** 2)
    cand = ((delta_e > STONE_DELTA_E_MIN) & (item > 0)).astype(np.uint8) * 255

    # A real stone is uniformly coloured; a residual lighting/compression
    # artefact that still clears the distance threshold isn't. Reject
    # connected components whose own delta-E values are too spread out
    # (coefficient of variation) to be one solid colour — but only above a
    # minimum size: on a small region the CV statistic is itself unreliable
    # (few pixels -> high sampling variance in std/mean), and a small region
    # isn't a room-filling lighting gradient to begin with. Verified: a real
    # small melee stone can read a HIGHER raw CV than a large illumination
    # artefact purely from this small-sample noise — gating by size, not
    # just tightening the threshold, is what avoids that false rejection.
    n_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n_m):
        area = stats_m[i, cv2.CC_STAT_AREA]
        if area < STONE_CV_GATE_FRAC * item_area:
            continue
        region = labels_m == i
        vals = delta_e[region]
        mean = float(vals.mean())
        cv_ratio = float(vals.std()) / mean if mean > 0 else 0.0
        if cv_ratio > STONE_CV_MAX:
            cand[region] = 0

    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ELLIPSE_3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, ELLIPSE_7)
    return cand


def _region_confidence(
    region: np.ndarray, hsv: np.ndarray, edge_mag: np.ndarray, lap_mag: np.ndarray,
    item: np.ndarray, metal_ref: np.ndarray, area: int, med_area: float,
) -> tuple[float, dict]:
    """
    Combine five independent, complementary signals into one 0-1 confidence
    — this replaces the old hard "which detector matched" bucket with an
    actual measure of how sure the pipeline is that a candidate region is a
    real stone rather than, say, a specular highlight on polished metal.
    """
    region_u8 = region.astype(np.uint8) * 255

    # 1. Boundary contrast: a set stone has a sharp bezel/prong edge; a
    #    highlight fades smoothly into the surrounding metal.
    ring = (cv2.dilate(region_u8, ELLIPSE_5) > 0) & ~region
    boundary_grad = float(edge_mag[ring].mean()) if ring.any() else 0.0
    interior_grad = float(edge_mag[region].mean()) if region.any() else 0.0
    edge_score = _clip01((boundary_grad - interior_grad) / 55.0)

    # 2. Local colour/brightness contrast against the immediate surrounding
    #    metal (sampled from a ring just outside the region, not the whole
    #    photo) — a real inset stone differs a lot from its own setting; a
    #    highlight barely differs from the metal it sits on.
    ring9 = (cv2.dilate(region_u8, ELLIPSE_9) > 0) & ~region & (item > 0)
    ring_hsv = [float(v) for v in (
        hsv[ring9].astype(np.float32).mean(axis=0) if ring9.sum() > 20 else metal_ref
    )]
    cand_hsv = [float(v) for v in hsv[region].astype(np.float32).mean(axis=0)]
    dh = min(abs(cand_hsv[0] - ring_hsv[0]), 180 - abs(cand_hsv[0] - ring_hsv[0]))
    contrast = dh * 2.0 + abs(cand_hsv[1] - ring_hsv[1]) + abs(cand_hsv[2] - ring_hsv[2]) * 0.5
    local_contrast_score = _clip01(contrast / 90.0)

    # 3. Shape regularity: convex-hull solidity + a bounded aspect ratio —
    #    gemstones (round/oval/faceted) are fairly convex; scratches,
    #    reflections and glare streaks are elongated or ragged.
    contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
        solidity = _clip01(cv2.contourArea(c) / hull_area)
        (rw, rh) = cv2.minAreaRect(c)[1]
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        shape_score = solidity * _clip01(1.4 - 0.15 * max(0.0, aspect - 2))
        perim = cv2.arcLength(c, True)
        circularity = 4 * math.pi * area / (perim ** 2) if perim > 0 else 0.0
    else:
        shape_score = 0.0

    # 4. Identity signature — the actual fingerprint that separates a real
    #    stone from a specular highlight once colour alone can't (a diamond
    #    is colourless too): either dense internal facet sparkle, OR a
    #    genuinely round/oval smooth outline (pearl/cabochon). A highlight
    #    on a curved polished band is smooth AND irregularly shaped, so it
    #    clears neither bar.
    facet_frac = float((lap_mag[region] > CLS_LAPLACE_EDGE).mean()) if region.any() else 0.0
    facet_signal = _clip01(facet_frac / CLS_FACET_FRAC_REF)
    round_signal = _clip01((circularity - CIRCULARITY_FLOOR) / (CIRCULARITY_CONFIDENT - CIRCULARITY_FLOOR))
    identity_score = max(facet_signal, round_signal)

    # 5. Size consistency vs. the other candidates in the same photo — melee
    #    stones cluster in size; one huge outlier (a broad reflective sheet)
    #    is penalised, on a log scale so a genuine solitaire isn't punished.
    ratio = area / max(med_area, 1.0)
    size_score = _clip01(1.0 - abs(math.log(max(ratio, 1e-3))) / math.log(6))

    confidence = _clip01(
        W_EDGE * edge_score + W_CONTRAST * local_contrast_score
        + W_SHAPE * shape_score + W_IDENTITY * identity_score + W_SIZE * size_score
    )
    return confidence, {
        "edge_score": round(edge_score, 3), "local_contrast_score": round(local_contrast_score, 3),
        "shape_score": round(shape_score, 3), "identity_score": round(identity_score, 3),
        "size_score": round(size_score, 3),
    }


def _detect_stones(
    bgr: np.ndarray, hsv: np.ndarray, edge_mag: np.ndarray, lap_mag: np.ndarray,
    item: np.ndarray, eligible: np.ndarray,
) -> tuple[list[dict], np.ndarray, str]:
    """
    One unified candidate pool, scored by _region_confidence. Candidates
    below UNCERTAIN_THRESHOLD are discarded entirely — not counted, not
    drawn — which is the actual fix for plain metal glare being labelled a
    doubtful stone: it now has to earn a number or a '?', not default to one.
    `eligible` (⊆ item) excludes large backdrop-coloured interior regions
    (e.g. a ring's own opening) from candidate generation; area_pct and the
    metal-colour reference still use the full item so they read correctly.

    The classical colour-threshold pass (`_candidate_mask` +
    connectedComponents) always runs first — cheap, and its blobs double as
    seed points for app/models/ml_stone_detection.py's MobileSAM refinement,
    which then REPLACES these as the actual region boundaries when available
    (fixes touching-stone undercounting and facet-reflection overcounting
    that a colour threshold alone can't). Falls back to the classical blobs
    directly if the model is unavailable.

    Returns (stones, labels, detection_mode) where labels is the region map
    the caller uses to redraw each stone's contour for the overlay, and
    detection_mode is "ml_sam" or "classical" — surfaced to the report/UI so
    which pass produced a given case's stone count is never hidden.
    """
    item_bool = item > 0
    item_area = max(1, int(item_bool.sum()))
    metal_ref = (np.median(hsv[item_bool], axis=0) if item_bool.any()
                 else np.array([20.0, 80.0, 150.0]))
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    metal_lab = (np.median(lab_full[item_bool], axis=0) if item_bool.any()
                 else np.array([180.0, 128.0, 128.0], dtype=np.float32))

    cand = _candidate_mask(bgr, eligible, metal_lab)
    n, labels, cc_stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    detection_mode = "classical"

    sized = []
    for i in range(1, n):
        area = int(cc_stats[i, cv2.CC_STAT_AREA])
        if MIN_GEM_AREA_FRAC * item_area <= area <= MAX_STONE_AREA_FRAC * item_area:
            sized.append((i, area))
    med_area = float(np.median([a for _, a in sized])) if sized else 1.0

    try:
        ml_result = ml_stone_detection.ml_connected_components(
            bgr, cand, item_bool, min_area=MIN_GEM_AREA_FRAC * item_area, ref_area=med_area,
        )
    except Exception as e:
        logger.warning("ML stone detection errored (%s) — using classical detection", e)
        ml_result = None
    if ml_result is not None and len(ml_result) == 4:
        n, labels, cc_stats, cents = ml_result
        detection_mode = "ml_sam"
        sized = []
        for i in range(1, n):
            area = int(cc_stats[i, cv2.CC_STAT_AREA])
            if MIN_GEM_AREA_FRAC * item_area <= area <= MAX_STONE_AREA_FRAC * item_area:
                sized.append((i, area))
        med_area = float(np.median([a for _, a in sized])) if sized else 1.0

    stones = []
    for i, area in sized:
        region = labels == i
        confidence, feats = _region_confidence(region, hsv, edge_mag, lap_mag, item, metal_ref, area, med_area)
        if confidence < UNCERTAIN_THRESHOLD:
            continue
        # Identify the stone by nearest-reference colour match, not a hard-
        # coded hue band — see _classify_stone_color.
        stone_name, hue_class, match_conf = _classify_stone_color(bgr[region])
        colourless_ceiling = (
            MAX_COLOURLESS_AREA_FRAC_HIGH_CONTRAST
            if feats["local_contrast_score"] >= HIGH_CONTRAST_FLOOR
            else MAX_COLOURLESS_AREA_FRAC
        )
        if hue_class == "colourless" and area > colourless_ceiling * item_area:
            continue   # too large, AND blends into its own setting — glare, not a stone
        status = "confirmed" if confidence >= CONFIDENT_THRESHOLD else "uncertain"
        x  = int(cc_stats[i, cv2.CC_STAT_LEFT]); y  = int(cc_stats[i, cv2.CC_STAT_TOP])
        bw = int(cc_stats[i, cv2.CC_STAT_WIDTH]); bh = int(cc_stats[i, cv2.CC_STAT_HEIGHT])
        stones.append({
            "area_pct":     round(area / item_area * 100, 2),
            "hue_class":    hue_class,
            "stone_name":   stone_name,
            "match_confidence": match_conf,
            "confidence":   round(confidence, 3),
            "status":       status,
            "bbox":         [x, y, bw, bh],
            "centroid":     [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
            "_label":       i,
        })

    stones.sort(key=lambda s: -s["area_pct"])
    return stones, labels, detection_mode


def _item_bbox(item_bool: np.ndarray) -> Optional[list[int]]:
    ys, xs = np.where(item_bool)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _draw_grid(bgr: np.ndarray, item_bbox: list[int], grid_n: int = 4) -> np.ndarray:
    """N×N grid over the item's bounding box — a forensic-exhibit style
    overlay used to count/locate detected stones per cell (app/utils/gem_grid.py
    turns this same geometry into stats)."""
    overlay = bgr.copy()
    x0, y0, x1, y1 = item_bbox
    colour = (150, 150, 150)
    for i in range(1, grid_n):
        x = int(x0 + (x1 - x0) * i / grid_n)
        cv2.line(overlay, (x, y0), (x, y1), colour, 1, cv2.LINE_AA)
        y = int(y0 + (y1 - y0) * i / grid_n)
        cv2.line(overlay, (x0, y), (x1, y), colour, 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 1, cv2.LINE_AA)
    return overlay


def _stones_overlay(bgr: np.ndarray, item: np.ndarray, labels: np.ndarray, stones: list[dict]) -> np.ndarray:
    """Presentation stage: backdrop dimmed, confirmed stones outlined and
    numbered solid, uncertain ones outlined in amber with a '?' — genuinely
    the minority now, since a candidate has to clear a confidence floor just
    to be drawn at all."""
    overlay = bgr.copy()
    overlay[item == 0] //= 3
    for idx, s in enumerate(stones, start=1):
        region = (labels == s["_label"]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if s["status"] == "confirmed":
            colour = (GEM_OUTLINE_BGR.get(s["hue_class"], GEM_OUTLINE_BGR["other"])
                      if s["hue_class"] != "colourless" else CONFIRMED_COLOURLESS_BGR)
            label = str(idx)
        else:
            colour = UNCERTAIN_OUTLINE_BGR
            label = f"{idx}?"
        cv2.drawContours(overlay, contours, -1, colour, 2)
        cx, cy = int(s["centroid"][0]), int(s["centroid"][1])
        ox = cx - 5 * len(label)
        cv2.putText(overlay, label, (ox, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (ox, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _histogram_image(
    grey: np.ndarray, item: np.ndarray, t1: int, t2: int, t3: int,
    width: int = 640, height: int = 160,
) -> np.ndarray:
    """
    Luminance histogram of the ITEM pixels (background excluded), painted in
    material-class colours with the three threshold cut lines overlaid and a
    baseline axis. A tally of how many item pixels fall at each brightness,
    dark (left) → bright (right). Kept for technical verification of where the
    four material bands are split.
    """
    baseline = height - 16          # y of the x-axis; bars grow upward from here
    top_pad = 20                    # headroom for the T-labels
    vals = grey[item > 0]
    bins = np.bincount(vals.ravel(), minlength=256).astype(np.float64)
    peak = bins.max() or 1.0

    img = np.full((height, width, 3), 250, np.uint8)          # light backdrop
    bar_w = width / 256.0
    for b in range(256):
        cls = 0 if b < t1 else 1 if b < t2 else 2 if b < t3 else 3
        colour = CLASS_COLOURS_RGB[cls][::-1]                 # RGB -> BGR
        bh = int(round(bins[b] / peak * (baseline - top_pad)))
        if bh > 0:
            x0, x1 = int(b * bar_w), max(int(b * bar_w) + 1, int((b + 1) * bar_w))
            cv2.rectangle(img, (x0, baseline - bh), (x1, baseline), colour, -1)

    # baseline axis line under the bars
    cv2.line(img, (0, baseline), (width, baseline), (120, 120, 120), 1, cv2.LINE_AA)

    for t, label in ((t1, "T1"), (t2, "T2"), (t3, "T3")):
        x = int(t / 255.0 * width)
        for y in range(2, baseline, 8):                       # dashed cut line
            cv2.line(img, (x, y), (x, min(y + 4, baseline)), (60, 60, 60), 1)
        cv2.putText(img, f"{label}={t}", (min(x + 3, width - 52), 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "dark", (4, height - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "bright", (width - 44, height - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def _masked_otsu(grey: np.ndarray, item: np.ndarray) -> int:
    """Otsu threshold computed on item pixels only."""
    vals = grey[item > 0]
    if vals.size < 64:
        vals = grey.reshape(-1)
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def _sobel_magnitude(grey: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _hue_map(hsv: np.ndarray, item_bool: np.ndarray) -> np.ndarray:
    """The literal 'convert to HSV to see the colour' view: every item pixel
    redrawn at its OWN hue but full saturation and brightness, background
    dimmed. This is what makes a stone's true colour difference from the
    gold obvious on sight, independent of how brightly/evenly the photo was
    lit — the same signal the detection pipeline itself measures (see
    STONE_DELTA_E_MIN), made visible rather than left as internal numbers."""
    pure = np.empty_like(hsv)
    pure[..., 0] = hsv[..., 0]
    pure[..., 1] = 255
    pure[..., 2] = 255
    bgr_pure = cv2.cvtColor(pure, cv2.COLOR_HSV2BGR)
    bgr_pure[~item_bool] //= 4
    return bgr_pure


def _run_pipeline(
    bgr: np.ndarray,
    t1: Optional[int] = None,
    t2: Optional[int] = None,
    t3: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return ({stage_name: BGR image}, stats). All stats are computed on the
    item only (background removed via known-backdrop distance), and dark
    inclusion regions are cross-validated against independently detected
    stone candidates — a text claim of 'stones' can never explain an
    anomaly."""
    norm = _normalize_illumination(bgr)
    grey = _bt709_grey(norm)
    inverted = 255 - grey

    item, background_removed, stone_eligible = _item_mask(norm)

    # Exclude the fiducial calibration card (if present in-frame) from the
    # item entirely — its high-contrast black/white finder squares and
    # checksum strip would otherwise be swept into the item mask as "item"
    # pixels (they differ sharply from the backdrop, same as the jewellery
    # does), corrupting material composition, edge density, and stone
    # detection with a printed card instead of the jewellery itself.
    try:
        cbbox = card_bbox(norm)
    except Exception as e:
        cbbox = None
        logger.warning("Fiducial card detection (for exclusion) failed: %s", e)
    if cbbox:
        cx0, cy0, cx1, cy1 = cbbox
        item[cy0:cy1, cx0:cx1] = 0
        stone_eligible[cy0:cy1, cx0:cx1] = 0

    item_bool = item > 0
    item_area = max(1, int(item_bool.sum()))

    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    sobel = _sobel_magnitude(grey)
    laplacian = np.abs(cv2.Laplacian(grey, cv2.CV_64F))

    stones, stone_labels, stone_detection_mode = _detect_stones(
        norm, hsv, sobel.astype(np.float32), laplacian, item, stone_eligible)
    gems = [{"area_pct": s["area_pct"], "hue_class": s["hue_class"], "confidence": s["confidence"],
             "stone_name": s["stone_name"], "match_confidence": s["match_confidence"]}
            for s in stones if s["hue_class"] != "colourless"]
    colourless = [{"area_pct": s["area_pct"], "confidence": s["confidence"],
                   "kind": s["status"],  # "confirmed" | "uncertain"
                   "stone_name": s["stone_name"], "match_confidence": s["match_confidence"]}
                  for s in stones if s["hue_class"] == "colourless"]
    gem_area_pct = round(sum(g["area_pct"] for g in gems), 2)
    colourless_area_pct = round(sum(c["area_pct"] for c in colourless), 2)

    # Union of every kept stone region (confirmed + uncertain) — used below
    # to decide whether a dark luminance region is "explained" as a stone.
    stone_mask = np.zeros_like(item)
    for s in stones:
        stone_mask[stone_labels == s["_label"]] = 255

    otsu = _masked_otsu(grey, item)
    a1, a2, a3 = int(otsu * 0.55), otsu, int(otsu + (255 - otsu) * 0.55)
    t1, t2, t3 = t1 or a1, t2 or a2, t3 or a3
    if not 0 < t1 < t2 < t3 < 255:
        raise ValueError(f"Thresholds must satisfy 0 < T1 < T2 < T3 < 255, got {t1}, {t2}, {t3}")

    classes = np.digitize(grey, [t1, t2, t3]).astype(np.uint8)  # 0..3
    quantised = (classes * 85).astype(np.uint8)                 # 0/85/170/255 for display

    edges = sobel > EDGE_THRESHOLD

    material_rgb = np.zeros((*grey.shape, 3), dtype=np.uint8)
    for cls, colour in CLASS_COLOURS_RGB.items():
        material_rgb[classes == cls] = colour
    material_rgb[edges] = (255, 255, 255)
    material_rgb[~item_bool] = (45, 45, 45)          # backdrop excluded from analysis
    material = cv2.cvtColor(material_rgb, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(grey, cv2.COLORMAP_JET)

    # Dark inclusion regions (class 0) inside the item, cross-validated
    # against the independently detected (and confidence-filtered) stones:
    # overlapping = explained stone, non-overlapping = unexplained anomaly
    # (solder plug / exposed base metal).
    incl_mask = ((classes == 0) & item_bool).astype(np.uint8) * 255
    incl_mask = cv2.morphologyEx(incl_mask, cv2.MORPH_OPEN, ELLIPSE_5)
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(incl_mask, connectivity=8)
    min_area = MIN_GEM_AREA_FRAC * item_area
    explained, unexplained, unexplained_px = 0, 0, 0
    for i in range(1, n_labels):
        area = cc_stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        region = labels == i
        overlap = float((stone_mask[region] > 0).mean())
        if overlap >= INCLUSION_OVERLAP_EXPLAINED:
            explained += 1
        else:
            unexplained += 1
            unexplained_px += int(area)

    composition = {
        CLASS_NAMES[cls]: round(float((classes[item_bool] == cls).sum()) / item_area * 100, 1)
        for cls in CLASS_NAMES
    }

    gems_overlay = _stones_overlay(norm, item, stone_labels, stones)
    item_bbox = _item_bbox(item_bool)
    stages = {
        "original":  bgr,
        "grey":      cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR),
        "invert":    cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR),
        "threshold": cv2.cvtColor(quantised, cv2.COLOR_GRAY2BGR),
        "sobel":     cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR),
        "hsv":       _hue_map(hsv, item_bool),
        "material":  material,
        "gems":      gems_overlay,
        "stones_grid": _draw_grid(gems_overlay, item_bbox, grid_n=4) if item_bbox else gems_overlay,
        "heatmap":   heatmap,
        "histogram": _histogram_image(grey, item, t1, t2, t3),
    }
    # Public stones list — same data as gems/colourless, without the
    # internal connected-component label id used only for overlay drawing.
    stones_public = [{k: v for k, v in s.items() if k != "_label"} for s in stones]
    stats = {
        "thresholds":             {"t1": t1, "t2": t2, "t3": t3},
        "composition":            composition,
        "background_removed":     background_removed,
        "item_area_pct":          round(item_area / classes.size * 100, 1),
        "item_bbox":              item_bbox,
        "gem_regions":            len(gems),
        "gems":                   gems,
        "gem_area_pct":           gem_area_pct,
        "colourless_regions":     len(colourless),
        "colourless":             colourless,
        "colourless_area_pct":    colourless_area_pct,
        "stones":                 stones_public,
        "stone_detection_mode":   stone_detection_mode,
        "stones_confirmed":       sum(1 for s in stones if s["status"] == "confirmed"),
        "stones_uncertain":       sum(1 for s in stones if s["status"] == "uncertain"),
        "inclusions_explained":   explained,
        "inclusions_unexplained": unexplained,
        "unexplained_area_pct":   round(unexplained_px / item_area * 100, 1),
        "edge_density":           round(float(edges[item_bool].mean()) * 100, 1),
    }
    return stages, stats


def generate_xray(
    raw_bytes: bytes,
    out_dir: Path,
    t1: Optional[int] = None,
    t2: Optional[int] = None,
    t3: Optional[int] = None,
) -> dict:
    """Run the pipeline and save each stage as a PNG under out_dir.

    Returns {"stages": {name: path}, ...stats} with paths relative to the
    repo root (same convention as the other saved case media).
    """
    bgr = _load_bgr(raw_bytes)
    stages, stats = _run_pipeline(bgr, t1, t2, t3)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, img in stages.items():
        p = out_dir / f"{name}.png"
        cv2.imwrite(str(p), img)
        paths[name] = str(p)

    return {"stages": paths, **stats}


def xray_preview(
    raw_bytes: bytes,
    t1: Optional[int] = None,
    t2: Optional[int] = None,
    t3: Optional[int] = None,
) -> dict:
    """Run the pipeline and return stages as base64 PNG data URIs (no disk I/O).

    Used by POST /api/xray for interactive threshold tuning.
    """
    bgr = _load_bgr(raw_bytes)
    stages, stats = _run_pipeline(bgr, t1, t2, t3)

    # Photographic stages compress far better as JPEG; synthetic flat-colour
    # stages (threshold classes, material map, histogram) stay PNG.
    PNG_STAGES = {"threshold", "material", "histogram", "hsv"}
    encoded = {}
    for name, img in stages.items():
        if name in PNG_STAGES:
            ok, buf = cv2.imencode(".png", img)
            mime = "image/png"
        else:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            mime = "image/jpeg"
        if not ok:
            raise RuntimeError(f"Failed to encode stage '{name}'")
        encoded[name] = f"data:{mime};base64," + base64.b64encode(buf).decode()

    return {"stages": encoded, **stats}
