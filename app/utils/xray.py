"""
DSIP pseudo X-ray pipeline — material segmentation of jewellery photographs.

Stages 1-7 are classical image processing (no ML), stage 8 is a
classical/ML hybrid, stage 9 is an additive gold-vs-gems colour map:
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
  9. Gold vs gems map (_gold_gem_map) — paints kept stone regions as gems
     and classifies remaining item pixels as gold vs other via Lab ΔE to
     the item's own metal colour plus an HSV gold band. Visual / reporting
     only — does not change fusion or loan-decision risk.
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
W_EDGE     = float(os.getenv("STONE_W_EDGE", "0.20"))       # boundary contrast
W_CONTRAST = float(os.getenv("STONE_W_CONTRAST", "0.28"))   # local colour contrast
W_SHAPE    = float(os.getenv("STONE_W_SHAPE", "0.15"))      # convexity / aspect
W_IDENTITY = float(os.getenv("STONE_W_IDENTITY", "0.22"))   # facet sparkle or roundness
W_SIZE     = float(os.getenv("STONE_W_SIZE", "0.15"))       # size consistency
# edge_score normalisation divisor. On 640px-max, illumination-normalised
# photos the boundary-vs-interior gradient gap is genuinely compressed; 55
# was too harsh and systematically underscored real stones (measured on the
# demo panel — the single biggest driver of "no stone ever confirmed", since
# edge carried the largest weight yet was consistently the lowest signal).
STONE_EDGE_NORM = float(os.getenv("STONE_EDGE_NORM", "38.0"))
# Deliberately conservative: a false "confirmed" silently asserts something
# wrong, while a false "uncertain" still routes to officer review — so the
# bar to CONFIDENT stays high, and ties go to "uncertain", not the other way
# round. Calibrated on the demo panel: clear stones land ~0.64-0.77, plain-
# metal glare stays below, so the bar sits at 0.64 (with the weights above).
CONFIDENT_THRESHOLD = float(os.getenv("STONE_CONFIDENT_THRESHOLD", "0.64"))
UNCERTAIN_THRESHOLD = float(os.getenv("STONE_UNCERTAIN_THRESHOLD", "0.40"))

# Hard gap-artifact rejection (P1-4). A negative-space gap cut into the item
# silhouette (split-shank opening, openwork gap) is concave and ragged, so its
# convex-hull solidity is low. A real stone (round/oval/faceted) is convex, so
# its solidity is high. Below this solidity a backdrop-adjacent candidate is
# REJECTED OUTRIGHT — not merely down-weighted via the soft shape score — so a
# gap that happens to score well on the other signals still can't survive.
GAP_SOLIDITY_MIN = float(os.getenv("STONE_GAP_SOLIDITY_MIN", "0.70"))
# Backdrop-colour gap/glare signal. A candidate whose mean BGR is within this
# distance of the frame-border backdrop colour is either a gap (opening onto
# the backdrop) or specular glare — never a real stone by colour alone.
BACKDROP_STONE_DIST     = float(os.getenv("STONE_BACKDROP_DIST", "28.0"))
# For a backdrop-coloured candidate, this is the minimum fraction of its
# immediate surroundings that must be metal for it to be treated as glare-on-
# metal (capped to uncertain) rather than a gap opening onto the backdrop
# (rejected outright).
BACKDROP_GAP_METAL_FRAC = float(os.getenv("STONE_BACKDROP_GAP_METAL_FRAC", "0.85"))
# A backdrop-coloured candidate more elongated than this is a gap sliver /
# wedge / scratch (rejected), not a compact glare spot (capped to uncertain).
BACKDROP_GAP_ASPECT     = float(os.getenv("STONE_BACKDROP_GAP_ASPECT", "2.5"))

# AI-only stones are validated against the ornament's convex-hull envelope
# (which includes openwork gaps), dilated by this fraction of the item's short
# side so pavé flush to the edge — and the vision model's approximate centres —
# aren't clipped. See reconcile_stones.
STONE_AI_ENVELOPE_MARGIN_FRAC = float(os.getenv("STONE_AI_ENVELOPE_MARGIN_FRAC", "0.04"))

# ── Layer-A recall widening (manifold-outlier + two-sided chroma) ──────────
# The single-point lightness-deemphasised delta-E test above misses two real
# classes of stone: (a) stones whose colour sits CLOSE to the gold in delta-E
# yet clearly off the gold's own a*/b* chromaticity cloud (champagne/near-gold
# gems), and (b) colourless stones (diamond) that read low-chroma rather than
# "differently coloured". We OR in two extra recall paths so neither is lost:
#   - manifold outlier: a pixel that is NOT a member of the item's own gold
#     chromaticity manifold (the same Lab a*/b* Mahalanobis ellipse used for
#     gold-vs-gems, L* excluded so shadows on the same gold don't trigger it).
#   - two-sided chroma: a pixel whose chroma is clearly ABOVE the gold's own
#     chroma band (coloured gems) OR clearly BELOW it (colourless stones).
# The delta-E path is still OR'd in unchanged, so nothing previously detected
# regresses; the confidence gate + CV-uniformity rejection downstream still
# decide what is actually drawn, so widening recall here does not by itself
# add drawn false positives.
STONE_CHROMA_HI_K       = float(os.getenv("STONE_CHROMA_HI_K", "1.6"))    # gold_mean + k*std
STONE_CHROMA_LO_ABS     = float(os.getenv("STONE_CHROMA_LO_ABS", "12.0")) # absolute low-chroma floor
# Specular glints (very bright, near-neutral) are dropped before a region's
# colour is classified — a white facet flash must not drag a ruby toward
# "colourless".
STONE_SPECULAR_L      = float(os.getenv("STONE_SPECULAR_L", "225"))
STONE_SPECULAR_CHROMA = float(os.getenv("STONE_SPECULAR_CHROMA", "18"))

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

# ── Gold vs gems pixel map (additive stage — does not affect fusion) ────
# Warm gold / vivid gem / cool "other" — kept in sync with XRayView.jsx
# GOLD_GEM_LEGEND. Classification: kept stone regions = gem; remaining item
# pixels that match the item's own measured gold (Lab ΔE + HSV gold band)
# = gold; everything else on the item = other (solder, rhodium, unexplained).
GOLD_GEM_COLOURS_BGR = {
    "gold":  (40, 170, 230),   # warm Canara gold
    "gem":   (90, 40, 220),    # jewel magenta-red (fallback when no hue)
    "other": (110, 100, 90),   # cool slate
}
# Soft Lab match to the item's own metal colour (stones excluded from the
# reference so a large centre stone doesn't pull "gold" toward itself).
GOLD_LAB_MATCH_MAX = float(os.getenv("GOLD_LAB_MATCH_MAX", "24"))
# OpenCV HSV gold band (H is 0–179): yellow–amber jewellery metal.
GOLD_HSV_H_LO = int(os.getenv("GOLD_HSV_H_LO", "8"))
GOLD_HSV_H_HI = int(os.getenv("GOLD_HSV_H_HI", "38"))
GOLD_HSV_S_MIN = int(os.getenv("GOLD_HSV_S_MIN", "40"))
GOLD_HSV_V_MIN = int(os.getenv("GOLD_HSV_V_MIN", "45"))
# ── Robust gold chromaticity manifold ───────────────────────────────────
# Gold's LIGHTNESS swings from deep shadow to blown specular highlight, but
# its CHROMATICITY (Lab a*/b*) stays in a tight band. We fit that band from
# the item's OWN metal pixels and classify by Mahalanobis distance in a*/b*
# ONLY — L* excluded — so shadowed/over-lit gold still reads as gold. This
# replaces the single-median + fixed-ΔE test that inflated the "other"
# bucket with mis-lit gold (see _gold_gem_map).
GOLD_MANIFOLD_MAHALANOBIS_MAX = float(os.getenv("GOLD_MANIFOLD_MAHA_MAX", "3.0"))
GOLD_MIN_METAL_PIXELS = int(os.getenv("GOLD_MIN_METAL_PIXELS", "500"))
# Specular-highlight rescue: a near-white, low-chroma pixel on the metal is a
# reflection of the gold, not a stone. High L* + low chroma → gold. A lit
# ruby/sapphire is bright but HIGH chroma, so it is NOT rescued and stays gem.
GOLD_SPECULAR_L_MIN = float(os.getenv("GOLD_SPECULAR_L_MIN", "205"))
GOLD_SPECULAR_CHROMA_MAX = float(os.getenv("GOLD_SPECULAR_CHROMA_MAX", "22"))
# Per-hue gem paint colours (BGR) for the gold_gem stage overlay.
GEM_PAINT_BGR = {
    "red":        (50, 40, 210),
    "green":      (60, 140, 30),
    "blue":       (200, 90, 40),
    "other":      (160, 80, 180),
    "colourless": (210, 210, 220),
}


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
    # Defence in depth: an empty pixel sample (a zero-pixel region slipping
    # through) would crash cv2.cvtColor with an "empty src" assertion.
    if bgr_pixels is None or bgr_pixels.size == 0:
        return "unidentified", "other", 0.0
    lab_pixels = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    # Drop specular glints (very bright, near-neutral) before taking the
    # region's colour — a white facet flash must not drag a ruby/emerald
    # toward "colourless". If a region is ALL glare, keep everything (fall
    # back to the raw median rather than emptying the sample).
    chroma = np.sqrt((lab_pixels[:, 1] - 128.0) ** 2 + (lab_pixels[:, 2] - 128.0) ** 2)
    keep = ~((lab_pixels[:, 0] > STONE_SPECULAR_L) & (chroma < STONE_SPECULAR_CHROMA))
    lab_use = lab_pixels[keep] if bool(keep.any()) else lab_pixels
    median_lab = np.median(lab_use, axis=0)
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


def _candidate_mask(bgr: np.ndarray, item: np.ndarray, metal_lab: np.ndarray,
                    gold_manifold: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The human-eye rule, in code: a stone is whatever differs clearly in
    COLOUR from the item's own measured gold. Three complementary recall
    paths are OR'd so no real class of stone is lost (see the STONE_CHROMA_*
    / manifold comments above):
      1. lightness-deemphasised CIE Lab delta-E from the median gold — the
         original test, kept intact so nothing previously found regresses;
      2. manifold outlier — pixels NOT belonging to the item's own gold a*/b*
         chromaticity ellipse (recovers near-gold stones whose delta-E is
         small yet whose hue is clearly off the gold cloud);
      3. two-sided chroma — pixels clearly more chromatic than gold (coloured
         gems) OR clearly less (colourless/diamond).
    `gold_manifold` is the precomputed membership mask from _gold_membership;
    when None it is computed here (kept optional for backward-compatible calls
    and standalone testing). Recall is widened here on purpose — the
    confidence gate and CV-uniformity rejection downstream decide what is
    actually drawn, so this does not by itself add drawn false positives.
    """
    item_bool = item > 0
    item_area = max(1, int(item_bool.sum()))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # (1) lightness-deemphasised delta-E from the median gold (original path)
    dL = lab[..., 0] - metal_lab[0]
    da = lab[..., 1] - metal_lab[1]
    db = lab[..., 2] - metal_lab[2]
    delta_e = np.sqrt(LAB_LIGHTNESS_WEIGHT * dL ** 2 + da ** 2 + db ** 2)
    de_hit = delta_e > STONE_DELTA_E_MIN
    # (2) manifold outlier — not a member of the gold chromaticity ellipse
    manifold = gold_manifold if gold_manifold is not None else _gold_membership(lab, item_bool)
    maha_hit = (~manifold) if manifold is not None else np.zeros_like(de_hit)
    # (3) two-sided chroma test around the gold's own chroma band
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    if item_bool.any():
        g_mean = float(chroma[item_bool].mean())
        g_std = float(chroma[item_bool].std()) or 1.0
    else:
        g_mean, g_std = 0.0, 1.0
    chroma_hi = chroma > (g_mean + STONE_CHROMA_HI_K * g_std)
    chroma_lo = chroma < min(STONE_CHROMA_LO_ABS, max(0.0, g_mean - STONE_CHROMA_HI_K * g_std))
    chroma_hit = chroma_hi | chroma_lo
    cand = ((de_hit | maha_hit | chroma_hit) & item_bool).astype(np.uint8) * 255

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


def _is_enamel_region(hsv_pixels: np.ndarray, edge_pixels: np.ndarray) -> bool:
    """Meenakari enamel vs a faceted gemstone (P1-2).

    Vitreous enamel is a smooth, flat, opaque fill: uniform hue, uniform
    saturation, NO specular sparkle, and low internal edge density. A faceted
    stone fails at least one of these — facets sparkle (specular flashes),
    split light into hue/saturation variation, and pack the interior with
    micro-edges. Enamel must pass ALL four; a stone fails most.

    hsv_pixels: (N,3) H,S,V samples of the region. edge_pixels: (N,) edge
    magnitude at those pixels."""
    hsv = np.asarray(hsv_pixels, dtype=np.float32)
    if hsv.ndim != 2 or hsv.shape[0] < 8:
        return False
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    edges = np.asarray(edge_pixels, dtype=np.float32).ravel()

    hue_uniform  = float(np.std(h)) < 8.0
    sat_uniform  = float(np.std(s)) < 20.0
    no_specular  = float(np.mean(v > 240)) < 0.02          # no facet flashes
    edge_density = float(np.mean(edges > 30)) if edges.size else 0.0
    low_edges    = edge_density < 0.10                     # smooth, not faceted
    return bool(hue_uniform and sat_uniform and no_specular and low_edges)


def _detect_filigree(mask: np.ndarray) -> dict:
    """Filigree / Tarakashi openwork detection (P1-3).

    Openwork is thin metal wire enclosing many small gaps (jali). It shows up
    as a LOT of enclosed holes plus a low fill ratio (mostly air). A plain
    solitaire ring is also mostly-hollow (one big open centre) but has only ONE
    enclosed gap — low fill ratio ALONE must not trigger this, so both a high
    enclosed-gap count AND a low fill ratio are required."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    total_area = float((mask_u8 > 0).sum())
    if total_area < 1:
        return {"is_filigree": False, "enclosed_gap_count": 0, "fill_ratio": 1.0}

    # Enclosed holes = inner contours (those with a parent) in the hierarchy.
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        min_hole = max(4.0, 0.0004 * total_area)   # ignore pinhole noise
        for i, hci in enumerate(hierarchy[0]):
            if hci[3] != -1 and cv2.contourArea(contours[i]) >= min_hole:
                holes += 1

    # Fill ratio = material area / solid silhouette area (holes filled).
    ff = mask_u8.copy()
    fmask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, fmask, (0, 0), 255)
    filled = mask_u8 | cv2.bitwise_not(ff)
    filled_area = float((filled > 0).sum()) or total_area
    fill_ratio = total_area / filled_area

    is_filigree = holes >= 4 and fill_ratio < 0.85
    return {"is_filigree": bool(is_filigree), "enclosed_gap_count": int(holes),
            "fill_ratio": round(fill_ratio, 3)}


def _detect_multiple_items(mask: np.ndarray) -> dict:
    """Flag when the frame contains more than one separate item (e.g. an
    earring pair) — each needs its own valuation, and a single blended weight
    across two pieces is a common intake error."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    total = float((mask_u8 > 0).sum())
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    min_area = max(50.0, 0.02 * total)
    comps = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area), reverse=True)
    count = len(comps)
    likely_pair = bool(count == 2 and comps[1] / max(comps[0], 1) >= 0.6)
    return {"multiple_items_detected": bool(count >= 2), "component_count": int(count),
            "likely_pair": likely_pair}


def _watershed_split_touching(cand: np.ndarray) -> np.ndarray:
    """Split touching-but-not-merged stones (pavé / channel settings) via a
    distance-transform watershed, so a chain of adjacent stones isn't
    under-counted as one blob. A single isolated blob is returned untouched."""
    cand_bin = (np.asarray(cand) > 0).astype(np.uint8)
    if cand_bin.sum() == 0:
        return (cand_bin * 255).astype(np.uint8)

    dist = cv2.distanceTransform(cand_bin, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return (cand_bin * 255).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    n_fg, markers = cv2.connectedComponents(sure_fg)
    if n_fg <= 2:                       # 0 or 1 peak → nothing to split
        return (cand_bin * 255).astype(np.uint8)

    markers = markers + 1
    unknown = cv2.subtract(cand_bin * 255, sure_fg)
    markers[unknown == 255] = 0
    color = cv2.cvtColor(cand_bin * 255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    out = (cand_bin * 255).copy()
    out[markers == -1] = 0              # carve the watershed ridge lines
    return out


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
    edge_score = _clip01((boundary_grad - interior_grad) / STONE_EDGE_NORM)

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
    # Backdrop colour reference (from the frame border) — used to tell a
    # backdrop-coloured GAP or specular GLARE apart from a real stone (P1-4).
    backdrop_bgr = np.median(_border_patch(bgr).astype(np.float32), axis=0)
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    metal_lab = (np.median(lab_full[item_bool], axis=0) if item_bool.any()
                 else np.array([180.0, 128.0, 128.0], dtype=np.float32))

    gold_manifold = _gold_membership(lab_full, item_bool)
    cand = _candidate_mask(bgr, eligible, metal_lab, gold_manifold)
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
        # A component can carry a positive cc_stats area yet map to zero pixels
        # in `labels` when the SAM/CC stats and label array disagree (seen on a
        # multi-item earring-pair frame). An empty region crashes the downstream
        # colour classifier (cv2.cvtColor on an empty array) and triggers
        # mean-of-empty-slice warnings — skip it outright.
        if not region.any():
            continue
        # Hard gap-artifact rejection (P1-4): a negative-space gap (split-shank
        # opening, openwork gap) is concave/ragged, so its convex-hull solidity
        # is low. Reject it OUTRIGHT here rather than only down-weighting it via
        # the soft shape score, so a gap can't survive by scoring well on the
        # other signals. Real stones are convex (solidity ≫ threshold).
        region_u8 = (region.astype(np.uint8)) * 255
        gcontours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if gcontours:
            gc = max(gcontours, key=cv2.contourArea)
            ghull = cv2.contourArea(cv2.convexHull(gc)) or 1.0
            if (cv2.contourArea(gc) / ghull) < GAP_SOLIDITY_MIN:
                continue

        # Backdrop-colour signal (P1-4): a convex triangular split-shank wedge
        # slips past the solidity filter (a triangle is convex), and a small
        # specular highlight can look like a colourless stone. Both are the
        # BACKDROP colour. Discriminate by their surroundings:
        #   • surroundings open onto the exterior backdrop  → it's a GAP  → reject
        #   • surroundings are (almost) all metal           → it's GLARE → cap to uncertain
        cand_bgr = bgr[region].astype(np.float32).mean(axis=0)
        backdrop_dist = float(np.linalg.norm(cand_bgr - backdrop_bgr))
        force_uncertain = False
        if backdrop_dist < BACKDROP_STONE_DIST:
            (grw, grh) = cv2.minAreaRect(gc)[1] if gcontours else (1.0, 1.0)
            gap_aspect = max(grw, grh) / max(1.0, min(grw, grh))
            surround = (cv2.dilate(region_u8, ELLIPSE_9) > 0) & ~region
            surround_n = int(surround.sum())
            metal_frac = float((surround & item_bool).sum()) / max(surround_n, 1)
            # Elongated backdrop-coloured sliver (split-shank wedge, scratch) OR
            # one that opens onto the exterior backdrop → it's a gap → reject.
            if gap_aspect > BACKDROP_GAP_ASPECT or metal_frac < BACKDROP_GAP_METAL_FRAC:
                continue
            force_uncertain = True       # compact backdrop-coloured spot on metal — glare

        confidence, feats = _region_confidence(region, hsv, edge_mag, lap_mag, item, metal_ref, area, med_area)
        # Identify the stone by nearest-reference colour match, not a hard-
        # coded hue band — see _classify_stone_color.
        stone_name, hue_class, match_conf = _classify_stone_color(bgr[region])
        # Meenakari enamel vs a real set stone: a flat, uniform, non-sparkling
        # coloured fill is enamel (painted metal), not a gemstone — tag it so it
        # is excluded from the stone count and weight deduction (P1-2). Only
        # considered for clearly-coloured regions; colourless/faceted stones
        # keep their facet edges and fail the enamel test.
        material = "gemstone"
        if hue_class != "colourless" and _is_enamel_region(
                hsv[region].astype(np.float32), edge_mag[region].astype(np.float32)):
            material = "enamel"
        colourless_ceiling = (
            MAX_COLOURLESS_AREA_FRAC_HIGH_CONTRAST
            if feats["local_contrast_score"] >= HIGH_CONTRAST_FLOOR
            else MAX_COLOURLESS_AREA_FRAC
        )
        if hue_class == "colourless" and area > colourless_ceiling * item_area:
            continue   # too large, AND blends into its own setting — glare, not a stone
        # Below UNCERTAIN_THRESHOLD candidates are NO LONGER discarded: they are
        # retained tagged status="candidate" so the AI cross-confirmation step
        # (app/utils/stone_fusion.py) can promote or reject them. Layer A never
        # draws or counts a "candidate" on its own (see _run_pipeline, which
        # filters to confirmed+uncertain) — so keeping them changes NO
        # ML-only output; it only stops silently destroying evidence the AI
        # could rescue, which is the real fix for under-counting.
        below_uncertain = confidence < UNCERTAIN_THRESHOLD
        if below_uncertain:
            status = "candidate"
        else:
            status = "confirmed" if confidence >= CONFIDENT_THRESHOLD else "uncertain"
        # A backdrop-coloured spot on metal is glare — never assert it confirmed.
        if force_uncertain and status == "confirmed":
            status = "uncertain"
        x  = int(cc_stats[i, cv2.CC_STAT_LEFT]); y  = int(cc_stats[i, cv2.CC_STAT_TOP])
        bw = int(cc_stats[i, cv2.CC_STAT_WIDTH]); bh = int(cc_stats[i, cv2.CC_STAT_HEIGHT])
        stones.append({
            "area_pct":     round(area / item_area * 100, 2),
            "hue_class":    hue_class,
            "material":     material,
            "stone_name":   stone_name,
            "match_confidence": match_conf,
            "confidence":   round(confidence, 3),
            "status":       status,
            "below_uncertain": below_uncertain,
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


def _gold_membership(lab: np.ndarray, metal_bool: np.ndarray) -> Optional[np.ndarray]:
    """Robust gold classification via a chromaticity manifold.

    Fit a robust ellipse to the item's own metal pixels in Lab a*/b* space
    (lightness deliberately excluded), then classify every frame pixel by
    Mahalanobis distance to that ellipse. Returns a full-frame boolean mask,
    or None when there are too few metal pixels to fit a stable covariance
    (caller falls back to the legacy ΔE test — never a hard failure).
    """
    ab = lab[..., 1:3].astype(np.float32)          # a*, b* only
    samp = ab[metal_bool]
    if samp.shape[0] < GOLD_MIN_METAL_PIXELS:
        return None
    center = np.median(samp, axis=0)
    d = samp - center
    cov = np.cov(d, rowvar=False) + np.eye(2, dtype=np.float32) * 1e-3
    try:
        inv = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        return None
    diff = ab - center
    m2 = np.einsum("...i,ij,...j->...", diff, inv, diff)   # Mahalanobis²
    return m2 <= GOLD_MANIFOLD_MAHALANOBIS_MAX ** 2


def _gold_gem_map(
    bgr: np.ndarray,
    hsv: np.ndarray,
    item: np.ndarray,
    stone_mask: np.ndarray,
    stones: list[dict],
    stone_labels: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Officer-facing gold-vs-gems segmentation map.

    Additive visualisation only — does NOT feed fusion / risk scoring.
    Reuses the already-detected stone regions as the gem class, then
    classifies the remaining item pixels as gold vs other via:
      1. Lightness-deemphasised CIE Lab distance to the item's own metal
         colour (median Lab over non-stone item pixels), and
      2. An HSV gold-hue band as a secondary accept (yellow–amber metal
         that Lab may undershoot under strong speculars).

    Returns (BGR visualisation, gold_gem_split stats).
    """
    item_bool = item > 0
    item_area = max(1, int(item_bool.sum()))
    gem_bool = (stone_mask > 0) & item_bool
    metal_bool = item_bool & ~gem_bool

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if metal_bool.any():
        metal_lab = np.median(lab[metal_bool], axis=0)
    elif item_bool.any():
        metal_lab = np.median(lab[item_bool], axis=0)
    else:
        metal_lab = np.array([180.0, 128.0, 128.0], dtype=np.float32)

    # Legacy ΔE test — retained as the fallback when the manifold can't fit.
    dL = lab[..., 0] - metal_lab[0]
    da = lab[..., 1] - metal_lab[1]
    db = lab[..., 2] - metal_lab[2]
    delta_e = np.sqrt(LAB_LIGHTNESS_WEIGHT * dL ** 2 + da ** 2 + db ** 2)
    lab_gold = delta_e <= GOLD_LAB_MATCH_MAX

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_gold = (
        (h >= GOLD_HSV_H_LO) & (h <= GOLD_HSV_H_HI)
        & (s >= GOLD_HSV_S_MIN) & (v >= GOLD_HSV_V_MIN)
    )

    # Primary: robust chromaticity manifold. L* excluded, so mis-lit gold
    # stays gold; specular highlights rescued explicitly below.
    manifold = _gold_membership(lab, metal_bool)
    if manifold is not None:
        L = lab[..., 0]
        chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
        specular = (L >= GOLD_SPECULAR_L_MIN) & (chroma <= GOLD_SPECULAR_CHROMA_MAX)
        gold_bool = metal_bool & (manifold | specular | hsv_gold)
        gold_method = "manifold"
    else:
        gold_bool = metal_bool & (lab_gold | hsv_gold)
        gold_method = "delta_e_fallback"
    other_bool = metal_bool & ~gold_bool

    # Soften gold/other masks slightly so the paint doesn't look speckled.
    gold_u8 = (gold_bool.astype(np.uint8) * 255)
    other_u8 = (other_bool.astype(np.uint8) * 255)
    gold_u8 = cv2.morphologyEx(gold_u8, cv2.MORPH_CLOSE, ELLIPSE_3)
    other_u8 = cv2.morphologyEx(other_u8, cv2.MORPH_OPEN, ELLIPSE_3)
    gold_bool = (gold_u8 > 0) & metal_bool
    other_bool = (other_u8 > 0) & metal_bool & ~gold_bool

    # Visualisation: dimmed original + class paints + gem outlines.
    base = (bgr.astype(np.float32) * 0.35).astype(np.uint8)
    base[~item_bool] = (base[~item_bool].astype(np.float32) * 0.35).astype(np.uint8)
    out = base.copy()

    gold_c = np.array(GOLD_GEM_COLOURS_BGR["gold"], dtype=np.float32)
    other_c = np.array(GOLD_GEM_COLOURS_BGR["other"], dtype=np.float32)

    if gold_bool.any():
        # Keep some of the photo's texture under a warm gold wash.
        blend = bgr[gold_bool].astype(np.float32) * 0.45 + gold_c * 0.55
        out[gold_bool] = np.clip(blend, 0, 255).astype(np.uint8)

    if other_bool.any():
        blend = bgr[other_bool].astype(np.float32) * 0.40 + other_c * 0.60
        out[other_bool] = np.clip(blend, 0, 255).astype(np.uint8)

    # Paint each kept stone in its hue-class colour (or saturated original).
    for s in stones:
        region = (stone_labels == s["_label"]) & item_bool
        if not region.any():
            continue
        paint = np.array(
            GEM_PAINT_BGR.get(s.get("hue_class"), GOLD_GEM_COLOURS_BGR["gem"]),
            dtype=np.float32,
        )
        # Colourless stones: keep brighter photo texture so diamonds/pearls
        # still read as clear rather than opaque paint.
        if s.get("hue_class") == "colourless":
            blend = bgr[region].astype(np.float32) * 0.70 + paint * 0.30
        else:
            blend = bgr[region].astype(np.float32) * 0.30 + paint * 0.70
        out[region] = np.clip(blend, 0, 255).astype(np.uint8)

        region_u8 = region.astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge = (0, 255, 255) if s.get("status") == "uncertain" else (255, 255, 255)
        cv2.drawContours(out, contours, -1, edge, 2, cv2.LINE_AA)

    gold_pct = round(float(gold_bool.sum()) / item_area * 100, 1)
    gem_pct = round(float(gem_bool.sum()) / item_area * 100, 1)
    other_pct = round(max(0.0, 100.0 - gold_pct - gem_pct), 1)
    # Re-normalise from actual other pixels if morphology drifted counts.
    other_pct = round(float(other_bool.sum()) / item_area * 100, 1)
    # Tiny float drift — force sum ≈ 100 on the three reported buckets.
    reported = gold_pct + gem_pct + other_pct
    if reported > 0 and abs(reported - 100.0) > 0.2:
        scale = 100.0 / reported
        gold_pct = round(gold_pct * scale, 1)
        gem_pct = round(gem_pct * scale, 1)
        other_pct = round(max(0.0, 100.0 - gold_pct - gem_pct), 1)

    split = {
        "gold_pct": gold_pct,
        "gem_pct": gem_pct,
        "other_pct": other_pct,
        "method": "lab_delta_e+hsv_gold_band+stone_mask",
        "gold_method": gold_method,
        "stones_used": len(stones),
    }
    return out, split


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

    all_stones, stone_labels, stone_detection_mode = _detect_stones(
        norm, hsv, sobel.astype(np.float32), laplacian, item, stone_eligible)
    # Layer-A surfaces only DRAWN stones (confirmed + uncertain). "candidate"
    # regions (below UNCERTAIN_THRESHOLD) are retained in `all_stones` for the
    # optional AI cross-confirmation step only, and are excluded from every
    # ML-only output below — so this file's external result is unchanged when
    # no AI fusion runs. The full labelled set travels to the route via `ctx`.
    stones = [s for s in all_stones if s.get("status") != "candidate"]
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
    gold_gem_img, gold_gem_split = _gold_gem_map(
        norm, hsv, item, stone_mask, stones, stone_labels)
    item_bbox = _item_bbox(item_bool)
    stages = {
        "original":  bgr,
        "grey":      cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR),
        "invert":    cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR),
        "threshold": cv2.cvtColor(quantised, cv2.COLOR_GRAY2BGR),
        "sobel":     cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR),
        "hsv":       _hue_map(hsv, item_bool),
        "material":  material,
        "gold_gem":  gold_gem_img,
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
        "gold_gem_split":         gold_gem_split,
        # Openwork (filigree/Tarakashi) and multi-item detection — surfaced so
        # downstream (tarnish/density/acoustic gating, officer UI) can react.
        "filigree":               _detect_filigree(item_bool),
        "multiple_items":         _detect_multiple_items(item_bool),
    }
    # Reconciliation context: the raw arrays + FULL labelled stone set (incl.
    # below-uncertain "candidate" regions filtered out of `stats` above) that
    # the async route needs to fuse in AI detections WITHOUT re-running the
    # pipeline / SAM. Never persisted — the route pops it. See reconcile_stones.
    ctx = {
        "norm": norm, "item": item, "item_bool": item_bool, "hsv": hsv,
        "stone_labels": stone_labels, "all_stones": all_stones,
        "item_bbox": item_bbox, "item_area": item_area,
        "detection_mode": stone_detection_mode,
    }
    return stages, stats, ctx


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
    stages, stats, _ctx = _run_pipeline(bgr, t1, t2, t3)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, img in stages.items():
        p = out_dir / f"{name}.png"
        cv2.imwrite(str(p), img)
        paths[name] = str(p)

    return {"stages": paths, **stats}


# Photographic stages compress far better as JPEG; synthetic flat-colour
# stages (threshold classes, material map, histogram, gold_gem) stay PNG.
_PNG_STAGES = {"threshold", "material", "histogram", "hsv", "gold_gem"}


def _encode_stage(name: str, img: np.ndarray) -> str:
    """Encode one BGR stage image as a base64 data URI (JPEG unless synthetic)."""
    if name in _PNG_STAGES:
        ok, buf = cv2.imencode(".png", img)
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        mime = "image/jpeg"
    if not ok:
        raise RuntimeError(f"Failed to encode stage '{name}'")
    return f"data:{mime};base64," + base64.b64encode(buf).decode()


def xray_preview(
    raw_bytes: bytes,
    t1: Optional[int] = None,
    t2: Optional[int] = None,
    t3: Optional[int] = None,
    _return_ctx: bool = False,
):
    """Run the pipeline and return stages as base64 data URIs (no disk I/O).

    Used by POST /api/xray for interactive threshold tuning. When
    `_return_ctx` is True, also returns the reconciliation context dict so the
    async analysis route can fuse in AI stone detections; default False keeps
    the original single-dict return for every existing caller.
    """
    bgr = _load_bgr(raw_bytes)
    stages, stats, ctx = _run_pipeline(bgr, t1, t2, t3)

    encoded = {name: _encode_stage(name, img) for name, img in stages.items()}

    result = {"stages": encoded, **stats}
    if _return_ctx:
        # Stash the FULL-RESOLUTION source alongside the (MAX_SIDE-capped) norm so
        # the AI stone-vision layer can crop small items from real pixels instead
        # of the downscaled frame — the normalisation cap (640px) blurs the few-
        # pixel accent/pavé stones a catalogue photo carries, and no upscaling
        # recovers detail that was thrown away before the crop. `source_to_norm`
        # maps a source-pixel coordinate back into norm space so fused detections
        # stay in the same coordinate frame as the ML result. Best-effort only.
        try:
            full = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if full is not None and full.shape[1] > ctx["norm"].shape[1]:
                ctx["source_bgr"] = full
                ctx["source_to_norm"] = ctx["norm"].shape[1] / float(full.shape[1])
        except Exception as e:  # noqa: BLE001 — optional detail boost, never fatal
            logger.warning("Full-res source decode for AI vision failed (%s)", e)
        return result, ctx
    return result


def reconcile_stones(ctx: dict, ai_stones) -> tuple[dict, dict]:
    """Fuse AI vision stone detections into the ML result and re-render the
    stone overlays. Returns (stats_patch, stage_patch) the async analysis route
    merges into the xray result and its encoded stages.

    `ai_stones` is None when the AI layer is off/failed -> ML-only passthrough
    (the drawn set is unchanged; only agreement tags + a `stone_agreement`
    summary are added). Guarded — on any error returns empty patches so the
    caller keeps the untouched ML result.
    """
    try:
        from app.utils import stone_fusion
        norm = ctx["norm"]; item = ctx["item"]; item_bool = ctx["item_bool"]
        labels = ctx["stone_labels"]; all_stones = ctx["all_stones"]
        item_bbox = ctx["item_bbox"]; item_area = max(1, int(ctx["item_area"]))

        # Envelope = the ornament silhouette INCLUDING openwork gaps (convex
        # hull of the metal mask). AI-only stones are validated against this, not
        # item_bool, so a stone floated in an openwork gap (pavé kite frame,
        # prong-set cluster) — which sits in a HOLE of the metal mask — is not
        # dropped. Falls back to item_bool if the hull can't be built.
        item_envelope = item_bool
        try:
            ys_e, xs_e = np.where(item_bool)
            if len(xs_e) >= 3:
                hull = cv2.convexHull(np.column_stack([xs_e, ys_e]).astype(np.int32))
                env = np.zeros(item_bool.shape, dtype=np.uint8)
                cv2.fillConvexPoly(env, hull, 1)
                # Dilate by a small margin so pavé stones flush to the ornament's
                # edge (and the vision model's slightly-approximate centres) are
                # not clipped by a hull that hugs the outermost metal pixel.
                # Tunable — larger recovers more edge stones at some risk of
                # accepting an off-item reflection the AI called a stone.
                Hh, Ww = item_bool.shape
                margin = max(4, int(STONE_AI_ENVELOPE_MARGIN_FRAC * min(Hh, Ww)))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
                item_envelope = (cv2.dilate(env, k) > 0)
        except Exception as e:  # noqa: BLE001 — geometry only, never fatal
            logger.warning("Item envelope (hull) failed (%s) — using metal mask", e)

        ai_mask_fn = None
        if ai_stones:
            def ai_mask_fn(ai):   # SAM-precise boundary for AI-only stones
                return ml_stone_detection.sam_mask_at_point(norm, ai["centroid"], item_envelope)

        stones, labels_out, meta = stone_fusion.reconcile(
            all_stones, labels, ai_stones, item_bool, ai_mask_fn, item_envelope=item_envelope)

        # Re-render overlays from the fused label map + stones.
        gems_overlay = _stones_overlay(norm, item, labels_out, stones)
        grid = _draw_grid(gems_overlay, item_bbox, grid_n=4) if item_bbox else gems_overlay
        stage_patch = {
            "gems": _encode_stage("gems", gems_overlay),
            "stones_grid": _encode_stage("stones_grid", grid),
        }

        # The gold-vs-gems split reuses the stone regions as its gem class, so
        # it MUST be recomputed from the fused set. _run_pipeline computed it
        # from the ML-only mask, before the AI layer had contributed anything —
        # leaving a diamond-set ring reading ~97% gold / ~2% gems even though
        # the AI had found every stone. Rebuild the gem mask from the fused
        # label map so the split, and its stage image, match the stone list the
        # officer is actually shown.
        gold_gem_split = None
        try:
            hsv = ctx.get("hsv")
            if hsv is not None:
                fused_stone_mask = (labels_out > 0).astype(np.uint8) * 255
                gold_gem_img, gold_gem_split = _gold_gem_map(
                    norm, hsv, item, fused_stone_mask, stones, labels_out)
                stage_patch["gold_gem"] = _encode_stage("gold_gem", gold_gem_img)
        except Exception as e:  # noqa: BLE001 — visualisation only, never fatal
            logger.warning("Gold/gem split recompute failed (%s) — keeping ML-only split", e)
            gold_gem_split = None

        # Recompute the gems/colourless summaries over the fused drawn set —
        # same shapes as _run_pipeline, plus the agreement fields.
        gems = [{"area_pct": s["area_pct"], "hue_class": s["hue_class"],
                 "confidence": s["confidence"], "stone_name": s["stone_name"],
                 "match_confidence": s.get("match_confidence", 0.0),
                 "agreement": s.get("agreement", "ml_only"),
                 "gem_type": s.get("gem_type", ""), "colour": s.get("colour", ""),
                 "ai_confidence": s.get("ai_confidence", 0.0)}
                for s in stones if s["hue_class"] != "colourless"]
        colourless = [{"area_pct": s["area_pct"], "confidence": s["confidence"],
                       "kind": s["status"], "stone_name": s["stone_name"],
                       "match_confidence": s.get("match_confidence", 0.0),
                       "agreement": s.get("agreement", "ml_only"),
                       "ai_confidence": s.get("ai_confidence", 0.0)}
                      for s in stones if s["hue_class"] == "colourless"]
        stones_public = [{k: v for k, v in s.items() if k != "_label"} for s in stones]

        detection_mode = "ml_ai" if meta.get("ai_used") else ctx.get("detection_mode", "classical")
        stats_patch = {
            "stones": stones_public,
            "gem_regions": len(gems),
            "gems": gems,
            "gem_area_pct": round(sum(g["area_pct"] for g in gems), 2),
            "colourless": colourless,
            "colourless_regions": len(colourless),
            "colourless_area_pct": round(sum(c["area_pct"] for c in colourless), 2),
            "stones_confirmed": sum(1 for s in stones if s["status"] == "confirmed"),
            "stones_uncertain": sum(1 for s in stones if s["status"] == "uncertain"),
            "stone_detection_mode": detection_mode,
            "stone_agreement": meta,
        }
        if gold_gem_split is not None:
            stats_patch["gold_gem_split"] = gold_gem_split
        return stats_patch, stage_patch
    except Exception as e:  # noqa: BLE001 — never break the request path
        logger.warning("Stone AI reconciliation failed (%s) — keeping ML-only result", e)
        return {}, {}
