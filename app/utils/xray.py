"""
DSIP pseudo X-ray pipeline — material segmentation of jewellery photographs.

Classical image-processing stages (no ML):
  1. BT.709 luminance greyscale
  2. Intensity inversion (radiographic negative)
  3. Multi-level thresholding into 4 material classes (Otsu-anchored)
  4. Sobel gradient magnitude (material boundaries)
  5. False-colour material map (threshold classes + edge overlay)
  6. Jet heatmap of raw luminance
"""
import base64
import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_SIDE = 640
EDGE_THRESHOLD = 70          # on the 0-255 normalised Sobel magnitude
MIN_GEM_AREA_FRAC = 0.002    # ≥0.2% of item area — beadwork stones are small

# Background subtraction: the backdrop colour is estimated from the frame
# border (SOP: photograph on a plain fixed-colour backdrop — the light box
# enforces this, not evaluator memory). Weighted HSV distance above this
# threshold = item pixel. Calibration parameter, env-overridable.
BACKDROP_DIST_THRESHOLD = float(os.getenv("BACKDROP_DIST_THRESHOLD", "60"))

# Gold hue band in OpenCV H units (0-179 = degrees/2); saturated pixels
# outside this band inside the item are gem candidates.
GOLD_HUE_CV = (12, 40)
GEM_SAT_MIN = 45     # pale/pastel stones (e.g. mint emerald beads) sit near S~60
GEM_VAL_MIN = 60     # shadow guard: hue is noise in near-black pixels

# Colourless stone candidates (pearl/diamond/CZ): bright + low-saturation.
# Distinguished from metal specular highlights by SHAPE (pearls: near-circular
# with smooth interior gradient) or TEXTURE (faceted stones: dense micro-edge
# sparkle). These are flagged for officer confirmation, never silently
# absorbed into metal.
CLS_SAT_MAX      = 55
CLS_VAL_MIN      = 165
CLS_LAPLACE_EDGE = 40     # |Laplacian| above this = micro-edge pixel
CLS_FACET_FRAC   = 0.15   # micro-edge fraction above this = faceted sparkle
CLS_SMOOTH_FRAC  = 0.08   # below this + round = pearl-like
CLS_CIRCULARITY  = 0.72   # 4πA/P² threshold for "round"
INCLUSION_OVERLAP_EXPLAINED = 0.25   # ≥25% overlap with a detected gem = explained

# Material class colours (RGB) — kept in sync with XRayView.jsx legend
CLASS_COLOURS_RGB = {
    0: (220, 38, 38),    # darkest  → gemstone body (red)
    1: (15, 118, 110),   # low-mid  → solder / joints / shadowed metal (teal)
    2: (30, 58, 95),     # mid-high → metal body (navy)
    3: (217, 119, 6),    # brightest→ specular facets (gold)
}
CLASS_NAMES = {0: "gemstone", 1: "joint", 2: "metal", 3: "facet"}


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


def _item_mask(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Separate the item from a plain backdrop. The backdrop colour is the
    median HSV of the frame border; item pixels are those whose weighted
    HSV distance from it exceeds BACKDROP_DIST_THRESHOLD.
    Returns (uint8 mask, background_removed). Falls back to the full frame
    when segmentation is implausible (item <3% or >97% of the frame).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, w = hsv.shape[:2]
    b = max(4, int(0.05 * min(h, w)))
    border = np.concatenate([
        hsv[:b].reshape(-1, 3), hsv[-b:].reshape(-1, 3),
        hsv[:, :b].reshape(-1, 3), hsv[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    dh = np.abs(hsv[..., 0] - bg[0])
    dh = np.minimum(dh, 180 - dh)                    # hue is circular
    dist = dh * 2.0 + np.abs(hsv[..., 1] - bg[1]) * 1.0 + np.abs(hsv[..., 2] - bg[2]) * 0.5

    mask = (dist > BACKDROP_DIST_THRESHOLD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # keep components ≥1% of frame AND ≥15% of the largest one — true
    # multi-piece items (earring pairs) are comparable in size, while stray
    # props (a flower, a coin) are not. Components touching the frame border
    # are dropped too: props, fingers and fabric folds enter from an edge,
    # while a centred item under the light box does not. Fill holes after.
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    if n > 1:
        largest = max(cc_stats[i, cv2.CC_STAT_AREA] for i in range(1, n))
        for i in range(1, n):
            area = cc_stats[i, cv2.CC_STAT_AREA]
            x, y = cc_stats[i, cv2.CC_STAT_LEFT], cc_stats[i, cv2.CC_STAT_TOP]
            bw, bh = cc_stats[i, cv2.CC_STAT_WIDTH], cc_stats[i, cv2.CC_STAT_HEIGHT]
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            if area >= 0.01 * mask.size and area >= 0.15 * largest and not touches_border:
                keep[labels == i] = 255
        if not keep.any():
            # everything touches the border (item larger than frame) — keep
            # the largest component rather than falling back to full frame
            big = max(range(1, n), key=lambda i: cc_stats[i, cv2.CC_STAT_AREA])
            keep[labels == big] = 255
    keep = _fill_holes(keep)

    frac = float((keep > 0).mean())
    if frac < 0.03 or frac > 0.97:
        return np.full(mask.shape, 255, np.uint8), False
    return keep, True


# Hue-class boundaries in OpenCV H units (0-179 = degrees/2)
def _classify_hue(h: float) -> str:
    if h < 10 or h >= 160:
        return "red"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    return "other"


GEM_OUTLINE_BGR = {
    "red":   (60, 60, 230),
    "green": (80, 200, 80),
    "blue":  (230, 140, 40),
    "other": (200, 200, 200),
}


def _gem_candidate_mask(
    bgr: np.ndarray, item: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """
    Independent gem detection (no text input): saturated colour clusters
    inside the item whose hue falls outside the gold band. Returns
    (mask, per-gem list [{area_pct, hue_class}]). This is the cross-check
    that DSIP inclusion regions must agree with before being 'explained',
    and the source of the stone-fraction estimate for composition analysis.
    Limitation: colourless stones (diamond, pearl) are low-saturation and
    are not detected by this colour rule.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    non_gold = (hch < GOLD_HUE_CV[0]) | (hch > GOLD_HUE_CV[1])
    cand = (
        (sch > GEM_SAT_MIN) & (vch > GEM_VAL_MIN) & non_gold & (item > 0)
    ).astype(np.uint8) * 255
    # gentle open removes specks; close merges adjacent small stones
    # (bead fringes) into clusters that pass the area filter as one region
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    item_area = max(1, int((item > 0).sum()))
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand)
    gems = []
    for i in range(1, n):
        area = int(cc_stats[i, cv2.CC_STAT_AREA])
        if area < MIN_GEM_AREA_FRAC * item_area:
            continue
        region = labels == i
        keep[region] = 255
        hue = float(np.median(hch[region]))
        gems.append({
            "area_pct":  round(area / item_area * 100, 2),
            "hue_class": _classify_hue(hue),
        })
    gems.sort(key=lambda g: -g["area_pct"])
    return keep, gems


def _colourless_candidates(
    bgr: np.ndarray, grey: np.ndarray, item: np.ndarray, gem_mask: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """
    Detect colourless stone candidates (pearls, diamonds, CZ) that the colour
    clusterer cannot see. Bright low-saturation blobs inside the item are
    classified by shape/texture:
      pearl-like : near-circular contour, smooth interior gradient
      faceted    : dense micro-edge sparkle (local Laplacian energy)
      ambiguous  : neither signature — could be a metal specular highlight
    All are flagged for officer confirmation; none are treated as confirmed.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sch, vch = hsv[..., 1], hsv[..., 2]
    cand = (
        (sch < CLS_SAT_MAX) & (vch > CLS_VAL_MIN) & (item > 0) & (gem_mask == 0)
    ).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    micro_edge = np.abs(cv2.Laplacian(grey, cv2.CV_64F)) > CLS_LAPLACE_EDGE
    item_area = max(1, int((item > 0).sum()))

    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand)
    regions = []
    for i in range(1, n):
        area = int(cc_stats[i, cv2.CC_STAT_AREA])
        # too small = noise; too large = lighting sheet, not a stone
        if area < 0.001 * item_area or area > 0.08 * item_area:
            continue
        region = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        circularity = 4 * math.pi * area / (perim ** 2) if perim > 0 else 0.0
        facet_frac = float(micro_edge[labels == i].mean())

        if circularity >= CLS_CIRCULARITY and facet_frac <= CLS_SMOOTH_FRAC:
            kind = "pearl-like"
        elif facet_frac >= CLS_FACET_FRAC:
            kind = "faceted"
        else:
            kind = "ambiguous"   # possibly a specular highlight — confirm

        keep[labels == i] = 255
        regions.append({
            "area_pct":    round(area / item_area * 100, 2),
            "kind":        kind,
            "circularity": round(circularity, 2),
            "facet_frac":  round(facet_frac, 3),
        })
    regions.sort(key=lambda r: -r["area_pct"])
    return keep, regions


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


def _gems_overlay(
    bgr: np.ndarray, item: np.ndarray, gem_mask: np.ndarray,
    cls_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Presentation stage: backdrop dimmed, coloured gems outlined and
    numbered, colourless candidates outlined in white with a '?' — those
    need officer confirmation."""
    overlay = bgr.copy()
    overlay[item == 0] //= 3
    hch = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0]
    n, labels, cc_stats, cents = cv2.connectedComponentsWithStats(gem_mask, connectivity=8)
    idx = 0
    order = sorted(range(1, n), key=lambda i: -cc_stats[i, cv2.CC_STAT_AREA])
    for i in order:
        idx += 1
        region = (labels == i).astype(np.uint8) * 255
        colour = GEM_OUTLINE_BGR[_classify_hue(float(np.median(hch[labels == i])))]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, colour, 2)
        cx, cy = map(int, cents[i])
        cv2.putText(overlay, str(idx), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if cls_mask is not None:
        n2, labels2, cc2, cents2 = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for i in sorted(range(1, n2), key=lambda i: -cc2[i, cv2.CC_STAT_AREA]):
            idx += 1
            region = (labels2 == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
            cx, cy = map(int, cents2[i])
            cv2.putText(overlay, f"{idx}?", (cx - 9, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"{idx}?", (cx - 9, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


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


def _run_pipeline(
    bgr: np.ndarray,
    t1: Optional[int] = None,
    t2: Optional[int] = None,
    t3: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return ({stage_name: BGR image}, stats). All stats are computed on the
    item only (background removed via known-backdrop distance), and dark
    inclusion regions are cross-validated against independently detected gem
    candidates — a text claim of 'stones' can never explain an anomaly."""
    grey = _bt709_grey(bgr)
    inverted = 255 - grey

    item, background_removed = _item_mask(bgr)
    item_bool = item > 0
    item_area = max(1, int(item_bool.sum()))

    gem_cand, gems = _gem_candidate_mask(bgr, item)
    gem_candidate_regions = len(gems)
    gem_area_pct = round(sum(g["area_pct"] for g in gems), 2)

    cls_mask, colourless = _colourless_candidates(bgr, grey, item, gem_cand)
    colourless_area_pct = round(sum(c["area_pct"] for c in colourless), 2)

    otsu = _masked_otsu(grey, item)
    a1, a2, a3 = int(otsu * 0.55), otsu, int(otsu + (255 - otsu) * 0.55)
    t1, t2, t3 = t1 or a1, t2 or a2, t3 or a3
    if not 0 < t1 < t2 < t3 < 255:
        raise ValueError(f"Thresholds must satisfy 0 < T1 < T2 < T3 < 255, got {t1}, {t2}, {t3}")

    classes = np.digitize(grey, [t1, t2, t3]).astype(np.uint8)  # 0..3
    quantised = (classes * 85).astype(np.uint8)                 # 0/85/170/255 for display

    sobel = _sobel_magnitude(grey)
    edges = sobel > EDGE_THRESHOLD

    material_rgb = np.zeros((*grey.shape, 3), dtype=np.uint8)
    for cls, colour in CLASS_COLOURS_RGB.items():
        material_rgb[classes == cls] = colour
    material_rgb[edges] = (255, 255, 255)
    material_rgb[~item_bool] = (45, 45, 45)          # backdrop excluded from analysis
    material = cv2.cvtColor(material_rgb, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(grey, cv2.COLORMAP_JET)

    # Dark inclusion regions (class 0) inside the item, cross-validated
    # against the independent gem detection: overlapping = explained stone,
    # non-overlapping = unexplained anomaly (solder plug / exposed base metal).
    incl_mask = ((classes == 0) & item_bool).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    incl_mask = cv2.morphologyEx(incl_mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(incl_mask, connectivity=8)
    min_area = MIN_GEM_AREA_FRAC * item_area
    explained, unexplained, unexplained_px = 0, 0, 0
    for i in range(1, n_labels):
        area = cc_stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        region = labels == i
        overlap = float((gem_cand[region] > 0).mean())
        if overlap >= INCLUSION_OVERLAP_EXPLAINED:
            explained += 1
        else:
            unexplained += 1
            unexplained_px += int(area)

    composition = {
        CLASS_NAMES[cls]: round(float((classes[item_bool] == cls).sum()) / item_area * 100, 1)
        for cls in CLASS_NAMES
    }

    stages = {
        "original":  bgr,
        "grey":      cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR),
        "invert":    cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR),
        "threshold": cv2.cvtColor(quantised, cv2.COLOR_GRAY2BGR),
        "sobel":     cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR),
        "material":  material,
        "gems":      _gems_overlay(bgr, item, gem_cand, cls_mask),
        "heatmap":   heatmap,
        "histogram": _histogram_image(grey, item, t1, t2, t3),
    }
    stats = {
        "thresholds":             {"t1": t1, "t2": t2, "t3": t3},
        "composition":            composition,
        "background_removed":     background_removed,
        "item_area_pct":          round(item_area / classes.size * 100, 1),
        "gem_regions":            gem_candidate_regions,
        "gems":                   gems,
        "gem_area_pct":           gem_area_pct,
        "colourless_regions":     len(colourless),
        "colourless":             colourless,
        "colourless_area_pct":    colourless_area_pct,
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
    PNG_STAGES = {"threshold", "material", "histogram"}
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
