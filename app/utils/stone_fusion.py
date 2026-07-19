"""
Strength-based reconciliation of the ML/CV stone detector (app/utils/xray.py +
MobileSAM) with the AI vision detector (app/llm/gem_vision.py).

The AI vision model is the PRIMARY judge of what a stone is and how many there
are; the ML/CV detector is the SECONDARY, supporting source (pixel-precise
boundaries + a safety net that keeps a stone the AI missed from vanishing).
Each source is still trusted where it is strong:
  - AI vision (Kimi/Gemini)   -> AUTHORITATIVE COUNT & gem TYPE/COLOUR.
  - MobileSAM / classical CV  -> pixel-precise BOUNDARY & location; a demoted
                                 fallback for stones the AI did not name.

Fusion table (AI-primary; STONE_AI_PRIMARY=1, the default — see
docs/superpowers/specs/2026-07-18-accurate-stone-detection-design.md):

  situation                | boundary        | type/colour | status/badge
  -------------------------+-----------------+-------------+---------------------------
  matched (both saw it)    | ML/SAM mask     | AI          | both    -> confirmed
  AI-only (ML missed it)   | SAM@point/bbox  | AI          | ai_only -> confirmed*
  ML-only (AI missed it)   | ML/SAM mask     | classifier  | ml_only -> demoted to
                           |                 |             |            uncertain (kept)

  * AI-only is confirmed when the AI is at least STONE_AI_CONFIRM_MIN confident
    AND the detection is physically plausible (see guards). Below that floor it
    stays "uncertain". The AI's SEMANTIC call is trusted; only physical
    impossibility is rejected.

Physical guards (about geometry, NOT about distrusting the AI's judgment):
  - AI-only stones must fall inside the item and be a plausible size, else dropped
    (this only stops the model boxing the whole ornament as one "stone").
  - A below-uncertain ML "candidate" only surfaces if the AI confirms it.

STRICT AI-ONLY (STONE_AI_ONLY=1, the default): the stone SET is exactly what the
AI reports. An ML detection the AI did NOT corroborate is DROPPED, not demoted —
the ML/CV pass only lends its precise boundary to stones the AI also saw. This is
"strictly AI-based": the AI alone decides how many stones there are and what each
one is. Crucially it also means AI detection does NOT depend on the ML background
separation — a stone the AI sees in the raw photo is reported even when the
backdrop could not be removed (e.g. a busy/leafy background). Set STONE_AI_ONLY=0
to instead KEEP an uncorroborated ML stone as demoted secondary evidence.

Set STONE_AI_PRIMARY=0 to fall back to the older balanced "best-of-both" policy
where AI-only stones are always flagged and only colourless ML-only is demoted.

Pure & deterministic; never performs I/O and never raises to the request path.
`ai_stones is None` (AI layer off/failed) => ML-only passthrough, labels & drawn
stones identical to today plus an `agreement="ml_only"` tag.
"""
import logging
import os
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# An AI-only detection larger than this fraction of the item is almost
# certainly the model boxing the whole ornament, not one stone — dropped.
AI_ONLY_MAX_AREA_FRAC = 0.60
# Minimum IoU of AI bbox vs ML bbox to count as the same stone when the AI
# centre does not land inside an ML mask.
MATCH_IOU_MIN = 0.30
# Nearest-centroid fallback: a vision model estimates a stone's centre only
# approximately (often off by more than the stone's own size), so the match
# distance scales with the larger stone's radius (MATCH_DIST_K × radius) —
# but is CAPPED at a fraction of the item's short side so a single huge
# (glare) ML region cannot swallow a distant AI point.
MATCH_DIST_K = float(os.getenv("STONE_MATCH_DIST_K", "2.5"))
MATCH_DIST_CAP_FRAC = float(os.getenv("STONE_MATCH_DIST_CAP_FRAC", "0.25"))
# Colourless (glare-prone) is ML's weakest class. When the AI vision check has
# run and did NOT corroborate a colourless ML detection, it is downgraded from
# "confirmed" to flagged rather than asserted — coloured ML-only stones are
# kept as-is (AI can miss a small coloured stone; ML colour there is reliable).
DOWNGRADE_UNCONFIRMED_COLOURLESS = os.getenv("STONE_DOWNGRADE_UNCONFIRMED_COLOURLESS", "1") != "0"

# ── AI-primary policy ────────────────────────────────────────────────────────
# The AI vision model is the authoritative judge of count + type/colour; the ML
# detector is secondary. Set STONE_AI_PRIMARY=0 for the older balanced policy.
AI_PRIMARY = os.getenv("STONE_AI_PRIMARY", "1") != "0"
# STRICT AI-ONLY: when the AI vision layer runs, the stone SET is exactly what
# the AI reports — the ML detector is used ONLY to refine the boundary of a
# stone the AI also saw (a matched pair). ML detections the AI did NOT
# corroborate are dropped entirely, not merely demoted. This is what "strictly
# AI-based stone detection" means: the AI decides how many stones there are and
# what each one is; the ML/CV pass contributes pixels, never opinions.
# Set STONE_AI_ONLY=0 to keep uncorroborated ML stones as secondary evidence
# (the AI-primary "demote but keep" policy).
AI_ONLY = os.getenv("STONE_AI_ONLY", "1") != "0"
# An AI-only stone at/above this confidence (and physically plausible) is trusted
# as CONFIRMED rather than merely flagged. Below it, it stays "uncertain".
AI_CONFIRM_MIN = float(os.getenv("STONE_AI_CONFIRM_MIN", "0.45"))


def _equiv_radius(bbox: Optional[list]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return 0.5 * max(float(bbox[2]), float(bbox[3]))


def _point_in_bbox(px, py, bbox, margin=0.0) -> bool:
    if px is None or not bbox or len(bbox) < 4:
        return False
    x, y, w, h = bbox
    mx, my = margin * w, margin * h
    return (x - mx) <= px <= (x + w + mx) and (y - my) <= py <= (y + h + my)


def _bbox_iou(a: Optional[list], b: Optional[list]) -> float:
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix), max(0.0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match(ai: dict, ml_stones: list[dict], labels: np.ndarray,
           item_short_side: float) -> Optional[int]:
    """Return the index in ml_stones the AI detection corresponds to, or None.
    Priority: AI centre in an ML mask -> AI centre in an ML bbox (or vice
    versa) -> best bbox IoU -> nearest centroid within a size-aware, capped
    radius (tolerates the vision model's approximate centres without letting a
    huge glare region capture distant points)."""
    H, W = labels.shape
    cx, cy = ai.get("centroid", [None, None])[:2]
    ai_bbox = ai.get("bbox")
    r_ai = _equiv_radius(ai_bbox)
    dist_cap = MATCH_DIST_CAP_FRAC * item_short_side if item_short_side > 0 else max(H, W)

    # 1. AI centre inside an ML mask (strongest signal) — unless that ML region
    #    is far larger than the AI's own box, i.e. a small stone the AI pointed
    #    at merely overlaps a big glare/metal blob the ML pass over-segmented.
    if cx is not None and 0 <= int(cy) < H and 0 <= int(cx) < W:
        lab = int(labels[int(cy), int(cx)])
        if lab > 0:
            for idx, s in enumerate(ml_stones):
                if s.get("_label") == lab:
                    r_ml = _equiv_radius(s.get("bbox"))
                    if r_ml <= 3.0 * max(r_ai, 1.0):
                        return idx
                    break   # too big -> fall through to the other match tiers

    # 2. bbox containment either direction (AI centre in ML bbox, or ML centre
    #    in AI bbox) — but reject if the ML region is far larger than the AI
    #    box (a huge glare blob whose bbox merely happens to contain the point).
    for idx, s in enumerate(ml_stones):
        r_ml = _equiv_radius(s.get("bbox"))
        too_big = r_ml > 3.0 * max(r_ai, 1.0)
        if too_big:
            continue
        mc = s.get("centroid")
        if _point_in_bbox(cx, cy, s.get("bbox"), margin=0.25):
            return idx
        if mc and _point_in_bbox(mc[0], mc[1], ai_bbox, margin=0.25):
            return idx

    # 3. best bbox IoU
    best_idx, best_iou = None, MATCH_IOU_MIN
    for idx, s in enumerate(ml_stones):
        iou = _bbox_iou(ai_bbox, s.get("bbox"))
        if iou >= best_iou:
            best_idx, best_iou = idx, iou
    if best_idx is not None:
        return best_idx

    # 4. nearest centroid within a size-aware, capped radius
    if cx is None:
        return None
    best_idx, best_d = None, None
    for idx, s in enumerate(ml_stones):
        mc = s.get("centroid")
        if not mc:
            continue
        r_ml = _equiv_radius(s.get("bbox"))
        allow = min(MATCH_DIST_K * max(r_ml, r_ai, 4.0), dist_cap)
        d = float(np.hypot(mc[0] - cx, mc[1] - cy))
        if d <= allow and (best_d is None or d < best_d):
            best_idx, best_d = idx, d
    return best_idx


def _combined_both(ml_conf: float, ai_conf: float) -> float:
    if AI_PRIMARY:
        # AI leads; ML agreement adds a smaller corroboration bonus. Both saw it,
        # so keep a solid floor.
        return round(min(1.0, 0.55 + 0.35 * ai_conf + 0.10 * ml_conf), 3)
    return round(min(1.0, 0.55 + 0.45 * max(ml_conf, ai_conf)), 3)


def _ai_only_mask(ai: dict, item_bool: np.ndarray,
                  ai_mask_fn: Optional[Callable[[dict], Optional[np.ndarray]]]) -> Optional[np.ndarray]:
    """Precise mask for an AI-only stone: prefer the SAM-at-point callback,
    else the AI bbox (a small disc around the centre if no bbox), clipped to
    the item."""
    if ai_mask_fn is not None:
        try:
            m = ai_mask_fn(ai)
        except Exception as e:  # noqa: BLE001 — never a hard block
            logger.warning("ai_mask_fn failed (%s) — using bbox mask", e)
            m = None
        if m is not None and m.any():
            return m & item_bool
    H, W = item_bool.shape
    mask = np.zeros((H, W), bool)
    bb = ai.get("bbox")
    if bb and bb[2] > 0 and bb[3] > 0:
        x, y, w, h = bb
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        mask[y0:y1, x0:x1] = True
    else:
        cx, cy = ai.get("centroid", [None, None])[:2]
        if cx is None:
            return None
        r = max(3, int(min(H, W) * 0.03))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return mask & item_bool


def reconcile(
    ml_stones: list[dict],
    labels: np.ndarray,
    ai_stones: Optional[list[dict]],
    item_bool: np.ndarray,
    ai_mask_fn: Optional[Callable[[dict], Optional[np.ndarray]]] = None,
) -> tuple[list[dict], np.ndarray, dict]:
    """Reconcile ML and AI stone detections. See module docstring for rules.

    ml_stones : full labelled set from xray._detect_stones (includes below-
                uncertain "candidate" stones, which surface only if AI confirms).
    labels    : int label map; region of ml stone s is (labels == s["_label"]).
    ai_stones : from gem_vision.detect_gems, or None when the AI layer is off.
    item_bool : item mask (H,W bool).
    ai_mask_fn: optional SAM-at-point callback for AI-only boundaries.

    Returns (stones, labels_out, meta).
    """
    item_area = max(1, int(item_bool.sum()))
    ys, xs = np.where(item_bool)
    if len(xs):
        item_short_side = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    else:
        item_short_side = float(min(item_bool.shape))
    drawn = [s for s in ml_stones if s.get("status") != "candidate"]

    # ── AI layer off/failed: ML-only passthrough (behaviour identical to today
    #    plus the agreement tag). Candidates never surface. ──────────────────
    if ai_stones is None:
        out = []
        for s in drawn:
            t = dict(s)
            t["agreement"] = "ml_only"
            t["ml_confidence"] = s.get("confidence", 0.0)
            t["ai_confidence"] = 0.0
            out.append(t)
        meta = {"mode": "ml_only", "ai_used": False,
                "n_both": 0, "n_ml_only": len(out), "n_ai_only": 0}
        return out, labels, meta

    labels_out = labels.copy()
    next_label = int(labels_out.max()) + 1
    matched_ml_idx: set[int] = set()
    result: list[dict] = []

    # ── AI detections: match to ML (incl. candidates) or add as AI-only ──────
    for ai in ai_stones:
        idx = _match(ai, ml_stones, labels_out, item_short_side)
        if idx is not None:
            ml = ml_stones[idx]
            matched_ml_idx.add(idx)
            ml_conf = ml.get("confidence", 0.0)
            ai_conf = ai.get("ai_confidence", 0.0)
            t = dict(ml)                        # SAM boundary + geometry
            t["agreement"] = "both"
            # The AI is the primary judge, so its confidence governs the final
            # status even for a matched pair: a stone the AI is unsure of
            # (< AI_CONFIRM_MIN) is only "uncertain", even though the ML pass
            # also fired there. This stops a low-confidence AI guess that happens
            # to coincide with an ML glare/specular blob (e.g. phantom "diamonds"
            # on a polished band) from being asserted as CONFIRMED.
            t["status"] = ("confirmed"
                           if (not AI_PRIMARY or ai.get("ai_confidence", 0.0) >= AI_CONFIRM_MIN)
                           else "uncertain")
            t["hue_class"] = ai.get("hue_class", ml.get("hue_class", "other"))
            t["stone_name"] = ai.get("gem_type", ml.get("stone_name", "stone"))
            t["gem_type"] = ai.get("gem_type", "")
            t["colour"] = ai.get("colour", "")
            t["shape"] = ai.get("shape", "")
            t["ml_confidence"] = ml_conf
            t["ai_confidence"] = ai_conf
            t["confidence"] = _combined_both(ml_conf, ai_conf)
            t.pop("below_uncertain", None)
            result.append(t)
        else:
            # AI-only — validate then add, always flagged.
            cx, cy = ai.get("centroid", [None, None])[:2]
            if cx is None:
                continue
            H, W = item_bool.shape
            if not (0 <= int(cy) < H and 0 <= int(cx) < W and item_bool[int(cy), int(cx)]):
                continue                          # centre outside the item -> reject
            mask = _ai_only_mask(ai, item_bool, ai_mask_fn)
            if mask is None or not mask.any():
                continue
            area = int(mask.sum())
            if area > AI_ONLY_MAX_AREA_FRAC * item_area:
                continue                          # boxed the whole ornament -> reject
            lbl = next_label; next_label += 1
            labels_out[mask] = lbl
            ys, xs = np.where(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            ai_conf = ai.get("ai_confidence", 0.0)
            if AI_PRIMARY:
                # AI is the authoritative judge — trust its semantic call. The
                # detection already passed the physical guards (inside item,
                # plausible size), so confirm it once the AI is confident enough;
                # only a low-confidence AI call stays flagged for review.
                ai_status = "confirmed" if ai_conf >= AI_CONFIRM_MIN else "uncertain"
                ai_final_conf = round(min(1.0, 0.45 + 0.55 * ai_conf), 3)
            else:
                ai_status = "uncertain"           # legacy: AI-only NEVER confirmed
                ai_final_conf = round(0.5 * ai_conf, 3)
            result.append({
                "_label": lbl,
                "area_pct": round(area / item_area * 100, 2),
                "hue_class": ai.get("hue_class", "other"),
                "stone_name": ai.get("gem_type", "stone"),
                "gem_type": ai.get("gem_type", ""),
                "colour": ai.get("colour", ""),
                "shape": ai.get("shape", ""),
                "match_confidence": round(ai_conf, 3),
                "confidence": ai_final_conf,
                "status": ai_status,
                "agreement": "ai_only",
                "ml_confidence": 0.0,
                "ai_confidence": ai_conf,
                "bbox": [x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)],
                "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            })

    # ── ML drawn stones the AI did not match -> ML-only. Below-uncertain
    #    "candidate" stones are skipped unless already promoted via an AI match
    #    above. Indexed against ml_stones so the matched-set test is by identity,
    #    not dict value.
    #    STRICT AI-ONLY: the AI owns the stone set, so an uncorroborated ML
    #    detection is dropped entirely (the ML pass only lent its boundary to
    #    stones the AI ALSO saw, handled in the matched branch above). ──────────
    for orig_idx, s in enumerate(ml_stones):
        if orig_idx in matched_ml_idx or s.get("status") == "candidate":
            continue
        if AI_ONLY:
            continue                       # strictly AI-based: drop ML-only stones
        t = dict(s)
        t["agreement"] = "ml_only"
        t["ml_confidence"] = s.get("confidence", 0.0)
        t["ai_confidence"] = 0.0
        t.pop("below_uncertain", None)
        if AI_PRIMARY:
            # The AI is the primary judge and did NOT corroborate this ML stone,
            # so it is secondary evidence — demote a would-be "confirmed" to
            # "uncertain" (never promote), but keep it: the AI can overlook a
            # small coloured stone and we must not lose a real one.
            if t.get("status") == "confirmed":
                t["status"] = "uncertain"
        elif (DOWNGRADE_UNCONFIRMED_COLOURLESS and t.get("hue_class") == "colourless"
                and t.get("status") == "confirmed"):
            # Legacy balanced policy: only the glare-prone colourless class is
            # demoted when the AI does not corroborate it.
            t["status"] = "uncertain"
        result.append(t)

    result.sort(key=lambda s: -s.get("area_pct", 0.0))
    meta = {
        "mode": "ml_ai",
        "ai_used": True,
        "ai_primary": AI_PRIMARY,
        "ai_only": AI_ONLY,
        "primary": "ai" if AI_PRIMARY else "balanced",
        "n_both": sum(1 for s in result if s["agreement"] == "both"),
        "n_ml_only": sum(1 for s in result if s["agreement"] == "ml_only"),
        "n_ai_only": sum(1 for s in result if s["agreement"] == "ai_only"),
    }
    return result, labels_out, meta
