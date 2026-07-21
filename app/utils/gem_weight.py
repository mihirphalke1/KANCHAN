"""Estimated gem carat weight from measured stone size — the novel weight
USP of the Gold-vs-Gems feature.

When a fiducial calibration card gives real-world scale (px_per_mm), each
detected stone's pixel size is converted to a face-up area in mm², then to an
estimated carat weight via a physically-grounded volume model:

    volume_mm³ ≈ face_area_mm² × depth_mm × fill_factor
    depth_mm   ≈ DEPTH_RATIO × equivalent_face_diameter
    weight_g   = volume_mm³ / 1000 × specific_gravity
    weight_ct  = weight_g / 0.2

Specific gravity is looked up per NAMED stone (ruby, emerald, coral, jade, ...)
from app/utils/xray.py's colour classification, not a single blended value per
hue bucket — see STONE_SG_BY_NAME. A "red" stone used to be priced as if it
were ruby (SG 4.00) even when it was coral (SG 2.65, ~50% lighter); a
colourless stone used to get one arbitrary point value even though diamond
(3.52) and an undetected CZ simulant (5.90) can't be told apart by colour
alone — that ambiguity is now reflected honestly as a wide low/high range
instead of a falsely precise point estimate.

The estimate is reported as a RANGE (material SG and cut depth both swept over
a plausible band) with an explicit caveat — it assumes a typical cut and is
never a substitute for unmounting. This is the SECONDARY, volumetric method
for net-gold-weight deduction (app/utils/ltv.compute_net_gold_weight prefers
the bounded jeweller trade-deduction table when a calibration card is present,
falling back to this volumetric estimate — capped — only when it isn't).

Without a card there is no scale, so carats are omitted and only the stone's
relative size (% of item) is reported — never a fabricated number.
"""
import logging
import math
import os

logger = logging.getLogger(__name__)

# Specific gravity (g/cm³) by legacy hue class — mirrors STONE_DENSITIES in
# app/utils/composition.py. This is a LAST-RESORT fallback only, used when the
# stone couldn't be matched to a specific named reference (see
# STONE_SG_BY_NAME below) — averaging every stone in a hue bucket to one
# number is what let a coral (2.65) get weighed as if it were a ruby (4.00), a
# ~50% overestimate. Kept local so this module has no import cycle.
STONE_SG = {
    "red":        4.00,   # ruby / garnet (corundum-ish)
    "blue":       4.00,   # sapphire
    "green":      2.72,   # emerald (beryl)
    "other":      3.00,   # generic coloured stone
    "colourless": 3.30,   # unconfirmed diamond/pearl/CZ midpoint
}
DEFAULT_SG = 3.00

# Named-stone specific gravity (GIA/Webster gemological references), keyed by
# app/utils/xray.py's `stone_name` (the finer 16-way colour match, not the
# 4-way legacy bucket). This is the actual fix for over/under-estimated stone
# weight: colour reliably tells a ruby from a coral, an emerald from a jade,
# a topaz from a citrine — lumping them into one hue-bucket SG was silently
# assuming every "red" stone is as dense as ruby (a coral bead came out ~50%
# overweight) and every "green"/"blue"/"other" stone was priced off a single
# arbitrary member of its bucket.
#
# Each entry is (sg_low, sg, sg_high):
#   - For colourless stones, colour ALONE cannot distinguish diamond from a
#     cubic-zirconia simulant (or, less commonly, white sapphire) without a
#     thermal/electrical diamond tester — that's real gemology, not a gap in
#     this code — so "diamond" is given an honestly WIDE range spanning
#     genuine diamond (3.52) through an undetected CZ (5.90) rather than a
#     false-precision point estimate. Pearl's warmer tone is colour-distinct
#     from diamond/CZ/white sapphire, so it gets a tight range.
#   - For colour-identifiable stones (ruby, garnet, coral, emerald, jade,
#     peridot, sapphire, aquamarine, turquoise, amethyst, topaz, citrine,
#     onyx) the range reflects normal natural variation within that specific
#     material, not cross-material ambiguity.
STONE_SG_BY_NAME = {
    "diamond":        (3.52, 3.52, 5.90),
    "white_sapphire": (3.97, 4.00, 4.05),
    "pearl":          (2.60, 2.71, 2.85),
    "ruby":           (3.97, 4.00, 4.05),
    "garnet":         (3.60, 3.95, 4.30),
    "coral":          (2.60, 2.65, 2.70),
    "emerald":        (2.68, 2.76, 2.80),
    "jade":           (2.95, 3.15, 3.35),
    "peridot":        (3.28, 3.34, 3.48),
    "sapphire":       (3.97, 4.00, 4.05),
    "aquamarine":     (2.68, 2.72, 2.80),
    "turquoise":      (2.60, 2.76, 2.90),
    "amethyst":       (2.63, 2.65, 2.66),
    "topaz":          (3.49, 3.53, 3.57),
    "citrine":        (2.63, 2.65, 2.66),
    "onyx":           (2.58, 2.65, 2.70),
}
# Minimum colour-match confidence (from xray._classify_stone_color) required to
# trust the specific named-stone SG over the coarser hue-bucket fallback — a
# low-confidence name is a weak guess and the wider bucket average is safer.
STONE_NAME_SG_MIN_CONFIDENCE = float(os.getenv("GEM_STONE_NAME_MIN_CONF", "0.30"))


def _sg_range_for(hue: str, stone_name: str, match_confidence: float) -> tuple[float, float, float]:
    """Resolve (sg_low, sg, sg_high) for a stone: prefer the named-stone
    reference when the colour match is confident enough, else the hue-bucket
    fallback (as a flat range, since the bucket midpoint is all we have)."""
    if (stone_name in STONE_SG_BY_NAME
            and (match_confidence or 0.0) >= STONE_NAME_SG_MIN_CONFIDENCE):
        return STONE_SG_BY_NAME[stone_name]
    sg = STONE_SG.get(hue, DEFAULT_SG)
    return (sg, sg, sg)

# Cut-geometry assumptions (all env-overridable, all documented).
DEPTH_RATIO = float(os.getenv("GEM_DEPTH_RATIO", "0.62"))   # total depth / face diameter
FILL_FACTOR = float(os.getenv("GEM_FILL_FACTOR", "0.58"))   # pavilion/crown taper vs a prism
DEPTH_RATIO_LOW = float(os.getenv("GEM_DEPTH_RATIO_LOW", "0.50"))
DEPTH_RATIO_HIGH = float(os.getenv("GEM_DEPTH_RATIO_HIGH", "0.75"))
G_PER_CARAT = 0.2


def _carat_for(area_mm2: float, sg: float, depth_ratio: float) -> float:
    """Face area (mm²) + SG + a depth ratio → estimated carat weight."""
    eq_diameter = 2.0 * math.sqrt(max(area_mm2, 0.0) / math.pi)
    depth_mm = depth_ratio * eq_diameter
    volume_mm3 = area_mm2 * depth_mm * FILL_FACTOR
    weight_g = volume_mm3 / 1000.0 * sg
    return weight_g / G_PER_CARAT


def estimate_gem_weights(stones, px_per_mm=None, declared_karat=0):
    """See module docstring. `stones` is the DSIP stones list."""
    stones = stones or []
    out_stones = []
    have_scale = bool(px_per_mm) and px_per_mm > 0
    total = total_lo = total_hi = 0.0

    for s in stones:
        hue = s.get("hue_class", "other")
        stone_name = s.get("stone_name", "unidentified")
        match_conf = s.get("match_confidence", 0.0)
        sg_lo, sg, sg_hi = _sg_range_for(hue, stone_name, match_conf)
        entry = {
            "hue_class": hue,
            "stone_name": stone_name,
            "status": s.get("status"),
            "area_pct": s.get("area_pct"),
            "sg": sg,
            "sg_low": sg_lo,
            "sg_high": sg_hi,
            "diameter_mm": None,
            "est_carat": None,
            "est_carat_low": None,
            "est_carat_high": None,
        }
        if have_scale:
            bx, by, bw, bh = s.get("bbox", [0, 0, 0, 0])
            # Equivalent face area from the bbox, discounted for the fact that
            # a stone does not fill its bounding rectangle (ellipse ≈ 0.785).
            area_px = bw * bh * math.pi / 4.0
            area_mm2 = area_px / (px_per_mm ** 2)
            entry["diameter_mm"] = round(((bw + bh) / 2.0) / px_per_mm, 2)
            # Low/high sweeps BOTH cut depth and material SG together — for a
            # colour-ambiguous colourless stone (diamond vs. an undetected CZ),
            # the SG spread is the dominant uncertainty and previously wasn't
            # reflected in the range at all (only depth ratio varied, so a CZ
            # misread as "diamond" landed the point estimate ~40% under truth
            # with a low/high band that never widened to admit it).
            ct = _carat_for(area_mm2, sg, DEPTH_RATIO)
            ct_lo = _carat_for(area_mm2, sg_lo, DEPTH_RATIO_LOW)
            ct_hi = _carat_for(area_mm2, sg_hi, DEPTH_RATIO_HIGH)
            entry["est_carat"] = round(ct, 3)
            entry["est_carat_low"] = round(ct_lo, 3)
            entry["est_carat_high"] = round(ct_hi, 3)
            total += ct
            total_lo += ct_lo
            total_hi += ct_hi
        out_stones.append(entry)

    if have_scale:
        note = ("Estimated from photo size + calibration card. Assumes a typical "
                "cut depth — approximate, not a substitute for unmounting.")
    else:
        note = ("No calibration card in the photo, so real-world size is unknown. "
                "Showing each stone's relative size only; carat weight needs the card.")

    return {
        "scale_source": "fiducial_card" if have_scale else None,
        "px_per_mm": px_per_mm if have_scale else None,
        "stones": out_stones,
        "total_carat": round(total, 3) if have_scale else None,
        "total_carat_low": round(total_lo, 3) if have_scale else None,
        "total_carat_high": round(total_hi, 3) if have_scale else None,
        "n_stones": len(out_stones),
        "note": note,
    }
