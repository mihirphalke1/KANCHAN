"""Estimated gem carat weight from measured stone size — the novel weight
USP of the Gold-vs-Gems feature.

When a fiducial calibration card gives real-world scale (px_per_mm), each
detected stone's pixel size is converted to a face-up area in mm², then to an
estimated carat weight via a physically-grounded volume model:

    volume_mm³ ≈ face_area_mm² × depth_mm × fill_factor
    depth_mm   ≈ DEPTH_RATIO × equivalent_face_diameter
    weight_g   = volume_mm³ / 1000 × specific_gravity
    weight_ct  = weight_g / 0.2

The estimate is reported as a RANGE (depth ratio swept over a plausible band)
with an explicit caveat — it assumes a typical cut and is never a substitute
for unmounting. Advisory/informational ONLY: it does not feed LTV, fusion, or
the loan decision.

Without a card there is no scale, so carats are omitted and only the stone's
relative size (% of item) is reported — never a fabricated number.
"""
import logging
import math
import os

logger = logging.getLogger(__name__)

# Specific gravity (g/cm³) by legacy hue class — mirrors STONE_DENSITIES in
# app/utils/composition.py. Kept local so this module has no import cycle and
# can be reasoned about on its own.
STONE_SG = {
    "red":        4.00,   # ruby / garnet (corundum-ish)
    "blue":       4.00,   # sapphire
    "green":      2.72,   # emerald (beryl)
    "other":      3.00,   # generic coloured stone
    "colourless": 3.30,   # unconfirmed diamond/pearl/CZ midpoint
}
DEFAULT_SG = 3.00

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
        sg = STONE_SG.get(hue, DEFAULT_SG)
        entry = {
            "hue_class": hue,
            "stone_name": s.get("stone_name", "unidentified"),
            "status": s.get("status"),
            "area_pct": s.get("area_pct"),
            "sg": sg,
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
            ct = _carat_for(area_mm2, sg, DEPTH_RATIO)
            ct_lo = _carat_for(area_mm2, sg, DEPTH_RATIO_LOW)
            ct_hi = _carat_for(area_mm2, sg, DEPTH_RATIO_HIGH)
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
