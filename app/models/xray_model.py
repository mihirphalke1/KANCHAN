"""
X-ray modality — risk scoring from the DSIP pseudo X-ray segmentation stats.

Three explainable features derived from classical image processing:
  1. Surface heterogeneity — Shannon entropy of the 4-class material
     composition. Polished genuine gold is dominated by one luminance band;
     plated/composite items segment into a patchwork of classes.
  2. Edge density — fraction of pixels above the Sobel threshold. A polished
     genuine surface is smooth; casting seams, pitting and wear on plated
     items produce dense gradient boundaries.
  3. Unexplained dark inclusions — dark regions that do NOT spatially overlap
     an independently detected gem candidate. The item description can add
     suspicion (declared stones that aren't there) but can never explain an
     anomaly away: two independent CV passes must agree before a dark region
     counts as a stone. This blocks a colluding evaluator from laundering a
     solder plug as "that's a gem".
"""
import json
import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Visual-channel fusion blend (image CNN vs DSIP), single source of truth for
# analyze.py and rebuild_fusion.py. Default: DSIP only — the CNN probe is
# proxy-trained (catalog vs steel-defect images) and carries no defensible
# evidence; deterministic classical CV is the decision-grade visual signal.
# Set VISUAL_BLEND_XRAY < 1 (and USE_CNN_PROBE=1) to re-enable the CNN.
VISUAL_BLEND_XRAY  = float(os.getenv("VISUAL_BLEND_XRAY", "1.0"))
VISUAL_BLEND_IMAGE = round(1.0 - VISUAL_BLEND_XRAY, 4)

# Keyword classes checked against the MATCHING detector — a declared coloured
# stone is verified by the colour clusterer, a declared colourless stone
# (pearl/diamond) by the shape/texture detector. Generic words match either.
COLOURED_KEYWORDS = (
    "ruby", "emerald", "sapphire", "kundan", "navratna", "navratan",
    "jadau", "meenakari", "coral", "garnet", "topaz",
)
COLOURLESS_KEYWORDS = (
    "pearl", "moti", "diamond", "heera", "solitaire", "polki", "zircon",
)
GENERIC_STONE_KEYWORDS = ("stone", "gem")

# Feature calibration. Preferred source: data/dsip_calibration.json, measured
# from a known-genuine photo panel (scripts/calibrate_dsip.py). The constants
# below are documented FALLBACKS only, used when no calibration file exists.
ENTROPY_BASELINE = 0.30   # normalised entropy of a clean single-material item
ENTROPY_SPAN     = 0.45
EDGE_BASELINE    = 2.5    # % edge pixels typical of a polished surface
EDGE_SPAN        = 9.0

_DSIP_CAL_PATH = Path("data/dsip_calibration.json")
_dsip_cal_cache: dict | None = None


def _dsip_calibration() -> dict:
    """Measured baselines when available; documented fallbacks otherwise."""
    global _dsip_cal_cache
    if _dsip_cal_cache is None:
        cal = {}
        if _DSIP_CAL_PATH.exists():
            try:
                cal = json.loads(_DSIP_CAL_PATH.read_text())
            except Exception as e:
                logger.warning("Could not read DSIP calibration: %s", e)
        _dsip_cal_cache = {
            "entropy_baseline": cal.get("entropy_baseline", ENTROPY_BASELINE),
            "entropy_span":     cal.get("entropy_span",     ENTROPY_SPAN),
            "edge_baseline":    cal.get("edge_baseline",    EDGE_BASELINE),
            "edge_span":        cal.get("edge_span",        EDGE_SPAN),
            "calibrated":       bool(cal),
        }
    return _dsip_cal_cache

W_HETEROGENEITY = 0.45
W_EDGE          = 0.35
W_INCLUSION     = 0.20


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _composition_entropy(composition: dict) -> float:
    """Normalised Shannon entropy (0..1) over the material class fractions."""
    fracs = [max(v, 0.0) / 100.0 for v in composition.values()]
    total = sum(fracs) or 1.0
    h = -sum((p / total) * math.log(p / total) for p in fracs if p > 0)
    return h / math.log(len(composition)) if len(composition) > 1 else 0.0


def analyze_xray(xray_stats: dict | None, item_description: str = "") -> dict:
    """
    Turn generate_xray() stats into a modality_score dict.
    Returns risk_score, confidence, mode, human-readable signals and features.
    """
    if not xray_stats or "composition" not in xray_stats:
        return {"risk_score": 0.5, "confidence": "low", "mode": "no_xray"}

    composition   = xray_stats["composition"]
    edge_density  = float(xray_stats.get("edge_density", 0.0))
    gem_regions   = int(xray_stats.get("gem_regions", 0))          # independent CV detection
    colourless    = xray_stats.get("colourless", [])
    cls_regions   = int(xray_stats.get("colourless_regions", 0))
    unexplained   = int(xray_stats.get("inclusions_unexplained", 0))
    explained     = int(xray_stats.get("inclusions_explained", 0))
    unexpl_pct    = float(xray_stats.get("unexplained_area_pct", 0.0))
    bg_removed    = bool(xray_stats.get("background_removed", False))

    desc = (item_description or "").lower()
    declared_coloured   = any(k in desc for k in COLOURED_KEYWORDS)
    declared_colourless = any(k in desc for k in COLOURLESS_KEYWORDS)
    declared_generic    = any(k in desc for k in GENERIC_STONE_KEYWORDS)
    stones_declared = declared_coloured or declared_colourless or declared_generic

    cal = _dsip_calibration()
    entropy = _composition_entropy(composition)
    heterogeneity_risk = _clip01((entropy - cal["entropy_baseline"]) / cal["entropy_span"])
    edge_risk = _clip01((edge_density - cal["edge_baseline"]) / cal["edge_span"])

    # A stone-set item is legitimately multi-material — but only when the
    # stones are DETECTED, never merely declared.
    if gem_regions > 0 or cls_regions > 0:
        heterogeneity_risk *= 0.4

    signals = []
    if not bg_removed:
        signals.append(
            "Background could not be separated — stats include backdrop pixels; "
            "re-photograph on a plain fixed-colour backdrop"
        )

    # Risk comes only from dark regions with NO corresponding detected gem.
    inclusion_risk = _clip01(0.20 * unexplained + unexpl_pct / 8.0)

    # Declared-but-undetected mismatches, checked against the MATCHING
    # detector. Suspicion adds; a claim can never subtract.
    if declared_coloured and gem_regions == 0:
        inclusion_risk = _clip01(inclusion_risk + 0.35)
        signals.append(
            "Description declares coloured stones but the colour clusterer found none"
        )
    if declared_colourless and cls_regions == 0:
        inclusion_risk = _clip01(inclusion_risk + 0.35)
        signals.append(
            "Description declares pearls/diamonds but no colourless stone candidates were detected"
        )
    if declared_generic and not (declared_coloured or declared_colourless) \
            and gem_regions == 0 and cls_regions == 0:
        inclusion_risk = _clip01(inclusion_risk + 0.35)
        signals.append(
            "Description declares stones but neither detector found any"
        )

    # Colourless candidates always require confirmation — undeclared ones
    # doubly so (a bright inclusion can wear a stone's clothes).
    if cls_regions > 0:
        kinds = {}
        for c in colourless:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        kind_txt = ", ".join(f"{v}× {k}" for k, v in kinds.items())
        if declared_colourless:
            signals.append(
                f"{cls_regions} colourless stone candidate(s) detected ({kind_txt}) — "
                "consistent with declared stones; stone identity not required, density range assumed"
            )
        else:
            signals.append(
                f"{cls_regions} UNDECLARED colourless stone candidate(s) detected ({kind_txt}) — "
                "officer must confirm these are stones, not bright inclusions or surface damage"
            )
    if explained > 0:
        signals.append(
            f"{explained} dark region(s) overlap independently detected gems — explained as stones"
        )
    if unexplained > 0:
        signals.append(
            f"{unexplained} dark region(s) ({unexpl_pct}% of item) have NO corresponding detected gem "
            f"— possible solder plugs or exposed base metal, regardless of description"
        )

    ref = "calibrated genuine panel" if cal["calibrated"] else "reference values"
    if heterogeneity_risk > 0.4:
        signals.append(
            f"Material mix (entropy {entropy:.2f}) is above the {ref} — surface splits into unusual luminance bands"
        )
    if edge_risk > 0.4:
        signals.append(
            f"Surface texture ({edge_density}% edges) is above the {ref} — rougher than genuine items of this kind"
        )
    if not signals:
        signals.append("Uniform segmentation consistent with a homogeneous polished surface")

    risk = _clip01(
        W_HETEROGENEITY * heterogeneity_risk
        + W_EDGE * edge_risk
        + W_INCLUSION * inclusion_risk
    )

    return {
        "risk_score": round(risk, 4),
        "confidence": "medium" if abs(risk - 0.5) > 0.25 else "low",
        "mode":       "dsip_xray",
        "signals":    signals,
        "features": {
            "composition_entropy":    round(entropy, 4),
            "heterogeneity_risk":     round(heterogeneity_risk, 4),
            "edge_risk":              round(edge_risk, 4),
            "inclusion_risk":         round(inclusion_risk, 4),
            "stones_declared":        stones_declared,
            "gem_regions_detected":   gem_regions,
            "colourless_detected":    cls_regions,
            "inclusions_explained":   explained,
            "inclusions_unexplained": unexplained,
            "background_removed":     bg_removed,
        },
    }
