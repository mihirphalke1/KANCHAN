"""
Spatial acoustic mapping — multi-point tap-test grid.

A single tap gives one ring frequency: a bulk-average signal. It cannot see
a LOCALIZED fraud — a genuine gold shell with a hidden dense core (tungsten,
lead) fused into only part of the item, where the overall density still
lands close enough to pass. Tapping the item at several distinct points and
comparing ring frequency ACROSS taps closes this gap: a solid, homogeneous
genuine item rings at a consistent pitch everywhere; a composite item rings
position-dependently — a tap directly over a hidden core resonates
differently than a tap elsewhere on the same piece.

This is a capture-protocol extension of the existing single-tap acoustic
modality (extract_ring_frequency / ring_frequency_check in
acoustic_physics.py) — same physics, same usability gates, no new model.
"""
import logging

from app.models.acoustic_physics import extract_ring_frequency

logger = logging.getLogger(__name__)

# Spread threshold: (max-min)/mean dominant frequency across taps that each
# individually cleared the usability gate. Two taps on a genuine solid item
# vary only by minor geometry/impact-point effects (a few percent); a hidden
# partial core produces a larger, structured spread. Calibration-gated like
# the single-tap ring check — informational until measured on a known-
# genuine multi-tap reference panel.
SPREAD_FLAG_RATIO = 0.12


def analyze_spatial_taps(taps: list[dict]) -> dict:
    """
    `taps`: [{"position": str, "audio_bytes": bytes}, ...] — 2 to 6 tap
    recordings at distinct, evaluator-labelled points on the item.
    """
    if not taps or len(taps) < 2:
        return {
            "risk_score": 0.5, "confidence": "low", "mode": "insufficient_taps",
            "taps": [], "n_usable": 0,
        }

    results = []
    for t in taps:
        try:
            ring = extract_ring_frequency(t["audio_bytes"])
        except Exception as e:
            logger.warning("Spatial tap extraction failed for %s: %s", t.get("position"), e)
            ring = {"dominant_freq_hz": None, "snr_db": None, "quality": "error"}
        results.append({"position": t.get("position", "?"), **ring})

    usable = [r for r in results if r.get("quality") == "usable" and r.get("dominant_freq_hz")]
    if len(usable) < 2:
        return {
            "risk_score": 0.5, "confidence": "low", "mode": "insufficient_usable_taps",
            "taps": results, "n_usable": len(usable),
        }

    freqs  = [r["dominant_freq_hz"] for r in usable]
    f_mean = sum(freqs) / len(freqs)
    spread_ratio = (max(freqs) - min(freqs)) / f_mean if f_mean else 0.0
    flag = spread_ratio > SPREAD_FLAG_RATIO
    risk = min(1.0, max(0.0, (spread_ratio - SPREAD_FLAG_RATIO) / SPREAD_FLAG_RATIO)) if flag else 0.0

    high_point = max(usable, key=lambda r: r["dominant_freq_hz"])
    low_point  = min(usable, key=lambda r: r["dominant_freq_hz"])

    return {
        "risk_score": round(risk, 4),
        "confidence": "medium" if len(usable) >= 3 else "low",
        "mode":       "spatial_map",
        "taps":       results,
        "n_usable":   len(usable),
        "mean_freq_hz":   round(f_mean, 1),
        "spread_ratio":   round(spread_ratio, 4),
        "spatial_inconsistency_flag": flag,
        "reason": (
            f"Ring pitch varies {spread_ratio*100:.1f}% across {len(usable)} tap points "
            f"(highest at '{high_point['position']}' {high_point['dominant_freq_hz']:.0f} Hz, "
            f"lowest at '{low_point['position']}' {low_point['dominant_freq_hz']:.0f} Hz) — "
            "a position-dependent stiffness change suggests a localized hidden core, not a uniform alloy."
            if flag else
            f"Ring pitch is consistent across {len(usable)} tap points (spread {spread_ratio*100:.1f}%) — "
            "no evidence of a localized hidden core."
        ),
    }
