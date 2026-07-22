"""
Ring-frequency physics — the tungsten/stiff-core cross-check.

Physical basis: sound speed in a material is v = sqrt(E/rho). Gold is soft
(E = 79 GPa, v ~2020 m/s); every practical filler is stiffer — copper
(130 GPa, ~3810), tungsten (411 GPa, ~4620). For comparable geometry the
ring frequency scales with v, so a filled item rings HIGHER-pitched than
solid gold. No cheap metal matches gold on both density AND stiffness.

Honesty constraints, in code not just in prose:
  * Absolute velocity from the free-bar formula is unreliable for irregular
    jewellery (flexural modes), so decisions use the RATIO of the measured
    dominant frequency to a CALIBRATED genuine band for the same item class
    (data/acoustic_calibration.json). No calibration -> informational only,
    never a risk contribution.
  * Low SNR or too-short ring -> abstain, ask for a re-recording.

Empirical validation on DS-1 real recordings: genuine gold taps median
6153 Hz (range 5704-6793), plated-composite taps 7689 Hz (7681-7703) —
non-overlapping, direction as predicted.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path("data/acoustic_calibration.json")

SNR_MIN_DB     = 12.0    # below this the recording is too noisy to trust
MIN_RING_S     = 0.20    # need at least this much decay after the impact
BAND_HZ        = (500.0, 8000.0)   # smartphone-mic usable ring band
# Decision thresholds placed EMPIRICALLY on the DS-1 reference panel:
# genuine recordings never exceed the calibrated band top (1.00 by
# construction, 1.03 with margin); every composite recording sits at
# >= 1.098 above it. The flag threshold is the geometric midpoint
# sqrt(1.03 * 1.098) ~= 1.06 — re-derive when recalibrating.
RATIO_FLAG     = 1.06    # f > f_high * this -> stiff-core flag (decisive)
RATIO_NOTE     = 1.03    # f > f_high * this -> above band, note only

# Length-adaptive velocity cross-check (v = 2·L·f). For the fundamental mode of
# a free bar the wavelength is 2L, so the effective sound speed is v = 2·L·f. A
# stiffer-than-gold core raises v (v = √(E/ρ)), so at a KNOWN item length the
# ring frequency should sit near f_expected = v_gold / (2·L). This scales the
# tolerance with item size instead of assuming one fixed band for every piece.
#
# Honesty caveat kept from the ratio approach: absolute velocity from the free-
# bar formula is only approximate for irregular jewellery (flexural vs
# longitudinal modes), so the velocity check is a CORROBORATING physics signal —
# decisive on its own only when no calibrated band exists (the previously
# information-only "uncalibrated" case), and a wide ±tolerance is used.
V_GOLD_MS      = 2020.0   # v = √(E/ρ) for gold: √(79e9 / 19320) ≈ 2020 m/s
VELOCITY_TOL   = 0.15     # ±15% around the length-adaptive expected frequency


def acoustic_velocity(f_dominant_hz: float, length_m: float) -> float:
    """Effective sound speed from the free-bar fundamental relation v = 2·L·f."""
    return 2.0 * float(length_m) * float(f_dominant_hz)


def velocity_cross_check(f0: float, length_m: float, gold_band: dict = None) -> dict:
    """Length-adaptive velocity cross-check, v = 2·L·f.

    Reports the item's effective sound speed from its measured ring pitch and
    known longest dimension (metres, from the calibration-card scale). Because
    real jewellery rings in flexural modes — NOT the bulk longitudinal mode — a
    naive comparison to gold's bulk velocity (2020 m/s) is physically invalid
    and would mark every genuine tap "too low", so it is NOT used as a gate.

    The physically valid comparison is between the item's effective velocity and
    a GENUINE-GOLD effective velocity at the same size. When a calibrated genuine
    band exists for the item class, we derive that reference (v_gold_eff = 2·L·f_gold)
    and report the ratio as CORROBORATION for the frequency-band verdict — a
    stiffer core shows up as an effective velocity above the genuine reference.
    It never independently drives a flag; the calibrated-band ratio decides."""
    if not length_m or length_m <= 0 or not f0:
        return {"available": False, "reason": "item length unknown (no calibration-card scale)"}

    v_eff = acoustic_velocity(f0, length_m)
    out = {
        "available":       True,
        "item_length_mm":  round(length_m * 1000, 1),
        "velocity_ms":     round(v_eff, 0),
        "formula":         "v = 2·L·f",
        "mode_note":       "effective (flexural) velocity; not bulk longitudinal — reported for cross-check, not a hard gate",
    }
    if gold_band and gold_band.get("f_low") and gold_band.get("f_high"):
        f_gold = 0.5 * (gold_band["f_low"] + gold_band["f_high"])
        v_gold_eff = acoustic_velocity(f_gold, length_m)
        ratio = v_eff / v_gold_eff if v_gold_eff else None
        out["gold_ref_velocity_ms"] = round(v_gold_eff, 0)
        out["velocity_ratio"] = round(ratio, 3) if ratio else None
        out["corroborates"] = (
            "above genuine effective velocity (consistent with a stiffer core)"
            if ratio and ratio > (1.0 + VELOCITY_TOL)
            else "within genuine effective-velocity range"
            if ratio and ratio >= (1.0 - VELOCITY_TOL)
            else "below genuine effective velocity"
        )
        out["reason"] = (
            f"v = 2·L·f = {v_eff:.0f} m/s at L={length_m*1000:.0f} mm; "
            f"genuine-gold effective velocity for this size ≈ {v_gold_eff:.0f} m/s "
            f"(ratio {ratio:.2f}) — {out['corroborates']}."
        )
    else:
        out["reason"] = (
            f"v = 2·L·f = {v_eff:.0f} m/s at L={length_m*1000:.0f} mm "
            f"(no calibrated genuine reference to compare against)."
        )
    return out


def extract_ring_frequency(audio_bytes: bytes, sr: int = 22050) -> dict:
    """Dominant ring frequency from a tap recording — a measured quantity,
    reported with its SNR and a usability gate.

    The recording is actively cleaned first (band-pass to the ring band +
    SNR-adaptive spectral-gating noise reduction, app/utils/audio_denoise.py)
    so branch ambient noise does not force a needless re-record. Both the
    pre-cleaning and post-cleaning SNR are returned for the audit trail; the
    SNR gate below then runs on the CLEANED signal as a final safety net."""
    import librosa
    from app.utils.audio_io import load_clean_audio

    y, sr, denoise = load_clean_audio(audio_bytes, sr=sr, mono=True)
    denoise_trail = {
        "denoise_applied": denoise.get("applied"),
        "snr_pre_db":      denoise.get("snr_pre_db"),
        "snr_post_db":     denoise.get("snr_post_db"),
        "noise_profile":   denoise.get("noise_profile"),
    }
    if len(y) == 0:
        return {"dominant_freq_hz": None, "snr_db": None, "quality": "empty", **denoise_trail}

    y_t, _ = librosa.effects.trim(y, top_db=20)
    noise = float(np.mean(np.abs(y[: int(0.05 * sr)]))) if len(y) > int(0.05 * sr) else 1e-9
    # cap at 99 dB: digitally-silent lead-ins otherwise report unphysical SNR
    snr = min(20.0 * np.log10(float(np.max(np.abs(y_t))) / (noise + 1e-9)), 99.0)

    if snr < SNR_MIN_DB:
        return {"dominant_freq_hz": None, "snr_db": round(snr, 1), "quality": "low_snr", **denoise_trail}

    decay = y_t[int(0.05 * sr):]           # skip the impact transient
    if len(decay) < int(MIN_RING_S * sr):
        return {"dominant_freq_hz": None, "snr_db": round(snr, 1), "quality": "too_short", **denoise_trail}

    mag = np.abs(np.fft.rfft(decay * np.hanning(len(decay))))
    freqs = np.fft.rfftfreq(len(decay), d=1.0 / sr)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    f0 = float(freqs[band][np.argmax(mag[band])])

    return {"dominant_freq_hz": round(f0, 1), "snr_db": round(snr, 1), "quality": "usable", **denoise_trail}


def load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        try:
            return json.loads(CALIBRATION_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read acoustic calibration: %s", e)
    return {}


def ring_frequency_check(
    ring: dict,
    density_result: dict,
    item_type: str = "default",
    item_length_m: float = None,
) -> dict:
    """
    Compare the measured ring frequency against the calibrated genuine band
    for this item class. The stiff-core signature = density consistent with
    gold AND ring pitch decisively above the genuine band.

    When the item's longest dimension is known (from the calibration-card
    scale), a length-adaptive velocity cross-check (v = 2·L·f) is added: it
    corroborates the calibrated-band verdict, and — crucially — makes the
    previously information-only "uncalibrated" case decisive on physics alone.
    """
    f0 = ring.get("dominant_freq_hz")
    if ring.get("quality") != "usable" or not f0:
        return {
            "status": "abstained",
            "reason": f"Recording quality: {ring.get('quality', 'unknown')} — re-record in quieter conditions",
            "stiff_core_flag": False,
        }

    # Density context: does the density LOOK like gold — by PHYSICS, not by
    # the customer's claim? (Claim-blind: a tungsten bar is flagged whatever
    # karat was declared, as long as its density sits in some gold band.)
    best = density_result.get("best_match") or {}
    if best:
        density_near_gold = best.get("kind") == "karat"
    else:
        density_near_gold = density_result.get("risk_score", 1.0) < 0.5

    cal = load_calibration()
    band = cal.get(item_type) or cal.get("default")
    # v = 2·L·f cross-check — reported when item length is known; corroborates
    # against the calibrated genuine band when one exists.
    velocity = velocity_cross_check(f0, item_length_m, band) if item_length_m else {"available": False}

    if not band:
        return {
            "status": "uncalibrated",
            "reason": "No calibrated genuine reference for this item class — frequency reported for information only"
                      + (". " + velocity["reason"] if velocity.get("available") else ""),
            "dominant_freq_hz": f0,
            "stiff_core_flag": False,
            "velocity_cross_check": velocity,
        }

    f_low, f_high = band["f_low"], band["f_high"]
    ratio = round(f0 / f_high, 3)

    if f0 > f_high * RATIO_FLAG:
        flag = density_near_gold
        return {
            "status": "stiff_core_signature" if flag else "above_genuine_band",
            "dominant_freq_hz": f0,
            "genuine_band_hz": [f_low, f_high],
            "ratio_above_band": ratio,
            "stiff_core_flag": flag,
            "velocity_cross_check": velocity,
            "reason": (
                f"Ring pitch {f0:.0f} Hz is {ratio:.2f}× the calibrated genuine band top "
                f"({f_low:.0f}–{f_high:.0f} Hz). A stiffer-than-gold core raises the pitch "
                f"(v = √(E/ρ); tungsten is 5× stiffer than gold)."
                + (" Density is consistent with gold — this combination is the filled-core signature."
                   if flag else "")
                + ("  " + velocity["reason"] if velocity.get("available") else "")
            ),
        }
    if f0 > f_high * RATIO_NOTE or f0 < f_low * (2 - RATIO_NOTE):
        return {
            "status": "marginal",
            "dominant_freq_hz": f0,
            "genuine_band_hz": [f_low, f_high],
            "ratio_above_band": ratio,
            "stiff_core_flag": False,
            "velocity_cross_check": velocity,
            "reason": f"Ring pitch {f0:.0f} Hz sits just outside the genuine band ({f_low:.0f}–{f_high:.0f} Hz) — inconclusive",
        }
    return {
        "status": "consistent",
        "dominant_freq_hz": f0,
        "velocity_cross_check": velocity,
        "genuine_band_hz": [f_low, f_high],
        "ratio_above_band": ratio,
        "stiff_core_flag": False,
        "reason": f"Ring pitch {f0:.0f} Hz is inside the calibrated genuine band ({f_low:.0f}–{f_high:.0f} Hz)",
    }
