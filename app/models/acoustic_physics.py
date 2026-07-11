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
import io
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


def extract_ring_frequency(audio_bytes: bytes, sr: int = 22050) -> dict:
    """Dominant ring frequency from a tap recording — a measured quantity,
    reported with its SNR and a usability gate."""
    import librosa

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    if len(y) == 0:
        return {"dominant_freq_hz": None, "snr_db": None, "quality": "empty"}

    y_t, _ = librosa.effects.trim(y, top_db=20)
    noise = float(np.mean(np.abs(y[: int(0.05 * sr)]))) if len(y) > int(0.05 * sr) else 1e-9
    snr = 20.0 * np.log10(float(np.max(np.abs(y_t))) / (noise + 1e-9))

    if snr < SNR_MIN_DB:
        return {"dominant_freq_hz": None, "snr_db": round(snr, 1), "quality": "low_snr"}

    decay = y_t[int(0.05 * sr):]           # skip the impact transient
    if len(decay) < int(MIN_RING_S * sr):
        return {"dominant_freq_hz": None, "snr_db": round(snr, 1), "quality": "too_short"}

    mag = np.abs(np.fft.rfft(decay * np.hanning(len(decay))))
    freqs = np.fft.rfftfreq(len(decay), d=1.0 / sr)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    f0 = float(freqs[band][np.argmax(mag[band])])

    return {"dominant_freq_hz": round(f0, 1), "snr_db": round(snr, 1), "quality": "usable"}


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
) -> dict:
    """
    Compare the measured ring frequency against the calibrated genuine band
    for this item class. The stiff-core signature = density consistent with
    gold AND ring pitch decisively above the genuine band.
    """
    f0 = ring.get("dominant_freq_hz")
    if ring.get("quality") != "usable" or not f0:
        return {
            "status": "abstained",
            "reason": f"Recording quality: {ring.get('quality', 'unknown')} — re-record in quieter conditions",
            "stiff_core_flag": False,
        }

    cal = load_calibration()
    band = cal.get(item_type) or cal.get("default")
    if not band:
        return {
            "status": "uncalibrated",
            "reason": "No calibrated genuine reference for this item class — frequency reported for information only",
            "dominant_freq_hz": f0,
            "stiff_core_flag": False,
        }

    f_low, f_high = band["f_low"], band["f_high"]
    ratio = round(f0 / f_high, 3)

    # Density context: does the density LOOK like gold? (that is exactly when
    # the stiff-core check matters — a fake that passes the weight test)
    density_near_gold = density_result.get("risk_score", 1.0) < 0.5

    if f0 > f_high * RATIO_FLAG:
        flag = density_near_gold
        return {
            "status": "stiff_core_signature" if flag else "above_genuine_band",
            "dominant_freq_hz": f0,
            "genuine_band_hz": [f_low, f_high],
            "ratio_above_band": ratio,
            "stiff_core_flag": flag,
            "reason": (
                f"Ring pitch {f0:.0f} Hz is {ratio:.2f}× the calibrated genuine band top "
                f"({f_low:.0f}–{f_high:.0f} Hz). A stiffer-than-gold core raises the pitch "
                f"(v = √(E/ρ); tungsten is 5× stiffer than gold)."
                + (" Density is consistent with gold — this combination is the filled-core signature."
                   if flag else "")
            ),
        }
    if f0 > f_high * RATIO_NOTE or f0 < f_low * (2 - RATIO_NOTE):
        return {
            "status": "marginal",
            "dominant_freq_hz": f0,
            "genuine_band_hz": [f_low, f_high],
            "ratio_above_band": ratio,
            "stiff_core_flag": False,
            "reason": f"Ring pitch {f0:.0f} Hz sits just outside the genuine band ({f_low:.0f}–{f_high:.0f} Hz) — inconclusive",
        }
    return {
        "status": "consistent",
        "dominant_freq_hz": f0,
        "genuine_band_hz": [f_low, f_high],
        "ratio_above_band": ratio,
        "stiff_core_flag": False,
        "reason": f"Ring pitch {f0:.0f} Hz is inside the calibrated genuine band ({f_low:.0f}–{f_high:.0f} Hz)",
    }
