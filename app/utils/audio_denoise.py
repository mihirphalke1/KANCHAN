"""
Tap-recording pre-processing — active noise cleanup BEFORE feature extraction.

The acoustic modality carries the tungsten/stiff-core claim, and in a real
branch (other officers, ceiling fans, HVAC hum, street noise through open
windows) a large fraction of raw recordings would fail the downstream SNR gate
and force a re-record. This module cleans the recording first, so the SNR gate
in acoustic_physics.py becomes a final safety net rather than the only defence:

  1. Band-pass to the physically relevant ring band (~500 Hz–8 kHz, the same
     BAND_HZ the ring analysis uses) — rejects low-frequency rumble (HVAC, foot
     traffic) and high-frequency hiss outside the band the jewellery rings in.
  2. Spectral-gating noise reduction (noisereduce, a stationary/non-stationary
     spectral-subtraction wrapper on librosa). The leading silence of the
     recording — the window before the tap transient — is used as the noise
     profile when it is long enough; otherwise noisereduce estimates a
     stationary profile from the whole clip.

Both the pre-cleaning and post-cleaning SNR are returned so the case record has
a debuggable trail ("why was this flagged low quality"). The cleanup is
best-effort: any failure returns the band-passed (or original) signal with a
reason, never raises — a bad clean must never be worse than no clean.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The ring band the whole acoustic pipeline works in. Kept in sync with
# acoustic_physics.BAND_HZ (imported lazily to avoid a cycle at import time).
_BAND_LOW_HZ = 500.0
_BAND_HIGH_HZ = 8000.0

# Leading-silence window used as the noise reference. A browser MediaRecorder
# tap clip almost always has >=120 ms of room tone before the impact.
_NOISE_LEAD_S = 0.20
_MIN_NOISE_LEAD_S = 0.08

# Above this pre-cleaning SNR the recording is already clean, and aggressive
# spectral-gating noise reduction would only strip discriminative timbre and
# add musical-noise artefacts (measured: it degrades genuine/composite MFCC
# separation on clean lab taps). So spectral gating is applied ONLY when the
# recording is actually noisy; the band-pass — which is physically motivated and
# helps regardless — is always applied. ACOUSTIC_DENOISE_FORCE=1 gates always.
_NOISE_GATE_SNR_DB = 30.0


def _snr_db(y: np.ndarray, sr: int) -> Optional[float]:
    """Peak-signal-to-noise-floor estimate in dB, using the leading window as
    the noise reference — the same shape acoustic_physics uses, so pre/post
    numbers are comparable."""
    if y is None or len(y) == 0:
        return None
    lead = y[: int(_MIN_NOISE_LEAD_S * sr)]
    noise = float(np.mean(np.abs(lead))) if len(lead) else 1e-9
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return None
    return round(min(20.0 * np.log10(peak / (noise + 1e-9)), 99.0), 1)


def _bandpass(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass. high is clamped below Nyquist."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    lo = max(1e-4, low / nyq)
    hi = min(0.999, high / nyq)
    if lo >= hi:
        return y
    sos = butter(4, [lo, hi], btype="band", output="sos")
    # filtfilt needs padlen < signal length; guard very short clips.
    if len(y) <= 12:
        return y
    return sosfiltfilt(sos, y).astype(np.float32)


def clean_tap_audio(y: np.ndarray, sr: int) -> Tuple[np.ndarray, dict]:
    """Clean a tap recording in place-safe fashion. Returns (y_clean, info)
    where info records the band, the noise-profile source, and pre/post SNR."""
    info = {
        "applied":       False,
        "band_hz":       [_BAND_LOW_HZ, _BAND_HIGH_HZ],
        "noise_profile": None,
        "snr_pre_db":    _snr_db(y, sr),
        "snr_post_db":   None,
        "reason":        "",
    }
    if y is None or len(y) < int(0.05 * sr):
        info["reason"] = "clip too short to clean"
        info["snr_post_db"] = info["snr_pre_db"]
        return y, info

    try:
        y_band = _bandpass(np.asarray(y, dtype=np.float32), sr, _BAND_LOW_HZ, _BAND_HIGH_HZ)
    except Exception as e:
        logger.warning("Band-pass failed (%s) — using raw signal", e)
        y_band = np.asarray(y, dtype=np.float32)
        info["reason"] = f"bandpass_failed:{e}"

    # SNR-adaptive: skip spectral gating on already-clean recordings.
    import os
    snr_pre = info["snr_pre_db"]
    force_gate = os.getenv("ACOUSTIC_DENOISE_FORCE", "0").strip() in ("1", "true", "True")
    if not force_gate and snr_pre is not None and snr_pre >= _NOISE_GATE_SNR_DB:
        info["applied"] = True
        info["noise_profile"] = "bandpass_only_clean"
        info["reason"] = f"band-pass only (SNR {snr_pre} dB already clean, spectral gating skipped)"
        info["snr_post_db"] = _snr_db(y_band, sr)
        return np.asarray(y_band, dtype=np.float32), info

    y_clean = y_band
    try:
        import noisereduce as nr

        lead_n = int(_NOISE_LEAD_S * sr)
        if len(y_band) > lead_n and lead_n >= int(_MIN_NOISE_LEAD_S * sr):
            noise_clip = y_band[:lead_n]
            y_clean = nr.reduce_noise(y=y_band, sr=sr, y_noise=noise_clip, stationary=True)
            info["noise_profile"] = "leading_silence"
        else:
            y_clean = nr.reduce_noise(y=y_band, sr=sr, stationary=True)
            info["noise_profile"] = "stationary_estimate"
        info["applied"] = True
        info["reason"] = info["reason"] or "bandpass + spectral-gating noise reduction"
    except ImportError:
        # No noisereduce — fall back to a simple stationary spectral subtraction
        # using the leading-silence magnitude spectrum as the noise floor.
        try:
            y_clean = _spectral_subtract(y_band, sr)
            info["applied"] = True
            info["noise_profile"] = "leading_silence_spectral_subtract"
            info["reason"] = info["reason"] or "bandpass + spectral subtraction (noisereduce unavailable)"
        except Exception as e:
            logger.warning("Spectral subtraction fallback failed (%s) — using band-passed signal", e)
            info["reason"] = info["reason"] or f"denoise_skipped:{e}"
    except Exception as e:
        logger.warning("noisereduce failed (%s) — using band-passed signal", e)
        y_clean = y_band
        info["reason"] = info["reason"] or f"denoise_failed:{e}"

    y_clean = np.asarray(y_clean, dtype=np.float32)
    info["snr_post_db"] = _snr_db(y_clean, sr)
    return y_clean, info


def _spectral_subtract(y: np.ndarray, sr: int) -> np.ndarray:
    """Minimal STFT spectral subtraction using the leading silence as the
    noise magnitude estimate. Pure-numpy fallback for when noisereduce isn't
    installed."""
    n_fft = 1024
    hop = n_fft // 4
    lead_n = max(int(_MIN_NOISE_LEAD_S * sr), n_fft)
    lead = y[:lead_n] if len(y) > lead_n else y

    def _stft(sig):
        win = np.hanning(n_fft)
        frames = []
        for start in range(0, max(1, len(sig) - n_fft + 1), hop):
            frames.append(np.fft.rfft(sig[start:start + n_fft] * win))
        if not frames:
            frames.append(np.fft.rfft(np.pad(sig, (0, n_fft - len(sig))) * win))
        return np.array(frames)

    noise_mag = np.mean(np.abs(_stft(lead)), axis=0)
    spec = _stft(y)
    mag = np.abs(spec)
    phase = np.angle(spec)
    clean_mag = np.maximum(mag - noise_mag[None, :], 0.05 * mag)

    win = np.hanning(n_fft)
    out = np.zeros(len(y) + n_fft)
    wsum = np.zeros(len(y) + n_fft)
    for i, start in enumerate(range(0, len(out) - n_fft, hop)):
        if i >= len(clean_mag):
            break
        frame = np.fft.irfft(clean_mag[i] * np.exp(1j * phase[i]), n=n_fft)
        out[start:start + n_fft] += frame * win
        wsum[start:start + n_fft] += win ** 2
    wsum[wsum < 1e-8] = 1.0
    return (out / wsum)[: len(y)].astype(np.float32)
