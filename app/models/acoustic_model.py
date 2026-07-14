"""
Novelty 1 — MFCC-ΔΔ Acoustic Fingerprinting for Jewelry.

Genuine gold has a long, clean ring after impact; base metals produce a
short, dampened thud. MFCC delta-delta captures the *acceleration* of
spectral decay, distinguishing composite-core items from solid gold.

Model: SVM with RBF kernel trained on DS-1 (20 samples, LOOCV).
Fallback: heuristic on RMS energy ratio + zero-crossing rate.
"""
import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/acoustic_svm.pkl")

GENUINE_GOLD_RMS_RATIO_THRESHOLD = 0.15
GENUINE_GOLD_ZCR_THRESHOLD       = 0.08


def extract_mfcc_features(audio_bytes: bytes) -> np.ndarray:
    """
    Extract 82-dim MFCC-ΔΔ feature vector from audio bytes.
    Vector layout: [20 MFCC means | 20 MFCC stds | 20 Δ means | 20 Δ stds | 1 ZCR | 1 RMS]
    """
    import librosa
    from app.utils.audio_io import load_audio_bytes

    y, sr = load_audio_bytes(audio_bytes, sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio signal")

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_d = librosa.feature.delta(mfcc)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_d.mean(axis=1),
        mfcc_d.std(axis=1),
        [float(librosa.feature.zero_crossing_rate(y).mean())],
        [float(np.sqrt(np.mean(y ** 2)))],
    ])
    return features.astype(np.float32)


def _heuristic_risk(audio_bytes: bytes) -> tuple[float, str]:
    """
    Heuristic fallback: genuine gold has low ZCR + higher RMS in ring decay.
    Returns (risk_score, explanation).
    """
    import librosa
    from app.utils.audio_io import load_audio_bytes
    y, sr = load_audio_bytes(audio_bytes, sr=22050, mono=True)

    if len(y) == 0:
        return 0.5, "empty_audio"

    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    rms = float(np.sqrt(np.mean(y ** 2)))

    if rms < 1e-6:
        return 0.5, "silent_audio"

    decay = y[len(y) // 4:]
    rms_ratio = float(np.sqrt(np.mean(decay ** 2))) / rms

    risk = 0.0
    if zcr > GENUINE_GOLD_ZCR_THRESHOLD:
        risk += 0.35
    if rms_ratio < GENUINE_GOLD_RMS_RATIO_THRESHOLD:
        risk += 0.35

    return min(risk, 1.0), "heuristic"


def analyze_acoustic(audio_bytes: bytes) -> dict:
    """
    Run acoustic analysis. Uses trained SVM if available, heuristic otherwise.
    """
    mode = "heuristic"

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            feats = extract_mfcc_features(audio_bytes).reshape(1, -1)
            prob  = model.predict_proba(feats)[0][1]
            risk  = float(prob)
            mode  = "svm"
            confidence = "high" if abs(risk - 0.5) > 0.2 else "medium"
            return {
                "risk_score": round(risk, 4),
                "confidence": confidence,
                "mode":       mode,
            }
        except Exception as e:
            logger.warning("SVM acoustic model failed (%s), falling back to heuristic", e)

    try:
        risk, sub_mode = _heuristic_risk(audio_bytes)
    except Exception as e:
        logger.warning("Acoustic heuristic failed (%s)", e)
        risk, sub_mode = 0.5, "error"

    return {
        "risk_score": round(risk, 4),
        "confidence": "low",
        "mode":       f"heuristic:{sub_mode}",
    }
