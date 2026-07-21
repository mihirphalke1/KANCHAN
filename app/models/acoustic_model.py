"""
Novelty 1 — MFCC-ΔΔ Acoustic Fingerprinting for Jewelry.

Genuine gold has a long, clean ring after impact; base metals produce a
short, dampened thud. MFCC delta-delta captures the *acceleration* of
spectral decay, distinguishing composite-core items from solid gold.

Model: SVM with RBF kernel trained on DS-1 (20 samples, LOOCV).
Fallback: heuristic on RMS energy ratio + zero-crossing rate.
"""
import json
import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/acoustic_svm.pkl")
META_PATH  = Path("models/acoustic_svm_meta.json")


def _model_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read acoustic model meta: %s", e)
    # No sidecar => legacy model of unknown provenance; trust it (prior behaviour).
    return {"reliable": True, "note": "no meta sidecar — legacy model"}

GENUINE_GOLD_RMS_RATIO_THRESHOLD = 0.15
GENUINE_GOLD_ZCR_THRESHOLD       = 0.08


FEATURE_DIM = 122   # [20 MFCC μ | 20 MFCC σ | 20 Δ μ | 20 Δ σ | 20 ΔΔ μ | 20 ΔΔ σ | ZCR | RMS]


def extract_mfcc_features(audio_bytes: bytes) -> np.ndarray:
    """
    Extract the 122-dim MFCC-ΔΔ feature vector from audio bytes.

    Vector layout:
      [20 MFCC μ | 20 MFCC σ | 20 Δ μ | 20 Δ σ | 20 ΔΔ μ | 20 ΔΔ σ | 1 ZCR | 1 RMS]

    Both first-order (Δ, spectral velocity) AND second-order (ΔΔ, spectral
    acceleration) MFCC derivatives are included — the ΔΔ term is the actual
    "acceleration of spectral decay" the model is named for. Audio is band-pass
    filtered and noise-reduced (app/utils/audio_denoise.py) before extraction so
    branch ambient noise does not corrupt the fingerprint.
    """
    import librosa
    from app.utils.audio_io import load_clean_audio

    y, sr, _denoise = load_clean_audio(audio_bytes, sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio signal")

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_d.mean(axis=1),
        mfcc_d.std(axis=1),
        mfcc_dd.mean(axis=1),
        mfcc_dd.std(axis=1),
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
            meta = _model_meta()
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            feats = extract_mfcc_features(audio_bytes).reshape(1, -1)
            prob  = model.predict_proba(feats)[0][1]
            risk  = float(prob)
            if not meta.get("reliable", True):
                # Data-limited model (DS-1 absent): do NOT inject an unreliable
                # prediction into fusion. Report it for information only and stay
                # neutral so the ring-frequency physics carries the acoustic
                # verdict. See scripts/retrain_acoustic_svm.py.
                return {
                    "risk_score":     0.5,
                    "confidence":     "low",
                    "mode":           "svm_data_limited",
                    "svm_raw_risk":   round(risk, 4),
                    "svm_loocv":      meta.get("loocv"),
                    "note":           meta.get("note", "acoustic SVM data-limited — deferring to ring physics"),
                }
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
