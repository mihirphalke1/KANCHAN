"""
Shared audio-bytes loading — writes to a temp file before decoding.

librosa/soundfile can only sniff a compressed container (webm/opus, ogg,
m4a) from a real file path — handing the same bytes to librosa.load() as a
BytesIO buffer fails outright with "Format not recognised" even though
they decode fine from disk, because the audioread/ffmpeg fallback shells
out to ffmpeg with a filename, not a stream. Every tap recording captured
via the browser's MediaRecorder is webm/opus, so this was hitting on every
real request — not an edge case — and silently degrading every acoustic
score to the neutral 0.5 default via the existing exception handlers.
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def load_audio_bytes(audio_bytes: bytes, sr: int = 22050, mono: bool = True):
    """Decode audio bytes of any container librosa/ffmpeg supports (wav,
    webm/opus, ogg, m4a, mp3, ...) via a temp file. Returns (y, sr)."""
    import librosa

    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return librosa.load(path, sr=sr, mono=mono)
    finally:
        try:
            os.unlink(path)
        except OSError as e:
            logger.warning("Could not remove temp audio file %s: %s", path, e)


def load_clean_audio(audio_bytes: bytes, sr: int = 22050, mono: bool = True):
    """Decode a tap recording AND actively clean it (band-pass + spectral-gating
    noise reduction) before any feature extraction. Returns (y_clean, sr, info)
    where info carries pre/post SNR and the noise-profile source for the audit
    trail. Denoising is on by default; ACOUSTIC_DENOISE=0 returns the raw signal
    with info["applied"]=False so the old behaviour is one env var away."""
    y, sr = load_audio_bytes(audio_bytes, sr=sr, mono=mono)
    if os.getenv("ACOUSTIC_DENOISE", "1").strip() not in ("1", "true", "True"):
        return y, sr, {"applied": False, "reason": "disabled (ACOUSTIC_DENOISE=0)",
                       "snr_pre_db": None, "snr_post_db": None}
    try:
        from app.utils.audio_denoise import clean_tap_audio
        y_clean, info = clean_tap_audio(y, sr)
        return y_clean, sr, info
    except Exception as e:
        logger.warning("Audio cleanup failed (%s) — using raw signal", e)
        return y, sr, {"applied": False, "reason": f"error:{e}",
                       "snr_pre_db": None, "snr_post_db": None}
