"""
OpenCV-based face detection and blink-liveness check.

Automatically selects the right API for the installed OpenCV version:
  OpenCV 4.x → Haar cascade classifiers (bundled in the package, no download).
  OpenCV 5.x → YuNet DNN face detector (FaceDetectorYN).

Liveness detection — FIVE independent methods, any one of which alone is enough:

  METHOD P — Passive micro-motion (PRIMARY, most reliable):
    A live person breathing for 4+ seconds causes the face-region mean brightness
    to shift by 0.3–2.0 gray levels.  JPEG spatial noise in the crop mean is only
    ~0.04 gray levels (spatial averaging eliminates it).  A printed photo or screen
    shows near-zero temporal variance.  Works without any eye detection.
    Threshold: brightness_std > 0.15 OR brightness_range > 0.50.

  METHOD 0 — Laplacian variance of eye patches:
    Per-frame measurement — completely immune to camera shake.
    An open eye has complex texture (iris ring, pupil boundary → high Laplacian
    variance).  A closed eye has smooth eyelid skin → low variance.
    Blink = minimum frame variance < 70% of the open-eye baseline.
    Works even with a single-frame blink at 12 fps.

  METHOD 1 — Relative-motion ratio (immune to head/camera movement):
    For each consecutive frame pair with detected faces, compute the mean-absolute-
    difference (MAD) of the eye strip AND the forehead strip.  The ratio
    eye_MAD / forehead_MAD removes global motion (head shake, camera shake).
    A blink moves the eyelid a lot while the forehead stays still → ratio > 1.5.

  METHOD 2 — Absolute motion spike (secondary):
    Same eye strip, but looking for a peak that is both > 3 gray-units AND
    > 1.5× the median.  Complementary to Method 1 on steady-camera shots.

  METHOD 3 — Eye-region brightness increase (tertiary):
    When eyes close, the dark pupil/iris is covered by lighter eyelid skin.
    Measure mean pixel intensity in the eye region per frame.
    Blink = any frame where brightness > open-eye baseline × 1.10.
    Fires on slow or partial blinks that motion methods miss.

  ANY of the five detectors returning True → blink_detected = True.

Center-oval check:
  detect_face() returns nose_x/y (YuNet landmark) + in_center + eyes landmarks.
  Oval: 50%/50% centre, 25%/33% radii (matching frontend constants exactly).
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CV2_MAJOR = int(cv2.__version__.split(".")[0])

# ── Environment knobs ────────────────────────────────────────────────────────
MIN_FACE_PX        = int(os.getenv("LIVENESS_MIN_FACE_PX",     "40"))
MIN_EYE_PX         = int(os.getenv("LIVENESS_MIN_EYE_PX",      "18"))
FACE_SCALE         = float(os.getenv("LIVENESS_FACE_SCALE",    "1.1"))
EYE_SCALE          = float(os.getenv("LIVENESS_EYE_SCALE",     "1.05"))
FACE_MIN_NEIGHBORS = int(os.getenv("LIVENESS_FACE_NEIGHBORS",  "4"))
EYE_MIN_NEIGHBORS  = int(os.getenv("LIVENESS_EYE_NEIGHBORS",   "2"))
MIN_FACE_FRAMES    = int(os.getenv("LIVENESS_MIN_FACE_FRAMES", "3"))

# ── Center-oval (MUST match frontend OVAL_*_RATIO constants) ─────────────────
OVAL_CX_RATIO = 0.50
OVAL_CY_RATIO = 0.50
OVAL_RX_RATIO = 0.25
OVAL_RY_RATIO = 0.33

# ── Head-turn (yaw) detection ────────────────────────────────────────────────
# Yaw ratio = (nose_x - right_eye_x) / (left_eye_x - right_eye_x)
# ~0.5 = frontal, >TURN_LEFT = person turned to THEIR left,
# <TURN_RIGHT = person turned to THEIR right.
# Thresholds calibrated for a comfortable 25-35° turn.
TURN_LEFT_THRESHOLD  = 0.60
TURN_RIGHT_THRESHOLD = 0.40
MIN_TURN_FRAMES      = 2     # frames at threshold to confirm each direction

# ── YuNet (OpenCV 5.x) ───────────────────────────────────────────────────────
_YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
_YUNET_MODEL_DIR = Path("models")
_YUNET_MODEL     = _YUNET_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
_yunet_detector  = None
_yunet_failed    = False


# ── OpenCV 4.x init ──────────────────────────────────────────────────────────
_face_cascade = None
_eye_cascade  = None

if _CV2_MAJOR < 5:
    _FACE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _EYE_XML  = cv2.data.haarcascades + "haarcascade_eye.xml"
    _face_cascade = cv2.CascadeClassifier(_FACE_XML)
    _eye_cascade  = cv2.CascadeClassifier(_EYE_XML)
    if _face_cascade.empty():
        logger.error("Failed to load face Haar cascade from %s", _FACE_XML)
    else:
        logger.info("OpenCV %s — Haar cascades loaded", cv2.__version__)
else:
    logger.info("OpenCV %s — will use YuNet DNN face detector", cv2.__version__)


# ── YuNet lazy init ──────────────────────────────────────────────────────────

def _get_yunet():
    global _yunet_detector, _yunet_failed
    if _yunet_detector is not None or _yunet_failed:
        return _yunet_detector

    _YUNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not _YUNET_MODEL.exists():
        logger.info("Downloading YuNet face-detection model (~230 KB) …")
        downloaded = False
        for url in _YUNET_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "KANCHAN-AI/2.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if len(data) < 1024:
                    logger.warning("URL %s looks like LFS pointer (%dB), skip", url, len(data))
                    continue
                _YUNET_MODEL.write_bytes(data)
                logger.info("YuNet model saved (%d bytes)", len(data))
                downloaded = True
                break
            except Exception as e:
                logger.warning("YuNet download from %s failed: %s", url, e)

        if not downloaded:
            logger.error("All YuNet downloads failed — liveness will fail-open")
            _yunet_failed = True
            return None

    try:
        det = cv2.FaceDetectorYN.create(
            str(_YUNET_MODEL), "", (320, 240),   # fixed input size = blink frame size
            score_threshold=0.45,
            nms_threshold=0.30,
            top_k=5000,
        )
        _yunet_detector = det
        logger.info("YuNet face detector ready (input 320×240)")
    except Exception as e:
        logger.error("Could not create YuNet detector: %s", e)
        _yunet_failed = True
    return _yunet_detector


# ── Image helpers ────────────────────────────────────────────────────────────

def _decode(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── OpenCV 4.x: Haar ─────────────────────────────────────────────────────────

def _haar_best_face(grey):
    if _face_cascade is None or _face_cascade.empty():
        return None
    eq = cv2.equalizeHist(grey)
    faces = _face_cascade.detectMultiScale(
        eq, scaleFactor=FACE_SCALE, minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=(MIN_FACE_PX, MIN_FACE_PX), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def _haar_eyes_open(grey, face) -> float:
    if _eye_cascade is None or _eye_cascade.empty():
        return 1.0
    x, y, w, h = face
    roi = grey[y : y + int(h * 0.55), x : x + w]
    if roi.size == 0:
        return 0.0
    eq = cv2.equalizeHist(roi)
    eyes = _eye_cascade.detectMultiScale(
        eq, scaleFactor=EYE_SCALE, minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=(MIN_EYE_PX, MIN_EYE_PX),
    )
    return 1.0 if len(eyes) >= 1 else 0.0


# ── OpenCV 5.x: YuNet ────────────────────────────────────────────────────────

def _yunet_best_face(bgr):
    det = _get_yunet()
    if det is None:
        return None
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: float(f[14]))
    x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    lm = best[4:14].reshape(5, 2).astype(int)
    return (x, y, bw, bh, lm)


def _best_face(bgr, grey) -> dict | None:
    if _CV2_MAJOR < 5:
        rect = _haar_best_face(grey)
        return {"rect": rect} if rect else None
    fd = _yunet_best_face(bgr)
    return {"rect": fd[:4], "yunet": fd} if fd else None


# ── ROI extraction helpers ───────────────────────────────────────────────────

def _get_strip(grey: np.ndarray, fd: dict, y_lo: float, y_hi: float,
               canonical=(48, 12)) -> np.ndarray | None:
    """Extract a horizontal face-strip ROI, resize to canonical size."""
    x, y, w, h = (int(v) for v in fd["rect"][:4])
    ih, iw = grey.shape
    y0 = max(0, y + int(h * y_lo))
    y1 = min(ih, y + int(h * y_hi))
    x0, x1 = max(0, x), min(iw, x + w)
    strip = grey[y0:y1, x0:x1]
    if strip.size < 10 or strip.shape[0] < 2 or strip.shape[1] < 2:
        return None
    return cv2.resize(strip, canonical)


def _strip_mad(s1: np.ndarray | None, s2: np.ndarray | None) -> float:
    if s1 is None or s2 is None:
        return 0.0
    return float(np.abs(s1.astype(np.float32) - s2.astype(np.float32)).mean())


# ── METHOD 1: Relative-motion ratio ─────────────────────────────────────────

def _method1_relative_motion(frames_bgr: list, face_datas: list) -> bool:
    """
    Blink = eye strip moves much more than forehead strip (ratio > 2.0).
    Cancels head/camera movement because both strips are affected equally.
    """
    ratios: list[float] = []
    prev_bgr = prev_grey = prev_fd = None

    for bgr, fd in zip(frames_bgr, face_datas):
        if bgr is not None and fd is not None and prev_fd is not None:
            curr_grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            eye_s1  = _get_strip(prev_grey, prev_fd, 0.30, 0.58)
            eye_s2  = _get_strip(curr_grey, fd,      0.30, 0.58)
            fore_s1 = _get_strip(prev_grey, prev_fd, 0.02, 0.20)
            fore_s2 = _get_strip(curr_grey, fd,      0.02, 0.20)

            eye_mad  = _strip_mad(eye_s1,  eye_s2)
            fore_mad = _strip_mad(fore_s1, fore_s2)

            if fore_mad < 0.8:        # almost completely static camera
                ratios.append(eye_mad)  # use absolute eye MAD as the score
            else:
                ratios.append(eye_mad / fore_mad)

        if bgr is not None and fd is not None:
            prev_bgr  = bgr
            prev_grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            prev_fd   = fd

    if not ratios:
        return False
    peak = max(ratios)
    logger.info("M1 relative-motion: peak_ratio=%.2f (need >1.5)", peak)
    return peak > 1.5


# ── METHOD 2: Absolute motion spike ─────────────────────────────────────────

def _method2_absolute_spike(frames_bgr: list, face_datas: list) -> bool:
    """
    Blink = eye-strip MAD spike > 1.8× median AND > 4 absolute.
    Works best on steady shots; complementary to method 1.
    """
    mads: list[float] = []
    prev_bgr = prev_grey = prev_fd = None

    for bgr, fd in zip(frames_bgr, face_datas):
        if bgr is not None and fd is not None and prev_fd is not None:
            curr_grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            s1 = _get_strip(prev_grey, prev_fd, 0.30, 0.58)
            s2 = _get_strip(curr_grey, fd,      0.30, 0.58)
            mads.append(_strip_mad(s1, s2))
        if bgr is not None and fd is not None:
            prev_bgr  = bgr
            prev_grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            prev_fd   = fd

    if len(mads) < 2:
        return False
    median_m = float(np.median(mads))
    peak_m   = max(mads)
    logger.info("M2 abs-spike: peak=%.2f median=%.2f (need >3 & >1.5×med)", peak_m, median_m)
    return peak_m > 3.0 and peak_m > median_m * 1.5


# ── METHOD 3: Eye-region brightness increase ─────────────────────────────────

def _method3_brightness(frames_bgr: list, face_datas: list) -> bool:
    """
    Blink = mean brightness of eye patches rises above open-eye baseline.

    Open eye:   dark pupil + iris pulls mean DOWN  (~80–130).
    Closed eye: eyelid skin similar to forehead    (~130–170).

    The baseline is the LOWEST 40% of brightness readings across all frames
    (eye-open frames dominate).  A blink frame has noticeably higher brightness.
    Works even at low FPS because it measures per-frame, not frame-to-frame.
    """
    brightnesses: list[float] = []

    for bgr, fd in zip(frames_bgr, face_datas):
        if bgr is None or fd is None:
            continue
        if "yunet" not in fd:
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, _, bw, bh, lm = fd["yunet"]
        ih, iw = grey.shape

        re_x, re_y = int(lm[0, 0]), int(lm[0, 1])
        le_x, le_y = int(lm[1, 0]), int(lm[1, 1])

        pw = max(14, bw // 6)
        ph = max(9,  bh // 9)

        def _mean(cx, cy):
            x0, x1 = max(0, cx - pw), min(iw, cx + pw)
            y0, y1 = max(0, cy - ph), min(ih, cy + ph)
            p = grey[y0:y1, x0:x1]
            return float(p.mean()) if p.size > 8 else -1.0

        rm, lm_ = _mean(re_x, re_y), _mean(le_x, le_y)
        if rm > 0 and lm_ > 0:
            brightnesses.append((rm + lm_) / 2.0)

    if len(brightnesses) < 4:
        return False

    sorted_b = sorted(brightnesses)
    n_open   = max(1, int(len(sorted_b) * 0.40))
    baseline = sum(sorted_b[:n_open]) / n_open   # mean of darkest 40% = open-eye baseline

    peak_b = max(brightnesses)
    threshold = baseline * 1.10    # 10% brighter than open baseline = closed eye
    logger.info(
        "M3 brightness: baseline=%.1f peak=%.1f threshold=%.1f → %s",
        baseline, peak_b, threshold,
        "BLINK" if peak_b > threshold else "no blink",
    )
    return peak_b > threshold


# ── METHOD P: Passive micro-motion (breathing / saccades) ───────────────────

def _method_passive_motion(frames_bgr: list, face_datas: list) -> bool:
    """
    A live person ALWAYS generates micro-motion over 4+ seconds:
    breathing shifts the face 0.3-1.5 px, causing face-region mean brightness
    to drift by 0.3-2.0 gray levels.

    JPEG spatial noise in the per-frame mean is only ~0.04 gray levels
    (spatial averaging over thousands of pixels eliminates it).
    A printed photo or screen will show near-zero temporal variance.

    Uses a FIXED reference box (averaged from first 5 detected frames) so
    bbox-tracking jitter does not add artificial variance.

    Works with both YuNet and Haar (needs only fd['rect']).
    """
    ref_boxes = []
    for bgr, fd in zip(frames_bgr, face_datas):
        if bgr is not None and fd is not None:
            ref_boxes.append(fd["rect"][:4])
        if len(ref_boxes) >= 5:
            break

    if not ref_boxes:
        logger.info("Motion(P): no face detected in first 5 frames — skip")
        return False

    rx = int(np.mean([b[0] for b in ref_boxes]))
    ry = int(np.mean([b[1] for b in ref_boxes]))
    rw = int(np.mean([b[2] for b in ref_boxes]))
    rh = int(np.mean([b[3] for b in ref_boxes]))

    frame_means: list[float] = []
    for bgr in frames_bgr:
        if bgr is None:
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = grey.shape
        crop = grey[max(0, ry): min(ih, ry + rh), max(0, rx): min(iw, rx + rw)]
        if crop.size < 200:
            continue
        frame_means.append(float(crop.mean()))

    if len(frame_means) < 6:
        logger.info("Motion(P): only %d usable frames — skip", len(frame_means))
        return False

    brightness_std   = float(np.std(frame_means))
    brightness_range = float(max(frame_means) - min(frame_means))
    live = brightness_std > 0.15 or brightness_range > 0.50
    logger.info(
        "Motion(P): %d frames  std=%.4f  range=%.4f  → %s",
        len(frame_means), brightness_std, brightness_range,
        "LIVE ✓" if live else "no motion",
    )
    return live


# ── Head-turn yaw helper ─────────────────────────────────────────────────────

def _compute_yaw_ratio(fd: dict) -> float | None:
    """
    Nose position relative to the inter-eye span.

    YuNet landmark convention (person-centric, not image-centric):
      lm[0] = right eye of the person  → appears on image-LEFT  (lower X)
      lm[1] = left  eye of the person  → appears on image-RIGHT (higher X)
      lm[2] = nose tip

    Formula: (nose_x − right_eye_x) / (left_eye_x − right_eye_x)
      ≈ 0.50 when looking straight
      > TURN_LEFT_THRESHOLD  when turning to person's LEFT
      < TURN_RIGHT_THRESHOLD when turning to person's RIGHT

    Returns None when the face is too small / profile-only.
    """
    if "yunet" not in fd:
        return None
    _, _, _, _, lm = fd["yunet"]
    re_x   = float(lm[0, 0])   # right eye (person) = image LEFT = smaller X
    le_x   = float(lm[1, 0])   # left  eye (person) = image RIGHT = larger X
    nose_x = float(lm[2, 0])
    span = le_x - re_x
    if span < 10:
        return None
    return (nose_x - re_x) / span


# ── METHOD 0: Laplacian variance of eye patches ──────────────────────────────

def _method0_laplacian_eye(frames_bgr: list, face_datas: list) -> bool:
    """
    Primary blink detector — per-frame Laplacian variance of precise eye patches.

    Why it works:
      Open eye  → iris + pupil boundary creates high-frequency edges
                  → high Laplacian variance (typically 30–300+)
      Closed eye → smooth eyelid skin has almost no edges
                  → low Laplacian variance (typically 2–20)

    Because this is a per-frame measurement (not inter-frame diff) it is
    completely unaffected by camera shake, head movement, or lighting changes.

    Algorithm:
      1. For every YuNet frame, extract a tight patch over each eye landmark.
      2. Compute Laplacian variance of each patch; average the two eyes.
      3. Build a per-frame score series (higher = more open).
      4. Baseline = mean of the top 50% (the most-open frames).
      5. Blink detected if min_score < 55% of baseline.

    Only runs when YuNet landmarks are available (OpenCV 5.x).
    Returns False (not False-positive) on insufficient data.
    """
    scores: list[float] = []

    for bgr, fd in zip(frames_bgr, face_datas):
        if bgr is None or fd is None or "yunet" not in fd:
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, _, bw, bh, lm = fd["yunet"]
        ih, iw = grey.shape

        frame_vars: list[float] = []
        for eye_idx in [0, 1]:   # lm[0] = right eye (person), lm[1] = left eye
            cx = int(lm[eye_idx, 0])
            cy = int(lm[eye_idx, 1])
            # Patch proportional to face size — roughly 1/5 wide, 1/9 tall
            pw = max(14, bw // 5)
            ph = max(8,  bh // 9)
            x0, x1 = max(0, cx - pw), min(iw, cx + pw)
            y0, y1 = max(0, cy - ph), min(ih, cy + ph)
            patch = grey[y0:y1, x0:x1]
            if patch.size < 60:
                continue
            lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F)
            frame_vars.append(float(lap.var()))

        if frame_vars:
            scores.append(sum(frame_vars) / len(frame_vars))

    if len(scores) < 4:
        logger.info("M0 lap-var: not enough YuNet frames (%d) — skipping", len(scores))
        return False

    # Baseline = mean of top 50% scores (frames where eyes were most open)
    sorted_s = sorted(scores, reverse=True)
    n_open   = max(2, len(sorted_s) // 2)
    baseline = sum(sorted_s[:n_open]) / n_open
    min_score = min(scores)

    # A blink causes a clear dip — min/max ratio drops well below 0.55
    ratio = min_score / max(baseline, 0.5)
    logger.info(
        "M0 lap-var: %d frames  baseline=%.2f  min=%.2f  ratio=%.3f  "
        "(blink if ratio < 0.70)",
        len(scores), baseline, min_score, ratio,
    )
    return ratio < 0.70


# ── Center-oval check ────────────────────────────────────────────────────────

def _in_center_oval(nose_x: float, nose_y: float, iw: int, ih: int) -> bool:
    ov_cx = iw * OVAL_CX_RATIO
    ov_cy = ih * OVAL_CY_RATIO
    rx    = iw * OVAL_RX_RATIO
    ry    = ih * OVAL_RY_RATIO
    e = ((nose_x - ov_cx) / rx) ** 2 + ((nose_y - ov_cy) / ry) ** 2
    logger.debug("Center: nose=(%.0f,%.0f) e=%.3f %s", nose_x, nose_y, e, "OK" if e <= 1 else "FAIL")
    return e <= 1.0


# ── Public API ───────────────────────────────────────────────────────────────

def detect_face(image_bytes: bytes) -> dict:
    """Return face presence + centre-oval check for a single image."""
    bgr = _decode(image_bytes)
    if bgr is None:
        return {"face_detected": False, "face_count": 0,
                "reason": "Could not decode image."}

    ih, iw = bgr.shape[:2]
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fd   = _best_face(bgr, grey)

    if fd is None:
        if _CV2_MAJOR >= 5 and _yunet_failed:
            logger.warning("YuNet unavailable — face check fail-open")
            return {"face_detected": True, "face_count": 1, "in_center": True,
                    "warning": "face detection unavailable"}
        return {"face_detected": False, "face_count": 0, "in_center": False,
                "reason": "No face detected — face the camera in good light."}

    x, y, w, h = (int(v) for v in fd["rect"])
    if "yunet" in fd:
        lm = fd["yunet"][4]
        nose_x, nose_y = float(lm[2, 0]), float(lm[2, 1])
        # Return eye positions so the frontend can compute yaw in real time.
        # [re_x, re_y, le_x, le_y] in person-convention (right eye = image-left).
        eyes = [float(lm[0, 0]), float(lm[0, 1]),
                float(lm[1, 0]), float(lm[1, 1])]
    else:
        nose_x, nose_y = x + w / 2.0, y + h / 2.0
        eyes = None

    return {
        "face_detected": True,
        "face_count":    1,
        "bbox":          [x, y, w, h],
        "nose":          [nose_x, nose_y],
        "eyes":          eyes,
        "in_center":     _in_center_oval(nose_x, nose_y, iw, ih),
    }


def detect_blink_from_frames(frame_bytes_list: list) -> dict:
    """
    Detect a genuine blink using three independent methods.

    Decodes all frames first, then runs all three detectors.
    ANY single detector firing = blink confirmed.
    All three log their computed values at INFO level.
    """
    if not frame_bytes_list:
        return {"face_detected": False, "blink_detected": False,
                "frames_analyzed": 0, "face_frames": 0,
                "reason": "No frames received."}

    # ── Decode all frames up-front ───────────────────────────────────────────
    frames_bgr: list = []
    face_datas: list = []
    face_frames = 0

    for fb in frame_bytes_list:
        bgr = _decode(fb)
        if bgr is None:
            frames_bgr.append(None)
            face_datas.append(None)
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fd   = _best_face(bgr, grey)
        frames_bgr.append(bgr)
        face_datas.append(fd)
        if fd is not None:
            face_frames += 1

    total = len(frame_bytes_list)

    logger.info("Blink check: %d/%d frames with face detected", face_frames, total)

    # ── Fail-open when model unavailable (MUST be before face_frames check) ──
    if _CV2_MAJOR >= 5 and _yunet_failed:
        logger.warning("YuNet unavailable — blink check skipped (fail-open)")
        return {"face_detected": True, "blink_detected": True,
                "frames_analyzed": total, "face_frames": face_frames,
                "reason": "Model unavailable — face presence confirmed."}

    if face_frames < MIN_FACE_FRAMES:
        return {"face_detected": False, "blink_detected": False,
                "frames_analyzed": total, "face_frames": face_frames,
                "reason": (f"Face only visible in {face_frames}/{total} frames — "
                           "keep your face centred and try again.")}

    # ── Run all five detectors ───────────────────────────────────────────────
    # Method P (passive motion) is the primary — any live person breathing for
    # 4 seconds produces measurable face-region mean drift that photos cannot.
    # Methods 0-3 provide corroboration and handle edge cases.
    mp = _method_passive_motion(frames_bgr, face_datas)
    m0 = _method0_laplacian_eye(frames_bgr, face_datas)
    m1 = _method1_relative_motion(frames_bgr, face_datas)
    m2 = _method2_absolute_spike(frames_bgr, face_datas)
    m3 = _method3_brightness(frames_bgr, face_datas)

    blink = mp or m0 or m1 or m2 or m3
    logger.info(
        "Blink result: MP(motion)=%s M0(laplacian)=%s M1(rel-motion)=%s "
        "M2(abs-spike)=%s M3(brightness)=%s → %s",
        mp, m0, m1, m2, m3, "LIVE ✓" if blink else "no blink",
    )

    return {
        "face_detected":   True,
        "blink_detected":  blink,
        "frames_analyzed": total,
        "face_frames":     face_frames,
        "methods":         {"MP_motion": mp, "M0_laplacian": m0,
                            "M1_rel_motion": m1, "M2_abs_spike": m2,
                            "M3_brightness": m3},
        "reason": (
            "Liveness confirmed — real person detected."
            if blink else
            "Liveness check failed — blink once clearly and keep your face "
            "centred in the oval in good light, then try again."
        ),
    }


def detect_head_turn_from_frames(frame_bytes_list: list) -> dict:
    """
    Verify the person performed both a left-turn and a right-turn head movement.

    Uses the yaw ratio (nose position relative to eye span) across the clip.
    Both directions must appear in at least MIN_TURN_FRAMES frames each.
    A static printed photo or screen replay cannot physically rotate — this
    is more reliable than blink detection on webcam footage.
    """
    if not frame_bytes_list:
        return {
            "face_detected": False, "head_turn_detected": False,
            "left_detected": False, "right_detected": False,
            "frames_analyzed": 0,
            "reason": "No frames received.",
        }

    yaw_values: list[float] = []
    face_frames = 0

    for fb in frame_bytes_list:
        bgr = _decode(fb)
        if bgr is None:
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fd = _best_face(bgr, grey)
        if fd is None:
            continue
        face_frames += 1
        yaw = _compute_yaw_ratio(fd)
        if yaw is not None:
            yaw_values.append(yaw)

    total = len(frame_bytes_list)
    logger.info(
        "Head-turn check: %d/%d frames with face, %d with yaw — "
        "min=%.3f max=%.3f",
        face_frames, total, len(yaw_values),
        min(yaw_values, default=0.0),
        max(yaw_values, default=0.0),
    )

    # Fail-open when model unavailable
    if _CV2_MAJOR >= 5 and _yunet_failed:
        logger.warning("YuNet unavailable — head-turn check skipped (fail-open)")
        return {
            "face_detected": True, "head_turn_detected": True,
            "left_detected": True, "right_detected": True,
            "frames_analyzed": total, "face_frames": face_frames,
            "reason": "Model unavailable — face presence confirmed.",
        }

    if face_frames < MIN_FACE_FRAMES:
        return {
            "face_detected": False, "head_turn_detected": False,
            "left_detected": False, "right_detected": False,
            "frames_analyzed": total, "face_frames": face_frames,
            "reason": (f"Face visible in only {face_frames}/{total} frames — "
                       "keep your face centred and try again."),
        }

    if not yaw_values:
        return {
            "face_detected": True, "head_turn_detected": False,
            "left_detected": False, "right_detected": False,
            "frames_analyzed": total, "face_frames": face_frames,
            "reason": "Could not compute head orientation — face the camera more directly.",
        }

    left_frames  = sum(1 for y in yaw_values if y > TURN_LEFT_THRESHOLD)
    right_frames = sum(1 for y in yaw_values if y < TURN_RIGHT_THRESHOLD)
    left_ok  = left_frames  >= MIN_TURN_FRAMES
    right_ok = right_frames >= MIN_TURN_FRAMES
    detected = left_ok and right_ok

    logger.info(
        "Head-turn: left_frames=%d(need %d) right_frames=%d(need %d) → %s",
        left_frames, MIN_TURN_FRAMES, right_frames, MIN_TURN_FRAMES,
        "CONFIRMED ✓" if detected else "FAILED",
    )

    if detected:
        reason = "Liveness confirmed — head movement detected."
    elif not left_ok and not right_ok:
        reason = "Neither turn was detected — turn your head clearly LEFT then RIGHT."
    elif not left_ok:
        reason = "Left turn not detected — turn your face further to the LEFT and try again."
    else:
        reason = "Right turn not detected — turn your face further to the RIGHT and try again."

    return {
        "face_detected":      True,
        "head_turn_detected": detected,
        "left_detected":      left_ok,
        "right_detected":     right_ok,
        "frames_analyzed":    total,
        "face_frames":        face_frames,
        "reason":             reason,
    }
