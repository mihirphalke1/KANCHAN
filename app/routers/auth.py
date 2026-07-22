"""
POST /api/auth/login    — evaluator ID + PIN → bearer token
POST /api/auth/selfie   — capture session selfie (face-validated)
POST /api/auth/check-frame     — real-time single-frame face detection
POST /api/auth/check-liveness  — blink detection across a burst of frames
GET  /api/auth/me       — who am I?

Liveness gate (two places):

  1. /check-frame: called by the browser every 500 ms during camera preview.
     Returns face_detected + bbox so the UI can show a live "face found" ring.

  2. /check-liveness: called once after the browser captures ~3 s of frames
     at 8 fps while the evaluator blinks. Uses the smoothed eye-state sequence
     to confirm the selfie comes from a live person, not a printed photo or a
     screen replay.

  The /selfie endpoint then receives the actual photo and runs face detection
  one final time as a server-side safety net — even if someone bypasses the
  browser-side liveness check, the backend rejects a selfie with no human face.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.auth import attach_selfie, authenticate, create_session, require_login
from app.rate_limit import LOGIN_RATE_LIMIT, limiter
from app.utils.liveness import detect_blink_from_frames, detect_face, detect_head_turn_from_frames

logger = logging.getLogger(__name__)
router = APIRouter()

SESSIONS_DIR = Path("data/sessions")

# Set to "0" in .env to disable the face-presence check on /selfie during
# development when you don't have a webcam handy.
import os
ENFORCE_FACE_ON_SELFIE = os.getenv("ENFORCE_FACE_ON_SELFIE", "1") != "0"


@router.post("/auth/login")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login(request: Request, evaluator_id: str = Form(...), pin: str = Form(...)):
    evaluator = authenticate(evaluator_id, pin)
    if not evaluator:
        raise HTTPException(status_code=401, detail="Invalid evaluator ID or PIN")
    session = create_session(evaluator)
    return JSONResponse(content={
        "token":           session["token"],
        "evaluator_id":    session["evaluator_id"],
        "name":            session["name"],
        "branch_id":       session["branch_id"],
        "role":            session["role"],
        "selfie_required": True,
    })


@router.post("/auth/check-frame")
async def check_frame(
    frame: UploadFile = File(...),
    session: dict = Depends(require_login),
):
    """Real-time face detection on a single camera frame.

    Called by the browser every ~500 ms while the liveness capture component
    is active. Returns face_detected and the bounding box so the UI can render
    a live face-detection ring. Does NOT require a selfie to have been captured
    yet — this endpoint IS part of the selfie capture flow.
    """
    raw = await frame.read()
    if not raw:
        return JSONResponse(content={"face_detected": False,
                                     "reason": "empty frame received"})
    try:
        result = detect_face(raw)
    except Exception as e:
        logger.warning("Face detection error in check-frame: %s", e)
        return JSONResponse(content={"face_detected": False,
                                     "reason": "detection error — try again"})
    return JSONResponse(content=result)


@router.post("/auth/check-liveness")
async def check_liveness(
    frames: list[UploadFile] = File(default=[]),
    session: dict = Depends(require_login),
):
    """Blink-liveness check across a burst of frames.

    The browser captures frames at ~8 fps for 3 seconds (≈ 24 frames) while
    asking the evaluator to blink once naturally. All frames are sent here.
    The backend runs the smoothed eye-state blink detector and returns whether
    a genuine blink was found. Does NOT require a selfie yet.
    """
    if not frames:
        return JSONResponse(
            status_code=422,
            content={"face_detected": False, "blink_detected": False,
                     "reason": "No frames received — the browser did not send any frames."},
        )

    frame_bytes_list = []
    for f in frames:
        raw = await f.read()
        if raw:
            frame_bytes_list.append(raw)

    if not frame_bytes_list:
        return JSONResponse(
            status_code=422,
            content={"face_detected": False, "blink_detected": False,
                     "reason": "All received frames were empty."},
        )

    try:
        result = detect_blink_from_frames(frame_bytes_list)
    except Exception as e:
        logger.error("Liveness detection error: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"face_detected": False, "blink_detected": False,
                     "reason": "Liveness check failed due to a server error — please try again."},
        )

    return JSONResponse(content=result)


@router.post("/auth/check-turn")
async def check_turn(
    frames: list[UploadFile] = File(default=[]),
    session: dict = Depends(require_login),
):
    """Head-turn liveness verification across a burst of frames.

    The browser records ~10 fps while the evaluator turns their head left
    then right.  This endpoint verifies both directions were performed —
    a printed photo or screen replay cannot physically rotate, making this
    a reliable anti-spoofing signal.
    """
    if not frames:
        return JSONResponse(
            status_code=422,
            content={"face_detected": False, "head_turn_detected": False,
                     "reason": "No frames received — the browser did not send any frames."},
        )

    frame_bytes_list = []
    for f in frames:
        raw = await f.read()
        if raw:
            frame_bytes_list.append(raw)

    if not frame_bytes_list:
        return JSONResponse(
            status_code=422,
            content={"face_detected": False, "head_turn_detected": False,
                     "reason": "All received frames were empty."},
        )

    try:
        result = detect_head_turn_from_frames(frame_bytes_list)
    except Exception as e:
        logger.error("Head-turn detection error: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"face_detected": False, "head_turn_detected": False,
                     "reason": "Liveness check failed due to a server error — please try again."},
        )

    return JSONResponse(content=result)


@router.post("/auth/selfie")
async def upload_selfie(
    selfie: UploadFile = File(...),
    session: dict = Depends(require_login),
):
    """Accept and persist the session selfie photo.

    Runs face detection as a server-side safety net: even if the browser-side
    liveness gate is somehow bypassed, a selfie with no human face is rejected
    here. This is the second of the two OpenCV check points.

    Disable the face check in development by setting ENFORCE_FACE_ON_SELFIE=0.
    """
    token = session["token"]
    selfie_bytes = await selfie.read()
    if not selfie_bytes:
        raise HTTPException(status_code=422, detail="Empty selfie upload")

    # ── Server-side face validation (second OpenCV checkpoint) ──────────────
    if ENFORCE_FACE_ON_SELFIE:
        try:
            face_result = detect_face(selfie_bytes)
        except Exception as e:
            logger.warning("Face detection failed on selfie upload: %s", e)
            face_result = {"face_detected": True}   # fail-open on detection error

        if not face_result.get("face_detected"):
            reason = face_result.get(
                "reason",
                "No human face detected in the selfie — retake the photo facing the camera directly.",
            )
            raise HTTPException(
                status_code=422,
                detail=f"Selfie rejected: {reason}",
            )

        if face_result.get("warning"):
            logger.warning("Multiple faces in selfie for session %s: %s",
                           token[:8], face_result["warning"])

    session_dir = SESSIONS_DIR / token
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "selfie.jpg"
    path.write_bytes(selfie_bytes)
    attach_selfie(token, str(path))
    return {"ok": True}


@router.get("/auth/me")
async def me(session: dict = Depends(require_login)):
    return JSONResponse(content={
        "evaluator_id":    session["evaluator_id"],
        "name":            session["name"],
        "branch_id":       session["branch_id"],
        "role":            session["role"],
        "selfie_captured": bool(session.get("selfie_path")),
        "login_time":      session["login_time"],
    })
