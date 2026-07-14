"""
POST /api/auth/login, POST /api/auth/selfie, GET /api/auth/me
Evaluator session endpoints — see app/auth.py for the session model.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.auth import attach_selfie, authenticate, create_session, require_login

logger = logging.getLogger(__name__)
router = APIRouter()

SESSIONS_DIR = Path("data/sessions")


@router.post("/auth/login")
async def login(evaluator_id: str = Form(...), pin: str = Form(...)):
    evaluator = authenticate(evaluator_id, pin)
    if not evaluator:
        raise HTTPException(status_code=401, detail="Invalid evaluator ID or PIN")
    session = create_session(evaluator)
    return JSONResponse(content={
        "token":          session["token"],
        "evaluator_id":   session["evaluator_id"],
        "name":           session["name"],
        "branch_id":      session["branch_id"],
        "role":           session["role"],
        "selfie_required": True,
    })


@router.post("/auth/selfie")
async def upload_selfie(selfie: UploadFile = File(...), session: dict = Depends(require_login)):
    token = session["token"]
    session_dir = SESSIONS_DIR / token
    session_dir.mkdir(parents=True, exist_ok=True)
    selfie_bytes = await selfie.read()
    if not selfie_bytes:
        raise HTTPException(status_code=422, detail="Empty selfie upload")
    path = session_dir / "selfie.jpg"
    path.write_bytes(selfie_bytes)
    attach_selfie(token, str(path))
    return {"ok": True}


@router.get("/auth/me")
async def me(session: dict = Depends(require_login)):
    return JSONResponse(content={
        "evaluator_id":     session["evaluator_id"],
        "name":             session["name"],
        "branch_id":        session["branch_id"],
        "role":             session["role"],
        "selfie_captured":  bool(session.get("selfie_path")),
        "login_time":       session["login_time"],
    })
