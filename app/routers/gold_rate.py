"""Gold-rate feed endpoints (P4-17).

GET  /api/gold-rate          — current cached rate (+ staleness).
POST /api/gold-rate/refresh  — pull the live IBJA/equivalent feed into the cache
                               (admin/branch-manager only); static file is the
                               fallback if the feed is unset/unreachable.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.auth import require_login
from app.utils.ltv import load_gold_rate, refresh_gold_rate

router = APIRouter()


@router.get("/gold-rate")
async def get_gold_rate(session: dict = Depends(require_login)):
    return JSONResponse(content=load_gold_rate())


@router.post("/gold-rate/refresh")
async def post_gold_rate_refresh(session: dict = Depends(require_login)):
    if session.get("role") != "branch_manager":
        raise HTTPException(status_code=403,
                            detail="Only a branch manager may refresh the gold rate.")
    return JSONResponse(content=refresh_gold_rate())
