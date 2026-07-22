"""Gold-rate feed endpoints (P4-17).

GET  /api/gold-rate          — current cached rate (+ staleness) + the
                               per-karat rate table derived from it.
POST /api/gold-rate/refresh  — pull the live IBJA/equivalent feed into the cache
                               (admin/branch-manager only); static file is the
                               fallback if the feed is unset/unreachable.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.auth import require_login
from app.utils.ltv import KARAT_FINENESS, karat_rate_per_gram, load_gold_rate, refresh_gold_rate

router = APIRouter()


def _with_karat_table(rate: dict) -> dict:
    """Bullion feeds (IBJA and equivalents) publish one 999/24K rate; 22K/18K/
    14K are derived by BIS fineness, the same convention any jeweller's rate
    card uses — not four separate live feeds."""
    rate_24k = rate.get("rate_per_gram_inr", 0)
    return {
        **rate,
        "karat_rates_per_gram": {str(k): karat_rate_per_gram(k, rate_24k) for k in KARAT_FINENESS},
    }


@router.get("/gold-rate")
async def get_gold_rate(session: dict = Depends(require_login)):
    return JSONResponse(content=_with_karat_table(load_gold_rate()))


@router.post("/gold-rate/refresh")
async def post_gold_rate_refresh(session: dict = Depends(require_login)):
    if session.get("role") != "branch_manager":
        raise HTTPException(status_code=403,
                            detail="Only a branch manager may refresh the gold rate.")
    return JSONResponse(content=_with_karat_table(refresh_gold_rate()))
