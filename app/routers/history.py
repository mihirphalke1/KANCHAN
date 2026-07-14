import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class CustomerUpdate(BaseModel):
    # officer_name is deliberately absent: it's audit-trail data resolved
    # from the authenticated evaluator session at analysis time (see
    # app/auth.py), not a customer detail — this model can never overwrite
    # it, regardless of what a client sends.
    name:         Optional[str] = None
    account_no:   Optional[str] = None
    loan_app_no:  Optional[str] = None
    phone:        Optional[str] = None
    email:        Optional[str] = None
    address:      Optional[str] = None
    notes:        Optional[str] = None

HISTORY_PATH = Path("data/case_history.json")


def _load() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except Exception:
        return []


def _save(history: list) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


@router.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    history = _load()
    history.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
    return JSONResponse(content={
        "total":  len(history),
        "offset": offset,
        "limit":  limit,
        "cases":  history[offset: offset + limit],
    })


@router.get("/history/{case_id}")
async def get_case(case_id: str):
    history = _load()
    for case in history:
        if case.get("case_id") == case_id:
            return JSONResponse(content=case)
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@router.patch("/history/{case_id}")
async def update_customer(case_id: str, update: CustomerUpdate):
    """Merge new/edited customer-info fields into a saved case, so details
    that weren't captured at intake (phone, email, address, notes, or a
    correction to the essentials) can be added afterwards from History."""
    history = _load()
    for case in history:
        if case.get("case_id") == case_id:
            customer = case.setdefault("customer", {})
            for k, v in update.model_dump(exclude_unset=True).items():
                if v is not None and v != "":
                    customer[k] = v
            _save(history)
            return JSONResponse(content=case)
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@router.delete("/history/{case_id}")
async def delete_case(case_id: str):
    history = _load()
    new_history = [c for c in history if c.get("case_id") != case_id]
    if len(new_history) == len(history):
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    _save(new_history)
    return {"deleted": case_id}
