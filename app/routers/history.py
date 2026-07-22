import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import require_login
from app.utils.approval import can_check, build_approval
from app.utils.hashchain import restamp_from, verify_chain, tip_hash

router = APIRouter()

IST = timezone(timedelta(hours=5, minutes=30))


class SignoffRequest(BaseModel):
    decision: str                      # "approve" | "reject"
    note:     Optional[str] = None


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


def _delete_roles() -> set:
    """Roles permitted to soft-delete a case, from HISTORY_DELETE_ROLES
    (comma-separated). Defaults to branch_manager only; empty => no one."""
    raw = os.getenv("HISTORY_DELETE_ROLES", "branch_manager")
    return {r.strip() for r in raw.split(",") if r.strip()}


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


def _is_deleted(case: dict) -> bool:
    return bool(case.get("deleted"))


@router.get("/history")
async def get_history(limit: int = 50, offset: int = 0, include_deleted: bool = False):
    history = _load()
    if not include_deleted:
        history = [c for c in history if not _is_deleted(c)]
    history.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
    return JSONResponse(content={
        "total":  len(history),
        "offset": offset,
        "limit":  limit,
        "cases":  history[offset: offset + limit],
    })


@router.get("/history/verify")
async def verify_history(session: dict = Depends(require_login)):
    """Verify the tamper-evident hash chain end to end. Any silent edit,
    reordering, or removal of a chained case is reported with the exact
    entry that first diverges."""
    result = verify_chain(_load())
    result["tip_hash"] = tip_hash(_load())
    return JSONResponse(content=result)


@router.get("/history/{case_id}")
async def get_case(case_id: str):
    history = _load()
    for case in history:
        if case.get("case_id") == case_id and not _is_deleted(case):
            return JSONResponse(content=case)
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@router.post("/history/{case_id}/signoff")
async def signoff_case(case_id: str, req: SignoffRequest,
                       session: dict = Depends(require_login)):
    """Maker-checker dual sign-off (P3-12). A BORDERLINE / HELD case cannot be
    closed on the maker's authority alone — a second, DIFFERENT authenticated
    officer must approve or reject it here. The action is recorded as a distinct
    `approvals` audit entry (separate from customer `amendments`) and the hash
    chain is re-stamped forward so the sign-off is tamper-evident."""
    decision = (req.decision or "").strip().lower()
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=422, detail="decision must be 'approve' or 'reject'")

    history = _load()
    for idx, case in enumerate(history):
        if case.get("case_id") != case_id:
            continue
        if _is_deleted(case):
            raise HTTPException(status_code=410, detail=f"Case {case_id} was deleted")

        # Older cases (pre-maker-checker) carry no approval block — derive one
        # from the stored verdict so they can still be gated retroactively.
        approval = case.get("approval")
        if approval is None:
            verdict = case.get("verdict", {})
            approval = build_approval(
                verdict.get("risk_level"), verdict.get("loan_action"),
                maker_id=(case.get("evaluator") or {}).get("evaluator_id"),
                maker_name=(case.get("evaluator") or {}).get("name"),
            )
            case["approval"] = approval

        ok, reason = can_check(session, approval)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

        now = datetime.now(IST).isoformat()
        approval["status"]       = "approved" if decision == "approve" else "rejected"
        approval["decision"]     = approval["status"]
        approval["checker_id"]   = session.get("evaluator_id")
        approval["checker_name"] = session.get("name")
        approval["signed_at"]    = now
        approval["note"]         = req.note
        approval["closable"]     = True
        case.setdefault("approvals", []).append({
            "at":       now,
            "by":       session.get("evaluator_id"),
            "by_name":  session.get("name"),
            "role":     session.get("role"),
            "decision": approval["status"],
            "note":     req.note,
        })
        restamp_from(history, idx)
        _save(history)
        return JSONResponse(content={"case_id": case_id, "approval": approval})

    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@router.patch("/history/{case_id}")
async def update_customer(case_id: str, update: CustomerUpdate,
                          session: dict = Depends(require_login)):
    """Merge new/edited customer-info fields into a saved case, so details
    that weren't captured at intake (phone, email, address, notes, or a
    correction to the essentials) can be added afterwards from History.

    Authenticated: every amendment is attributed to the evaluator making it,
    recorded in the case's own `amendments` audit log, and the hash chain is
    re-stamped forward so the ledger stays internally consistent."""
    history = _load()
    for idx, case in enumerate(history):
        if case.get("case_id") == case_id:
            if _is_deleted(case):
                raise HTTPException(status_code=410, detail=f"Case {case_id} was deleted")
            customer = case.setdefault("customer", {})
            changed = {}
            for k, v in update.model_dump(exclude_unset=True).items():
                if v is not None and v != "":
                    before = customer.get(k)
                    if before != v:
                        changed[k] = {"from": before, "to": v}
                        customer[k] = v
            if changed:
                case.setdefault("amendments", []).append({
                    "at":           datetime.now(IST).isoformat(),
                    "by":           session.get("evaluator_id"),
                    "by_name":      session.get("name"),
                    "fields":       changed,
                })
                # Re-stamp the chain from this entry forward so the edit is
                # reflected in the hashes rather than silently breaking them.
                restamp_from(history, idx)
                _save(history)
            return JSONResponse(content=case)
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@router.delete("/history/{case_id}")
async def delete_case(case_id: str, session: dict = Depends(require_login)):
    """Soft-delete a case. Authenticated and role-restricted
    (HISTORY_DELETE_ROLES). The record is TOMBSTONED, not erased: it stays on
    disk with a `deleted` marker (who/when) and remains verifiable in the hash
    chain — it just no longer appears in History or reports. This closes the
    original hole (an unauthenticated DELETE that silently rewrote the ledger)
    while keeping deletions themselves auditable."""
    role = session.get("role")
    if role not in _delete_roles():
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' is not permitted to delete cases — requires one of: {', '.join(sorted(_delete_roles())) or '(none)'}",
        )
    history = _load()
    for case in history:
        if case.get("case_id") == case_id:
            if _is_deleted(case):
                return {"deleted": case_id, "already_deleted": True}
            case["deleted"] = {
                "at":      datetime.now(IST).isoformat(),
                "by":      session.get("evaluator_id"),
                "by_name": session.get("name"),
            }
            _save(history)
            return {"deleted": case_id}
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
