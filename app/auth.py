"""
Evaluator authentication + session management — the Evaluator Integrity Layer.

No database exists in this project (data/*.json flat files are the
persistence layer throughout) — evaluators and sessions follow the same
pattern. Deliberately simple (evaluator ID + PIN, bearer token) because the
goal for a bank pilot is closing the "officer_name is a free-text field
anyone can type" hole, not building a general-purpose IAM system Canara's
own systems will eventually replace this with.

A session is only usable for analysis once a selfie has been captured for
it (require_session) — every case is then traceable to a specific person who
was physically present at login, not just a typed name.
"""
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

EVALUATORS_PATH = Path("data/evaluators.json")
SESSIONS_PATH   = Path("data/sessions.json")
SESSIONS_DIR    = Path("data/sessions")

SESSION_TTL_HOURS = 12


def _hash_pin(pin: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{pin}".encode()).hexdigest()


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _seed_evaluators() -> list:
    """Demo evaluator roster for a pilot — PINs are salted + hashed, never
    stored in the clear. In production this table is provisioned by branch
    admin (or synced from Canara's HR/IAM system), not seeded in code."""
    seed = [
        {"evaluator_id": "EMP-1001", "name": "Mihir Phalke", "pin": "1234", "branch_id": "BLR-001", "role": "assessment_officer"},
        {"evaluator_id": "EMP-1002", "name": "Ananya Rao",      "pin": "4321", "branch_id": "BLR-001", "role": "assessment_officer"},
        {"evaluator_id": "EMP-9001", "name": "Suresh Bhat",     "pin": "9999", "branch_id": "BLR-001", "role": "branch_manager"},
    ]
    records = []
    for e in seed:
        salt = secrets.token_hex(8)
        records.append({
            "evaluator_id": e["evaluator_id"], "name": e["name"], "branch_id": e["branch_id"],
            "role": e["role"], "salt": salt, "pin_hash": _hash_pin(e["pin"], salt),
        })
    _save_json(EVALUATORS_PATH, records)
    logger.info("Seeded %d demo evaluators at %s (default PINs — for pilot demo only)",
                len(records), EVALUATORS_PATH)
    return records


def load_evaluators() -> list:
    if not EVALUATORS_PATH.exists():
        return _seed_evaluators()
    return _load_json(EVALUATORS_PATH, [])


def authenticate(evaluator_id: str, pin: str) -> Optional[dict]:
    for e in load_evaluators():
        if e["evaluator_id"].lower() == evaluator_id.lower().strip():
            if _hash_pin(pin, e["salt"]) == e["pin_hash"]:
                return e
            return None
    return None


def create_session(evaluator: dict) -> dict:
    token = secrets.token_hex(24)
    session = {
        "token":        token,
        "evaluator_id": evaluator["evaluator_id"],
        "name":         evaluator["name"],
        "branch_id":    evaluator["branch_id"],
        "role":         evaluator["role"],
        "login_time":   datetime.now(IST).isoformat(),
        "selfie_path":  None,
    }
    sessions = _load_json(SESSIONS_PATH, {})
    sessions[token] = session
    _save_json(SESSIONS_PATH, sessions)
    return session


def get_session(token: str) -> Optional[dict]:
    sessions = _load_json(SESSIONS_PATH, {})
    session = sessions.get(token)
    if not session:
        return None
    try:
        login_time = datetime.fromisoformat(session["login_time"])
    except Exception:
        return None
    if datetime.now(IST) - login_time > timedelta(hours=SESSION_TTL_HOURS):
        return None
    return session


def attach_selfie(token: str, selfie_path: str) -> None:
    sessions = _load_json(SESSIONS_PATH, {})
    if token in sessions:
        sessions[token]["selfie_path"] = selfie_path
        _save_json(SESSIONS_PATH, sessions)


def _resolve_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header — please log in")
    return authorization.removeprefix("Bearer ").strip()


def require_login(authorization: str = Header(default=None)) -> dict:
    """FastAPI dependency: any authenticated evaluator, selfie or not."""
    token = _resolve_token(authorization)
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid — please log in again")
    return session


def require_session(authorization: str = Header(default=None)) -> dict:
    """FastAPI dependency for evidence-producing routes: the evaluator must
    also have captured a session selfie — every case must be traceable to a
    specific person physically present at login, not just a typed name."""
    session = require_login(authorization)
    if not session.get("selfie_path"):
        raise HTTPException(
            status_code=403,
            detail="Session selfie not captured — capture a session selfie before analysing items",
        )
    return session
