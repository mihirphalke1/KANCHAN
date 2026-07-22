"""
Per-borrower aggregate pledging cap — RBI Gold Collateral Directions, 2025,
Para 16 (P4-15).

A single borrower may not pledge more than a hard aggregate weight of gold
collateral across ALL their active loans (not just the loan in front of the
officer): 1 kg of ornaments and 50 g of gold coins per borrower. This function
sums a borrower's already-pledged net gold weight — from the mock core-banking
adapter and prior cases in the local ledger — adds the item being appraised
now, and flags a breach so the officer cannot silently exceed the cap by
splitting a pledge across visits or branches.

The core-banking side is a mock adapter (data/mock_core_banking.json) for the
pilot; the same interface swaps to the real CBS lookup later.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CBS_PATH = Path("data/mock_core_banking.json")
HISTORY_PATH = Path("data/case_history.json")

# RBI Para 16 aggregate caps per borrower.
ORNAMENT_CAP_G = 1000.0   # 1 kg of ornaments
COIN_CAP_G     = 50.0     # 50 g of specially-minted gold coins


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
    return default


def _cbs_pledged_weight(account_no: str) -> float:
    """Already-pledged net gold weight for this account from core banking.
    The mock records `existing_pledged_gold_g` where known; absent => 0."""
    cbs = _load_json(CBS_PATH, {})
    rec = cbs.get(account_no) or {}
    try:
        return float(rec.get("existing_pledged_gold_g", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ledger_pledged_weight(account_no: str, exclude_case_id: str = None) -> float:
    """Sum net gold weight from prior ACCEPTED/HOLD cases for this account in the
    local ledger — pledges made through KANCHAN that CBS may not yet reflect."""
    if not account_no:
        return 0.0
    total = 0.0
    for case in _load_json(HISTORY_PATH, []):
        if case.get("case_id") == exclude_case_id or case.get("deleted"):
            continue
        cust = case.get("customer") or {}
        if (cust.get("account_no") or "").strip() != account_no.strip():
            continue
        # Only count loans that weren't declined.
        action = ((case.get("verdict") or {}).get("loan_action") or "").upper()
        if action == "DECLINE":
            continue
        ltv = case.get("ltv") or {}
        try:
            total += float(ltv.get("net_gold_weight_g", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def check_pledging_cap(account_no: str, current_net_weight_g: float,
                       item_type: str = "ornament", exclude_case_id: str = None) -> dict:
    """Aggregate this borrower's pledged gold across CBS + the local ledger,
    add the item being appraised now, and check it against the RBI Para 16 cap.
    Never raises; returns a structured result even when no account is given."""
    is_coin = str(item_type).lower() in ("coin", "coins", "gold_coin")
    cap = COIN_CAP_G if is_coin else ORNAMENT_CAP_G
    current = float(current_net_weight_g or 0.0)

    if not account_no:
        return {
            "checked": False, "cap_g": cap, "item_type": "coin" if is_coin else "ornament",
            "current_g": round(current, 2), "existing_g": 0.0, "aggregate_g": round(current, 2),
            "exceeds": False,
            "reason": "No borrower account provided — aggregate pledging cap not checked.",
        }

    existing = _cbs_pledged_weight(account_no) + _ledger_pledged_weight(account_no, exclude_case_id)
    aggregate = existing + current
    exceeds = aggregate > cap
    return {
        "checked":     True,
        "account_no":  account_no,
        "item_type":   "coin" if is_coin else "ornament",
        "cap_g":       cap,
        "current_g":   round(current, 2),
        "existing_g":  round(existing, 2),
        "aggregate_g": round(aggregate, 2),
        "exceeds":     bool(exceeds),
        "reason": (
            f"Borrower's aggregate pledged gold would be {aggregate:.0f} g "
            f"({existing:.0f} g already pledged + {current:.0f} g now), exceeding the RBI Para 16 "
            f"{'coin' if is_coin else 'ornament'} cap of {cap:.0f} g. Loan must not be sanctioned as-is."
            if exceeds else
            f"Aggregate pledged gold {aggregate:.0f} g is within the RBI Para 16 cap of {cap:.0f} g "
            f"({existing:.0f} g already pledged + {current:.0f} g now)."
        ),
        "source": "RBI Lending Against Gold Collateral Directions, 2025 — Para 16 aggregate pledging cap",
    }
