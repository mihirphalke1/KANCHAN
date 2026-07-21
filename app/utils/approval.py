"""Maker-checker dual sign-off (P3-12).

A BORDERLINE / HOLD verdict is exactly the ambiguous case a single officer
should not be able to close alone — it's where a corrupt or pressured maker has
the most room to wave a doubtful item through. RBI's own gold-loan guidance and
standard banking segregation-of-duties require a second, independent authorised
officer to sign off before such a case is closed.

This module is the pure policy core (which verdicts need a checker, and whether
a proposed checker is eligible); the HTTP surface lives in
app/routers/history.py (POST /history/{case_id}/signoff) and the block is first
stamped onto a case in app/routers/analyze.py at analysis time.
"""
import os


def maker_checker_roles() -> set:
    """Roles allowed to act as the CHECKER, from MAKER_CHECKER_ROLES
    (comma-separated). Empty (the default) => any authenticated officer other
    than the maker may check — segregation of duties is enforced regardless by
    the maker≠checker rule. Set e.g. "branch_manager,senior_officer" to require
    a more senior second signature."""
    raw = os.getenv("MAKER_CHECKER_ROLES", "")
    return {r.strip() for r in raw.split(",") if r.strip()}


def requires_maker_checker(risk_level: str, loan_action: str) -> bool:
    """A second sign-off is mandated when the verdict is not a clean pass — any
    BORDERLINE outcome or any HELD loan action. A clean GENUINE/APPROVE or an
    outright REJECT/DECLINE is unambiguous and closes on the maker alone."""
    return (risk_level or "").upper() == "BORDERLINE" or (loan_action or "").upper() == "HOLD"


def build_approval(risk_level: str, loan_action: str,
                   maker_id: str | None, maker_name: str | None) -> dict:
    """The approval block stamped onto a case at analysis time. `pending_checker`
    means the case CANNOT be closed until a second officer signs off."""
    required = requires_maker_checker(risk_level, loan_action)
    return {
        "maker_checker_required": required,
        "status":       "pending_checker" if required else "not_required",
        "closable":     not required,          # closable only once (if needed) a checker signs
        "maker_id":     maker_id,
        "maker_name":   maker_name,
        "checker_id":   None,
        "checker_name": None,
        "decision":     None,                  # "approved" | "rejected" once signed
        "signed_at":    None,
        "note":         None,
    }


def can_check(session: dict, approval: dict) -> tuple[bool, str]:
    """Is this authenticated officer eligible to sign off this case?
    Returns (ok, reason_if_not)."""
    if not approval or not approval.get("maker_checker_required"):
        return False, "This case does not require a second sign-off."
    if approval.get("status") in ("approved", "rejected"):
        return False, "This case has already been signed off."
    checker_id = session.get("evaluator_id")
    if checker_id and checker_id == approval.get("maker_id"):
        return False, ("Segregation of duties: the checker must be a different officer "
                       "from the maker who assessed the case.")
    roles = maker_checker_roles()
    if roles and session.get("role") not in roles:
        return False, (f"Role '{session.get('role')}' may not act as checker — "
                       f"requires one of: {', '.join(sorted(roles))}.")
    return True, ""
