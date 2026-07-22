"""
Multi-branch / regional admin dashboard (P-C, mobile-audit Section C).

Everything in History and Dashboard is scoped to a single case or a single
branch officer's own view. This router adds the aggregate, cross-branch view
a regional/HO admin actually needs: verdict distribution, Benford alerts,
contradiction-flag rate, cross-branch pHash duplicate hits, and officer
activity — each filterable by branch/region and exportable as CSV.

RBAC: a `branch_manager` is always scoped to their OWN branch_id regardless of
any branch_id/region_id query param (a branch manager should not be able to
pull another branch's aggregate numbers just by editing the URL). Only roles
in ADMIN_FULL_ACCESS_ROLES (default "admin") see across branches/regions.
Both roles must additionally be in ADMIN_DASHBOARD_ROLES to reach this router
at all — an assessment_officer gets a 403.

No new data store: this reads the same flat-file sources every other router
already writes (case_history.json, density_log.csv via benford/monitor.py,
photo_hashes.json), so there is nothing to keep in sync.
"""
import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import require_login
from app.benford.monitor import run_benford_test
from app.routers.history import _is_deleted, _load as _load_history
from app.utils.numpy_safe import numpy_safe
from app.utils.phash import DUPLICATE_HAMMING_THRESHOLD, HASH_INDEX_PATH, hamming_distance

router = APIRouter()

IST = timezone(timedelta(hours=5, minutes=30))

BRANCHES_PATH = Path("data/branches.json")

# Clustering all photo hashes is O(n^2); past this size fall back to exact-hash
# grouping only (still catches the common case: same file re-uploaded).
DUPLICATE_CLUSTER_MAX_N = 3000


def _dashboard_roles() -> set:
    raw = os.getenv("ADMIN_DASHBOARD_ROLES", "branch_manager,admin")
    return {r.strip() for r in raw.split(",") if r.strip()}


def _full_access_roles() -> set:
    raw = os.getenv("ADMIN_FULL_ACCESS_ROLES", "admin")
    return {r.strip() for r in raw.split(",") if r.strip()}


def _load_branches() -> dict:
    if BRANCHES_PATH.exists():
        try:
            return json.loads(BRANCHES_PATH.read_text())
        except Exception:
            return {}
    return {}


def _require_dashboard_access(session: dict) -> dict:
    role = session.get("role")
    if role not in _dashboard_roles():
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' cannot access the admin dashboard — requires one of: "
                   f"{', '.join(sorted(_dashboard_roles())) or '(none)'}",
        )
    return session


def _resolve_scope(session: dict, branch_id: Optional[str], region_id: Optional[str],
                    branches: dict) -> dict:
    """Which branch_ids this request is allowed to see, honouring RBAC."""
    full_access = session.get("role") in _full_access_roles()
    all_branch_ids = set(branches.keys())

    if not full_access:
        own = session.get("branch_id")
        return {"branch_ids": {own} if own else set(), "restricted_to_own_branch": True,
                "region_id": branches.get(own, {}).get("region_id")}

    if branch_id:
        return {"branch_ids": {branch_id}, "restricted_to_own_branch": False, "region_id": region_id}
    if region_id:
        ids = {bid for bid, meta in branches.items() if meta.get("region_id") == region_id}
        return {"branch_ids": ids, "restricted_to_own_branch": False, "region_id": region_id}
    return {"branch_ids": all_branch_ids, "restricted_to_own_branch": False, "region_id": None}


def _case_evaluator_id(case: dict) -> Optional[str]:
    ev = case.get("evaluator") or {}
    return ev.get("evaluator_id")


def _verdict_distribution(cases: list, branches: dict) -> list:
    by_branch: dict = {}
    for c in cases:
        b = c.get("branch_id") or "default"
        row = by_branch.setdefault(b, {"branch_id": b, "genuine": 0, "borderline": 0, "reject": 0, "total": 0})
        rl = (c.get("verdict") or {}).get("risk_level", "").upper()
        if rl == "GENUINE":
            row["genuine"] += 1
        elif rl == "BORDERLINE":
            row["borderline"] += 1
        elif rl == "REJECT":
            row["reject"] += 1
        row["total"] += 1
    out = []
    for b, row in by_branch.items():
        meta = branches.get(b, {})
        row["branch_name"] = meta.get("name", b)
        row["region_id"] = meta.get("region_id")
        row["region_name"] = meta.get("region_name")
        out.append(row)
    out.sort(key=lambda r: -r["total"])
    return out


def _benford_alerts(branch_ids: set, cases: list) -> list:
    alerts = []
    for b in sorted(branch_ids):
        result = run_benford_test(branch_id=b)
        if result["status"] != "insufficient_data":
            alerts.append({"scope_type": "branch", "branch_id": b, **result})

    evaluator_ids = sorted({eid for c in cases if (eid := _case_evaluator_id(c))})
    for eid in evaluator_ids:
        result = run_benford_test(evaluator_id=eid)
        if result["status"] == "anomaly":
            alerts.append({"scope_type": "evaluator", "evaluator_id": eid, **result})
    alerts.sort(key=lambda a: (not a.get("alert"), a.get("p_value") if a.get("p_value") is not None else 1.0))
    return alerts


def _contradiction_rate(cases: list, branches: dict) -> list:
    by_branch: dict = {}
    for c in cases:
        b = c.get("branch_id") or "default"
        row = by_branch.setdefault(b, {"branch_id": b, "flagged": 0, "total": 0})
        row["total"] += 1
        if (c.get("contradiction") or {}).get("flags"):
            row["flagged"] += 1
    out = []
    for b, row in by_branch.items():
        meta = branches.get(b, {})
        row["branch_name"] = meta.get("name", b)
        row["rate"] = round(row["flagged"] / row["total"], 4) if row["total"] else 0.0
        out.append(row)
    out.sort(key=lambda r: -r["rate"])
    return out


def _duplicate_feed(branch_ids: set, all_cases: list) -> list:
    if not HASH_INDEX_PATH.exists():
        return []
    try:
        index = json.loads(HASH_INDEX_PATH.read_text())
    except Exception:
        return []
    if not index:
        return []

    case_branch = {c.get("case_id"): (c.get("branch_id") or "default") for c in all_cases}

    # Union-find over hamming-close hashes (or exact-match grouping if too large).
    n = len(index)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if n <= DUPLICATE_CLUSTER_MAX_N:
        for i in range(n):
            for j in range(i + 1, n):
                if hamming_distance(index[i]["hash"], index[j]["hash"]) <= DUPLICATE_HAMMING_THRESHOLD:
                    union(i, j)
    else:
        by_hash: dict = {}
        for i, entry in enumerate(index):
            by_hash.setdefault(entry["hash"], []).append(i)
        for group in by_hash.values():
            for i in group[1:]:
                union(group[0], i)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(index[i])

    feed = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        case_ids = sorted({m["case_id"] for m in members})
        if len(case_ids) < 2:
            continue
        member_branches = sorted({case_branch.get(cid, "unknown") for cid in case_ids})
        if not (set(member_branches) & branch_ids):
            continue
        feed.append({
            "case_ids": case_ids,
            "branches": member_branches,
            "cross_branch": len(member_branches) > 1,
            "image_roles": sorted({m.get("image_role") for m in members if m.get("image_role")}),
            "occurrences": len(members),
        })
    feed.sort(key=lambda f: (not f["cross_branch"], -f["occurrences"]))
    return feed


def _officer_activity(cases: list) -> list:
    by_officer: dict = {}
    for c in cases:
        ev = c.get("evaluator") or {}
        eid = ev.get("evaluator_id") or "unknown"
        row = by_officer.setdefault(eid, {
            "evaluator_id": eid, "name": ev.get("name", "-"), "branch_id": ev.get("branch_id") or c.get("branch_id"),
            "total": 0, "genuine": 0, "borderline": 0, "reject": 0,
            "signed_off": 0, "rejected_signoffs": 0, "pending_checker": 0,
        })
        row["total"] += 1
        rl = (c.get("verdict") or {}).get("risk_level", "").upper()
        if rl == "GENUINE":
            row["genuine"] += 1
        elif rl == "BORDERLINE":
            row["borderline"] += 1
        elif rl == "REJECT":
            row["reject"] += 1
        approval = c.get("approval") or {}
        if approval.get("status") == "pending_checker":
            row["pending_checker"] += 1
        elif approval.get("status") in ("approved", "rejected"):
            row["signed_off"] += 1
            if approval.get("status") == "rejected":
                row["rejected_signoffs"] += 1

    out = []
    for row in by_officer.values():
        row["override_rate"] = round(row["rejected_signoffs"] / row["signed_off"], 4) if row["signed_off"] else 0.0
        out.append(row)
    out.sort(key=lambda r: -r["total"])
    return out


def _build_overview(session: dict, branch_id: Optional[str], region_id: Optional[str]) -> dict:
    branches = _load_branches()
    scope = _resolve_scope(session, branch_id, region_id, branches)
    branch_ids = scope["branch_ids"]

    all_cases = [c for c in _load_history() if not _is_deleted(c)]
    cases = [c for c in all_cases if (c.get("branch_id") or "default") in branch_ids] if branch_ids else []

    return {
        "generated_at": datetime.now(IST).isoformat(),
        "scope": {
            "role": session.get("role"),
            "restricted_to_own_branch": scope["restricted_to_own_branch"],
            "branch_ids": sorted(branch_ids),
            "region_id": scope["region_id"],
        },
        "available_branches": (
            {branch_id_: meta for branch_id_, meta in branches.items()}
            if session.get("role") in _full_access_roles()
            else {b: branches.get(b, {}) for b in branch_ids}
        ),
        "total_cases_in_scope": len(cases),
        "verdict_distribution": _verdict_distribution(cases, branches),
        "benford_alerts":       _benford_alerts(branch_ids, cases),
        "contradiction_rate":   _contradiction_rate(cases, branches),
        "duplicate_feed":       _duplicate_feed(branch_ids, all_cases),
        "officer_activity":     _officer_activity(cases),
    }


@router.get("/admin/overview")
async def admin_overview(
    branch_id: Optional[str] = Query(default=None),
    region_id: Optional[str] = Query(default=None),
    session: dict = Depends(require_login),
):
    _require_dashboard_access(session)
    return JSONResponse(content=numpy_safe(_build_overview(session, branch_id, region_id)))


@router.get("/admin/export")
async def admin_export(
    view: str = Query(..., pattern="^(verdicts|benford|contradiction|duplicates|officers)$"),
    branch_id: Optional[str] = Query(default=None),
    region_id: Optional[str] = Query(default=None),
    session: dict = Depends(require_login),
):
    _require_dashboard_access(session)
    data = _build_overview(session, branch_id, region_id)

    view_map = {
        "verdicts":     ("verdict_distribution", ["branch_id", "branch_name", "region_name", "genuine", "borderline", "reject", "total"]),
        "benford":      ("benford_alerts", ["scope_type", "branch_id", "evaluator_id", "status", "n_samples", "p_value", "chi_square", "alert", "message"]),
        "contradiction": ("contradiction_rate", ["branch_id", "branch_name", "flagged", "total", "rate"]),
        "duplicates":   ("duplicate_feed", ["case_ids", "branches", "cross_branch", "image_roles", "occurrences"]),
        "officers":     ("officer_activity", ["evaluator_id", "name", "branch_id", "total", "genuine", "borderline", "reject", "signed_off", "rejected_signoffs", "pending_checker", "override_rate"]),
    }
    key, fields = view_map[view]
    rows = data[key]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat = dict(row)
        for k, v in flat.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v)
        writer.writerow(flat)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="kanchan_admin_{view}.csv"'},
    )
