"""
Seed a Head-Office "admin" evaluator so the multi-branch admin dashboard
(app/routers/admin.py) has a role that can actually see every branch/region —
existing evaluators are all scoped to BLR-001 as assessment_officer/branch_manager.

Idempotent: skips if EMP-9999 already exists. Run once:
    venv/bin/python scripts/seed_regional_admin.py
"""
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import EVALUATORS_PATH, _hash_pin, load_evaluators  # noqa: E402

ADMIN_ID   = "EMP-9999"
ADMIN_PIN  = "5555"
ADMIN_NAME = "Head Office Admin (Demo)"


def main():
    evaluators = load_evaluators()
    if any(e["evaluator_id"] == ADMIN_ID for e in evaluators):
        print(f"{ADMIN_ID} already present — nothing to do.")
        return
    salt = secrets.token_hex(8)
    evaluators.append({
        "evaluator_id": ADMIN_ID,
        "name":         ADMIN_NAME,
        "branch_id":    "HO-000",
        "role":         "admin",
        "salt":         salt,
        "pin_hash":     _hash_pin(ADMIN_PIN, salt),
    })
    EVALUATORS_PATH.write_text(json.dumps(evaluators, indent=2))
    print(f"Seeded {ADMIN_ID} ({ADMIN_NAME}), role=admin, branch_id=HO-000, PIN={ADMIN_PIN} (demo only).")


if __name__ == "__main__":
    main()
