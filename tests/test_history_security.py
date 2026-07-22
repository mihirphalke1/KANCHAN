"""P2 security: authenticated tamper-evident case-history ledger.

Covers the DELETE auth + role gate, chain-preserving soft delete, the verify
endpoint, and that a direct tamper is detected. Uses an isolated history file
so it never touches real case data.
"""
import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("HISTORY_DELETE_ROLES", "branch_manager")

from fastapi.testclient import TestClient

import app.main as m
import app.routers.history as H
from app.auth import attach_selfie, create_session, load_evaluators
from app.utils.hashchain import append_with_chain, verify_chain


class HistorySecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("data/_test_case_history.json")
        self._orig = H.HISTORY_PATH
        H.HISTORY_PATH = self.tmp
        self.tmp.write_text("[]")
        self.client = TestClient(m.app)
        evs = load_evaluators()
        officer = next(e for e in evs if e["role"] == "assessment_officer")
        manager = next(e for e in evs if e["role"] == "branch_manager")
        self.otok = create_session(officer)["token"]; attach_selfie(self.otok, "x")
        self.mtok = create_session(manager)["token"]; attach_selfie(self.mtok, "x")

    def tearDown(self):
        H.HISTORY_PATH = self._orig
        self.tmp.unlink(missing_ok=True)

    def _seed(self, case_id="SEC1"):
        hist = H._load()
        case = {"case_id": case_id, "timestamp": "z", "verdict": "ACCEPT",
                "customer": {"name": "T"}}
        append_with_chain(hist, case)
        hist.append(case)
        H.HISTORY_PATH.write_text(json.dumps(hist, indent=2))

    def _seed_borderline(self, case_id="MC1", maker_id="EMP-1001"):
        from app.utils.approval import build_approval
        hist = H._load()
        case = {"case_id": case_id, "timestamp": "z",
                "verdict": {"risk_level": "BORDERLINE", "loan_action": "HOLD"},
                "evaluator": {"evaluator_id": maker_id, "name": "Maker"},
                "approval": build_approval("BORDERLINE", "HOLD", maker_id, "Maker"),
                "customer": {"name": "T"}}
        append_with_chain(hist, case)
        hist.append(case)
        H.HISTORY_PATH.write_text(json.dumps(hist, indent=2))

    def test_delete_requires_auth(self):
        self.assertEqual(self.client.delete("/api/history/x").status_code, 401)

    def test_delete_role_gated(self):
        self._seed()
        r = self.client.delete("/api/history/SEC1",
                               headers={"Authorization": f"Bearer {self.otok}"})
        self.assertEqual(r.status_code, 403)

    def test_manager_soft_delete_hides_but_preserves_chain(self):
        self._seed()
        r = self.client.delete("/api/history/SEC1",
                               headers={"Authorization": f"Bearer {self.mtok}"})
        self.assertEqual(r.status_code, 200)
        # hidden from default list, present with include_deleted
        listed = [c["case_id"] for c in self.client.get("/api/history").json()["cases"]]
        self.assertNotIn("SEC1", listed)
        all_ = [c["case_id"] for c in self.client.get("/api/history?include_deleted=true").json()["cases"]]
        self.assertIn("SEC1", all_)
        # chain still verifies (soft delete excluded from hash)
        v = self.client.get("/api/history/verify",
                            headers={"Authorization": f"Bearer {self.mtok}"}).json()
        self.assertTrue(v["ok"], v.get("reason"))

    def test_patch_requires_auth_and_logs_amendment(self):
        self._seed()
        self.assertEqual(
            self.client.patch("/api/history/SEC1", json={"phone": "9"}).status_code, 401)
        r = self.client.patch("/api/history/SEC1", json={"phone": "999"},
                              headers={"Authorization": f"Bearer {self.otok}"})
        self.assertEqual(r.status_code, 200)
        case = r.json()
        self.assertEqual(case["customer"]["phone"], "999")
        self.assertEqual(len(case["amendments"]), 1)
        v = self.client.get("/api/history/verify",
                            headers={"Authorization": f"Bearer {self.mtok}"}).json()
        self.assertTrue(v["ok"], v.get("reason"))

    def test_signoff_requires_auth(self):
        self._seed_borderline()
        self.assertEqual(
            self.client.post("/api/history/MC1/signoff", json={"decision": "approve"}).status_code, 401)

    def test_signoff_blocks_maker_self_approval(self):
        # maker == the officer session (EMP-1001) -> segregation of duties.
        self._seed_borderline(maker_id="EMP-1001")
        r = self.client.post("/api/history/MC1/signoff", json={"decision": "approve"},
                             headers={"Authorization": f"Bearer {self.otok}"})
        self.assertEqual(r.status_code, 403)
        self.assertIn("Segregation", r.json()["detail"])

    def test_second_officer_can_sign_off_and_chain_holds(self):
        self._seed_borderline(maker_id="EMP-1001")
        r = self.client.post("/api/history/MC1/signoff",
                             json={"decision": "approve", "note": "reviewed"},
                             headers={"Authorization": f"Bearer {self.mtok}"})
        self.assertEqual(r.status_code, 200, r.text)
        ap = r.json()["approval"]
        self.assertEqual(ap["status"], "approved")
        self.assertTrue(ap["closable"])
        self.assertNotEqual(ap["checker_id"], ap["maker_id"])
        # distinct audit entry recorded
        case = self.client.get("/api/history/MC1").json()
        self.assertEqual(len(case["approvals"]), 1)
        # hash chain still verifies after the restamp
        v = self.client.get("/api/history/verify",
                            headers={"Authorization": f"Bearer {self.mtok}"}).json()
        self.assertTrue(v["ok"], v.get("reason"))

    def test_signoff_rejects_bad_decision(self):
        self._seed_borderline()
        r = self.client.post("/api/history/MC1/signoff", json={"decision": "maybe"},
                             headers={"Authorization": f"Bearer {self.mtok}"})
        self.assertEqual(r.status_code, 422)

    def test_tamper_detected(self):
        self._seed()
        hist = H._load()
        hist[0]["verdict"] = "REJECT"          # silent edit, no re-stamp
        H.HISTORY_PATH.write_text(json.dumps(hist))
        v = verify_chain(H._load())
        self.assertFalse(v["ok"])
        self.assertEqual(v["broken_case"], "SEC1")


if __name__ == "__main__":
    unittest.main()
