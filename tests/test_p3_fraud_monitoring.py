"""P3 fraud-monitoring: per-evaluator Benford, maker-checker, fraud_scenario."""
import unittest

from app.benford.monitor import run_benford_test, first_significant_digit
from app.utils.approval import (build_approval, requires_maker_checker,
                                can_check, maker_checker_roles)
from app.utils.fraud_scenario import map_scenarios


class BenfordPerEvaluatorTests(unittest.TestCase):
    def test_scope_echoed(self):
        r = run_benford_test(evaluator_id="EMP-XYZ", values=None, min_samples=30)
        self.assertIn("scope", r)
        self.assertEqual(r["scope"].get("evaluator_id"), "EMP-XYZ")

    def test_explicit_clean_values_ok(self):
        # Log-uniform values are Benford-compliant by construction.
        import numpy as np
        rng = np.random.default_rng(1)
        vals = (10.0 ** rng.uniform(-1, 2.5, size=200)).tolist()
        r = run_benford_test(values=vals, min_samples=30)
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["alert"])

    def test_anomalous_values_alert(self):
        # Leading digit forced to 1/2 -> breaks the chi-squared fit.
        vals = [1.0 + (i % 100) / 100.0 for i in range(120)]
        r = run_benford_test(values=vals, min_samples=30)
        self.assertTrue(r["alert"])
        self.assertEqual(r["status"], "anomaly")

    def test_insufficient_data_carries_scope(self):
        r = run_benford_test(values=[1, 2, 3], min_samples=30)
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIn("scope", r)

    def test_first_significant_digit(self):
        self.assertEqual(first_significant_digit(0.0453), 4)
        self.assertEqual(first_significant_digit(913.2), 9)
        self.assertIsNone(first_significant_digit(0.0))


class MakerCheckerTests(unittest.TestCase):
    def test_borderline_requires_checker(self):
        self.assertTrue(requires_maker_checker("BORDERLINE", "HOLD"))
        self.assertTrue(requires_maker_checker("GENUINE", "HOLD"))
        self.assertFalse(requires_maker_checker("GENUINE", "APPROVE"))
        self.assertFalse(requires_maker_checker("REJECT", "DECLINE"))

    def test_build_approval_pending_when_required(self):
        a = build_approval("BORDERLINE", "HOLD", "EMP-1001", "Ramesh")
        self.assertTrue(a["maker_checker_required"])
        self.assertEqual(a["status"], "pending_checker")
        self.assertFalse(a["closable"])
        self.assertEqual(a["maker_id"], "EMP-1001")

    def test_build_approval_not_required_closable(self):
        a = build_approval("GENUINE", "APPROVE", "EMP-1001", "Ramesh")
        self.assertFalse(a["maker_checker_required"])
        self.assertEqual(a["status"], "not_required")
        self.assertTrue(a["closable"])

    def test_checker_must_differ_from_maker(self):
        a = build_approval("BORDERLINE", "HOLD", "EMP-1001", "Ramesh")
        ok, reason = can_check({"evaluator_id": "EMP-1001", "role": "assessment_officer"}, a)
        self.assertFalse(ok)
        self.assertIn("Segregation", reason)

    def test_different_officer_can_check(self):
        a = build_approval("BORDERLINE", "HOLD", "EMP-1001", "Ramesh")
        ok, reason = can_check({"evaluator_id": "EMP-1002", "role": "assessment_officer"}, a)
        self.assertTrue(ok, reason)

    def test_cannot_check_when_not_required(self):
        a = build_approval("GENUINE", "APPROVE", "EMP-1001", "Ramesh")
        ok, _ = can_check({"evaluator_id": "EMP-1002", "role": "assessment_officer"}, a)
        self.assertFalse(ok)

    def test_role_gate_enforced_when_configured(self):
        import os
        os.environ["MAKER_CHECKER_ROLES"] = "branch_manager"
        try:
            self.assertEqual(maker_checker_roles(), {"branch_manager"})
            a = build_approval("BORDERLINE", "HOLD", "EMP-1001", "Ramesh")
            ok, reason = can_check({"evaluator_id": "EMP-1002", "role": "assessment_officer"}, a)
            self.assertFalse(ok)
            self.assertIn("checker", reason.lower())
            ok2, _ = can_check({"evaluator_id": "EMP-9001", "role": "branch_manager"}, a)
            self.assertTrue(ok2)
        finally:
            del os.environ["MAKER_CHECKER_ROLES"]


class FraudScenarioTests(unittest.TestCase):
    def test_tungsten_maps_to_core_fill(self):
        r = map_scenarios({"density": {"karat_verdict": "TUNGSTEN_BLIND_SPOT"},
                           "stiff_core_flag": True})
        codes = {m["code"] for m in r["matched"]}
        self.assertIn("TUNGSTEN_CORE", codes)

    def test_low_density_maps_to_plating(self):
        r = map_scenarios({"density": {"karat_verdict": "LOW_DENSITY",
                                       "closest_fake": "brass"}})
        codes = {m["code"] for m in r["matched"]}
        self.assertIn("GOLD_PLATED_BASE_METAL", codes)

    def test_misdeclared_maps_to_under_karat_not_plating(self):
        r = map_scenarios({"density": {"karat_verdict": "LOW_DENSITY",
                                       "misdeclared_purity": True}})
        codes = {m["code"] for m in r["matched"]}
        self.assertIn("UNDER_KARAT", codes)
        self.assertNotIn("GOLD_PLATED_BASE_METAL", codes)

    def test_stone_fraction_triggers_inflation(self):
        r = map_scenarios({"n_stones": 5, "stone_weight_fraction": 0.3})
        codes = {m["code"] for m in r["matched"]}
        self.assertIn("STONE_WEIGHT_INFLATION", codes)

    def test_pledging_breach_maps_to_repledge(self):
        r = map_scenarios({"pledging_exceeds": True})
        codes = {m["code"] for m in r["matched"]}
        self.assertIn("REPLEDGE_OR_CAP_BREACH", codes)

    def test_no_signals_no_match(self):
        r = map_scenarios({"density": {"karat_verdict": "IN_RANGE"}})
        self.assertEqual(r["matched"], [])
        self.assertIn("No spurious-gold scenario", r["summary"])

    def test_catalog_shape(self):
        r = map_scenarios({})
        self.assertEqual(len(r["all"]), 8)   # Sl.1..Sl.8
        self.assertTrue(all("sl" in row and "code" in row for row in r["all"]))


if __name__ == "__main__":
    unittest.main()
