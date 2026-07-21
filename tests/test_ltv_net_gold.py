"""Regression: a miscalibrated per-stone volumetric estimate must never zero
out (or nearly zero out) net gold weight.

Real bug (2026-07-21): a calibration-card scale wobble made gem_weight's
volumetric size→carat estimate blow up (weight scales with diameter³) to
77.7 ct for 3 stones on a 10 g item, while the independent, bounded jeweller
trade-deduction table (gem_grid) correctly priced the same stones at 0.27 g.
compute_net_gold_weight used the unbounded number, capped it at the full
gross weight, and reported net_gold_weight_g = 0 — visibly wrong to any
officer, since a 10 g item obviously isn't 100% stones.
"""
import unittest

from app.utils.ltv import compute_net_gold_weight, MAX_STONE_FRACTION


class NetGoldWeightTests(unittest.TestCase):
    def test_miscalibrated_volumetric_estimate_does_not_zero_net_gold(self):
        gem_weight_result = {"total_carat": 77.714, "total_carat_high": 90.0, "n_stones": 3}
        gem_grid_result = {"total_deduction_g": 0.27, "total_stones": 3}

        net = compute_net_gold_weight(10.0, gem_weight_result, None, gem_grid_result)

        self.assertEqual(net["method"], "trade_deduction")
        self.assertAlmostEqual(net["stone_weight_g"], 0.27)
        self.assertAlmostEqual(net["net_gold_weight_g"], 9.73)
        self.assertIn("calibration_flag", net)

    def test_trade_deduction_is_primary_when_available(self):
        # Volumetric (0.09 * 5 = 0.45g / 0.2 ct-per-g => 0.45g) and trade-table
        # (0.09g) estimates agree closely enough not to trip the flag.
        gem_weight_result = {"total_carat": 0.45, "n_stones": 2}
        gem_grid_result = {"total_deduction_g": 0.09, "total_stones": 2}
        net = compute_net_gold_weight(5.0, gem_weight_result, None, gem_grid_result)
        self.assertEqual(net["method"], "trade_deduction")
        self.assertNotIn("calibration_flag", net)   # estimates roughly agree here

    def test_volumetric_fallback_is_capped_without_grid_result(self):
        # No gem_grid_result at all (e.g. no calibration card for the grid step)
        # — the volumetric estimate alone must still never claim more than
        # MAX_STONE_FRACTION of the gross weight.
        gem_weight_result = {"total_carat": 500.0, "n_stones": 3}   # absurdly miscalibrated
        net = compute_net_gold_weight(10.0, gem_weight_result, None, None)
        self.assertEqual(net["method"], "stone_weight_deduction")
        self.assertLessEqual(net["stone_weight_g"], 10.0 * MAX_STONE_FRACTION + 1e-9)
        self.assertGreaterEqual(net["net_gold_weight_g"], 10.0 * (1 - MAX_STONE_FRACTION) - 1e-9)
        self.assertIn("calibration_flag", net)

    def test_no_stones_falls_through_to_gross_weight(self):
        net = compute_net_gold_weight(10.0, None, None, None)
        self.assertEqual(net["method"], "gross_weight")
        self.assertEqual(net["net_gold_weight_g"], 10.0)


if __name__ == "__main__":
    unittest.main()
