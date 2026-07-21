"""
Tests for the stone-detection robustness layer:
  - app/llm/gem_vision.py    — multi-vote consensus + small-crop upscaling
                               (the fix for "one junk run zeroes out a good one"
                               and "tiny pavé stones missed at native size").
  - app/utils/stone_fusion.py — graceful degradation when the AI vision judge
                               returns an EMPTY set but the CV/ML pass is
                               confident (the "diamond ring -> 0 stones" failure).

All pure / network-free.

Run with: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

import app.llm.gem_vision as gv
import app.utils.stone_fusion as sf


def _det(x, y, conf, r=6, gem="diamond", hue="colourless"):
    return {"centroid": [x, y], "bbox": [x - r, y - r, 2 * r, 2 * r],
            "ai_confidence": conf, "gem_type": gem, "hue_class": hue}


class TestVoteConsensus(unittest.TestCase):
    def setUp(self):
        # Deterministic voting params regardless of the ambient .env.
        self._saved = (gv.GEM_VISION_AGREE_FLOOR, gv.GEM_VISION_CLUSTER_FRAC)
        gv.GEM_VISION_AGREE_FLOOR = 0.6
        gv.GEM_VISION_CLUSTER_FRAC = 0.03

    def tearDown(self):
        gv.GEM_VISION_AGREE_FLOOR, gv.GEM_VISION_CLUSTER_FRAC = self._saved

    def test_full_agreement_keeps_confidence(self):
        # Two votes both see the same two stones (nearly identical positions).
        a = [_det(100, 100, 0.9), _det(140, 100, 0.8)]
        b = [_det(101, 99, 0.9), _det(139, 101, 0.8)]
        out = gv._cluster_votes([a, b], n_success=2, short_side=400)
        self.assertEqual(len(out), 2)
        for s in out:
            self.assertEqual(s["vote_agreement"], 1.0)
            self.assertEqual(s["votes_seen"], 2)

    def test_minority_stone_dropped_when_majority_exists(self):
        # Strict-majority set: two runs agree on 2 stones; a 3rd only one run saw
        # is dropped (this is what stops dense-pavé union inflation, 46 -> 126).
        a = [_det(100, 100, 0.9), _det(140, 100, 0.8), _det(180, 100, 0.7)]
        b = [_det(101, 99, 0.9), _det(139, 101, 0.8)]     # missed the 3rd
        out = gv._cluster_votes([a, b], n_success=2, short_side=400)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(s["vote_agreement"] == 1.0 for s in out))
        self.assertNotIn(180, [s["centroid"][0] for s in out])

    def test_recovery_fallback_when_no_majority(self):
        # The headline failure: votes agree on NOTHING (one run empty) — rather
        # than report zero, recover the union at LOW agreement so the fusion
        # layer keeps them review-flagged, not asserted.
        a = [_det(100, 100, 0.9), _det(140, 100, 0.8)]
        out = gv._cluster_votes([a, []], n_success=2, short_side=400)
        self.assertEqual(len(out), 2)                     # recovered, not zeroed
        self.assertTrue(all(s["vote_agreement"] == 0.5 for s in out))
        self.assertTrue(all(s["ai_confidence"] > 0 for s in out))

    def test_single_vote_is_identity_passthrough(self):
        a = [_det(100, 100, 0.9), _det(140, 100, 0.75)]
        out = gv._cluster_votes([a], n_success=1, short_side=400)
        self.assertEqual(sorted(round(s["ai_confidence"], 3) for s in out),
                         [0.75, 0.9])
        self.assertTrue(all(s["vote_agreement"] == 1.0 for s in out))

    def test_within_run_duplicate_not_counted_as_agreement(self):
        # A stone listed twice in ONE run is a within-run dedup issue, not
        # cross-run agreement — it must not read as "seen in 2 votes".
        a = [_det(100, 100, 0.9), _det(102, 101, 0.85)]   # same stone, one run
        out = gv._cluster_votes([a], n_success=1, short_side=400)
        # Both collapse into one cluster because no second vote to pair with,
        # but agreement is bounded by the single successful vote.
        self.assertTrue(all(s["votes_seen"] == 1 for s in out))


class TestUpscale(unittest.TestCase):
    def setUp(self):
        self._saved = gv.GEM_VISION_MIN_SIDE
        gv.GEM_VISION_MIN_SIDE = 640

    def tearDown(self):
        gv.GEM_VISION_MIN_SIDE = self._saved

    def test_small_crop_upscaled_to_min_side(self):
        small = np.zeros((120, 90, 3), np.uint8)
        up = gv._upscale_for_vision(small)
        self.assertEqual(min(up.shape[:2]), 640)
        # Aspect ratio preserved.
        self.assertAlmostEqual(up.shape[1] / up.shape[0], 90 / 120, places=2)

    def test_large_crop_not_downscaled(self):
        big = np.zeros((800, 700, 3), np.uint8)
        self.assertEqual(gv._upscale_for_vision(big).shape[:2], (800, 700))

    def test_disabled_when_min_side_zero(self):
        gv.GEM_VISION_MIN_SIDE = 0
        small = np.zeros((120, 90, 3), np.uint8)
        self.assertEqual(gv._upscale_for_vision(small).shape[:2], (120, 90))


class TestGracefulDegradation(unittest.TestCase):
    def _fixture(self):
        labels = np.zeros((50, 50), np.int32)
        labels[10:20, 10:20] = 1
        labels[30:40, 30:40] = 2
        item = np.ones((50, 50), bool)
        ml = [
            {"_label": 1, "confidence": 0.72, "status": "confirmed",
             "hue_class": "colourless", "stone_name": "diamond", "area_pct": 4.0,
             "bbox": [10, 10, 10, 10], "centroid": [15, 15]},
            {"_label": 2, "confidence": 0.45, "status": "uncertain",
             "hue_class": "colourless", "stone_name": "diamond", "area_pct": 4.0,
             "bbox": [30, 30, 10, 10], "centroid": [35, 35]},
        ]
        return ml, labels, item

    def setUp(self):
        self._saved = (sf.RESCUE_ML_ON_AI_EMPTY, sf.RESCUE_ML_MIN_CONF, sf.AI_ONLY)
        sf.RESCUE_ML_ON_AI_EMPTY = True
        sf.RESCUE_ML_MIN_CONF = 0.60
        sf.AI_ONLY = True

    def tearDown(self):
        sf.RESCUE_ML_ON_AI_EMPTY, sf.RESCUE_ML_MIN_CONF, sf.AI_ONLY = self._saved

    def test_ai_empty_rescues_confident_ml_as_review(self):
        ml, labels, item = self._fixture()
        out, _, meta = sf.reconcile(ml, labels, [], item)
        # Confident (0.72) rescued, weak (0.45) dropped.
        self.assertEqual(len(out), 1)
        self.assertEqual(meta["n_rescued_ai_empty"], 1)
        self.assertTrue(meta["ai_empty_ml_disagree"])
        s = out[0]
        self.assertEqual(s["agreement"], "ml_only_ai_empty")
        self.assertEqual(s["status"], "uncertain")     # never asserted confirmed
        self.assertTrue(s["needs_review"])

    def test_ai_none_is_unchanged_ml_passthrough(self):
        ml, labels, item = self._fixture()
        out, _, meta = sf.reconcile(ml, labels, None, item)
        self.assertEqual(len(out), 2)                  # nothing dropped
        self.assertEqual(meta["mode"], "ml_only")

    def test_nonempty_ai_stays_authoritative_no_rescue(self):
        ml, labels, item = self._fixture()
        ai = [{"gem_type": "ruby", "colour": "red", "hue_class": "red",
               "ai_confidence": 0.9, "centroid": [15, 15], "bbox": [10, 10, 10, 10]}]
        out, _, meta = sf.reconcile(ml, labels, ai, item)
        self.assertEqual(meta["n_rescued_ai_empty"], 0)
        self.assertFalse(meta["ai_empty_ml_disagree"])
        self.assertEqual(meta["n_both"], 1)            # AI owns the set

    def test_rescue_can_be_disabled(self):
        sf.RESCUE_ML_ON_AI_EMPTY = False
        ml, labels, item = self._fixture()
        out, _, meta = sf.reconcile(ml, labels, [], item)
        self.assertEqual(len(out), 0)                  # strict AI-only: all dropped
        self.assertEqual(meta["n_rescued_ai_empty"], 0)


class TestVoteConsensusGate(unittest.TestCase):
    """A low-consensus AI stone is kept but never asserted confirmed."""
    def setUp(self):
        self._saved = (sf.AI_PRIMARY, sf.AI_CONFIRM_MIN)
        sf.AI_PRIMARY = True
        sf.AI_CONFIRM_MIN = 0.45

    def tearDown(self):
        sf.AI_PRIMARY, sf.AI_CONFIRM_MIN = self._saved

    def _ai(self, low_consensus):
        return [{"gem_type": "diamond", "colour": "white", "hue_class": "colourless",
                 "ai_confidence": 0.9, "low_consensus": low_consensus,
                 "centroid": [25, 25], "bbox": [20, 20, 10, 10]}]

    def test_agreed_high_conf_confirmed(self):
        labels = np.zeros((50, 50), np.int32)
        out, _, _ = sf.reconcile([], labels, self._ai(False), np.ones((50, 50), bool))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "confirmed")
        self.assertNotIn("needs_review", out[0])

    def test_low_consensus_capped_to_review(self):
        labels = np.zeros((50, 50), np.int32)
        out, _, meta = sf.reconcile([], labels, self._ai(True), np.ones((50, 50), bool))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "uncertain")   # high conf, but low consensus
        self.assertTrue(out[0]["needs_review"])
        self.assertEqual(meta["n_needs_review"], 1)


class TestOpenworkEnvelope(unittest.TestCase):
    """AI-only stones floated in openwork gaps must be validated against the
    ornament envelope, not the pixel-precise metal mask."""
    def setUp(self):
        self._saved = (sf.AI_PRIMARY, sf.AI_ONLY, sf.AI_CONFIRM_MIN)
        sf.AI_PRIMARY = True
        sf.AI_ONLY = True
        sf.AI_CONFIRM_MIN = 0.45

    def tearDown(self):
        sf.AI_PRIMARY, sf.AI_ONLY, sf.AI_CONFIRM_MIN = self._saved

    def _scene(self):
        # A ring-like metal mask with a big empty CENTRE (openwork). The stone
        # sits in that centre hole — outside item_bool, inside the hull envelope.
        H = W = 100
        item = np.zeros((H, W), bool)
        item[20:80, 20:80] = True
        item[35:65, 35:65] = False            # openwork centre (a hole)
        envelope = np.ones((H, W), bool)       # hull would fill the whole square
        ai = [{"gem_type": "diamond", "hue_class": "colourless", "ai_confidence": 0.9,
               "centroid": [50, 50], "bbox": [46, 46, 8, 8]}]   # in the hole
        return item, envelope, ai

    def test_openwork_stone_dropped_without_envelope(self):
        item, _env, ai = self._scene()
        labels = np.zeros(item.shape, np.int32)
        out, _, _ = sf.reconcile([], labels, ai, item)          # envelope defaults to item_bool
        self.assertEqual(len(out), 0)                            # falls in the hole -> dropped

    def test_openwork_stone_kept_with_envelope(self):
        item, env, ai = self._scene()
        labels = np.zeros(item.shape, np.int32)
        out, _, _ = sf.reconcile([], labels, ai, item, item_envelope=env)
        self.assertEqual(len(out), 1)                            # envelope covers the gap
        self.assertEqual(out[0]["agreement"], "ai_only")

    def test_sam_overgrow_falls_back_to_bbox(self):
        item, env, ai = self._scene()
        labels = np.zeros(item.shape, np.int32)
        # ai_mask_fn (SAM) over-grows to the whole envelope (> area cap); the
        # tight bbox must rescue the stone instead of it being dropped.
        big = np.ones(item.shape, bool)
        out, _, _ = sf.reconcile([], labels, ai, item, ai_mask_fn=lambda a: big, item_envelope=env)
        self.assertEqual(len(out), 1)
        # Kept via bbox fallback -> its area is the small bbox, not the whole item.
        self.assertLess(out[0]["area_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
