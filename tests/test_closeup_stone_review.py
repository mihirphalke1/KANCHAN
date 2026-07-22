"""
Tests for the close-up stone photo review layer:
  - app/utils/stone_fusion.py::review_closeup_photos
  - app/utils/xray.py::rebuild_gem_summaries

The real-world problem this solves: the calibration card only has a scale
reference in the PRIMARY photo, so a ring/bangle that can't show both the
card and its small stones clearly in one frame needs a second, close-up
photo of just the stones. That close-up has no scale of its own, so it must
never be allowed to invent stone size/weight — only flag a higher count, or
refine a vague "colourless" stone to a confirmed type.

Run with: python -m unittest discover -s tests -v
"""
import unittest

from app.utils.stone_fusion import review_closeup_photos
from app.utils.xray import rebuild_gem_summaries


def _stone(hue_class="colourless", area_pct=2.0, confidence=0.7, status="confirmed"):
    return {"hue_class": hue_class, "area_pct": area_pct, "confidence": confidence,
            "status": status, "stone_name": "stone"}


def _gem(hue_class, conf=0.9):
    return {"hue_class": hue_class, "ai_confidence": conf}


class TestReviewCloseupPhotos(unittest.TestCase):
    def test_no_closeups_is_a_noop(self):
        primary = [_stone(), _stone()]
        r = review_closeup_photos(primary, [])
        self.assertEqual(r["stones"], primary)
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["reviews"], [])

    def test_higher_closeup_count_only_flags_never_fabricates(self):
        primary = [_stone(), _stone()]
        closeups = [{"photo_index": 1, "gems": [_gem("red"), _gem("red"), _gem("red")]}]
        r = review_closeup_photos(primary, closeups)
        # Never invent a 3rd stone position/size from an unscaled photo.
        self.assertEqual(len(r["stones"]), 2)
        self.assertEqual(len(r["flags"]), 1)
        self.assertIn("photo #2", r["flags"][0])
        self.assertIn("3 stone(s)", r["flags"][0])

    def test_lower_or_equal_closeup_count_does_not_flag(self):
        primary = [_stone(), _stone(), _stone()]
        closeups = [{"photo_index": 1, "gems": [_gem("red"), _gem("red")]}]
        r = review_closeup_photos(primary, closeups)
        self.assertEqual(r["flags"], [])

    def test_unambiguous_closeup_refines_colourless_type(self):
        primary = [_stone(hue_class="colourless"), _stone(hue_class="colourless")]
        closeups = [{"photo_index": 1, "gems": [_gem("red", 0.9), _gem("red", 0.85)]}]
        r = review_closeup_photos(primary, closeups)
        self.assertTrue(all(s["hue_class"] == "red" for s in r["stones"]))
        self.assertTrue(all(s.get("type_refined_from_photo") == 2 for s in r["stones"]))

    def test_low_confidence_closeup_does_not_refine(self):
        primary = [_stone(hue_class="colourless")]
        closeups = [{"photo_index": 1, "gems": [_gem("red", 0.3)]}]
        r = review_closeup_photos(primary, closeups)
        self.assertEqual(r["stones"][0]["hue_class"], "colourless")

    def test_mixed_types_in_closeup_does_not_refine(self):
        # Ambiguous close-up (two different hue classes) must not guess a mapping.
        primary = [_stone(hue_class="colourless")]
        closeups = [{"photo_index": 1, "gems": [_gem("red", 0.9), _gem("green", 0.9)]}]
        r = review_closeup_photos(primary, closeups)
        self.assertEqual(r["stones"][0]["hue_class"], "colourless")

    def test_already_typed_stone_is_not_overwritten(self):
        # A stone the primary photo already classified confidently keeps its type
        # even if a close-up disagrees — refinement only fills in "colourless".
        primary = [_stone(hue_class="blue")]
        closeups = [{"photo_index": 1, "gems": [_gem("red", 0.95)]}]
        r = review_closeup_photos(primary, closeups)
        self.assertEqual(r["stones"][0]["hue_class"], "blue")

    def test_never_touches_size_or_area(self):
        primary = [_stone(area_pct=3.3)]
        closeups = [{"photo_index": 1, "gems": [_gem("red", 0.9)]}]
        r = review_closeup_photos(primary, closeups)
        self.assertEqual(r["stones"][0]["area_pct"], 3.3)


class TestRebuildGemSummaries(unittest.TestCase):
    def test_splits_by_hue_class(self):
        stones = [_stone(hue_class="red", area_pct=1.0),
                  _stone(hue_class="colourless", area_pct=2.0)]
        out = rebuild_gem_summaries(stones)
        self.assertEqual(out["gem_regions"], 1)
        self.assertEqual(out["colourless_regions"], 1)
        self.assertAlmostEqual(out["gem_area_pct"], 1.0)
        self.assertAlmostEqual(out["colourless_area_pct"], 2.0)

    def test_refined_stones_move_bucket(self):
        # A stone refined from colourless -> red must show up in "gems", not
        # "colourless", after rebuild — this is the exact bug this function
        # exists to avoid (stale summaries after an in-place hue_class edit).
        stones = [_stone(hue_class="colourless", area_pct=1.5)]
        stones[0]["hue_class"] = "red"
        out = rebuild_gem_summaries(stones)
        self.assertEqual(out["gem_regions"], 1)
        self.assertEqual(out["colourless_regions"], 0)
        self.assertEqual(out["gems"][0]["hue_class"], "red")


if __name__ == "__main__":
    unittest.main()
