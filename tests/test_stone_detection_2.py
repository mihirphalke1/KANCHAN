"""Stone-detection hardening: empty-region crash guard, full-res + tiling
helpers, and the water-density table extension (P6)."""
import unittest

import numpy as np

from app.utils.xray import _classify_stone_color, xray_preview
from app.llm.gem_vision import (_tile_boxes, _pad_box, _scale_stone,
                                _merge_union, _same_stone)
from app.utils.density import water_density, WATER_DENSITY_TABLE


class ClassifyStoneColorGuardTests(unittest.TestCase):
    def test_empty_pixels_do_not_crash(self):
        # The bug: an empty region (zero pixels) reached cv2.cvtColor -> assert.
        name, bucket, conf = _classify_stone_color(np.empty((0, 3), dtype=np.uint8))
        self.assertEqual(name, "unidentified")
        self.assertEqual(conf, 0.0)

    def test_none_pixels_do_not_crash(self):
        name, bucket, conf = _classify_stone_color(None)
        self.assertEqual(name, "unidentified")

    def test_normal_pixels_still_classify(self):
        red = np.zeros((50, 3), dtype=np.uint8)
        red[:, 2] = 200          # BGR red channel
        name, bucket, conf = _classify_stone_color(red)
        self.assertIsInstance(name, str)
        self.assertGreaterEqual(conf, 0.0)


class TilingHelperTests(unittest.TestCase):
    def test_tile_boxes_cover_grid(self):
        boxes = _tile_boxes([0, 0, 100, 100], grid=2, overlap=0.1)
        self.assertEqual(len(boxes), 4)
        for b in boxes:
            self.assertTrue(0 <= b[0] < b[2] <= 100 and 0 <= b[1] < b[3] <= 100)

    def test_tile_boxes_overlap(self):
        # With overlap, adjacent tiles share a seam region.
        boxes = _tile_boxes([0, 0, 100, 100], grid=2, overlap=0.2)
        # top-left tile x1 should exceed 50 (the naive midline) due to overlap
        self.assertGreater(boxes[0][2], 50)

    def test_pad_box_clips_to_shape(self):
        b = _pad_box([10, 10, 90, 90], (100, 100), 0.5)
        self.assertGreaterEqual(b[0], 0)
        self.assertLessEqual(b[2], 100)

    def test_scale_stone_maps_coords(self):
        det = {"centroid": [100, 200], "bbox": [90, 190, 20, 20]}
        s = _scale_stone(det, 0.5)
        self.assertEqual(s["centroid"], [50, 100])
        self.assertEqual(s["bbox"], [45, 95, 10, 10])

    def test_scale_stone_identity(self):
        det = {"centroid": [100, 200], "bbox": [90, 190, 20, 20]}
        self.assertIs(_scale_stone(det, 1.0), det)

    def test_merge_union_dedups_same_stone(self):
        base = [{"centroid": [100, 100], "bbox": [95, 95, 10, 10], "ai_confidence": 0.9}]
        extra = [{"centroid": [101, 101], "bbox": [96, 96, 10, 10], "ai_confidence": 0.7}]  # same stone
        merged = _merge_union(base, extra, short_side=200.0)
        self.assertEqual(len(merged), 1)

    def test_merge_union_keeps_distinct(self):
        base = [{"centroid": [10, 10], "bbox": [5, 5, 10, 10], "ai_confidence": 0.9}]
        extra = [{"centroid": [180, 180], "bbox": [175, 175, 10, 10], "ai_confidence": 0.7}]
        merged = _merge_union(base, extra, short_side=200.0)
        self.assertEqual(len(merged), 2)
        # The recovered (extra) stone is flagged low-consensus.
        recovered = [m for m in merged if m["centroid"] == [180, 180]][0]
        self.assertTrue(recovered.get("low_consensus"))


class WaterDensityExtensionTests(unittest.TestCase):
    def test_table_spans_validator_range(self):
        temps = [t for t, _ in WATER_DENSITY_TABLE]
        self.assertLessEqual(min(temps), 0.0)
        self.assertGreaterEqual(max(temps), 45.0)

    def test_in_range_temps_interpolated_not_clamped(self):
        # 3 C and 43 C must differ from the old endpoints (10 C / 40 C), i.e.
        # no longer silently clamped.
        self.assertNotAlmostEqual(water_density(3.0), water_density(10.0), places=5)
        self.assertNotAlmostEqual(water_density(43.0), water_density(40.0), places=5)

    def test_monotonic_decrease_above_4c(self):
        self.assertGreater(water_density(20.0), water_density(40.0))


class EarringPairNoCrashTests(unittest.TestCase):
    """The earring-pair frame crashed the pipeline (empty region -> cvtColor).
    Regression: the pipeline must complete on a synthetic multi-blob frame."""
    def test_pipeline_completes_on_synthetic_frame(self):
        img = np.full((240, 240, 3), 220, dtype=np.uint8)
        # two bright metal blobs with darker 'stone' centres
        import cv2
        for cx in (70, 170):
            cv2.circle(img, (cx, 120), 30, (60, 180, 230), -1)
            cv2.circle(img, (cx, 120), 8, (230, 230, 230), -1)
        ok, buf = cv2.imencode(".jpg", img)
        result = xray_preview(buf.tobytes())
        self.assertIn("stages", result)


if __name__ == "__main__":
    unittest.main()
