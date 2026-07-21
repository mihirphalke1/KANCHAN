"""
Tests for the image-analysis edge-case fixes:
  - app/utils/fiducial.py   — grayscale white-balance ramp on the calibration card
  - app/utils/xray.py       — negative-space gaps, Meenakari enamel, isolated
                              highlights, touching-stone watershed, filigree,
                              multiple items
  - app/models/tarnish_model.py — antique patina vs contamination tarnish
  - app/models/image_model.py   — rose/white/antique gold alloy bands
  - app/utils/gem_grid.py   — enamel excluded from weight deduction

Run with: python -m unittest discover -s tests -v
"""
import unittest
from datetime import date

import cv2
import numpy as np


class TestFiducialWhiteBalance(unittest.TestCase):
    def _card_on_canvas(self, tint_gain=None):
        from app.utils.fiducial import generate_marker_png

        # Today's date, not a hardcoded one — detect_marker's checksum grace
        # window is relative to the real wall clock (date.today()), so a fixed
        # past date drifts out of the grace window as days pass.
        png = generate_marker_png("BLR-001", on_date=date.today())
        arr = np.frombuffer(png, dtype=np.uint8)
        card_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        canvas = np.full((900, 1200, 3), (200, 200, 200), dtype=np.uint8)
        h, w = card_img.shape[:2]
        canvas[100:100 + h, 100:100 + w] = card_img
        if tint_gain is not None:
            canvas = np.clip(canvas.astype(np.float32) * np.array(tint_gain), 0, 255).astype(np.uint8)
        return canvas

    def test_card_detected_and_scale_computed(self):
        from app.utils.fiducial import locate_card
        canvas = self._card_on_canvas()
        card = locate_card(canvas)
        self.assertIsNotNone(card)
        self.assertGreater(card["px_per_mm"], 0)

    def test_white_balance_applied_and_corrects_cast(self):
        from app.utils.fiducial import locate_card, sample_white_balance
        # Warm colour cast across the whole frame (incandescent-style: less
        # blue, more red) — including the card itself, as a real cast would.
        tinted = self._card_on_canvas(tint_gain=[0.75, 0.9, 1.15])
        card = locate_card(tinted)
        self.assertIsNotNone(card)
        wb = sample_white_balance(tinted, card)
        self.assertTrue(wb["applied"])
        gains = wb["gains_bgr"]
        # Blue was suppressed most by the cast, so correcting it back needs
        # the largest gain of the three.
        self.assertGreater(gains[0], gains[2])
        corrected = wb["corrected_image"]
        self.assertEqual(corrected.shape, tinted.shape)

    def test_fallback_when_card_not_detected(self):
        from app.utils.fiducial import sample_white_balance
        blank = np.full((400, 400, 3), 128, dtype=np.uint8)
        wb = sample_white_balance(blank)
        self.assertFalse(wb["applied"])
        self.assertTrue(np.array_equal(wb["corrected_image"], blank))
        self.assertIn("reason", wb)

    def test_checksum_tamper_evidence_still_works(self):
        # Regression guard: adding the grayscale ramp must not break the
        # pre-existing daily-checksum tamper-evidence feature.
        from app.utils.fiducial import detect_marker
        canvas = self._card_on_canvas()
        result = detect_marker(canvas, "BLR-001")
        self.assertTrue(result["detected"])
        self.assertTrue(result["checksum_valid"])


class TestEnamelDetection(unittest.TestCase):
    def test_uniform_flat_region_classified_as_enamel(self):
        from app.utils.xray import _is_enamel_region
        rng = np.random.default_rng(0)
        n = 2000
        # Flat, uniform, low-saturation-variance, no specular peaks, no edges.
        h = np.full(n, 8, dtype=np.float32) + rng.normal(0, 0.5, n)
        s = np.full(n, 200, dtype=np.float32) + rng.normal(0, 3, n)
        v = np.full(n, 140, dtype=np.float32) + rng.normal(0, 3, n)
        hsv_pixels = np.stack([h, s, v], axis=1)
        edge_pixels = np.zeros(n, dtype=np.float32)
        self.assertTrue(_is_enamel_region(hsv_pixels, edge_pixels))

    def test_faceted_stone_not_rejected_as_enamel(self):
        from app.utils.xray import _is_enamel_region
        rng = np.random.default_rng(1)
        n = 2000
        # Faceted stone: strong hue/sat variation, specular sparkle, dense edges.
        h = rng.uniform(0, 10, n).astype(np.float32)
        s = rng.uniform(120, 255, n).astype(np.float32)
        v = rng.uniform(60, 255, n).astype(np.float32)
        v[: n // 20] = 250  # specular facet flashes
        hsv_pixels = np.stack([h, s, v], axis=1)
        edge_pixels = rng.uniform(0, 255, n).astype(np.float32)  # dense facet edges
        self.assertFalse(_is_enamel_region(hsv_pixels, edge_pixels))

    def test_gem_grid_skips_enamel_stones(self):
        from app.utils.gem_grid import build_grid_stats
        stones = [
            {"bbox": [10, 10, 20, 20], "hue_class": "red", "status": "confirmed",
             "area_pct": 1.0, "material": "gemstone"},
            {"bbox": [50, 50, 20, 20], "hue_class": "green", "status": "confirmed",
             "area_pct": 1.0, "material": "enamel"},
        ]
        result = build_grid_stats([0, 0, 100, 100], stones, grid_n=4, px_per_mm=10.0)
        self.assertEqual(result["total_stones"], 1)
        # Only the gemstone should have contributed a deduction.
        self.assertGreater(result["total_deduction_g"], 0)


class TestFiligreeAndMultipleItems(unittest.TestCase):
    def test_solid_band_not_filigree(self):
        from app.utils.xray import _detect_filigree
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:150, 50:150] = True   # solid filled square
        result = _detect_filigree(mask)
        self.assertFalse(result["is_filigree"])

    def test_openwork_mesh_detected_as_filigree(self):
        from app.utils.xray import _detect_filigree
        mask = np.zeros((200, 200), dtype=bool)
        mask[40:160, 40:160] = True
        # Punch a grid of small enclosed holes through it — jali/filigree gaps.
        for y in range(50, 150, 12):
            for x in range(50, 150, 12):
                mask[y:y + 6, x:x + 6] = False
        result = _detect_filigree(mask)
        self.assertTrue(result["is_filigree"])

    def test_plain_ring_hollow_centre_not_filigree(self):
        """Regression: a normal solitaire ring is naturally mostly-hollow
        (round band, tapered shank, one open centre) and can read a low
        fill_ratio purely from that silhouette — that alone must not trigger
        filigree abstention. Real case: fill_ratio ~0.40, enclosed_gap_count
        0, incorrectly flagged is_filigree=True before this fix."""
        from app.utils.xray import _detect_filigree
        # A ring-like silhouette: a thick annulus (band) with one big hollow
        # centre — naturally low fill_ratio, but only ONE enclosed gap.
        mask_u8 = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask_u8, (100, 100), 70, 255, thickness=25)
        result = _detect_filigree(mask_u8 > 0)
        self.assertLessEqual(result["enclosed_gap_count"], 1)
        self.assertFalse(result["is_filigree"])

    def test_single_item_no_multi_warning(self):
        from app.utils.xray import _detect_multiple_items
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:150, 50:150] = True
        result = _detect_multiple_items(mask)
        self.assertFalse(result["multiple_items_detected"])

    def test_earring_pair_triggers_warning(self):
        from app.utils.xray import _detect_multiple_items
        mask = np.zeros((200, 400), dtype=bool)
        mask[50:150, 30:130] = True     # first earring
        mask[50:150, 270:370] = True    # second earring, similar size
        result = _detect_multiple_items(mask)
        self.assertTrue(result["multiple_items_detected"])
        self.assertEqual(result["component_count"], 2)
        self.assertTrue(result["likely_pair"])


class TestWatershedSplit(unittest.TestCase):
    def test_touching_circles_are_separated(self):
        from app.utils.xray import _watershed_split_touching
        cand = np.zeros((100, 160), dtype=np.uint8)
        # Tangent circles (distance between centres == sum of radii) — one
        # connected blob with a thin neck, exactly the pavé/channel-set
        # "touching, not merged" case watershed is meant to split.
        cv2.circle(cand, (50, 50), 30, 255, -1)
        cv2.circle(cand, (110, 50), 30, 255, -1)
        n_before, _ = cv2.connectedComponents(cand)
        self.assertEqual(n_before, 2)  # merged into one blob + background

        split = _watershed_split_touching(cand)
        n_after, _ = cv2.connectedComponents(split)
        self.assertGreaterEqual(n_after, 3)  # background + 2 separated stones

    def test_single_blob_left_unsplit(self):
        from app.utils.xray import _watershed_split_touching
        cand = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(cand, (50, 50), 30, 255, -1)
        split = _watershed_split_touching(cand)
        n_after, _ = cv2.connectedComponents(split)
        self.assertEqual(n_after, 2)  # background + the one stone, untouched


class TestTarnishVsPatina(unittest.TestCase):
    def _hsv_mask(self, hue, sat, val, shape=(200, 200), frac=0.3):
        hsv = np.zeros((*shape, 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = sat
        hsv[..., 2] = val
        mask = np.zeros(shape, dtype=bool)
        n_on = int(shape[0] * shape[1] * frac)
        mask.flat[:n_on] = True
        return hsv, mask

    def test_uniform_broad_dark_warm_is_patina(self):
        from app.models.tarnish_model import _classify_discoloration
        hsv, mask = self._hsv_mask(hue=25, sat=100, val=100, frac=0.4)
        result = _classify_discoloration(hsv, mask, mask.size)
        self.assertEqual(result["type"], "intentional_patina")
        self.assertLess(result["risk_multiplier"], 1.0)

    def test_localised_green_is_contamination(self):
        from app.models.tarnish_model import _classify_discoloration
        hsv, mask = self._hsv_mask(hue=60, sat=150, val=100, frac=0.1)
        result = _classify_discoloration(hsv, mask, mask.size)
        self.assertEqual(result["type"], "contamination_tarnish")
        self.assertEqual(result["risk_multiplier"], 1.0)

    def test_analyze_tarnish_end_to_end_patina_low_risk(self):
        from app.models.tarnish_model import analyze_tarnish
        # A whole solid-colour antique-patina-like image: uniform dark, warm,
        # desaturated gold — dark/desaturated enough to fall inside
        # analyze_tarnish's own tarnish_mask (v < val_min*0.6, s < sat_min*0.5
        # for karat 22: v < 66, s < 40) while still reading as a warm hue.
        hsv = np.zeros((150, 150, 3), dtype=np.uint8)
        hsv[..., 0] = 25
        hsv[..., 1] = 30
        hsv[..., 2] = 50
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        result = analyze_tarnish(buf.tobytes(), declared_karat=22)
        self.assertEqual(result["features"]["discoloration_type"], "intentional_patina")
        self.assertLess(result["risk_score"], 0.3)


class TestGoldAlloyColour(unittest.TestCase):
    def test_yellow_gold_matches(self):
        from app.models.image_model import _classify_gold_alloy_hue
        alloy, risk = _classify_gold_alloy_hue(hue=30, sat=150, val=180)
        self.assertEqual(alloy, "yellow_gold")
        self.assertLess(risk, 0.5)

    def test_rose_gold_not_flagged(self):
        from app.models.image_model import _classify_gold_alloy_hue
        alloy, risk = _classify_gold_alloy_hue(hue=12, sat=100, val=180)
        self.assertEqual(alloy, "rose_gold")
        self.assertLess(risk, 0.5)

    def test_white_gold_not_flagged(self):
        from app.models.image_model import _classify_gold_alloy_hue
        alloy, risk = _classify_gold_alloy_hue(hue=90, sat=15, val=200)
        self.assertEqual(alloy, "white_gold")
        self.assertLess(risk, 0.5)

    def test_brass_like_colour_gets_high_risk(self):
        from app.models.image_model import _classify_gold_alloy_hue
        # Strong blue/purple hue — matches no known gold alloy band.
        alloy, risk = _classify_gold_alloy_hue(hue=120, sat=200, val=180)
        self.assertEqual(alloy, "unmatched")
        self.assertGreaterEqual(risk, 0.5)


class TestGapArtifactRejection(unittest.TestCase):
    def test_concave_gap_touching_background_rejected(self):
        """A thin, concave, background-touching sliver (split-shank gap)
        should score low solidity + high background adjacency and never
        reach _region_confidence at all."""
        # Build an L-shaped (concave) region touching the mask edge.
        region = np.zeros((60, 60), dtype=np.uint8)
        region[10:50, 10:15] = 255
        region[45:50, 10:50] = 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
        solidity = cv2.contourArea(c) / hull_area
        from app.utils.xray import GAP_SOLIDITY_MIN
        self.assertLess(solidity, GAP_SOLIDITY_MIN)

    def test_oval_stone_passes_convexity_filter(self):
        region = np.zeros((60, 60), dtype=np.uint8)
        cv2.ellipse(region, (30, 30), (20, 12), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
        solidity = cv2.contourArea(c) / hull_area
        from app.utils.xray import GAP_SOLIDITY_MIN
        self.assertGreaterEqual(solidity, GAP_SOLIDITY_MIN)


class TestConvexWedgeAndSmallHighlightRealCase(unittest.TestCase):
    """
    End-to-end regression for a real production case (a bezel-set ruby ring
    photo): the gap-solidity filter alone missed a plain triangular
    split-shank wedge (it's convex), and the isolation-ratio-only highlight
    veto missed a small specular highlight sitting right next to bright gold
    (its immediate surround was already bright, so the brightness ratio
    looked unremarkable). Both need the backdrop-colour signal added
    alongside the existing ones.
    """

    def _synthetic_ring_photo(self):
        canvas = np.full((300, 300, 3), (250, 250, 250), dtype=np.uint8)  # white backdrop
        gold_bgr = (30, 170, 210)
        cv2.rectangle(canvas, (60, 60), (240, 240), gold_bgr, thickness=-1)
        # Convex wedge gap (split-shank opening) — backdrop colour, cut
        # into the gold silhouette. A triangle is already convex, so a pure
        # solidity check alone won't catch it.
        wedge = np.array([[150, 60], [140, 150], [160, 150]], dtype=np.int32)
        cv2.fillPoly(canvas, [wedge], (250, 250, 250))
        # Real gemstone, well clear of the wedge/highlight.
        cv2.circle(canvas, (110, 190), 22, (40, 40, 190), thickness=-1)  # red
        # Small, smooth, near-backdrop-coloured highlight sitting right on
        # the bright gold (not near any darker region) — mirrors the real
        # ruby/prong-junction glare case.
        cv2.circle(canvas, (205, 90), 6, (248, 248, 249), thickness=-1)
        return canvas

    def test_wedge_rejected_highlight_downgraded_stone_kept(self):
        import os
        from unittest.mock import patch
        from app.models import ml_stone_detection as M
        from app.utils import xray as X

        canvas = self._synthetic_ring_photo()
        with patch.object(M, "USE_ML_STONES", False):
            # _run_pipeline returns (stages, stats, ctx) — ctx was added later
            # for the AI-fusion reconciliation path; behaviour asserted below.
            _stages, stats, _ctx = X._run_pipeline(canvas)

        self.assertEqual(stats["stone_detection_mode"], "classical")
        stones = stats["stones"]

        # The wedge (backdrop-coloured, convex) must not appear as a stone
        # at all — bbox roughly (140-160, 60-150) in canvas coordinates.
        def overlaps_wedge(s):
            x, y, w, h = s["bbox"]
            return x < 165 and x + w > 135 and y < 155

        self.assertFalse(any(overlaps_wedge(s) for s in stones),
                          f"wedge gap should be rejected outright, got: {stones}")

        # The real ruby must still be found and confirmed.
        reds = [s for s in stones if s["hue_class"] == "red"]
        self.assertTrue(reds, f"real stone should still be detected, got: {stones}")
        self.assertEqual(reds[0]["status"], "confirmed")

        # The small highlight, if picked up as a candidate at all, must be
        # downgraded to uncertain, never asserted as a confirmed diamond.
        def near_highlight(s):
            x, y, w, h = s["bbox"]
            cx, cy = x + w / 2, y + h / 2
            return abs(cx - 205) < 15 and abs(cy - 90) < 15

        highlight_hits = [s for s in stones if near_highlight(s)]
        for s in highlight_hits:
            self.assertEqual(s["status"], "uncertain",
                              f"highlight must not be silently confirmed: {s}")


if __name__ == "__main__":
    unittest.main()
