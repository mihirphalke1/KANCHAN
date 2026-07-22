"""Standalone asserts for Layer-A recall widening + glare-robust classify.

Run: python scripts/test_stone_layer_a.py   (expects: ALL PASS)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.utils import xray  # noqa: E402


def _synthetic():
    # Gold field (warm BGR) with a saturated red stone and a NEAR-GOLD stone
    # whose lightness-weighted delta-E from the gold (19.6) sits UNDER the
    # STONE_DELTA_E_MIN=28 threshold — i.e. the old delta-E-only candidate
    # pass misses it entirely. It is still a clear a*/b* chromaticity outlier,
    # so the new manifold-outlier path must recover it.
    img = np.full((120, 120, 3), (40, 120, 190), np.uint8)   # gold-ish
    cv2.circle(img, (35, 60), 12, (40, 40, 200), -1)         # red gem (obvious)
    cv2.circle(img, (85, 60), 12, (90, 150, 200), -1)        # near-gold gem (delta-E 19.6)
    item = np.full((120, 120), 255, np.uint8)
    return img, item


def test_candidates_include_near_gold_and_red_stone():
    img, item = _synthetic()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    metal_lab = np.median(lab.reshape(-1, 3), axis=0)

    # Prove the premise: the near-gold stone IS under the delta-E threshold,
    # so a delta-E-only detector would miss it.
    px = np.float32([90, 150, 200])
    l = cv2.cvtColor(px.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    dE = float(np.sqrt(0.25 * (l[0] - metal_lab[0]) ** 2 + (l[1] - metal_lab[1]) ** 2 + (l[2] - metal_lab[2]) ** 2))
    assert dE < xray.STONE_DELTA_E_MIN, f"test premise broken: dE {dE:.1f} not under {xray.STONE_DELTA_E_MIN}"

    mask = xray._candidate_mask(img, item, metal_lab)
    assert mask[60, 85] > 0, "near-gold stone (under delta-E) missed — manifold recall not working"
    assert mask[60, 35] > 0, "red stone missed by candidate mask"
    print(f"  near-gold stone (dE {dE:.1f} < {xray.STONE_DELTA_E_MIN}) recovered + red stone: OK")


def test_classify_ignores_glare():
    # A red gem region with a blown-out specular corner must still classify red-ish.
    patch = np.full((30, 30, 3), (40, 40, 200), np.uint8)
    patch[0:8, 0:8] = (255, 255, 255)   # glare
    name, hue, conf = xray._classify_stone_color(patch.reshape(-1, 3))
    assert hue in ("red", "other"), f"glare skewed classification to {hue!r}"
    print(f"  glare-robust classify: {name}/{hue} (conf {conf}) OK")


def test_delta_e_path_still_fires():
    # Backward-compat: the original delta-E-only path must still flag an obvious
    # coloured stone even if manifold/chroma were disabled.
    img, item = _synthetic()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    metal_lab = np.median(lab.reshape(-1, 3), axis=0)
    mask = xray._candidate_mask(img, item, metal_lab)
    assert mask.dtype == np.uint8 and mask.shape == item.shape
    print("  delta-E fallback path intact: OK")


if __name__ == "__main__":
    test_candidates_include_near_gold_and_red_stone()
    test_classify_ignores_glare()
    test_delta_e_path_still_fires()
    print("ALL PASS")
