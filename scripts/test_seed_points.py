"""Standalone assert: touching same-size stones must yield >=2 SAM seeds.

Run: python scripts/test_seed_points.py   (expects: ALL PASS)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models import ml_stone_detection as ml  # noqa: E402


def test_two_touching_get_two_seeds():
    # Two heavily-overlapping equal discs merge into ONE elongated ("peanut")
    # blob. When this is the only blob in frame, the pipeline passes
    # ref_area == this blob's own area, so area/ref == 1.0 and the old 1.6x
    # oversized trigger never fires — the old code emitted a single centroid
    # seed and SAM could never split the pair. The new elongation trigger
    # (aspect >= SEED_ASPECT_SPLIT) must multi-seed it.
    m = np.zeros((70, 140), np.uint8)
    cv2.circle(m, (50, 35), 22, 255, -1)
    cv2.circle(m, (90, 35), 22, 255, -1)   # centres 40 apart, r22 -> overlap
    ref = float((m > 0).sum())             # the merged blob's OWN area (the real case)
    pts = ml._seed_points(m, min_area=50.0, ref_area=ref)
    assert len(pts) >= 2, f"touching stones produced {len(pts)} seed(s), expected >=2"
    print(f"  two touching stones -> {len(pts)} seeds: OK")


def test_single_round_stone_one_seed():
    # A single round stone must still be a single seed (no spurious splitting).
    m = np.zeros((80, 80), np.uint8)
    cv2.circle(m, (40, 40), 22, 255, -1)
    ref = float(np.pi * 22 * 22)
    pts = ml._seed_points(m, min_area=50.0, ref_area=ref)
    assert len(pts) == 1, f"single round stone produced {len(pts)} seeds, expected 1"
    print("  single round stone -> 1 seed: OK")


if __name__ == "__main__":
    test_two_touching_get_two_seeds()
    test_single_round_stone_one_seed()
    print("ALL PASS")
