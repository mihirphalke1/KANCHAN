"""Standalone checks for the carat estimator. Run: python scripts/test_gem_weight.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.gem_weight import estimate_gem_weights   # noqa: E402

# A ~6.5mm round ruby (bbox ~65px at 10 px/mm) should land in a sane carat
# range for a stone that size (roughly ~1 ct order of magnitude, not 0.01 or 100).
stones = [{"bbox": [0, 0, 65, 65], "area_pct": 5.0, "hue_class": "red",
           "stone_name": "ruby", "status": "confirmed"}]
r = estimate_gem_weights(stones, px_per_mm=10.0)
assert r["scale_source"] == "fiducial_card", r
s0 = r["stones"][0]
assert s0["diameter_mm"] == 6.5, s0
assert s0["est_carat_low"] < s0["est_carat"] < s0["est_carat_high"], s0
assert 0.3 < s0["est_carat"] < 5.0, f"carat out of sane range: {s0['est_carat']}"
assert r["total_carat"] == s0["est_carat"], r
print("with-scale OK:", s0["diameter_mm"], "mm ->", s0["est_carat"], "ct",
      f"({s0['est_carat_low']}-{s0['est_carat_high']})")

# No card → carats omitted, relative size kept, no fabricated number.
r2 = estimate_gem_weights(stones, px_per_mm=None)
assert r2["scale_source"] is None and r2["total_carat"] is None, r2
assert r2["stones"][0]["est_carat"] is None, r2
assert r2["stones"][0]["area_pct"] == 5.0, r2
print("no-scale OK: carats omitted, area_pct kept")

# Empty input is safe.
r3 = estimate_gem_weights([], px_per_mm=10.0)
assert r3["n_stones"] == 0 and r3["total_carat"] == 0.0, r3
print("empty OK")
print("ALL PASS")
