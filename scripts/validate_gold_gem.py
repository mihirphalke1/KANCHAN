"""Before/after validation of the gold-vs-gems colour split over the demo +
saved-case image panel. Run: python scripts/validate_gold_gem.py

Success signals (Task 1): the "other" bucket drops sharply on gold-dominant
items (mis-lit gold recovered as gold), and no image flips gold pixels into
the GEM class (gem_pct must not balloon)."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_ML_STONE_DETECTION", "0")   # classical, fast, deterministic

from app.utils.xray import _load_bgr, _run_pipeline  # noqa: E402

paths = sorted(
    glob.glob("data/demo/*.jpg")
    + glob.glob("data/demo/*.jpeg")
    + glob.glob("data/cases/*/img_0.jpg")
)
print(f"{'image':40s} {'bg':>5} {'gold%':>6} {'gem%':>6} {'other%':>7} {'method':>16} {'conf':>4}")
for p in paths:
    try:
        _, stats, _ = _run_pipeline(_load_bgr(open(p, "rb").read()))
    except Exception as e:
        print(f"{os.path.basename(p):40s} ERROR {e}")
        continue
    g = stats.get("gold_gem_split", {})
    print(f"{os.path.basename(p):40s} {str(stats.get('background_removed'))[:5]:>5} "
          f"{g.get('gold_pct', 0):>6} {g.get('gem_pct', 0):>6} {g.get('other_pct', 0):>7} "
          f"{g.get('gold_method', '-'):>16} {stats.get('stones_confirmed', 0):>4}")
