"""Diagnose stone detection on the images it used to fail on.

Runs the REAL request pipeline (xray_preview -> gem_vision.detect_gems ->
reconcile_stones) over every image in a folder (default
tests/fixtures/failing_stones/) and prints, per image, what each layer produced
and how the robustness layer changed the outcome:

  - background_removed (light/white catalogue shots often fail this)
  - AI votes: how many returned, and each vote's raw count
  - consensus count (after multi-vote merge + upscaling)
  - fused final: confirmed / uncertain / needs_review, agreement breakdown
  - whether graceful degradation fired (AI empty but CV confident)

Usage:
    python scripts/diagnose_failing_stones.py [folder]
    # compare with the robustness layer OFF:
    GEM_VISION_VOTES=1 GEM_VISION_MIN_SIDE=0 STONE_RESCUE_ML_ON_AI_EMPTY=0 \
        python scripts/diagnose_failing_stones.py

Unlike scripts/validate_stone_detection.py this loads .env, so the AI layer
actually runs when a key is configured.
"""
import asyncio
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from app.llm import gem_vision  # noqa: E402
from app.llm.gem_vision import detect_gems  # noqa: E402
from app.utils.xray import xray_preview, reconcile_stones  # noqa: E402


async def diagnose(path: str) -> None:
    raw = open(path, "rb").read()
    result, ctx = xray_preview(raw, _return_ctx=True)
    name = os.path.basename(path)
    bg = result.get("background_removed")
    ml_drawn = len([s for s in ctx["all_stones"] if s.get("status") != "candidate"])

    ai_stones = await detect_gems(
        ctx.get("norm"), ctx.get("item_bbox"),
        source_bgr=ctx.get("source_bgr"),
        source_to_norm=ctx.get("source_to_norm", 1.0),
    )
    stats_patch, _ = reconcile_stones(ctx, ai_stones)
    meta = stats_patch.get("stone_agreement", {})
    stones = stats_patch.get("stones", [])

    n_conf = sum(1 for s in stones if s.get("status") == "confirmed")
    n_unc = sum(1 for s in stones if s.get("status") == "uncertain")
    n_review = sum(1 for s in stones if s.get("needs_review"))
    ai_n = "None(fail)" if ai_stones is None else len(ai_stones)

    print(f"\n=== {name} ===")
    print(f"  background_removed : {bg}")
    print(f"  ML drawn (pre-AI)  : {ml_drawn}")
    print(f"  AI consensus stones: {ai_n}   (votes={gem_vision.GEM_VISION_VOTES}, "
          f"min_side={gem_vision.GEM_VISION_MIN_SIDE})")
    if ai_stones:
        agrees = [s.get("vote_agreement") for s in ai_stones if "vote_agreement" in s]
        if agrees:
            full = sum(1 for a in agrees if a >= 0.999)
            print(f"  consensus quality  : {full}/{len(agrees)} stones seen in ALL votes")
    print(f"  FUSED final        : {len(stones)} stones "
          f"({n_conf} confirmed, {n_unc} uncertain, {n_review} needs_review)")
    print(f"  agreement          : both={meta.get('n_both',0)} "
          f"ai_only={meta.get('n_ai_only',0)} ml_only={meta.get('n_ml_only',0)} "
          f"rescued={meta.get('n_rescued_ai_empty',0)}")
    if meta.get("ai_empty_ml_disagree"):
        print("  ⚠ DISAGREEMENT     : AI saw 0 stones; CV recovered some for review")
    if not bg and (ai_stones or n_review):
        print("  ✓ recovered stones despite background not separable")


async def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/failing_stones"
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
        + glob.glob(os.path.join(folder, "*.webp"))
    )
    if not paths:
        print(f"No images in {folder}/ — drop the failing photos there and re-run.")
        return
    prov = gem_vision._active_provider()
    print(f"AI provider: {prov or 'NONE (ML-only fallback)'}   images: {len(paths)}")
    for p in paths:
        try:
            await diagnose(p)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {os.path.basename(p)} ===\n  ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
