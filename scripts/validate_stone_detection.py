"""Validate stone detection (ML alone, or ML + AI cross-confirmation).

Runs the real pipeline the analysis route uses — xray_preview(_return_ctx=True)
-> gem_vision.detect_gems -> xray.reconcile_stones — over the demo panel and
prints, per image: ML-drawn count, whether AI ran, and the fused
matched / ML-only / AI-only breakdown + mode.

    python scripts/validate_stone_detection.py

HONEST NOTE: with no labelled ground-truth dataset, this measures ML<->AI
AGREEMENT and behaviour, NOT accuracy against a known answer. High agreement is
evidence the two independent methods corroborate each other; it is not a
certified accuracy figure. The AI columns only populate when GOOGLE_API_KEY is
set and USE_AI_STONE_CONFIRM != 0.
"""
import asyncio
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.utils.xray import xray_preview, reconcile_stones  # noqa: E402
from app.llm.gem_vision import detect_gems  # noqa: E402

PANEL = sorted(glob.glob("data/demo/*.jpg")) + sorted(glob.glob("data/demo/*.jpeg"))


async def one(path: str) -> dict:
    raw = open(path, "rb").read()
    result, ctx = xray_preview(raw, _return_ctx=True)
    if not result.get("background_removed"):
        return {"name": os.path.basename(path), "bg": False}
    ml_drawn = len([s for s in ctx["all_stones"] if s.get("status") != "candidate"])
    ai_stones = await detect_gems(ctx.get("norm"), ctx.get("item_bbox"))
    stats_patch, _ = reconcile_stones(ctx, ai_stones)
    meta = stats_patch.get("stone_agreement", {})
    return {
        "name": os.path.basename(path), "bg": True,
        "ml_drawn": ml_drawn,
        "ai": None if ai_stones is None else len(ai_stones),
        "mode": stats_patch.get("stone_detection_mode"),
        "both": meta.get("n_both", 0),
        "ml_only": meta.get("n_ml_only", 0),
        "ai_only": meta.get("n_ai_only", 0),
        "final": stats_patch.get("gem_regions", 0),
    }


async def main():
    ai_on = bool(os.getenv("GOOGLE_API_KEY")) and os.getenv("USE_AI_STONE_CONFIRM", "1") != "0"
    print(f"AI cross-confirmation: {'ON' if ai_on else 'OFF (ML-only)'}\n")
    print(f"{'image':22s} {'bg':4s} {'ML':>3s} {'AI':>3s} {'both':>4s} {'mlO':>4s} {'aiO':>4s} {'final':>5s}  mode")
    tot_both = tot_ai_only = tot_ml_only = 0
    for p in PANEL:
        r = await one(p)
        if not r["bg"]:
            print(f"{r['name']:22s} {'no':4s}   (background not separable — skipped)")
            continue
        ai = "-" if r["ai"] is None else str(r["ai"])
        print(f"{r['name']:22s} {'yes':4s} {r['ml_drawn']:>3d} {ai:>3s} "
              f"{r['both']:>4d} {r['ml_only']:>4d} {r['ai_only']:>4d} {r['final']:>5d}  {r['mode']}")
        tot_both += r["both"]; tot_ai_only += r["ai_only"]; tot_ml_only += r["ml_only"]
    if ai_on:
        denom = tot_both + tot_ai_only + tot_ml_only
        rate = (tot_both / denom * 100) if denom else 0.0
        print(f"\nAgreement (both / all detections): {tot_both}/{denom} = {rate:.0f}%")
        print("Reminder: agreement is corroboration between two independent methods,")
        print("not certified accuracy against a labelled ground truth.")
    else:
        print("\nSet GOOGLE_API_KEY (and USE_AI_STONE_CONFIRM=1) to populate the AI columns.")


if __name__ == "__main__":
    asyncio.run(main())
