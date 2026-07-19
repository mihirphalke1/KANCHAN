"""Standalone asserts for strength-based ML/AI stone fusion.

Run: python scripts/test_stone_fusion.py   (expects: ALL PASS)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.utils import stone_fusion as sf  # noqa: E402


def _scene():
    # 100x100 frame; one ML stone (label 1) as a filled square at (20..40,20..40).
    labels = np.zeros((100, 100), np.int32)
    labels[20:40, 20:40] = 1
    item = np.ones((100, 100), bool)
    ml_stones = [{
        "_label": 1, "area_pct": 4.0, "hue_class": "other", "stone_name": "unidentified",
        "match_confidence": 0.1, "confidence": 0.5, "status": "uncertain",
        "below_uncertain": False, "bbox": [20, 20, 20, 20], "centroid": [30.0, 30.0],
    }]
    return ml_stones, labels, item


def test_matched_pair_uses_sam_boundary_and_ai_type():
    ml, labels, item = _scene()
    ai = [{"index": 1, "gem_type": "ruby", "colour": "red", "hue_class": "red",
           "shape": "oval", "ai_confidence": 0.9, "centroid": [30, 30], "bbox": [20, 20, 20, 20]}]
    stones, out_labels, meta = sf.reconcile(ml, labels, ai, item)
    assert len(stones) == 1, stones
    s = stones[0]
    assert s["agreement"] == "both", s["agreement"]
    assert s["status"] == "confirmed"
    assert s["hue_class"] == "red" and s["stone_name"] == "ruby"   # AI owns type
    assert s["_label"] == 1                                        # SAM owns boundary
    assert s["ml_confidence"] == 0.5 and s["ai_confidence"] == 0.9
    assert meta["mode"] == "ml_ai" and meta["n_both"] == 1
    print("  matched pair: SAM boundary + AI type, confirmed: OK")


def test_ai_only_confirmed_when_ai_primary():
    # AI is the PRIMARY judge: a confident AI-only stone (inside the item, sane
    # size) is trusted -> CONFIRMED, not merely flagged. Under strict AI-only
    # (the default) the unmatched ML stone in the scene is DROPPED.
    ml, labels, item = _scene()
    ai = [{"index": 1, "gem_type": "emerald", "colour": "green", "hue_class": "green",
           "shape": "oval", "ai_confidence": 0.8, "centroid": [70, 70], "bbox": [62, 62, 16, 16]}]
    stones, out_labels, meta = sf.reconcile(ml, labels, ai, item)
    ai_only = [s for s in stones if s["agreement"] == "ai_only"]
    assert len(ai_only) == 1, [s["agreement"] for s in stones]
    s = ai_only[0]
    assert s["status"] == "confirmed", "confident AI-only must be CONFIRMED under AI-primary"
    assert s["hue_class"] == "green" and s["stone_name"] == "emerald"
    assert s["_label"] not in (0, 1), "AI-only needs a fresh label painted into the map"
    assert (out_labels == s["_label"]).any(), "AI-only region not painted into labels"
    assert s["area_pct"] > 0
    assert meta["n_ai_only"] == 1 and meta["mode"] == "ml_ai"
    assert meta["ai_primary"] is True and meta["primary"] == "ai"
    assert meta["ai_only"] is True and meta["n_ml_only"] == 0, "strict AI-only drops ML-only"
    print("  AI-only: confirmed under AI-primary, fresh label painted: OK")


def test_strict_ai_only_drops_uncorroborated_ml():
    # The grb101p.jpg situation: the AI finds a stone the ML missed, AND the ML
    # has a (different) detection the AI did not see. Strictly AI-based => the
    # result is EXACTLY the AI's stone; the uncorroborated ML detection is gone.
    ml, labels, item = _scene()                       # ML stone at [30,30]
    ml[0]["status"] = "confirmed"; ml[0]["hue_class"] = "colourless"
    ai = [{"index": 1, "gem_type": "ruby", "colour": "red", "hue_class": "red",
           "shape": "round", "ai_confidence": 0.8, "centroid": [70, 70], "bbox": [62, 62, 16, 16]}]
    stones, _, meta = sf.reconcile(ml, labels, ai, item)
    assert len(stones) == 1, f"strict AI-only should show only the AI stone: {stones}"
    assert stones[0]["agreement"] == "ai_only" and stones[0]["stone_name"] == "ruby"
    assert meta["n_ml_only"] == 0
    print("  strict AI-only: uncorroborated ML dropped, only AI stone shown: OK")


def test_ai_only_low_conf_stays_uncertain():
    # Below the confidence floor, even AI-primary keeps an AI-only stone flagged.
    ml, labels, item = _scene()
    ai = [{"index": 1, "gem_type": "emerald", "colour": "green", "hue_class": "green",
           "shape": "oval", "ai_confidence": 0.2, "centroid": [70, 70], "bbox": [62, 62, 16, 16]}]
    stones, _, _ = sf.reconcile(ml, labels, ai, item)
    s = [s for s in stones if s["agreement"] == "ai_only"][0]
    assert s["status"] == "uncertain", "low-confidence AI-only must stay flagged"
    print("  AI-only low confidence: stays uncertain: OK")


def test_ml_only_demoted_when_not_strict():
    # With STONE_AI_ONLY=0 (keep ML as secondary), the AI is still primary and
    # did NOT corroborate a confirmed coloured ML stone -> it is demoted to
    # secondary evidence (uncertain) but NOT deleted.
    saved = sf.AI_ONLY
    sf.AI_ONLY = False
    try:
        ml, labels, item = _scene()
        ml[0]["status"] = "confirmed"; ml[0]["hue_class"] = "red"; ml[0]["stone_name"] = "garnet"
        ai = [{"index": 1, "gem_type": "emerald", "colour": "green", "hue_class": "green",
               "shape": "oval", "ai_confidence": 0.8, "centroid": [70, 70], "bbox": [62, 62, 16, 16]}]
        stones, _, _ = sf.reconcile(ml, labels, ai, item)
        ml_only = [s for s in stones if s["agreement"] == "ml_only"]
        assert len(ml_only) == 1, "uncorroborated ML stone must be kept when not strict"
        assert ml_only[0]["status"] == "uncertain", "uncorroborated ML (AI-primary) must be demoted"
    finally:
        sf.AI_ONLY = saved
    print("  ML-only coloured (STONE_AI_ONLY=0): demoted to secondary (kept): OK")


def test_legacy_balanced_policy_flags_ai_only():
    # STONE_AI_PRIMARY=0 restores the older balanced policy: AI-only is NEVER
    # confirmed, and only colourless ML-only is downgraded.
    saved_p, saved_o = sf.AI_PRIMARY, sf.AI_ONLY
    sf.AI_PRIMARY = False; sf.AI_ONLY = False       # legacy keeps ML-only stones
    try:
        ml, labels, item = _scene()
        ml[0]["status"] = "confirmed"; ml[0]["hue_class"] = "red"
        ai = [{"index": 1, "gem_type": "emerald", "colour": "green", "hue_class": "green",
               "shape": "oval", "ai_confidence": 0.9, "centroid": [70, 70], "bbox": [62, 62, 16, 16]}]
        stones, _, meta = sf.reconcile(ml, labels, ai, item)
        ai_only = [s for s in stones if s["agreement"] == "ai_only"][0]
        assert ai_only["status"] != "confirmed", "legacy: AI-only must stay flagged"
        ml_only = [s for s in stones if s["agreement"] == "ml_only"][0]
        assert ml_only["status"] == "confirmed", "legacy: coloured ML-only stays as-is"
        assert meta["primary"] == "balanced"
    finally:
        sf.AI_PRIMARY = saved_p; sf.AI_ONLY = saved_o
    print("  legacy balanced policy (STONE_AI_PRIMARY=0): AI-only flagged: OK")


def test_strict_ai_finds_nothing_shows_nothing():
    # Strictly AI-based (default): the AI ran and returned zero stones -> the AI
    # is the authority, so nothing is drawn (the ML detection is not asserted).
    ml, labels, item = _scene()
    stones, _, meta = sf.reconcile(ml, labels, [], item)   # AI ran, found none
    assert stones == [], f"strict AI-only: AI found none => show none, got {stones}"
    assert meta["mode"] == "ml_ai" and meta["ai_used"] is True and meta["n_ml_only"] == 0
    print("  strict AI-only (AI found nothing): shows nothing: OK")


def test_ml_only_kept_when_ai_finds_nothing_not_strict():
    # With STONE_AI_ONLY=0 the ML detection is kept (as secondary) when the AI
    # ran but found nothing.
    saved = sf.AI_ONLY
    sf.AI_ONLY = False
    try:
        ml, labels, item = _scene()
        stones, _, meta = sf.reconcile(ml, labels, [], item)
        assert len(stones) == 1 and stones[0]["agreement"] == "ml_only"
        assert stones[0]["status"] == "uncertain"        # original ML status preserved
        assert meta["ai_used"] is True and meta["n_ml_only"] == 1
    finally:
        sf.AI_ONLY = saved
    print("  ML-only kept (STONE_AI_ONLY=0, AI found nothing): OK")


def test_ai_none_is_passthrough():
    ml, labels, item = _scene()
    stones, out_labels, meta = sf.reconcile(ml, labels, None, item)  # AI did NOT run
    assert len(stones) == 1 and stones[0]["agreement"] == "ml_only"
    assert meta["mode"] == "ml_only" and meta["ai_used"] is False
    assert np.array_equal(out_labels, labels)        # labels untouched
    print("  AI None -> ML passthrough, labels untouched: OK")


def test_ml_candidate_only_surfaces_if_ai_confirms():
    # A below-uncertain ML candidate is NOT drawn unless AI confirms it.
    ml, labels, item = _scene()
    ml[0]["status"] = "candidate"; ml[0]["below_uncertain"] = True
    # AI finds nothing -> candidate must NOT surface
    stones, _, _ = sf.reconcile(ml, labels, [], item)
    assert stones == [] or all(s["agreement"] != "ml_only" for s in stones), \
        "unconfirmed ML candidate must not be drawn"
    # AI confirms at the candidate location -> it surfaces as 'both'
    ai = [{"index": 1, "gem_type": "sapphire", "colour": "blue", "hue_class": "blue",
           "shape": "round", "ai_confidence": 0.85, "centroid": [30, 30], "bbox": [20, 20, 20, 20]}]
    stones2, _, meta2 = sf.reconcile(ml, labels, ai, item)
    assert len(stones2) == 1 and stones2[0]["agreement"] == "both"
    assert stones2[0]["hue_class"] == "blue"
    print("  ML candidate: hidden until AI confirms, then promoted: OK")


def test_ai_box_outside_item_rejected():
    ml, labels, item = _scene()
    item2 = np.zeros((100, 100), bool); item2[0:50, 0:50] = True   # item only top-left
    ai = [{"index": 1, "gem_type": "ruby", "colour": "red", "hue_class": "red",
           "shape": "oval", "ai_confidence": 0.9, "centroid": [80, 80], "bbox": [72, 72, 16, 16]}]
    stones, _, meta = sf.reconcile(ml, labels, ai, item2)
    assert meta["n_ai_only"] == 0, "AI detection outside the item must be rejected"
    print("  AI box outside item: rejected: OK")


def test_near_miss_matches_not_double_counts():
    # AI centre a bit off the ML centroid (Kimi estimates high) but clearly the
    # SAME small stone must MATCH, not create a phantom second stone.
    labels = np.zeros((640, 640), np.int32)
    labels[316:344, 403:428] = 1                     # small stone ~25x28
    item = np.zeros((640, 640), bool); item[220:423, 177:462] = True
    ml = [{"_label": 1, "area_pct": 1.0, "hue_class": "red", "stone_name": "garnet",
           "match_confidence": 0.3, "confidence": 0.7, "status": "confirmed",
           "below_uncertain": False, "bbox": [403, 316, 25, 28], "centroid": [415.1, 329.3]}]
    ai = [{"index": 1, "gem_type": "ruby", "colour": "red", "hue_class": "red",
           "shape": "pear", "ai_confidence": 0.6, "centroid": [418, 295],
           "bbox": [406, 282, 24, 26]}]                # ~34px above ML centroid
    stones, _, meta = sf.reconcile(ml, labels, ai, item)
    assert meta["n_both"] == 1 and meta["n_ai_only"] == 0, \
        f"near-miss should match, not double-count: {meta}"
    assert len(stones) == 1 and stones[0]["stone_name"] == "ruby"
    print("  near-miss AI/ML -> single 'both' stone (no phantom): OK")


def test_huge_blob_does_not_swallow_distant_ai_point():
    # A giant colourless ML blob must NOT match an AI point 80px from its
    # centroid just because the blob is large.
    labels = np.zeros((640, 640), np.int32)
    labels[220:423, 195:460] = 2                      # huge 265x203 blob
    item = np.zeros((640, 640), bool); item[220:423, 177:462] = True
    ml = [{"_label": 2, "area_pct": 5.0, "hue_class": "colourless", "stone_name": "white_sapphire",
           "match_confidence": 0.2, "confidence": 0.7, "status": "confirmed",
           "below_uncertain": False, "bbox": [195, 220, 265, 203], "centroid": [302.4, 291.1]}]
    ai = [{"index": 1, "gem_type": "ruby", "colour": "red", "hue_class": "red",
           "shape": "pear", "ai_confidence": 0.6, "centroid": [221, 295], "bbox": [209, 282, 24, 26]}]
    # Strict AI-only (default): the distant AI point must NOT match the huge blob
    # (matcher too_big guard), and the uncorroborated blob is dropped entirely.
    stones, _, meta = sf.reconcile(ml, labels, ai, item)
    assert meta["n_ai_only"] == 1, "distant AI point must NOT match the huge blob"
    assert not [s for s in stones if s["stone_name"] == "white_sapphire"], \
        "strict AI-only: uncorroborated glare blob must be dropped"
    # With STONE_AI_ONLY=0 the blob is kept but downgraded (AI didn't corroborate).
    saved = sf.AI_ONLY
    sf.AI_ONLY = False
    try:
        stones2, _, meta2 = sf.reconcile(ml, labels, ai, item)
        assert meta2["n_ai_only"] == 1
        blob = [s for s in stones2 if s["stone_name"] == "white_sapphire"][0]
        assert blob["status"] != "confirmed", "unconfirmed colourless (AI active) must be flagged"
    finally:
        sf.AI_ONLY = saved
    print("  huge blob: no false match; dropped when strict, downgraded when not: OK")


if __name__ == "__main__":
    test_matched_pair_uses_sam_boundary_and_ai_type()
    test_near_miss_matches_not_double_counts()
    test_huge_blob_does_not_swallow_distant_ai_point()
    test_ai_only_confirmed_when_ai_primary()
    test_strict_ai_only_drops_uncorroborated_ml()
    test_ai_only_low_conf_stays_uncertain()
    test_ml_only_demoted_when_not_strict()
    test_legacy_balanced_policy_flags_ai_only()
    test_strict_ai_finds_nothing_shows_nothing()
    test_ml_only_kept_when_ai_finds_nothing_not_strict()
    test_ai_none_is_passthrough()
    test_ml_candidate_only_surfaces_if_ai_confirms()
    test_ai_box_outside_item_rejected()
    print("ALL PASS")
