"""Standalone asserts for the offline (network-free) parts of gem_vision.

Run: python scripts/test_gem_vision.py   (expects: ALL PASS)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.llm import gem_vision as gv  # noqa: E402


def test_parse_valid_json_object():
    raw = ('```json\n{"stones":[{"gem_type":"ruby","colour":"red","shape":"oval",'
           '"confidence":0.9,"location":[0.5,0.5],"bbox":[0.4,0.4,0.2,0.2]}]}\n```')
    out = gv._parse_gems(raw, W=200, H=100, ox=0, oy=0)
    assert len(out) == 1, f"expected 1 stone, got {len(out)}"
    s = out[0]
    assert s["hue_class"] == "red" and s["gem_type"] == "ruby", s
    assert s["centroid"] == [100, 50], s["centroid"]
    assert s["bbox"] == [80, 40, 40, 20], s["bbox"]
    assert 0.0 <= s["ai_confidence"] <= 1.0
    print("  parse valid json object: OK")


def test_parse_bare_list_and_offset():
    raw = '[{"gem_type":"emerald","colour":"green","location":[0.0,0.0],"confidence":1}]'
    out = gv._parse_gems(raw, W=100, H=100, ox=10, oy=20)
    assert len(out) == 1 and out[0]["hue_class"] == "green"
    assert out[0]["centroid"] == [10, 20], out[0]["centroid"]   # offset applied
    print("  parse bare list + offset: OK")


def test_parse_reasoning_preamble_then_json():
    # Kimi K2.6 is a reasoning model: it thinks in prose (which may contain
    # stray braces like {x}) BEFORE the final JSON. _extract_json must recover
    # the last balanced JSON object regardless.
    raw = ('The user wants gemstones. Let me think {this has a brace}. '
           'I see one red stone.\n\n'
           '{"stones":[{"gem_type":"ruby","colour":"red","location":[0.5,0.5],"confidence":0.8}]}')
    out = gv._parse_gems(raw, W=200, H=100, ox=0, oy=0)
    assert len(out) == 1 and out[0]["gem_type"] == "ruby", out
    assert out[0]["centroid"] == [100, 50]
    print("  parse reasoning preamble + trailing json: OK")


def test_parse_garbage_returns_empty():
    assert gv._parse_gems("not json at all", 100, 100, 0, 0) == []
    assert gv._parse_gems("", 100, 100, 0, 0) == []
    print("  parse garbage -> []: OK")


def test_hue_bucket():
    assert gv._hue_bucket("deep red") == "red"
    assert gv._hue_bucket("colorless diamond") == "colourless"
    assert gv._hue_bucket("royal blue sapphire") == "blue"
    assert gv._hue_bucket("emerald") == "green"
    assert gv._hue_bucket("amethyst purple") == "other"
    print("  hue bucket mapping: OK")


if __name__ == "__main__":
    test_parse_valid_json_object()
    test_parse_bare_list_and_offset()
    test_parse_reasoning_preamble_then_json()
    test_parse_garbage_returns_empty()
    test_hue_bucket()
    print("ALL PASS")
