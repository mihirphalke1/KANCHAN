"""
Layer B — AI vision gem detector (Gemini 1.5 Flash).

A second, INDEPENDENT opinion on "how many gemstones are set in this ornament,
and what are they?" that cross-confirms the classical/ML stone detector in
app/utils/xray.py. A general vision model is strong at exactly the thing a
colour threshold is weak at: semantically finding, locating, and NAMING set
stones (including ones near the gold colour, touching pairs, and diamonds that
read as "colourless"). MobileSAM still owns pixel-precise BOUNDARIES; this
module owns semantic COUNT + TYPE. app/utils/stone_fusion.py reconciles them.

Deliberately guarded, same "never a hard block" contract as verdict_prompt.py:
  - no GOOGLE_API_KEY, or USE_AI_STONE_CONFIRM=0            -> returns None
  - any timeout / quota / malformed-JSON / exception        -> returns None
When None is returned the caller keeps the ML-only result unchanged.

Determinism: temperature 0 and response_mime_type application/json, so the
model is asked for machine-readable output and the same photo yields the same
answer run to run.

`_parse_gems` and `_hue_bucket` are PURE and network-free (unit-tested in
scripts/test_gem_vision.py); the network call lives only in `detect_gems`.
"""
import asyncio
import base64
import json
import logging
import os
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

USE_AI_STONE_CONFIRM = os.getenv("USE_AI_STONE_CONFIRM", "1") != "0"
# Provider: "fireworks" (Qwen3-VL, a grounding-native vision model) is primary;
# "gemini" is the fallback. "auto" picks whichever key is present (Fireworks
# first). Qwen3-VL is chosen over a general reasoning model because stone COUNT
# and POSITION are a visual-grounding task — Qwen3-VL is trained specifically
# for bounding-box + point grounding and counting, which is exactly what a
# per-stone bbox needs.
GEM_VISION_PROVIDER = os.getenv("GEM_VISION_PROVIDER", "auto").lower()
# Only FOUR vision models in the Fireworks public catalog are serverless-callable
# (supportsImageInput AND supportsServerless — see scripts/list_fw_models.py):
# kimi-k2p6, kimi-k2p7-code, minimax-m3, qwen3p7-plus. Every qwen*-vl and
# llama-vision entry is catalogued but supportsServerless=False, so it needs a
# dedicated deployment and 404s on the serverless chat endpoint.
#
# Measured on a real pavé ring, 6 runs each (scripts/compare_gem_vision.py),
# true count ~32-36 stones:
#   kimi-k2p6       14, 18                  122-187s   (least accurate, slowest)
#   kimi-k2p7-code  24, 20, 0, 20, 24, 18    46-153s   (1 run returned junk)
#   qwen3p7-plus    21, 35, 26, 22, 21, 27   49- 77s   (best mean, 0 failures)
# qwen3p7-plus wins on accuracy, reliability and latency, so it is the default.
FIREWORKS_VISION_MODEL = os.getenv("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/qwen3p7-plus")
GEMINI_VISION_MODEL = os.getenv("GEM_VISION_MODEL", "gemini-1.5-flash")
FIREWORKS_URL = os.getenv("FIREWORKS_URL", "https://api.fireworks.ai/inference/v1/chat/completions")
# Kimi REASONS at length before emitting the answer: on a real pavé ring it
# spent ~8.7k completion tokens and ~90s to enumerate every stone. Measured
# behaviour (same photo, same prompt):
#     4096 tokens  -> finish_reason=length, truncated mid-reasoning, 0 stones
#    16384 tokens  -> finish_reason=stop, 16 stones parsed
# so the budget must comfortably exceed the reasoning, and the timeout must
# exceed the ~90s call. Any overrun still just falls back to the ML-only result.
GEM_VISION_MAX_TOKENS = int(os.getenv("GEM_VISION_MAX_TOKENS", "16384"))
GEM_VISION_TIMEOUT_S = float(os.getenv("GEM_VISION_TIMEOUT_S", "180"))
# Pad the item crop a little so stones flush to the item edge aren't clipped.
_CROP_PAD_FRAC = 0.04

_PROMPT = (
    "You are a professional gemologist inspecting ONE gold ornament in the image.\n"
    "Your job is to COUNT and LOCATE every gemstone set into the metal — faceted "
    "stones, cabochons, pearls, and small beads/pavé — with one entry per physical "
    "stone.\n"
    "\n"
    "HOW TO COUNT ACCURATELY:\n"
    "- Scan the whole item systematically (e.g. left-to-right, top-to-bottom) so no "
    "region is skipped.\n"
    "- Count EACH distinct set stone exactly once, including tiny accent beads and "
    "pavé stones — small does not mean skip.\n"
    "- A single faceted stone shows several bright facets and internal reflections: "
    "that is still ONE stone, not many. Do NOT emit a separate entry per facet or "
    "per glare spot.\n"
    "- Conversely, do NOT merge a tight cluster of several small stones into one "
    "entry — if there are visibly separate stones, list them separately.\n"
    "\n"
    "WHAT IS NOT A STONE (never include these):\n"
    "- The gold/metal body itself, prongs, bezels, solder joints, engraving, filigree, "
    "or hallmark stamps.\n"
    "- Specular reflections, glare, or bright highlights ON the metal.\n"
    "- Do NOT read, infer from, or use any text, numbers, or watermark in the image.\n"
    "\n"
    "LOCATION PRECISION:\n"
    "- bbox must tightly enclose the stone (not the whole setting), and location is the "
    "stone's visual centre.\n"
    "- If unsure whether a small feature is a stone, still include it with a LOW "
    "confidence rather than dropping it.\n"
    "\n"
    "Return ONLY valid JSON of the form:\n"
    '{"stones":[{"gem_type":"ruby","colour":"red","shape":"oval","confidence":0.0-1.0,'
    '"location":[x,y],"bbox":[x,y,w,h]}]}\n'
    "where location is the stone centre and bbox is its tight bounding box, ALL as "
    "fractions 0..1 of THIS image's width/height (x,y = top-left of bbox). Return "
    '{"stones":[]} if there are genuinely no stones.'
)

# Colour/gem word -> legacy 4-way bucket used across composition.py and the UI.
_BUCKET_KEYWORDS = [
    ("red",        ("red", "ruby", "garnet", "crimson", "maroon", "pink", "coral", "rose")),
    ("green",      ("green", "emerald", "peridot", "jade", "olive")),
    ("blue",       ("blue", "sapphire", "aqua", "turquoise", "teal", "navy", "topaz blue")),
    ("colourless", ("colourless", "colorless", "white", "clear", "diamond", "transparent",
                    "pearl", "crystal")),
]


def _hue_bucket(colour_str: str) -> str:
    """Map a free-text colour/gem word to {red,green,blue,colourless,other}."""
    s = (colour_str or "").strip().lower()
    if not s:
        return "other"
    for bucket, words in _BUCKET_KEYWORDS:
        if any(w in s for w in words):
            return bucket
    return "other"


def _coerce_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _balanced_spans(s: str, opener: str, closer: str) -> list[str]:
    """All top-level balanced `opener..closer` substrings, in order."""
    spans, depth, start = [], 0, -1
    for k, ch in enumerate(s):
        if ch == opener:
            if depth == 0:
                start = k
            depth += 1
        elif ch == closer and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(s[start:k + 1])
    return spans


def _extract_json(text: str):
    """Robustly pull the JSON object/array out of a model reply that may include
    reasoning prose before it (Kimi K2.6 thinks out loud, then emits JSON). Tries
    the whole cleaned string, then each balanced {...}/[...] span from the LAST
    one backward (the answer is normally last), returning the first that parses."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    candidates = [cleaned]
    # Prefer the last balanced object/array (reasoning models put it at the end).
    candidates += list(reversed(_balanced_spans(cleaned, "{", "}")))
    candidates += list(reversed(_balanced_spans(cleaned, "[", "]")))
    for cand in candidates:
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            continue
    return None


def _parse_gems(text: str, W: int, H: int, ox: int, oy: int) -> list[dict]:
    """Parse the model's JSON reply into pixel-space gem dicts. Pure; any
    malformed input yields []. Coordinates in the reply are fractions 0..1 of
    the (cropped) image the model saw; they are mapped back to full-image
    pixels via the crop size (W,H) and origin (ox,oy)."""
    data = _extract_json(text)
    if data is None:
        return []

    if isinstance(data, dict):
        raw = data.get("stones") or data.get("gems") or data.get("results") or []
    elif isinstance(data, list):
        raw = data
    else:
        return []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        gem_type = str(s.get("gem_type") or s.get("type") or s.get("name") or "stone").strip()
        colour = str(s.get("colour") or s.get("color") or "").strip()
        shape = str(s.get("shape") or "").strip()
        conf = _coerce_float(s.get("confidence"), 0.5)
        conf = min(1.0, max(0.0, conf))
        hue_class = _hue_bucket(colour or gem_type)

        loc = s.get("location") or s.get("center") or s.get("centre")
        bbox = s.get("bbox") or s.get("box")
        cx = cy = None
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            cx = ox + _coerce_float(loc[0]) * W
            cy = oy + _coerce_float(loc[1]) * H
        px_bbox = None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bx = ox + _coerce_float(bbox[0]) * W
            by = oy + _coerce_float(bbox[1]) * H
            bw = _coerce_float(bbox[2]) * W
            bh = _coerce_float(bbox[3]) * H
            px_bbox = [int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))]
            if cx is None:
                cx, cy = bx + bw / 2.0, by + bh / 2.0
        if cx is None:
            continue   # nothing locatable — skip
        out.append({
            "index":         i + 1,
            "gem_type":      gem_type,
            "colour":        colour,
            "hue_class":     hue_class,
            "shape":         shape,
            "ai_confidence": round(conf, 3),
            "centroid":      [int(round(cx)), int(round(cy))],
            "bbox":          px_bbox,
        })
    return out


def _item_crop(image_bgr: np.ndarray, item_bbox: Optional[list[int]]):
    """Return (crop_bgr, W, H, ox, oy). item_bbox is [xmin,ymin,xmax,ymax] from
    xray._item_bbox, or None to use the whole frame."""
    H_full, W_full = image_bgr.shape[:2]
    if not item_bbox or len(item_bbox) != 4:
        return image_bgr, W_full, H_full, 0, 0
    x0, y0, x1, y1 = item_bbox
    padx = int((x1 - x0) * _CROP_PAD_FRAC)
    pady = int((y1 - y0) * _CROP_PAD_FRAC)
    x0 = max(0, x0 - padx); y0 = max(0, y0 - pady)
    x1 = min(W_full, x1 + padx); y1 = min(H_full, y1 + pady)
    if x1 <= x0 or y1 <= y0:
        return image_bgr, W_full, H_full, 0, 0
    return image_bgr[y0:y1, x0:x1], (x1 - x0), (y1 - y0), x0, y0


def _crop_to_data_uri(crop_bgr: np.ndarray) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _real_key(name: str) -> Optional[str]:
    """Return an API key from the environment only if it looks real — a value
    that is unset, blank, or still the `.env.example` placeholder
    (`your_..._here`) counts as absent. This keeps the layer from firing a
    doomed network call (and adding latency) on every case before falling back
    to ML-only when no key has actually been configured yet."""
    v = (os.getenv(name) or "").strip()
    if not v or (v.startswith("your_") and v.endswith("_here")):
        return None
    return v


def _active_provider() -> Optional[str]:
    """Which provider to use given the configured keys. 'fireworks' (Qwen3-VL)
    is preferred; 'gemini' is the fallback; None when no real key is available."""
    p = GEM_VISION_PROVIDER
    if p == "fireworks":
        return "fireworks" if _real_key("FIREWORKS_API_KEY") else None
    if p in ("gemini", "google"):
        return "gemini" if _real_key("GOOGLE_API_KEY") else None
    # auto
    if _real_key("FIREWORKS_API_KEY"):
        return "fireworks"
    if _real_key("GOOGLE_API_KEY"):
        return "gemini"
    return None


def _call_fireworks_sync(crop_bgr: np.ndarray) -> Optional[str]:
    """Blocking Fireworks (Kimi K2.6) vision call via the OpenAI-compatible
    chat endpoint. Returns raw text or None on any failure. Kimi is a reasoning
    model, so it emits thinking text before the JSON — _extract_json recovers
    the JSON regardless."""
    api_key = _real_key("FIREWORKS_API_KEY")
    if not api_key:
        return None
    data_uri = _crop_to_data_uri(crop_bgr)
    if data_uri is None:
        return None
    try:
        import requests
        # NOTE: deliberately NO "response_format": {"type": "json_object"}.
        # Measured on the live API: constraining Kimi to json_object makes it
        # consume the ENTIRE completion budget (16384 tokens, 200s) and still
        # return truncated output -> 0 stones. Unconstrained, the same request
        # finishes cleanly (finish_reason=stop, ~8.7k tokens) and returns the
        # full stone list. _extract_json pulls the JSON out of the trailing
        # reasoning prose, so the constraint bought nothing and cost everything.
        payload = {
            "model": FIREWORKS_VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            "temperature": 0.0,
            "max_tokens": GEM_VISION_MAX_TOKENS,
        }
        r = requests.post(
            FIREWORKS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=GEM_VISION_TIMEOUT_S,
        )
        if r.status_code != 200:
            logger.warning("Fireworks gem vision HTTP %s: %s — ML-only", r.status_code, r.text[:200])
            return None
        return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — never a hard block
        logger.warning("Fireworks gem vision failed (%s) — ML-only for this case", e)
        return None


def _call_gemini_sync(crop_bgr: np.ndarray) -> Optional[str]:
    """Blocking Gemini vision call. Returns raw text or None on any failure."""
    api_key = _real_key("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        from PIL import Image
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        resp = model.generate_content(
            [_PROMPT, pil],
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        return (resp.text or "").strip()
    except Exception as e:  # noqa: BLE001 — never a hard block
        logger.warning("Gemini gem vision failed (%s) — ML-only for this case", e)
        return None


def _call_vision_sync(crop_bgr: np.ndarray, provider: str) -> Optional[str]:
    if provider == "fireworks":
        return _call_fireworks_sync(crop_bgr)
    if provider == "gemini":
        return _call_gemini_sync(crop_bgr)
    return None


async def detect_gems(image_bgr: np.ndarray, item_bbox: Optional[list[int]]) -> Optional[list[dict]]:
    """AI cross-confirmation entry point.

    Returns a list of pixel-space gem dicts (see `_parse_gems`), or None when
    the layer is disabled / no key / any failure — in which case the caller
    keeps the ML-only result. Never raises.
    """
    if not USE_AI_STONE_CONFIRM:
        return None
    provider = _active_provider()
    if provider is None:
        return None
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    try:
        crop, W, H, ox, oy = _item_crop(image_bgr, item_bbox)
        text = await asyncio.wait_for(
            asyncio.to_thread(_call_vision_sync, crop, provider), timeout=GEM_VISION_TIMEOUT_S
        )
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.warning("Gem vision (%s) timed out/failed (%s) — ML-only", provider, e)
        return None
    if not text:
        return None
    return _parse_gems(text, W, H, ox, oy)
