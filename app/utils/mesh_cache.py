"""
Image-keyed cache for generated 3D models.

Generating a mesh on a public Hugging Face Space is the slow, rate-limited part
of the 3D path (cold starts, queues, per-token quota). The same item photo —
a re-uploaded demo image, the same ring shot twice, a retried case — should not
pay that cost twice. So we key finished, *coloured* GLBs on a perceptual hash
(dHash) of the primary photo: a near-identical photo reuses the stored model
instantly, with no Space call at all.

dHash (not a byte checksum) so recompression / minor crop / resize still hit
the same entry. Match threshold is deliberately tight — a visually different
ring must miss and generate its own model, never be served someone else's.

Cosmetic / additive path: a wrong-but-close cache hit is only a cosmetic
mismatch on an illustrative model; it never touches fusion or the loan verdict.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.utils.phash import compute_dhash, hamming_distance

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/mesh_cache")
INDEX_PATH = CACHE_DIR / "index.json"

USE_MESH_CACHE = os.getenv("USE_MESH_CACHE", "1").strip() not in ("0", "false", "False")
# Hamming distance (out of 64 bits) under which two photos share a model.
# 8 is tight enough to separate different rings, loose enough to survive
# recompression of the same shot.
CACHE_HAMMING = int(os.getenv("MESH_CACHE_HAMMING", "8"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index() -> list:
    if INDEX_PATH.exists():
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Could not read mesh cache index: %s", e)
    return []


def _save_index(index: list) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def lookup(image_bytes: bytes) -> Optional[dict]:
    """Closest cached model within the match threshold, or None.

    Returns the index entry plus `_path` (the on-disk GLB) and `_distance`."""
    if not USE_MESH_CACHE or not image_bytes:
        return None
    try:
        h = compute_dhash(image_bytes)
    except Exception as e:
        logger.warning("mesh cache hash failed: %s", e)
        return None
    best = None
    for entry in _load_index():
        try:
            d = hamming_distance(h, entry["hash"])
        except Exception:
            continue
        if d <= CACHE_HAMMING and (best is None or d < best["_distance"]):
            p = CACHE_DIR / entry["glb_file"]
            if p.exists():
                best = {**entry, "_distance": d, "_path": p}
    if best:
        logger.info("mesh cache HIT (distance %d) → %s", best["_distance"], best["glb_file"])
    return best


def store(image_bytes: bytes, glb_path: Path, color: Optional[dict], provider: str) -> None:
    """Add a finished coloured GLB to the cache, keyed by the photo's dHash."""
    if not USE_MESH_CACHE or not image_bytes:
        return
    try:
        h = compute_dhash(image_bytes)
    except Exception as e:
        logger.warning("mesh cache store hash failed: %s", e)
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{h}.glb"
        shutil.copy2(glb_path, CACHE_DIR / fname)
        index = [e for e in _load_index() if e.get("hash") != h]  # replace stale
        index.append({
            "hash": h,
            "glb_file": fname,
            "color": color,
            "provider": provider,
            "created": _now_iso(),
        })
        _save_index(index)
        logger.info("mesh cache STORE %s (provider %s)", fname, provider)
    except Exception as e:
        logger.warning("mesh cache store failed: %s", e)
