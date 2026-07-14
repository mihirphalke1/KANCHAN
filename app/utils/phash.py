"""
Perceptual-hash duplicate/reuse detection — the single highest-leverage
anti-fraud check in the Evaluator Integrity Layer.

A dishonest evaluator's easiest move is re-using an old genuine photo for a
new (fake) item. A difference-hash (dHash) is stable across resizing,
recompression and minor crops but changes sharply on real content changes,
so a match against ANY prior photo — from any case, any branch — surfaces a
reused image even if it was touched up before re-upload.

No new dependency: dHash is ~15 lines of PIL + numpy, both already required.
"""
import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HASH_INDEX_PATH = Path("data/photo_hashes.json")

# Hamming distance out of 64 bits below which two photos are treated as the
# same underlying image (allows for recompression / minor crop / resize).
DUPLICATE_HAMMING_THRESHOLD = 6


def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> str:
    """Difference hash: for each row, is pixel[i] brighter than pixel[i+1]?
    Robust to resizing/recompression, sensitive to real content changes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    arr = np.asarray(img, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in diff:
        value = (value << 1) | int(bit)
    nibbles = (hash_size * hash_size + 3) // 4
    return format(value, f"0{nibbles}x")


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def _load_index() -> list:
    if HASH_INDEX_PATH.exists():
        try:
            return json.loads(HASH_INDEX_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read photo hash index: %s", e)
    return []


def _save_index(index: list) -> None:
    HASH_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_INDEX_PATH.write_text(json.dumps(index, indent=2))


def check_duplicate(image_bytes: bytes) -> dict | None:
    """Compare against every photo hash ever recorded, system-wide (not just
    this case). Returns the closest prior match within the duplicate
    threshold, or None if this photo looks new."""
    try:
        h = compute_dhash(image_bytes)
    except Exception as e:
        logger.warning("pHash computation failed: %s", e)
        return None
    best = None
    for entry in _load_index():
        try:
            dist = hamming_distance(h, entry["hash"])
        except Exception:
            continue
        if dist <= DUPLICATE_HAMMING_THRESHOLD and (best is None or dist < best["distance"]):
            best = {**entry, "distance": dist}
    return best


def register_photo(image_bytes: bytes, *, case_id: str, image_role: str, timestamp: str) -> None:
    """Add this photo's hash to the system-wide index. Call only once a case
    is actually being persisted, so the index doesn't grow with abandoned
    uploads that never became a real case."""
    try:
        h = compute_dhash(image_bytes)
    except Exception as e:
        logger.warning("pHash registration failed: %s", e)
        return
    index = _load_index()
    index.append({"hash": h, "case_id": case_id, "image_role": image_role, "timestamp": timestamp})
    _save_index(index)
