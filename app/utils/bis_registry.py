"""
Mock BIS HUID registry — SIMULATED.

BIS does not expose a public developer API for HUID verification; the only
real lookup path today is the consumer-facing BIS Care app. This mock
reproduces the exact data contract Canara's own systems would call once a
BIS integration/MOU exists, backed by a small local reference file
(data/bis_mock_registry.json) instead of the real central database.

Never trusted alone — see app/models/hallmark_model.py for why a HUID
match/mismatch is only one signal among several.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("data/bis_mock_registry.json")


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read mock BIS registry: %s", e)
    return {}


def lookup_huid(huid: str) -> Optional[dict]:
    registry = _load_registry()
    record = registry.get((huid or "").strip().upper())
    if not record:
        return None
    return {**record, "simulated": True}
