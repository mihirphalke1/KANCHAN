"""
Gemstone grid — divides the item's bounding box into an N×N grid, buckets
detected stones (from app/utils/xray.py's DSIP pipeline) into cells, and —
when a fiducial scale reference (px_per_mm) is available — estimates each
stone's real-world diameter and a standard trade weight deduction.

No new detection model: this is a presentation + measurement layer over
stones the DSIP pipeline already finds independently of any description.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEDUCTION_TABLE_PATH = Path("data/stone_deduction_table.json")

# Standard trade deduction bands (grams) by estimated stone diameter (mm).
# Illustrative defaults — jewellers commonly use a flat per-size-band
# deduction rather than a computed volume, since exact stone density and cut
# vary. Always officer-overridable per stone in the UI; this is an assistive
# starting estimate, never an autonomous final number.
DEFAULT_DEDUCTION_TABLE = [
    {"max_diameter_mm": 2.0, "deduction_g": 0.005},
    {"max_diameter_mm": 4.0, "deduction_g": 0.020},
    {"max_diameter_mm": 6.0, "deduction_g": 0.045},
    {"max_diameter_mm": 999, "deduction_g": 0.090},
]


def load_deduction_table() -> list:
    if DEDUCTION_TABLE_PATH.exists():
        try:
            return json.loads(DEDUCTION_TABLE_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read stone deduction table: %s", e)
    return DEFAULT_DEDUCTION_TABLE


def _deduction_for_diameter(mm: float, table: list) -> float:
    for band in table:
        if mm <= band["max_diameter_mm"]:
            return band["deduction_g"]
    return table[-1]["deduction_g"]


def build_grid_stats(
    item_bbox: list[int] | None, stones: list[dict], grid_n: int = 4,
    px_per_mm: float | None = None,
) -> dict:
    """
    item_bbox: [x0, y0, x1, y1] of the segmented item (from the DSIP pipeline).
    stones:    the DSIP stones list, each with a pixel bbox [x, y, w, h].
    """
    if not item_bbox or stones is None:
        return {"grid_n": grid_n, "cells": [], "total_stones": 0,
                "px_per_mm": px_per_mm, "total_deduction_g": None}

    x0, y0, x1, y1 = item_bbox
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    table = load_deduction_table()

    cells = {(r, c): {"row": r, "col": c, "count": 0, "stones": []}
             for r in range(grid_n) for c in range(grid_n)}
    total_deduction = 0.0 if px_per_mm else None

    for s in stones:
        bx, by, bw, bh = s.get("bbox", [0, 0, 0, 0])
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        col = min(grid_n - 1, max(0, int((cx - x0) / w * grid_n)))
        row = min(grid_n - 1, max(0, int((cy - y0) / h * grid_n)))
        entry = {"hue_class": s.get("hue_class"), "status": s.get("status"),
                  "area_pct": s.get("area_pct")}
        if px_per_mm:
            diameter_mm = round(((bw + bh) / 2.0) / px_per_mm, 2)
            deduction_g = _deduction_for_diameter(diameter_mm, table)
            entry["diameter_mm"] = diameter_mm
            entry["deduction_g"] = deduction_g
            total_deduction += deduction_g
        cells[(row, col)]["count"] += 1
        cells[(row, col)]["stones"].append(entry)

    return {
        "grid_n":            grid_n,
        "cells":              list(cells.values()),
        "total_stones":       len(stones),
        "px_per_mm":          px_per_mm,
        "total_deduction_g":  round(total_deduction, 3) if total_deduction is not None else None,
        "deduction_table":    table,
        "scale_source":       "fiducial_card" if px_per_mm else None,
    }
