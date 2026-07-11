#!/usr/bin/env python3
"""
Calibrate the DSIP risk baselines from a panel of KNOWN-GENUINE photos.

The material-scan risk features (composition entropy, edge density) need a
"what does genuine look like" reference. Instead of asserted constants, this
script measures the distributions over a genuine photo panel and writes:

    baseline = 75th percentile of genuine   (most genuine items -> risk ~0)
    span     = 2 x (p99 - p75) of genuine   (the top of the genuine range
                                             maps to modest risk ~0.12; only
                                             values far beyond genuine reach 1)

to data/dsip_calibration.json. app/models/xray_model.py loads this file when
present and falls back to its documented defaults otherwise.

Usage:
    python3 scripts/calibrate_dsip.py <folder> [more folders...]
    # e.g. python3 scripts/calibrate_dsip.py data/raw/tanishq \
    #        data/raw/counterfeit_gold/gold/bare_gold/images

Only photos where the background separates cleanly are used (the same gate
the live pipeline applies).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.xray_model import _composition_entropy
from app.utils.xray import _load_bgr, _run_pipeline

OUT = ROOT / "data/dsip_calibration.json"
CAP_PER_FOLDER = 150


def main(folders: list[str]) -> None:
    entropies, edges = [], []
    used = skipped = 0
    for folder in folders:
        photos = sorted(Path(folder).rglob("*.jpg"))[:CAP_PER_FOLDER]
        print(f"{folder}: {len(photos)} photos")
        for i, p in enumerate(photos):
            try:
                _, stats = _run_pipeline(_load_bgr(p.read_bytes()))
            except Exception:
                skipped += 1
                continue
            if not stats.get("background_removed"):
                skipped += 1
                continue
            entropies.append(_composition_entropy(stats["composition"]))
            edges.append(stats["edge_density"])
            used += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(photos)}")

    if used < 20:
        raise SystemExit(f"Only {used} usable photos — need ≥20 for stable percentiles")

    def band(vals):
        v = np.array(vals)
        p75, p99 = float(np.percentile(v, 75)), float(np.percentile(v, 99))
        return round(p75, 4), round(max(2 * (p99 - p75), 1e-3), 4)

    e_base, e_span = band(entropies)
    d_base, d_span = band(edges)

    cal = {
        "entropy_baseline": e_base,
        "entropy_span":     e_span,
        "edge_baseline":    d_base,
        "edge_span":        d_span,
        "n_photos":         used,
        "skipped":          skipped,
        "calibrated_on":    str(date.today()),
        "sources":          folders,
        "genuine_stats": {
            "entropy": {"p50": round(float(np.percentile(entropies, 50)), 4),
                        "p75": e_base,
                        "p99": round(float(np.percentile(entropies, 99)), 4)},
            "edge_density": {"p50": round(float(np.percentile(edges, 50)), 4),
                             "p75": d_base,
                             "p99": round(float(np.percentile(edges, 99)), 4)},
        },
    }
    OUT.write_text(json.dumps(cal, indent=2))
    print(f"\nGenuine panel (n={used}):")
    print(f"  entropy      p50={cal['genuine_stats']['entropy']['p50']}  p75={e_base}  p99={cal['genuine_stats']['entropy']['p99']}  -> span {e_span}")
    print(f"  edge density p50={cal['genuine_stats']['edge_density']['p50']}  p75={d_base}  p99={cal['genuine_stats']['edge_density']['p99']}  -> span {d_span}")
    print(f"Written -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: calibrate_dsip.py <folder-of-genuine-photos> [...]")
    main(sys.argv[1:])
