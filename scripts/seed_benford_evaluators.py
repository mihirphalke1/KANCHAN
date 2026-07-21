"""Backfill + seed the density log with per-evaluator attribution (P3-11).

The Benford monitor can now slice by `evaluator_id` (a systematic density
anomaly localised to ONE officer is far stronger fraud evidence than a
branch-wide aggregate). For that slice to actually fire, the log needs the
`evaluator_id` column AND enough attributed rows per evaluator (>=30).

This script:
  1. Reads the existing density_log.csv (any older schema is tolerated).
  2. Adds an `evaluator_id` column, attributing existing rows round-robin to
     the real roster in data/evaluators.json (matching branch where possible).
  3. Tops each evaluator up to >=40 rows of Benford-clean submerged-weight
     readings (log-uniform across 0.05-500 g -> naturally Benford-compliant).
  4. Seeds ONE deliberately anomalous evaluator (EMP-1002) whose readings
     cluster on leading digit 1-2 (the "round-number fabrication" signature) so
     the per-evaluator chi-squared test demonstrably ALERTS in a demo while the
     branch aggregate still looks clean.

Idempotent-ish: re-running regenerates the synthetic rows but preserves any
real (case_id not starting with "seed"/"bf") rows. Safe to run before a demo.

    python scripts/seed_benford_evaluators.py
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "data" / "density_log.csv"
EVALUATORS = ROOT / "data" / "evaluators.json"

COLUMNS = ["timestamp", "case_id", "branch_id", "evaluator_id",
           "density", "measurement", "declared_karat"]

RNG = np.random.default_rng(20260722)

# Which evaluator is the planted bad actor, and how many rows each gets.
ANOMALOUS_EVALUATOR = "EMP-1002"
CLEAN_ROWS = 42
ANOM_ROWS = 45


def _load_roster() -> list[dict]:
    if EVALUATORS.exists():
        return json.loads(EVALUATORS.read_text())
    # Fallback matches auth._seed_evaluators()
    return [
        {"evaluator_id": "EMP-1001", "branch_id": "BLR-001"},
        {"evaluator_id": "EMP-1002", "branch_id": "BLR-001"},
        {"evaluator_id": "EMP-9001", "branch_id": "BLR-001"},
    ]


def _benford_clean_measurements(n: int) -> np.ndarray:
    """Log-uniform weights span many orders of magnitude -> Benford-compliant
    leading-digit distribution by construction."""
    exp = RNG.uniform(np.log10(0.05), np.log10(500.0), size=n)
    return np.round(10.0 ** exp, 4)


def _anomalous_measurements(n: int) -> np.ndarray:
    """Fabricated readings a corrupt officer might invent: values clustered so
    the leading digit is almost always 1 or 2 (people inventing 'plausible
    small numbers' over-produce low leading digits beyond Benford's already
    front-loaded expectation). This breaks the chi-squared fit."""
    vals = []
    for _ in range(n):
        lead = RNG.choice([1, 2, 1, 1, 2], )  # heavily 1/2
        frac = RNG.uniform(0, 1)
        scale = RNG.choice([0.1, 1.0, 10.0])
        vals.append(round((lead + frac) * scale, 4))
    return np.array(vals)


def _mk_rows(evaluator_id, branch_id, measurements, tag) -> pd.DataFrame:
    base = datetime(2026, 4, 1, 9, 0, 0)
    rows = []
    for i, m in enumerate(measurements):
        # density is cosmetic here (Benford tests `measurement`); keep it plausible
        density = round(float(RNG.uniform(10.0, 19.5)), 4)
        rows.append({
            "timestamp":      (base + timedelta(hours=i * 7 + hash(evaluator_id) % 5)).isoformat(),
            "case_id":        f"{tag}-{evaluator_id}-{i:03d}",
            "branch_id":      branch_id,
            "evaluator_id":   evaluator_id,
            "density":        density,
            "measurement":    float(m),
            "declared_karat": int(RNG.choice([18, 20, 22, 24])),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    roster = _load_roster()
    # Preserve real rows (not previously synthesised by a seed/backfill script).
    preserved = pd.DataFrame(columns=COLUMNS)
    if LOG.exists():
        old = pd.read_csv(LOG)
        if "evaluator_id" not in old.columns:
            old["evaluator_id"] = None
        for c in COLUMNS:
            if c not in old.columns:
                old[c] = None
        cid = old["case_id"].astype(str)
        is_synth = cid.str.startswith(("seed", "bf-", "anom-", "clean-"))
        preserved = old.loc[~is_synth, COLUMNS].copy()

    frames = [preserved] if len(preserved) else []
    for ev in roster:
        eid = ev["evaluator_id"]
        branch = ev.get("branch_id", "BLR-001")
        if eid == ANOMALOUS_EVALUATOR:
            frames.append(_mk_rows(eid, branch, _anomalous_measurements(ANOM_ROWS), "anom"))
        else:
            frames.append(_mk_rows(eid, branch, _benford_clean_measurements(CLEAN_ROWS), "clean"))

    out = pd.concat(frames, ignore_index=True)
    out = out[COLUMNS]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG, index=False)

    print(f"Wrote {len(out)} rows to {LOG.relative_to(ROOT)} "
          f"({len(preserved)} real preserved, {len(out) - len(preserved)} synthetic).")

    # Sanity: show the per-evaluator Benford verdicts we just seeded.
    from app.benford.monitor import run_benford_test
    print("\nPer-evaluator Benford:")
    for ev in roster:
        r = run_benford_test(evaluator_id=ev["evaluator_id"])
        print(f"  {ev['evaluator_id']:<10} n={r['n_samples']:<3} "
              f"status={r['status']:<17} p={r['p_value']}")
    br = run_benford_test(branch_id="BLR-001")
    print(f"  [branch BLR-001] n={br['n_samples']} status={br['status']} p={br['p_value']}")


if __name__ == "__main__":
    main()
