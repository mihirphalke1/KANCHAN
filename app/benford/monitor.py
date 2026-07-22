"""
Novelty 2 — Benford's Law on Physical Density Measurements.

Standard in financial fraud detection. First application to physical density
measurements at a bank branch. Catches *systematic* counterfeiting rings —
not individual items, but patterns across 30+ appraisals.

A branch's density log deviating from Benford's Law is statistical evidence
of organised fraud.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare

logger = logging.getLogger(__name__)

DENSITY_LOG_PATH = Path("data/density_log.csv")

BENFORD_PROBS = np.array([
    np.log10(1 + 1 / d) for d in range(1, 10)
])


def first_significant_digit(x: float) -> int | None:
    """Return first significant digit (1-9) of x, or None if invalid."""
    x = abs(x)
    if x == 0 or not np.isfinite(x):
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def run_benford_test(
    values: list[float] | None = None,
    branch_id: str | None = None,
    evaluator_id: str | None = None,
    min_samples: int = 30,
) -> dict:
    """
    Run Benford's Law chi-squared test.
    If values is None, loads from density_log.csv, optionally filtered by
    branch_id and/or evaluator_id.

    An `evaluator_id` filter is the fraud-relevant slice: a systematic
    density anomaly localised to ONE officer's appraisals — while the rest of
    the branch looks Benford-clean — is far stronger evidence of a single
    corrupt evaluator than a branch-wide aggregate, which dilutes one bad
    actor across many honest ones. Falls back gracefully if the log predates
    the `evaluator_id` column (older rows carry no attribution).

    Returns:
      status          "ok" | "anomaly" | "insufficient_data"
      scope           dict describing which slice was tested
      n_samples       int
      p_value         float
      chi_square      float
      alert           bool
      message         str
      digit_observed  list[float]  observed proportions for digits 1-9
      digit_expected  list[float]  Benford expected proportions
    """
    scope = {"branch_id": branch_id, "evaluator_id": evaluator_id}
    if values is None:
        if not DENSITY_LOG_PATH.exists():
            return _insufficient("No density log found.", scope)
        try:
            df = pd.read_csv(DENSITY_LOG_PATH)
            if branch_id:
                df = df[df["branch_id"] == branch_id]
            if evaluator_id:
                if "evaluator_id" not in df.columns:
                    return _insufficient(
                        "Density log has no evaluator_id column yet — "
                        "per-evaluator monitoring starts once attributed rows accrue.",
                        scope,
                    )
                df = df[df["evaluator_id"].astype(str) == str(evaluator_id)]
            col = "measurement" if "measurement" in df.columns else "density"
            values = df[col].dropna().tolist()
        except Exception as e:
            logger.error("Failed to load density log: %s", e)
            return _insufficient(f"Could not read density log: {e}", scope)

    if len(values) < min_samples:
        return _insufficient(f"Only {len(values)} samples (need ≥ {min_samples}).", scope)

    digits   = [first_significant_digit(v) for v in values]
    digits   = [d for d in digits if d is not None]
    n        = len(digits)
    observed = np.array([digits.count(d) for d in range(1, 10)], dtype=float)

    if observed.sum() == 0:
        return _insufficient("No valid digits extracted.", scope)

    expected = BENFORD_PROBS * n
    chi2, p  = chisquare(observed, expected)

    alert = p < 0.05
    _who = (f" for evaluator {evaluator_id}" if evaluator_id
            else f" for branch {branch_id}" if branch_id else "")
    return {
        "status":         "anomaly" if alert else "ok",
        "scope":          scope,
        "n_samples":      n,
        "p_value":        round(float(p), 6),
        "chi_square":     round(float(chi2), 4),
        "alert":          alert,
        "message": (
            f"⚠️ BATCH ALERT: Systematic density anomaly detected{_who} "
            f"({n} samples, p={p:.4f}). "
            + ("Possible corrupt evaluator — escalate to branch manager and audit their appraisals."
               if evaluator_id else
               "Possible organised counterfeiting ring — escalate to branch manager.")
            if alert else
            f"No batch anomaly detected{_who} ({n} samples, p={p:.4f})."
        ),
        "digit_observed": [round(v / n, 4) for v in observed.tolist()],
        "digit_expected": [round(float(v), 4) for v in BENFORD_PROBS.tolist()],
    }


DENSITY_LOG_COLUMNS = [
    "timestamp", "case_id", "branch_id", "evaluator_id",
    "density", "measurement", "declared_karat",
]


def append_density_reading(
    density: float,
    case_id: str,
    branch_id: str = "default",
    evaluator_id: str | None = None,
    declared_karat: int | None = None,
    weight_submerged: float | None = None,
) -> None:
    """
    Append a measurement to the log CSV.
    Uses weight_submerged as the Benford-tested column when available —
    submerged weights span multiple orders of magnitude (0.05g tiny ring →
    500g heavy necklace), giving a natural Benford-compliant distribution.
    Falls back to density if not provided.

    `evaluator_id` attributes each reading to the authenticated officer, so the
    Benford monitor can localise an anomaly to a single corrupt evaluator
    rather than only a whole branch. Written even when the existing file has an
    older (evaluator-less) header — pandas fills the gap on read.
    """
    DENSITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    measurement = weight_submerged if weight_submerged is not None else density
    row = pd.DataFrame([{
        "timestamp":       datetime.now().isoformat(),
        "case_id":         case_id,
        "branch_id":       branch_id,
        "evaluator_id":    evaluator_id,
        "density":         density,
        "measurement":     measurement,
        "declared_karat":  declared_karat,
    }], columns=DENSITY_LOG_COLUMNS)
    header = not DENSITY_LOG_PATH.exists()
    row.to_csv(DENSITY_LOG_PATH, mode="a", header=header, index=False)


def _insufficient(reason: str, scope: dict | None = None) -> dict:
    return {
        "status":         "insufficient_data",
        "scope":          scope or {},
        "n_samples":      0,
        "p_value":        None,
        "chi_square":     None,
        "alert":          False,
        "message":        reason,
        "digit_observed": [],
        "digit_expected": BENFORD_PROBS.round(4).tolist(),
    }
