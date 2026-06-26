#!/usr/bin/env python3
"""
Seed synthetic demo data for KANCHAN-AI.

Creates:
  - data/density_log.csv  — 60 synthetic density readings (50 genuine-like + 10 anomalous)
  - data/case_history.json — 5 sample case records for history display

Run once before demo. Safe to re-run (overwrites existing demo data).
"""
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

DENSITY_LOG_PATH = Path("data/density_log.csv")
HISTORY_PATH     = Path("data/case_history.json")

KARAT_NOMINALS = {22: 17.80, 18: 15.55, 24: 19.32, 14: 13.07}


def generate_benford_sample(n: int, scale: float = 15.0) -> list[float]:
    """Generate n values whose first digits approximately follow Benford's Law."""
    values = []
    base   = datetime.utcnow() - timedelta(days=30)
    for i in range(n):
        d   = np.random.choice(range(1, 10), p=np.array([np.log10(1 + 1/d) for d in range(1, 10)]))
        val = d * (10 ** np.random.uniform(-0.3, 0.3))
        values.append(round(val, 4))
    return values


def seed_density_log():
    """
    Seed density_log.csv with 60 appraisals.

    The Benford-tested column is `measurement` = submerged weight (grams).
    Submerged weights span ~0.05g (tiny studs) to ~470g (heavy ornaments),
    naturally producing first-digit distributions that follow Benford's Law.

    Genuine batch (50): item dry weights ~Benford distributed, sub derived from
    karat density. These naturally follow Benford's Law.
    Anomalous batch (10): suspicious, clustered values (all starting with 5-6)
    indicating fabricated measurements from a counterfeiting ring.
    """
    np.random.seed(42)
    karats     = np.random.choice([14, 18, 22, 24], size=60)
    branch_ids = np.random.choice(["BLR-001", "BLR-002", "MUM-003"], size=60)
    case_ids   = [uuid.uuid4().hex[:8] for _ in range(60)]
    base_time  = datetime.now() - timedelta(days=45)

    rows = []
    for i in range(60):
        karat   = int(karats[i])
        nominal = KARAT_NOMINALS[karat]

        if i < 50:
            # Genuine: dry weights sampled from Benford distribution
            # spanning 1g (stud earring) to 200g (heavy necklace)
            d = np.random.choice(range(1, 10), p=[np.log10(1+1/d) for d in range(1, 10)])
            dry = round(d * np.random.uniform(1, 30), 3)
            noise   = np.random.normal(0, 0.05)
            density = nominal + noise
            sub     = round(dry * (density - 1) / density, 4)
            measurement = sub
        else:
            # Anomalous: fabricated sub weights ALL starting with 8 or 9 —
            # a signature of rounded/fabricated measurements in a fraud ring
            prefix  = np.random.choice([8, 9])
            dry     = round(prefix * np.random.uniform(1.0, 1.09), 4)
            density = round(np.random.uniform(7.5, 10.5), 4)
            sub     = round(dry * (density - 1) / density, 4)
            measurement = sub

        rows.append({
            "timestamp":       (base_time + timedelta(hours=i * 7)).isoformat(),
            "case_id":         case_ids[i],
            "branch_id":       str(branch_ids[i]),
            "density":         round(density, 4),
            "measurement":     measurement,
            "declared_karat":  karat,
        })

    df = pd.DataFrame(rows)
    DENSITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DENSITY_LOG_PATH, index=False)
    print(f"[seed] density_log.csv — {len(rows)} rows (50 genuine-like, 10 anomalous)")


def seed_case_history():
    base_time = datetime.utcnow() - timedelta(hours=6)
    cases = [
        {
            "case_id": "demo0001",
            "timestamp": (base_time - timedelta(hours=5)).isoformat() + "Z",
            "item_description": "22K gold necklace with pendant",
            "declared_karat": 22,
            "branch_id": "BLR-001",
            "modality_scores": {
                "image":    {"risk_score": 0.12, "confidence": "low",  "mode": "heuristic"},
                "density":  {"risk_score": 0.05, "confidence": "high", "mode": "computed",
                             "measured_density": 17.76, "expected_low": 17.4, "expected_high": 18.1,
                             "deviation_pct": -0.22, "karat_verdict": "IN_RANGE",
                             "closest_fake": None, "tungsten_warning": False},
                "acoustic": {"risk_score": 0.18, "confidence": "low",  "mode": "heuristic:heuristic"},
                "streak":   {"risk_score": 0.20, "confidence": "low",  "mode": "heuristic"},
            },
            "contradiction": {
                "contradiction_score": 0.08, "mean_contradiction": 0.04,
                "flags": [], "cross_pairs": {},
            },
            "fusion": {"risk_score": 0.12, "mode": "heuristic", "shap_values": None},
            "benford": {"status": "ok", "n_samples": 50, "p_value": 0.34, "chi_square": 7.2,
                        "alert": False, "message": "No batch anomaly detected (50 samples, p=0.3400)."},
            "verdict": {
                "risk_level": "GENUINE", "confidence": "HIGH", "loan_action": "APPROVE",
                "plain_english": "All four analysis signals are consistent with genuine 22K gold. The measured density of 17.76 g/cm³ falls squarely within the expected range, and no acoustic anomalies were detected.",
                "action": "The item appears genuine. Proceed with loan approval subject to standard documentation checks.",
                "llm_provider": "heuristic",
            },
        },
        {
            "case_id": "demo0002",
            "timestamp": (base_time - timedelta(hours=3)).isoformat() + "Z",
            "item_description": "24K gold bangle (tungsten-core suspect)",
            "declared_karat": 24,
            "branch_id": "BLR-001",
            "modality_scores": {
                "image":    {"risk_score": 0.15, "confidence": "low",  "mode": "heuristic"},
                "density":  {"risk_score": 0.06, "confidence": "high", "mode": "computed",
                             "measured_density": 19.18, "expected_low": 19.1, "expected_high": 19.4,
                             "deviation_pct": -0.72, "karat_verdict": "IN_RANGE",
                             "closest_fake": "tungsten", "tungsten_warning": True},
                "acoustic": {"risk_score": 0.82, "confidence": "low",  "mode": "heuristic:heuristic"},
                "streak":   {"risk_score": 0.20, "confidence": "low",  "mode": "heuristic"},
            },
            "contradiction": {
                "contradiction_score": 0.76,
                "mean_contradiction": 0.32,
                "flags": [
                    "density↔acoustic: density suggests genuine but acoustic suggests fake — possible tungsten-core or gold-plated composite"
                ],
                "cross_pairs": {
                    "density↔acoustic": 0.76, "density↔image": 0.09,
                    "image↔streak": 0.05, "acoustic↔image": 0.67,
                    "density↔streak": 0.14, "acoustic↔streak": 0.62,
                },
            },
            "fusion": {"risk_score": 0.68, "mode": "heuristic", "shap_values": None},
            "benford": {"status": "ok", "n_samples": 50, "p_value": 0.34, "chi_square": 7.2,
                        "alert": False, "message": "No batch anomaly detected."},
            "verdict": {
                "risk_level": "BORDERLINE", "confidence": "MEDIUM", "loan_action": "HOLD",
                "plain_english": "⚠ TUNGSTEN BLIND-SPOT: The density test passes (19.18 g/cm³ matches 24K gold), but the acoustic ring is heavily dampened — a signature of a tungsten-core item plated with gold. Density alone cannot distinguish tungsten from 24K gold.",
                "action": "Hold the item. Request XRF spectrometry or acid test before approving. This is the key cross-modal contradiction scenario.",
                "llm_provider": "heuristic",
            },
        },
        {
            "case_id": "demo0003",
            "timestamp": (base_time - timedelta(hours=1)).isoformat() + "Z",
            "item_description": "18K gold ring (copper base metal)",
            "declared_karat": 18,
            "branch_id": "BLR-002",
            "modality_scores": {
                "image":    {"risk_score": 0.62, "confidence": "low",  "mode": "heuristic"},
                "density":  {"risk_score": 0.88, "confidence": "high", "mode": "computed",
                             "measured_density": 9.21, "expected_low": 15.2, "expected_high": 15.9,
                             "deviation_pct": -40.7, "karat_verdict": "LOW_DENSITY",
                             "closest_fake": "copper", "tungsten_warning": False},
                "acoustic": {"risk_score": 0.74, "confidence": "low",  "mode": "heuristic:heuristic"},
                "streak":   {"risk_score": 0.55, "confidence": "low",  "mode": "heuristic"},
            },
            "contradiction": {
                "contradiction_score": 0.26, "mean_contradiction": 0.12,
                "flags": [], "cross_pairs": {},
            },
            "fusion": {"risk_score": 0.81, "mode": "heuristic", "shap_values": None},
            "benford": {"status": "ok", "n_samples": 50, "p_value": 0.34, "chi_square": 7.2,
                        "alert": False, "message": "No batch anomaly detected."},
            "verdict": {
                "risk_level": "REJECT", "confidence": "HIGH", "loan_action": "DECLINE",
                "plain_english": "The density measurement of 9.21 g/cm³ is far below the expected range for 18K gold (15.2–15.9 g/cm³), consistent with copper base metal. The acoustic test also shows a dampened thud rather than a ring.",
                "action": "Do not approve the loan. Escalate to the branch gold appraiser for manual inspection and document the case.",
                "llm_provider": "heuristic",
            },
        },
    ]

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(cases, indent=2))
    print(f"[seed] case_history.json — {len(cases)} demo cases")


if __name__ == "__main__":
    seed_density_log()
    seed_case_history()
    print("\n[seed] Demo data ready.")
