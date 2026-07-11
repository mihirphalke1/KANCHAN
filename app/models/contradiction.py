"""
Novelty 3 — Cross-Modal Contradiction Detection.

Standard multimodal systems fuse AGREEMENT. We model DISAGREEMENT between
physically-correlated signal pairs as an explicit XGBoost feature.

Key case: tungsten-core fakes pass density (19.25 ≈ 24K) and visual
(gold-plated surface) but fail acoustic (dampened ring). The density↔acoustic
contradiction score catches what any single-modal test misses.
"""
import numpy as np

CORRELATED_PAIRS = [
    ("density", "acoustic"),
    ("density", "image"),
    ("image",   "streak"),
    ("acoustic", "image"),
    ("density", "streak"),
    ("acoustic", "streak"),
]

# Extra pairs used only for the human-facing summary and verdict boost.
# NOT part of compute_contradiction_features — the fusion XGBoost model was
# trained on the fixed 6-dim vector above and must not change shape.
XRAY_PAIRS = [
    ("density",  "xray"),
    ("image",    "xray"),
    ("acoustic", "xray"),
]

THRESHOLD = 0.35


def compute_contradiction_features(scores: dict[str, float]) -> np.ndarray:
    """
    Given per-modality risk scores, return a 6-dim contradiction feature vector.
    Each element = |score_A - score_B| for a correlated pair.
    Pairs where either modality is absent get value 0.0 (no information).
    """
    feats = []
    for a, b in CORRELATED_PAIRS:
        if a in scores and b in scores:
            feats.append(abs(scores[a] - scores[b]))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def contradiction_summary(scores: dict[str, float]) -> dict:
    """
    Compute contradiction features and flag any suspicious disagreements.

    Returns:
      contradiction_score  float   max pairwise disagreement
      mean_contradiction   float   mean pairwise disagreement
      flags                list    human-readable flag strings
      cross_pairs          dict    pair→disagreement magnitude
    """
    pairs = CORRELATED_PAIRS + XRAY_PAIRS
    feats = [
        abs(scores[a] - scores[b]) if a in scores and b in scores else 0.0
        for a, b in pairs
    ]
    cross_pairs = {f"{a}↔{b}": round(float(v), 4)
                   for (a, b), v in zip(pairs, feats)}

    contradiction_score = float(max(feats))
    mean_contradiction  = float(np.mean(feats))

    flags = []
    for (a, b), val in zip(pairs, feats):
        if val > THRESHOLD:
            sa = scores.get(a, 0.5)
            sb = scores.get(b, 0.5)
            low_sig  = a if sa < sb else b
            high_sig = b if sa < sb else a
            if "density" in (a, b) and "acoustic" in (a, b):
                cause = "tungsten-core or gold-plated composite"
            elif "xray" in (a, b):
                cause = "plated or composite surface structure"
            else:
                cause = "composite item"
            flags.append(
                f"{a}↔{b}: {low_sig} suggests genuine but {high_sig} suggests fake "
                f"— possible {cause}"
            )

    return {
        "contradiction_score": round(contradiction_score, 4),
        "mean_contradiction":  round(mean_contradiction, 4),
        "flags":               flags,
        "cross_pairs":         cross_pairs,
    }
