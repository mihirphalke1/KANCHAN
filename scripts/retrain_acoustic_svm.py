#!/usr/bin/env python3
"""
Retrain the acoustic SVM on the 122-dim MFCC-ΔΔ feature vector.

The original DS-1 dataset (data/raw/counterfeit_gold/) is not present in this
checkout, so this retrains on the labelled demo tap kit
(data/demo/test-kit/audio/) — genuine gold taps vs filled/composite taps. Small
(4 vs 4), so treat the reported LOOCV accuracy as indicative, not a benchmark;
the SVM is a SECONDARY acoustic signal (ring-frequency physics is primary).

Re-pickling here also clears the sklearn InconsistentVersionWarning (the shipped
pkl was trained on 1.4.2; this writes it on the installed version).

Output: models/acoustic_svm.pkl
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIO_DIR = Path("data/demo/test-kit/audio")
MODEL_PATH = Path("models/acoustic_svm.pkl")
META_PATH = Path("models/acoustic_svm_meta.json")

# A model is trusted as a fusion signal only when it was trained on enough
# labelled recordings AND cross-validates well. Below this it still loads and
# runs (so the pipeline is exercised end to end), but analyze_acoustic treats
# its output as informational, not decisive — the ring-frequency PHYSICS
# (empirically non-overlapping on real data) carries the acoustic verdict.
MIN_RELIABLE_SAMPLES = 16
MIN_RELIABLE_LOOCV = 0.75

GENUINE = ["genuine_gold_tap_1", "genuine_gold_tap_2", "genuine_gold_tap_3",
           "synthetic_bangle_genuine_tap"]
FAKE    = ["fake_composite_tap_1", "fake_composite_tap_2", "fake_composite_tap_3",
           "synthetic_bangle_filled_core_tap"]


def main():
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    from app.models.acoustic_model import extract_mfcc_features, FEATURE_DIM

    X, y = [], []
    for stem, label in [(s, 0) for s in GENUINE] + [(s, 1) for s in FAKE]:
        p = AUDIO_DIR / f"{stem}.wav"
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        feats = extract_mfcc_features(p.read_bytes())
        assert feats.shape[0] == FEATURE_DIM, f"expected {FEATURE_DIM}-dim, got {feats.shape[0]}"
        X.append(feats); y.append(label)

    X = np.array(X); y = np.array(y)
    print(f"Training on {len(X)} samples ({int((y == 0).sum())} genuine, "
          f"{int((y == 1).sum())} fake) × {X.shape[1]} features")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=42)),
    ])

    loocv = None
    if len(X) >= 4:
        scores = cross_val_score(pipe, X, y, cv=LeaveOneOut())
        loocv = float(scores.mean())
        print(f"LOOCV accuracy: {loocv:.2f} ({int(scores.sum())}/{len(scores)})")

    pipe.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    reliable = (len(X) >= MIN_RELIABLE_SAMPLES
                and loocv is not None and loocv >= MIN_RELIABLE_LOOCV)
    meta = {
        "feature_dim":   int(X.shape[1]),
        "n_train":       int(len(X)),
        "n_genuine":     int((y == 0).sum()),
        "n_fake":        int((y == 1).sum()),
        "loocv":         round(loocv, 3) if loocv is not None else None,
        "reliable":      bool(reliable),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "source":        "data/demo/test-kit/audio (DS-1 absent)",
        "note": (
            "Trusted as a fusion signal." if reliable else
            "DATA-LIMITED: too few labelled taps (DS-1 absent) for a trustworthy "
            "classifier — analyze_acoustic returns a neutral score and defers to "
            "the ring-frequency physics. Restore DS-1 and re-run to enable."
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved {MODEL_PATH} (n_features_in_={pipe.named_steps['svc'].n_features_in_})")
    print(f"Saved {META_PATH} (reliable={reliable})")


if __name__ == "__main__":
    main()
