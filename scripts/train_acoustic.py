#!/usr/bin/env python3
"""
Train acoustic SVM on DS-1 gold vs copper audio.

Requires DS-1 to be downloaded (run download_datasets.py first).
Uses leave-one-out cross-validation (20 samples total).

Output: models/acoustic_svm.pkl
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle

LOG = Path("AGENT_LOG.md")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"\n---\n[{datetime.now().isoformat()}] {msg}\n---\n")


def main():
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    gold_dir   = Path("data/raw/counterfeit_gold/plain_sound")
    if not gold_dir.exists():
        print("ERROR: DS-1 not found. Run scripts/download_datasets.py first.")
        return

    from app.models.acoustic_model import extract_mfcc_features

    gold_files   = sorted(gold_dir.glob("gold_*.wav"))
    copper_files = sorted(gold_dir.glob("copper_*.wav"))

    if not gold_files or not copper_files:
        print("ERROR: No WAV files found. Check DS-1 directory structure.")
        return

    X, y = [], []
    for f in gold_files:
        feats = extract_mfcc_features(f.read_bytes())
        X.append(feats); y.append(0)

    for f in copper_files:
        feats = extract_mfcc_features(f.read_bytes())
        X.append(feats); y.append(1)

    X = np.array(X); y = np.array(y)
    print(f"Dataset: {len(X)} samples ({y.sum()} fake, {(y==0).sum()} genuine)")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    ])

    loo    = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
    auc    = cross_val_score(model, X, y, cv=loo, scoring="roc_auc")
    print(f"LOOCV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"LOOCV AUC:      {auc.mean():.3f} ± {auc.std():.3f}")

    model.fit(X, y)
    out = Path("models/acoustic_svm.pkl")
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {out}")

    log(f"Phase 2.2 — Acoustic SVM trained\nSTATUS: DONE\nMODEL: models/acoustic_svm.pkl\nDATASET: DS-1 ({len(X)} samples)\nACCURACY: {scores.mean():.3f}\nAUC: {auc.mean():.3f}")


if __name__ == "__main__":
    main()
