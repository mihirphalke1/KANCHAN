#!/usr/bin/env python3
"""
Train XGBoost fusion model on processed modality outputs.

Requires data/processed/fusion_dataset.csv (created by build_image_dataset.py).
Run after all individual modality models are trained.

Output: models/fusion_xgb.pkl
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
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    fusion_csv = Path("data/processed/fusion_dataset.csv")
    if not fusion_csv.exists():
        print("ERROR: fusion_dataset.csv not found.")
        print("Run scripts/build_image_dataset.py to create it.")
        return

    df = pd.read_csv(fusion_csv)
    FEATURES = [
        "image_risk", "density_risk", "acoustic_risk", "streak_risk",
        "contra_density_acoustic", "contra_density_image",
        "contra_image_streak", "contra_acoustic_image",
        "contra_density_streak", "contra_acoustic_streak",
    ]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    X = df[FEATURES].values
    y = df["label"].values
    print(f"Dataset: {len(X)} samples, {y.sum()} fake, {(y==0).sum()} genuine")

    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        min_child_weight=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=min(5, y.sum()), shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    model.fit(X, y)
    out = Path("models/fusion_xgb.pkl")
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {out}")

    log(f"Phase 3.3 — XGBoost fusion trained\nSTATUS: DONE\nMODEL: models/fusion_xgb.pkl\nDATASET: {len(X)} samples\nAUC: {auc_scores.mean():.3f}")


if __name__ == "__main__":
    main()
