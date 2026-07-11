#!/usr/bin/env python3
"""
Rebuild the fusion training dataset WITHOUT label leakage, then retrain XGBoost.

The one rule: every feature value is produced by the exact code path that
produces it at inference time (app/routers/analyze.py -> analyze_fusion).

That means:
  * image risk      -> analyze_image() on a real photo, blended 0.65/0.35 with
                       analyze_xray() on the same photo (as analyze.py does),
                       or the 0.5 no-image default
  * acoustic risk   -> analyze_acoustic() on a real tap recording (DS-1),
                       or the 0.5 no-audio default
  * density risk    -> analyze_density() on SIMULATED WEIGHINGS of a true
                       material density: true rho -> (dry, submerged) weights
                       with rho_water(T) buoyancy and N(0, scale_sigma) noise
                       on each weighing. The physics is simulated; the risk
                       scoring path is the real one.
  * streak risk     -> 0.5 abstention everywhere (no real touchstone dataset
                       exists; DS-7 uncollected). The model must learn to not
                       rely on streak. When DS-7 is collected, add real rows.
  * contradiction   -> fusion_model.build_feature_vector(), i.e. the exact
                       inference function, never recomputed by hand.

Missing modalities occur at the SAME rate in both classes so that absence
carries no label information (the old dataset leaked labels through
hardcoded 0.1/0.75 streak and acoustic constants).

Row groups
  genuine : DS-1 bare_gold photos + genuine taps + in-band density
            DS-3 Tanishq photos (audio present p=AUDIO_P via genuine taps)
            no-photo rows (density + audio only)
  fake    : DS-1 bare_copper photos + composite taps + copper density
            DS-2 NEU defect photos (proxy for plated/defective surface)
            under-karat rows (18K metal declared as 22K)
            no-photo rows
  tungsten-like (label=1, documented proxy):
            gold-appearance photos (DS-1 bare_gold) + COMPOSITE tap audio +
            density inside the 24K band. Physical basis: interface damping is
            a property of any core-shell composite, whatever the filler; the
            DS-1 "copper" recordings are taps on plated composites and stand
            in for the acoustic signature of a filled item. The mixed pattern
            (density passes / acoustic fails) is what gives the contradiction
            features label-correlated variance.

Usage:  python3 scripts/rebuild_fusion.py [--rows-cap N]
Output: data/processed/fusion_dataset_v2.csv, models/fusion_xgb.pkl
        (previous model backed up to models/fusion_xgb.pkl.bak-leaky)
"""
import argparse
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.acoustic_model import analyze_acoustic
from app.models.density_model import analyze_density
from app.models.image_model import analyze_image
from app.models.xray_model import analyze_xray, VISUAL_BLEND_IMAGE, VISUAL_BLEND_XRAY
from app.models.fusion_model import build_feature_vector
from app.utils.xray import _load_bgr, _run_pipeline
from app.utils.density import water_density, SCALE_SIGMA_G

DS1 = ROOT / "data/raw/counterfeit_gold"
DS2 = ROOT / "data/raw/neu_defect/NEU-DET/train/images"
DS3 = ROOT / "data/raw/tanishq"
OUT = ROOT / "data/processed/fusion_dataset_v2.csv"

RNG = np.random.default_rng(42)

SCALE_SIGMA = SCALE_SIGMA_G   # g — same balance repeatability the scorer assumes
AUDIO_P     = 0.6      # fraction of rows with a tap recording (both classes)
ABSTAIN     = 0.5      # inference default for a missing modality

# True-density ranges (g/cm3). Karat bands per the app's own reference table;
# base metals per CRC Handbook values already in app/utils/density.py.
MATERIALS = {
    "gold_22k":  (17.40, 18.10, 22),
    "gold_18k":  (15.20, 15.90, 18),
    "gold_24k":  (19.10, 19.40, 24),
    "copper":    (8.40,  9.00,  22),   # copper/brass declared as 22K
    "underkarat": (15.20, 15.90, 22),  # 18K metal declared as 22K
    "tungsten":  (18.90, 19.30, 24),   # core-shell composite, passes 24K band
}


def simulated_density_risk(material: str) -> float:
    """
    True density -> two noisy weighings -> the real analyze_density path.
    The bath temperature is drawn but scoring uses the API default (25 C),
    which models the real-world case of an officer not measuring it.
    """
    lo, hi, declared = MATERIALS[material]
    rho_true = RNG.uniform(lo, hi)
    mass     = RNG.uniform(8.0, 25.0)                    # g
    volume   = mass / rho_true                           # cm3
    temp_c   = RNG.uniform(18.0, 30.0)                   # branch water bath
    sub_true = mass - volume * water_density(temp_c)     # buoyancy
    dry  = mass     + RNG.normal(0, SCALE_SIGMA)
    sub  = sub_true + RNG.normal(0, SCALE_SIGMA)
    if sub >= dry:                                       # noise crossed over
        sub = dry - 0.01
    return analyze_density(float(dry), float(sub), declared)["risk_score"]


def visual_risk_from_photo(path: Path, description: str) -> float:
    """Replicate analyze.py's visual channel blend exactly."""
    raw = path.read_bytes()
    image_risk = analyze_image([raw])["risk_score"]
    try:
        _, stats = _run_pipeline(_load_bgr(raw))
        xray_risk = analyze_xray(stats, description)["risk_score"]
    except Exception:
        return image_risk
    return round(VISUAL_BLEND_IMAGE * image_risk + VISUAL_BLEND_XRAY * xray_risk, 4)


def acoustic_pool(folder: Path) -> list[float]:
    """Risk score of every recording in a folder through the real model."""
    scores = []
    for wav in sorted(folder.glob("*.wav")):
        scores.append(analyze_acoustic(wav.read_bytes())["risk_score"])
    return scores


def make_row(visual, density, acoustic, label):
    feats = build_feature_vector(visual, density, acoustic, ABSTAIN)
    return list(map(float, feats)) + [label]


def sample_audio(pool: list[float]) -> float:
    """Real tap risk with prob AUDIO_P, else the inference no-audio default."""
    if RNG.random() < AUDIO_P and pool:
        return pool[RNG.integers(len(pool))]
    return ABSTAIN


def main(rows_cap: int):
    import pandas as pd

    print("Scoring DS-1 tap recordings through the real acoustic model...")
    genuine_taps   = acoustic_pool(DS1 / "wav/genuine")
    composite_taps = acoustic_pool(DS1 / "wav/fake")
    print(f"  genuine taps   ({len(genuine_taps)}): "
          f"mean risk {np.mean(genuine_taps):.3f}")
    print(f"  composite taps ({len(composite_taps)}): "
          f"mean risk {np.mean(composite_taps):.3f}")

    rows = []

    # ── genuine: DS-1 gold photos ──
    for p in sorted((DS1 / "gold/bare_gold/images").glob("*.jpg")):
        v = visual_risk_from_photo(p, "gold item")
        rows.append(make_row(v, simulated_density_risk("gold_22k"),
                             sample_audio(genuine_taps), 0))

    # ── genuine: Tanishq catalogue photos ──
    tanishq = sorted(DS3.rglob("*.jpg"))[:rows_cap]
    print(f"Scoring {len(tanishq)} Tanishq photos (visual = image + xray blend)...")
    for i, p in enumerate(tanishq):
        v = visual_risk_from_photo(p, "gold jewellery")
        mat = "gold_18k" if RNG.random() < 0.25 else "gold_22k"
        rows.append(make_row(v, simulated_density_risk(mat),
                             sample_audio(genuine_taps), 0))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(tanishq)}")

    # ── genuine: no-photo rows (density + audio only) ──
    for _ in range(60):
        rows.append(make_row(ABSTAIN, simulated_density_risk("gold_22k"),
                             sample_audio(genuine_taps), 0))

    # ── fake: DS-1 copper photos ──
    for p in sorted((DS1 / "gold/bare_copper/images").glob("*.jpg")):
        v = visual_risk_from_photo(p, "gold item")
        rows.append(make_row(v, simulated_density_risk("copper"),
                             sample_audio(composite_taps), 1))

    # ── fake: NEU surface-defect proxies ──
    neu = sorted(DS2.rglob("*.jpg"))
    neu = [neu[i] for i in RNG.choice(len(neu), size=min(rows_cap, len(neu)),
                                      replace=False)]
    print(f"Scoring {len(neu)} NEU defect photos...")
    for i, p in enumerate(neu):
        v = visual_risk_from_photo(p, "gold item")
        mat = "underkarat" if RNG.random() < 0.3 else "copper"
        rows.append(make_row(v, simulated_density_risk(mat),
                             sample_audio(composite_taps), 1))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(neu)}")

    # ── fake: no-photo rows ──
    for _ in range(60):
        mat = "underkarat" if RNG.random() < 0.3 else "copper"
        rows.append(make_row(ABSTAIN, simulated_density_risk(mat),
                             sample_audio(composite_taps), 1))

    # ── tungsten-like composites (documented proxy — see module docstring) ──
    gold_photos = sorted((DS1 / "gold/bare_gold/images").glob("*.jpg"))
    for _ in range(40):
        p = gold_photos[RNG.integers(len(gold_photos))]
        v = visual_risk_from_photo(p, "gold bar")
        # audio always present: the contradiction IS the signal here
        a = composite_taps[RNG.integers(len(composite_taps))]
        rows.append(make_row(v, simulated_density_risk("tungsten"), a, 1))

    FEATURES = [
        "image_risk", "density_risk", "acoustic_risk", "streak_risk",
        "contra_density_acoustic", "contra_density_image",
        "contra_image_streak", "contra_acoustic_image",
        "contra_density_streak", "contra_acoustic_streak",
    ]
    df = pd.DataFrame(rows, columns=FEATURES + ["label"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    n0, n1 = (df.label == 0).sum(), (df.label == 1).sum()
    print(f"\nDataset: {len(df)} rows ({n0} genuine, {n1} fake) -> {OUT}")

    # ── sanity: no feature may be class-constant (that was the leak) ──
    for f in FEATURES:
        g, k = df[df.label == 0][f], df[df.label == 1][f]
        if g.std() < 1e-9 and k.std() < 1e-9 and abs(g.mean() - k.mean()) > 0.01:
            raise SystemExit(f"LEAKAGE: {f} is a class-separating constant")
    print("Leakage check passed: no class-constant features.")

    # ── train ──
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = df[FEATURES].values, df["label"].values
    model = xgb.XGBClassifier(
        max_depth=3, n_estimators=50, min_child_weight=2,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {auc.mean():.3f} ± {auc.std():.3f}  (honest — expect < 1.0)")

    model.fit(X, y)

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {name:24s} {imp:.3f}")

    dst = ROOT / "models/fusion_xgb.pkl"
    bak = ROOT / "models/fusion_xgb.pkl.bak-leaky"
    if dst.exists() and not bak.exists():
        shutil.copy2(dst, bak)
        print(f"\nOld model backed up -> {bak}")
    with open(dst, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {dst}  ({datetime.now().isoformat(timespec='seconds')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-cap", type=int, default=200,
                    help="max photos per large dataset (DS-2, DS-3)")
    main(ap.parse_args().rows_cap)
