#!/usr/bin/env python3
"""
Full integration test for KANCHAN-AI production build.
Tests the API with real DS-1 files and verifies model behaviour.

Run after build_and_train.py completes:
    uvicorn app.main:app --port 8000 &
    python scripts/integration_test.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DS1  = ROOT / "data" / "raw" / "counterfeit_gold" / "gold"
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

RESULTS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    RESULTS.append((status, name, detail))
    symbol = "✓" if condition else "✗"
    print(f"  {symbol} [{status}] {name}" + (f" — {detail}" if detail else ""))


def test_models_exist() -> None:
    print("\n── Model files ──")
    for name in ("acoustic_svm.pkl", "image_probe.pkl", "fusion_xgb.pkl"):
        path = ROOT / "models" / name
        check(name, path.exists(), f"{path.stat().st_size // 1024}KB" if path.exists() else "MISSING")


def test_demo_fixtures() -> None:
    print("\n── Demo fixtures ──")
    demo = ROOT / "data" / "demo"
    for name in ("genuine_ring.jpg", "genuine_ring.wav", "fake_bangle.jpg", "fake_bangle.wav", "streak_genuine.jpg"):
        p = demo / name
        check(name, p.exists(), f"{p.stat().st_size // 1024}KB" if p.exists() else "MISSING")


def test_acoustic_model() -> None:
    print("\n── Acoustic model inference ──")
    sys.path.insert(0, str(ROOT))
    from app.models.acoustic_model import analyze_acoustic

    genuine_wav = sorted((DS1 / "plain_sound" / "original").glob("*.wav"))
    fake_wav    = sorted((DS1 / "plain_sound" / "copper").glob("*.wav"))

    if not genuine_wav or not fake_wav:
        check("DS-1 WAV files present", False, "No WAV files found")
        return

    g_score = analyze_acoustic(genuine_wav[0].read_bytes())
    f_score = analyze_acoustic(fake_wav[0].read_bytes())

    check("Genuine audio loaded",       g_score["mode"] in ("svm", "heuristic:heuristic"), g_score["mode"])
    check("Fake audio loaded",          f_score["mode"] in ("svm", "heuristic:heuristic"), f_score["mode"])
    check("Model mode is SVM",          g_score["mode"] == "svm", g_score["mode"])
    check("Genuine < fake risk score",  g_score["risk_score"] < f_score["risk_score"],
          f"genuine={g_score['risk_score']:.3f}, fake={f_score['risk_score']:.3f}")


def test_image_model() -> None:
    print("\n── Image model inference ──")
    sys.path.insert(0, str(ROOT))
    from app.models.image_model import analyze_image

    genuine_img = list((DS1 / "bare_gold" / "images").glob("*.jpg"))
    fake_img    = list((DS1 / "bare_copper" / "images").glob("*.jpg"))

    if not genuine_img or not fake_img:
        check("DS-1 images present", False, "No images found")
        return

    g = analyze_image([genuine_img[0].read_bytes()])
    f = analyze_image([fake_img[0].read_bytes()])

    check("Genuine image loaded",       g["mode"] in ("efficientnet", "heuristic"), g["mode"])
    check("Model mode is efficientnet", g["mode"] == "efficientnet", g["mode"])
    check("Scores in [0,1]",            0 <= g["risk_score"] <= 1 and 0 <= f["risk_score"] <= 1,
          f"genuine={g['risk_score']:.3f}, fake={f['risk_score']:.3f}")


def test_density_verdicts() -> None:
    print("\n── Density verdicts ──")
    sys.path.insert(0, str(ROOT))
    from app.models.density_model import analyze_density

    # Pure 22K: 17.88 g/cm³
    r22 = analyze_density(15.2, 14.35, 22)
    check("22K genuine low risk", r22["risk_score"] < 0.15,
          f"risk={r22['risk_score']:.3f}, density={r22['measured_density']:.2f}")

    # Copper fake declared as 22K: ~8.93 g/cm³
    r_cu = analyze_density(15.0, 13.31, 22)
    check("Copper fake high risk", r_cu["risk_score"] > 0.80,
          f"risk={r_cu['risk_score']:.3f}, density={r_cu['measured_density']:.2f}")

    # Tungsten declared as 24K: ~19.25 g/cm³ (known blind spot).
    # 50 g bar: the realistic tungsten-fake form factor, and heavy enough that
    # the balance can resolve the narrow 24K band (sigma ~0.05 g/cm³).
    r_tg = analyze_density(50.0, 47.41, 24)
    check("Tungsten passes density (documented blind spot)", r_tg["risk_score"] < 0.20,
          f"risk={r_tg['risk_score']:.3f}, density={r_tg['measured_density']:.2f}")
    check("Tungsten blind-spot flag fires", r_tg["tungsten_warning"] is True,
          f"verdict={r_tg['karat_verdict']}")


def test_api_analyze() -> None:
    print("\n── API /api/analyze (genuine 22K) ──")
    import requests

    genuine_img = list((DS1 / "bare_gold" / "images").glob("*.jpg"))
    genuine_wav = list((DS1 / "plain_sound" / "original").glob("*.wav"))

    if not genuine_img or not genuine_wav:
        check("DS-1 files available", False, "Missing files"); return

    try:
        files = {
            "images":    ("img.jpg", genuine_img[0].read_bytes(), "image/jpeg"),
            "audio":     ("audio.wav", genuine_wav[0].read_bytes(), "audio/wav"),
        }
        data = {
            "item_description": "Integration test — genuine 22K ring",
            "declared_karat":   "22",
            "weight_dry":       "15.2",
            "weight_submerged": "14.35",
            "branch_id":        "test",
        }
        resp = requests.post(API_BASE + "/api/analyze", files=files, data=data, timeout=30)
        check("API returns 200",             resp.status_code == 200, str(resp.status_code))
        if resp.status_code == 200:
            body = resp.json()
            verdict = body.get("verdict", {})
            check("Verdict present",         "risk_level" in verdict)
            check("Genuine verdict",         verdict.get("risk_level") == "GENUINE",
                  verdict.get("risk_level", "?"))
            check("Loan action APPROVE",     verdict.get("loan_action") == "APPROVE",
                  verdict.get("loan_action", "?"))
            check("Acoustic mode is SVM",    body["modality_scores"]["acoustic"]["mode"] == "svm",
                  body["modality_scores"]["acoustic"]["mode"])
            check("Photo material scan ran", body["modality_scores"]["xray"]["mode"] == "dsip_xray",
                  body["modality_scores"]["xray"]["mode"])
            check("Ring-pitch check ran",    "ring" in body["modality_scores"]["acoustic"],
                  body["modality_scores"]["acoustic"].get("ring", {}).get("status", "missing"))
    except requests.ConnectionError:
        check("API reachable", False, "Connection refused — is uvicorn running on port 8000?")


def test_api_tungsten() -> None:
    print("\n── API /api/analyze (tungsten-core 24K — STAR scenario) ──")
    import requests

    fake_img = list((DS1 / "bare_gold" / "images").glob("*.jpg"))   # visually looks like gold
    fake_wav = list((DS1 / "plain_sound" / "copper").glob("*.wav"))  # dampened acoustic = copper

    if not fake_img or not fake_wav:
        check("DS-1 files available", False, "Missing files"); return

    try:
        files = {
            "images": ("img.jpg", fake_img[0].read_bytes(), "image/jpeg"),
            "audio":  ("audio.wav", fake_wav[0].read_bytes(), "audio/wav"),
        }
        data = {
            "item_description": "Integration test — tungsten-core 24K bar",
            "declared_karat":   "24",
            "weight_dry":       "50.0",
            "weight_submerged": "47.41",  # buoyancy-corrected density ≈ 19.25 (passes 24K band)
            "branch_id":        "test",
        }
        resp = requests.post(API_BASE + "/api/analyze", files=files, data=data, timeout=30)
        check("Tungsten API returns 200", resp.status_code == 200, str(resp.status_code))
        if resp.status_code == 200:
            body = resp.json()
            verdict = body.get("verdict", {})
            risk = verdict.get("risk_level", "?")
            check("Tungsten NOT genuine",   risk != "GENUINE", risk)
            check("Contradiction detected", body["contradiction"]["contradiction_score"] > 0.30,
                  f"score={body['contradiction']['contradiction_score']:.3f}")
    except requests.ConnectionError:
        check("API reachable (tungsten)", False, "Connection refused")


def main() -> None:
    print("=" * 60)
    print("  KANCHAN-AI Integration Test")
    print("=" * 60)

    test_models_exist()
    test_demo_fixtures()
    test_acoustic_model()
    test_image_model()
    test_density_verdicts()
    test_api_analyze()
    test_api_tungsten()

    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    total  = len(RESULTS)

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed:
        print("\nFailed checks:")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
