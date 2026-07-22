"""Enumerate the real Fireworks public model catalog and flag vision models.

Uses the documented control-plane endpoint /v1/accounts/{account}/models
(public models live under the `fireworks` account) with pagination, rather
than guessing model IDs.
"""
import json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
import requests

KEY = os.getenv("FIREWORKS_API_KEY")
BASE = "https://api.fireworks.ai/v1/accounts/fireworks/models"


def fetch_all():
    out, token = [], None
    while True:
        params = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        r = requests.get(BASE, headers={"Authorization": f"Bearer {KEY}"},
                         params=params, timeout=60)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:300])
            return out
        j = r.json()
        out.extend(j.get("models", []))
        token = j.get("nextPageToken") or None
        if not token:
            return out


def main():
    models = fetch_all()
    print(f"total public models: {len(models)}\n")
    if models:
        print("sample record keys:", sorted(models[0].keys()))

    vision = []
    for m in models:
        blob = json.dumps(m).lower()
        name = m.get("name", "")
        # capability flags vary by record; check both flags and known families
        if ("image" in blob and "input" in blob) or m.get("supportsImageInput"):
            vision.append((name, "flag"))
        elif any(t in name.lower() for t in ("-vl", "vision", "llava", "pixtral", "kimi")):
            vision.append((name, "name"))

    print(f"\nVISION-CAPABLE CANDIDATES ({len(vision)}):")
    for name, why in sorted(vision):
        print(f"  [{why}] {name}")

    Path(ROOT / "scripts/_fw_models.json").write_text(json.dumps(models, indent=2))
    print("\nfull catalog -> scripts/_fw_models.json")


if __name__ == "__main__":
    main()
