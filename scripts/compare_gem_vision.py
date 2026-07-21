"""
Compare Fireworks vision models on real stone detection.

Runs app/llm/gem_vision.py's production prompt against a real case photo for
each candidate model and reports how many stones each returns, so the model
choice is made on measured behaviour rather than assumption.

Usage:  python scripts/compare_gem_vision.py data/cases/<id>/img_0.jpg
"""
import base64
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
# Explicit path — load_dotenv() with no args resolves relative to the CALLER's
# directory, which silently sends an empty key when run from elsewhere.
load_dotenv(ROOT / ".env", override=True)

import cv2  # noqa: E402
import requests  # noqa: E402

from app.llm.gem_vision import _PROMPT, _parse_gems  # noqa: E402

# The only vision models with supportsServerless=True in the Fireworks public
# catalog (see scripts/list_fw_models.py). Every qwen*-vl / llama-vision entry
# is supportsImageInput=True but supportsServerless=False — catalogued, yet only
# callable via a dedicated deployment, which is why they 404 on /v1/chat.
CANDIDATES = [
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/qwen3p7-plus",
]
RUNS = int(os.getenv("RUNS", "2"))   # repeat to expose run-to-run variance
URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/cases/8ce4f55e/img_0.jpg"
    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        print("No FIREWORKS_API_KEY found in .env")
        return

    bgr = cv2.imread(str(ROOT / img_path))
    if bgr is None:
        print("Could not read image:", img_path)
        return
    H, W = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    print(f"image {img_path}  ({W}x{H})\n")
    print(f"{'model':<44} {'http':>5} {'stones':>7} {'secs':>6}  note")
    print("-" * 92)

    for model in CANDIDATES:
      for run in range(RUNS):
        short = model.split("/")[-1] + f" #{run+1}"
        model_id = model
        t0 = time.time()
        try:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ]}],
                    "temperature": 0.0,
                    "max_tokens": 16384,
                },
                timeout=400,
            )
            dt = time.time() - t0
            if r.status_code != 200:
                msg = ""
                try:
                    msg = json.loads(r.text).get("error", {}).get("message", "")[:44]
                except Exception:
                    msg = r.text[:44]
                print(f"{short:<44} {r.status_code:>5} {'-':>7} {dt:>6.1f}  {msg}")
                continue
            txt = r.json()["choices"][0]["message"]["content"] or ""
            gems = _parse_gems(txt, W, H, 0, 0)
            note = "parsed OK" if gems else f"0 parsed (reply {len(txt)} chars)"
            print(f"{short:<44} {200:>5} {len(gems):>7} {dt:>6.1f}  {note}")
            if gems:
                Path(ROOT / f"scripts/_out_{short.replace(' ','_').replace('#','')}.json").write_text(
                    json.dumps(gems, indent=2))
        except Exception as e:
            print(f"{short:<44} {'ERR':>5} {'-':>7} {time.time()-t0:>6.1f}  {repr(e)[:44]}")


if __name__ == "__main__":
    main()
