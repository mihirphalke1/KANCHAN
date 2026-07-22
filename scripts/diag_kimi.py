"""Diagnose why Kimi's reply doesn't yield parseable stone JSON."""
import base64, json, os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
import cv2, requests
from app.llm.gem_vision import _PROMPT, _parse_gems

KEY = os.getenv("FIREWORKS_API_KEY")
MODEL = "accounts/fireworks/models/kimi-k2p6"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"

img = cv2.imread(str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/cases/8ce4f55e/img_0.jpg")))
H, W = img.shape[:2]
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

VARIANTS = [
    ("prod: 4096 + json_object", {"max_tokens": 4096, "response_format": {"type": "json_object"}}),
    ("16384 + json_object",      {"max_tokens": 16384, "response_format": {"type": "json_object"}}),
    ("16384 plain",              {"max_tokens": 16384}),
]

for name, extra in VARIANTS:
    t0 = time.time()
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": _PROMPT},
        {"type": "image_url", "image_url": {"url": uri}}]}],
        "temperature": 0.0}
    body.update(extra)
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"\n### {name}: HTTP {r.status_code} {r.text[:150]}")
            continue
        j = r.json()
        txt = j["choices"][0]["message"]["content"] or ""
        finish = j["choices"][0].get("finish_reason")
        usage = j.get("usage", {})
        gems = _parse_gems(txt, W, H, 0, 0)
        print(f"\n### {name}")
        print(f"    finish_reason={finish} completion_tokens={usage.get('completion_tokens')} "
              f"chars={len(txt)} secs={dt:.1f} parsed_stones={len(gems)}")
        print(f"    TAIL: ...{txt[-320:]!r}")
    except Exception as e:
        print(f"\n### {name}: ERR {e!r}")
