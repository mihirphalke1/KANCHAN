"""Probe which Fireworks vision models this account can actually call.

Sends a trivial 2-circle image to each candidate; a model that answers is
reachable and image-capable. Run before choosing a gem-vision model.
"""
import base64, json, os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)
import cv2, numpy as np, requests

KEY = os.getenv("FIREWORKS_API_KEY")
URL = "https://api.fireworks.ai/inference/v1/chat/completions"

CANDIDATES = [
    # Kimi
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2-instruct",
    # Qwen VL
    "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct",
    "accounts/fireworks/models/qwen3-vl-32b-instruct",
    "accounts/fireworks/models/qwen3-vl-8b-instruct",
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
    "accounts/fireworks/models/qwen2p5-vl-72b-instruct",
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
    "accounts/fireworks/models/qwen2p5-vl-7b-instruct",
    "accounts/fireworks/models/qwen2-vl-72b-instruct",
    # Llama vision
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
    "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "accounts/fireworks/models/llama4-scout-instruct-basic",
    # Others
    "accounts/fireworks/models/phi-3-vision-128k-instruct",
    "accounts/fireworks/models/firellava-13b",
    "accounts/fireworks/models/pixtral-12b",
    "accounts/fireworks/models/internvl3-78b",
    "accounts/fireworks/models/deepseek-vl2",
    "accounts/fireworks/models/glm-4v-9b",
]

img = np.full((200, 300, 3), 220, np.uint8)
cv2.circle(img, (90, 100), 35, (40, 40, 200), -1)
cv2.circle(img, (200, 100), 35, (40, 170, 40), -1)
ok, buf = cv2.imencode(".jpg", img)
uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

print(f"{'model':<52} {'http':>5} {'secs':>6}  answer/error")
print("-" * 100)
reachable = []
for m in CANDIDATES:
    short = m.split("/")[-1]
    t0 = time.time()
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"},
                          json={"model": m, "messages": [{"role": "user", "content": [
                              {"type": "text", "text": "How many circles? Answer with just the number."},
                              {"type": "image_url", "image_url": {"url": uri}}]}],
                              "temperature": 0, "max_tokens": 2048},
                          timeout=120)
        dt = time.time() - t0
        if r.status_code == 200:
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
            print(f"{short:<52} {200:>5} {dt:>6.1f}  {txt[:40]!r}")
            reachable.append(m)
        else:
            msg = json.loads(r.text).get("error", {}).get("message", "")[:40]
            print(f"{short:<52} {r.status_code:>5} {dt:>6.1f}  {msg}")
    except Exception as e:
        print(f"{short:<52} {'ERR':>5} {time.time()-t0:>6.1f}  {repr(e)[:40]}")

print("\nREACHABLE VISION MODELS:")
for m in reachable:
    print("  ", m)
