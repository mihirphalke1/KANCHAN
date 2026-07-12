# KANCHAN-AI Demo Playbook
## SuRaksha Cyber Hackathon 2.0 — IISc Bangalore Panel

**Duration:** 5–7 minutes
**Dev URL:** http://localhost:5173 (backend on 8001) · **Demo build:** `bash scripts/run_demo.sh` → http://localhost:8000

---

## Pre-Demo Checklist

- [ ] Backend: `python3 -m uvicorn app.main:app --reload --port 8001` (from repo root)
- [ ] Frontend: `cd frontend && npm run dev` → open http://localhost:5173
- [ ] Test kit present: `data/demo/test-kit/` (audio + photos)
- [ ] `data/density_log.csv` present (Benford-conformant seeds + real cases)
- [ ] Optional but stronger: recalibrate on your own items —
      5 taps on a known-genuine piece → `python3 scripts/calibrate_acoustic.py <folder>`
- [ ] Groq/Google key in `.env` for LLM-written explanations (heuristic fallback is fine too)
- [ ] Chrome, 100% zoom

**The one-line pitch:** *Physics, not black-box AI — three mandatory tests, every verdict
recomputable by hand, and no single test can ever approve an item.*

---

## Scenario 1 — Genuine item, full battery (60s)

- Photos: `test-kit/photos/gold_ring.jpg`
- Dry `20.00` · Submerged `18.88` · Water temp `25` · Karat `22K`
- Audio: `test-kit/audio/genuine_gold_tap_1.wav`

**Expect:** GENUINE / APPROVE, overall risk ~8%.

**Show:** expand "How This Result Was Reached" — ten plain-language steps.
Open "What Influenced This Decision": the numbers literally add up
(−1.10 weight, −0.93 sound, −0.42 photo → sum → 8%).

**Say:** *"Density 17.80 ± 0.11 — the ± is real uncertainty propagated from the balance.
Ring pitch inside the band we calibrated from genuine recordings. An officer can verify
every line by hand."*

---

## Scenario 2 — STAR: filled core the weight test cannot see (90s)

- Photos: `test-kit/photos/gold_bar.jpg`
- Dry `50.00` · Submerged `47.41` · Karat `24K`
- Audio: `test-kit/audio/fake_composite_tap_1.wav`

**Expect:** weight test PASSES (19.25 g/cm³, inside the 24K band — the tungsten blind
spot), then **REJECT** with the override reason on the verdict card:
*"Ring pitch 7681 Hz is 1.10× the calibrated genuine band top… a stiffer-than-gold core."*

**Say:** *"Tungsten matches gold's density to 0.36% — smaller than any branch scale's
error, and we prove that with the ± on screen. But no filler matches gold's stiffness:
v = √(E/ρ), tungsten is five times stiffer, so it rings higher. Validated 10 out of 10
on real composite recordings. The claim is irrelevant — declare any karat, the physics
identifies the density as gold-like and the pitch betrays the core."*

---

## Scenario 3 — Purity mis-declaration (45s)

- Photos: `gold_ring.jpg` · Audio: `genuine_gold_tap_1.wav`
- Dry `15.00` · Submerged `14.04` · Karat declared `22K` (metal is actually 18K)

**Expect:** REJECT at the declared karat, but the card says:
*"What the physics says it is: 18K gold (99.8% match)"* and the action is
*"re-run declared as 18K and revalue."*

**Say:** *"The declaration is a hypothesis, never trusted. Over-claiming purity on
genuine gold is the most common gold-loan fraud — we separate it from counterfeit
metal, because the bank's action differs: revalue versus refuse."*

Bonus: switch Karat to **"Not declared — identify from physics"** and re-run —
the system approves it *as 18K* with no claim at all.

---

## Scenario 4 — Stone-set jewellery valued correctly (60s)

- Photos: `test-kit/photos/gold_necklace.jpg`
- Dry `19.70` · Submerged `18.51` · Karat `22K`
- Audio: `genuine_gold_tap_1.wav`

**Expect:** raw density looks WRONG (16.5 — below the 22K band), but the Composition
card reconciles it: camera finds ~9% stones, physics implies ~9%, z = 0 →
GENUINE / APPROVE with **"Gold ≈ 19.36 g of 19.70 g"**.

**Show:** the Gem detection stage (stones outlined & numbered, pearls marked "?"),
then the Histogram stage (cut points sitting in the valleys).

**Say:** *"Nobody pledges a pure gold block. The mixture model ρ = (1−f)·ρ_gold + f·ρ_stone
inverts the density into grams of actual gold — the number the loan should be priced on.
And if physics demands more non-gold volume than the camera can see, that's a hidden
core and it flags."*

---

## Scenario 5 — The mandate (15s, do it live)

Remove the audio file from the form. The Analyse button greys out:
*"still missing: tap recording."*

**Say:** *"Known scams passed because one factor was trusted alone. Here one failing
test can reject, but no approval exists without all three."*

---

## Q&A Ammunition

- **Sources?** CRC Handbook (elements, water), IS 1417:2016 (BIS fineness), Webster/GIA
  (gem SG), JCGM 106 (scoring under uncertainty). Karat bands are DERIVED:
  `python3 -m app.utils.references` prints the mixture-rule derivation vs the table.
- **"X-ray"?** *"Not literal radiography — a structural-heterogeneity proxy inspired by
  how X-ray reveals internal structure, built from surface visual cues."*
- **ML anywhere?** Only two places, both non-deciding by default: the acoustic SVM
  (secondary to the ring physics) and the LLM explainer (written after the verdict,
  cannot alter it). Fusion is a hand-recomputable log-odds sum. We found and fixed a
  label-leakage bug in our own earlier XGBoost training — documented in the README.
- **Privacy?** No uploaded media is ever stored; the audit record is numeric + the
  full step-by-step trace.
- **Why not XRF?** ₹3–8 lakh, trained operators, surface-only (a plated shell passes
  XRF). We are the zero-hardware first-pass filter; XRF remains the escalation path.
- **Honest limitations:** colourless-stone detection is shape-based and imperfect;
  photo stone-fraction is a lower bound; ring check abstains on uncalibrated item
  classes; decision thresholds are declared policy dials pending pilot data.
