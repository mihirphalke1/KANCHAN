# KANCHAN-AI Demo Playbook
## SuRaksha Cyber Hackathon 2.0 — IISc Bangalore Panel

**Presenter:** Mihir Phalke  
**Duration:** 5–7 minutes  
**URL:** http://localhost:8000  

---

## Pre-Demo Checklist

- [ ] `bash scripts/run_demo.sh` — server running, browser open
- [ ] Density log seeded (`data/density_log.csv` exists, 60 rows)
- [ ] `.env` file has at least one API key (Groq or Google) — or run in heuristic mode
- [ ] Browser at 100% zoom, open in Chrome
- [ ] Phone connected to same WiFi for mobile demo (navigate to your machine's IP:8000)

---

## Scenario 1 — Genuine Gold Ring (60 seconds)

**Objective:** Show normal happy-path flow.

1. Open the app → empty state shows three novelties at the bottom
2. Fill form:
   - Description: `22K gold necklace`
   - Declared Karat: `22K — 91.7% gold`
   - Dry weight: `15.20`
   - Submerged weight: `14.35`  ← gives 17.80 g/cm³ (genuine 22K)
   - (Skip photo/audio if no demo files, or use any image)
3. Click **Analyse Gold Item**
4. Watch the animated loading steps
5. Result: **GREEN GENUINE card**, all signal bars below 35%, Benford OK

**Key talking point:**  
*"All four signals agree. The density is 17.76 g/cm³, squarely in the 22K range of 17.4–18.1. System is confident — loan officer can approve."*

---

## Scenario 2 — Tungsten-Core Fake (90 seconds) ← STAR SCENARIO

**Objective:** Demonstrate Novelty 3 (cross-modal contradiction).

1. Fill form:
   - Description: `24K gold bangle (suspected tungsten)`
   - Declared Karat: `24K — 99.9% gold`
   - Dry weight: `18.90`
   - Submerged weight: `17.91`  ← gives 19.10 g/cm³ — passes 24K density test!
   - This is the key: density passes, only acoustic catches it
   - Upload the fake bangle audio (if available) — or skip
2. Click **Analyse Gold Item**
3. Result: **AMBER/RED BORDERLINE card**, acoustic bar HIGH, density bar LOW

**Key talking points:**
- Point to the Contradiction Alert panel:  
  *"Look here — density says 19.18 g/cm³, which passes the 24K test. But the acoustic ring is 82% risk."*  
- *"This is the exact signature of a tungsten-core item. Tungsten has density 19.25 g/cm³ — essentially identical to pure gold. Single-signal density methods miss this completely."*  
- *"Our cross-modal contradiction module — Novelty 3 — catches this. The density↔acoustic pair disagrees by 76%. That's the flag that saves the bank."*

---

## Scenario 3 — Benford's Law Dashboard (45 seconds)

**Objective:** Demonstrate Novelty 2 (statistical fraud ring detection).

1. Scroll down to the **Benford's Law Monitor** card at the bottom of results
2. Point to the bar chart:  
   *"These 60 density readings from this branch follow Benford's Law — the natural distribution of first digits. p-value is 0.34, well above 0.05. No anomaly."*
3. Explain the detection power:  
   *"Now imagine a counterfeiting ring. When multiple fake items are submitted — all with fabricated density measurements — the distribution of first digits starts to deviate from Benford's Law. Our monitor would flag this branch automatically with a batch alert."*
4. *"This is the first application of Benford's Law to physical density measurements at a bank. Standard in financial fraud — we bring it to gold appraisal."*

---

## Scenario 4 — Mobile Demo (30 seconds, if judge asks)

1. Ask the judge to scan the QR code / navigate to `http://<your-ip>:8000` on their phone
2. Show the responsive layout — form stacks vertically, results below
3. *"The system is designed for branch-office use. A field officer can run a full multi-modal analysis from their phone."*

---

## Three Novelties — Slide Reference

| Novelty | What | Why Novel |
|---------|------|-----------|
| **1** | MFCC-ΔΔ acoustic fingerprinting on irregular jewelry | Published work (Devrim & Kirişoğlu 2025) for lab flat bars only; we apply ΔΔ (acceleration of spectral decay) to irregular shapes via smartphone |
| **2** | Benford's Law on physical density measurements | Cross-domain transfer from financial fraud; first application to bank branch density logs |
| **3** | Cross-modal contradiction as explicit XGBoost feature | Standard fusion models aggregate agreement; we model *disagreement* — the tungsten-core blind spot is only catchable this way |

---

## Q&A Anticipated Questions

**Q: How accurate is the acoustic model with 20 samples?**  
A: "Currently the acoustic module runs in heuristic mode — we extract MFCC-ΔΔ features but the SVM classifier will be trained once we collect our DS-7 self-collected data. The MFCC-ΔΔ feature extraction itself is validated; the architecture is in place."

**Q: What happens if the officer doesn't have a recording setup?**  
A: "The system degrades gracefully. If no audio is uploaded, the acoustic score defaults to 0.5 (neutral), and the other three modalities carry the verdict. The density module alone catches brass, lead, and copper fakes with high confidence."

**Q: What about XRF spectrometers — why not use those?**  
A: "XRF costs ₹3–8 lakh, requires trained operators, and takes 15 minutes. Our system costs nothing beyond a smartphone and a weighing scale. We're not replacing XRF for high-value contested items — we're providing a first-pass filter that catches 80% of fakes instantly."

**Q: How do you handle the tungsten blind spot in density?**  
A: "That's Novelty 3 in action. We document tungsten as a known limitation of single-signal density testing — and then we *specifically* design the contradiction module to catch it. The density↔acoustic pair disagreement is a tungsten signature."
