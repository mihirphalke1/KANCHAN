# KANCHAN-AI

**Karat Authentication via Neural Computation, Heuristics & Acoustic Novelties**

Branch-deployable, non-destructive spurious gold detection for Indian gold loan appraisal.
Built for **SuRaksha Cyber Hackathon 2.0** — Canara Bank / IISc Bangalore.

---

## Architecture

```
FastAPI backend  ──┬── /api/analyze    (POST: multipart form)
                   ├── /api/benford    (GET: population stats)
                   └── /api/history    (GET: case log)

Four modalities  ──┬── Density       (Archimedes: dry vs. submerged weight)
                   ├── Acoustic      (MFCC-ΔΔ + SVM, Novelty 1)
                   ├── Visual        (EfficientNet-B3 + LogReg probe)
                   └── Streak        (HSV LogReg on touchstone image)

Fusion           ──── XGBoost on 10 features (4 modality + 6 cross-pairs)
                       + SHAP explainability

Novel signals    ──┬── Benford's Law on submerged weights (Novelty 2)
                   └── Cross-modal contradiction detection (Novelty 3)

LLM verdict      ──── Groq (Llama-3 70B) → Gemini 1.5 Flash → heuristic fallback
```

---

## Quickstart

```bash
# Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`.

---

## Datasets (DS-1 through DS-7)

### DS-1 — Counterfeit Gold Audio
- **Source**: [Kaggle — mohammedlotfy50/counterfeit-gold](https://www.kaggle.com/datasets/mohammedlotfy50/counterfeit-gold)
- **Content**: 20 tap-test recordings — genuine gold vs. copper-plated fakes (m4a, converted to wav)
- **Use**: Train acoustic SVM on 82-dim MFCC-ΔΔ features; 6× augmentation → 120 training samples
- **Local path**: `data/raw/counterfeit_gold/`

### DS-2 — NEU Surface Defect Database
- **Source**: [Kaggle — kaustubhb999/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhb999/neu-surface-defect-database)
- **Content**: 1 800 labelled steel surface images (6 defect categories, 300 per class)
- **Use**: Fake/defect proxy class for EfficientNet-B3 visual probe
- **Local path**: `data/raw/neu_defect/NEU-DET/`

### DS-3 — Tanishq Jewellery Images
- **Source**: [Kaggle — ravirajsinh45/product-images-of-jewelry](https://www.kaggle.com/datasets/ravirajsinh45/product-images-of-jewelry)
- **Content**: 490 genuine product photos of gold jewellery
- **Use**: Genuine class images for EfficientNet-B3 visual probe
- **Local path**: `data/raw/tanishq/`

### DS-4 — Roboflow Jewelry Detection
- **Source**: [Roboflow Universe — valuable-object-detection/jewelry-dkgqg](https://universe.roboflow.com/valuable-object-detection/jewelry-dkgqg)
- **Content**: ~324 mixed jewellery images (YOLOv5 format, train/valid/test splits)
- **Use**: Additional genuine-class images to strengthen visual probe
- **Setup**: Add `ROBOFLOW_API_KEY` to `.env`, then run `python scripts/download_ds4.py`
- **Local path**: `data/raw/roboflow_jewelry/`

### DS-5 — ESC-50 Environmental Sounds
- **Source**: [GitHub — karoldvl/ESC-50](https://github.com/karoldvl/ESC-50)
- **Content**: 2 000 × 5-second environmental audio clips across 50 categories
- **Use**: Acoustic pipeline validation and negative-class sample augmentation
- **Local path**: `data/raw/esc50/`

### DS-6 — Banknote Authentication (UCI)
- **Source**: [UCI ML Repository #267](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
- **Content**: 1 372 rows, 4 wavelet-transform features + genuine/fake label
- **Use**: Benford's Law monitor baseline and XGBoost fusion cross-validation
- **Local path**: `data/raw/banknote/data_banknote_authentication.txt`

### DS-7 — Self-collected Streak Images
- **Status**: Not yet collected — streak module uses HSV heuristic as fallback
- **Target**: 80+ touchstone streak photos of genuine and fake gold items
- **When available**: Place in `data/raw/streak_genuine/` and `data/raw/streak_fake/`, then re-run `python scripts/build_and_train.py`
- **Note**: The `streak_logreg.pkl` model was trained on DS-1 item images (HSV proxy); real streak photos would substantially improve this modality's accuracy

---

## Roboflow Setup (DS-4)

```bash
# 1. Get your API key
#    https://app.roboflow.com → workspace → Settings → Roboflow API → copy key

# 2. Add to .env
echo "ROBOFLOW_API_KEY=rf_xxxxxxxxxxxx" >> .env

# 3. Download dataset
python scripts/download_ds4.py

# 4. Retrain (DS-4 images are auto-included in next run)
python scripts/build_and_train.py
```

---

## Trained Models

| File | Size | Algorithm | CV Score |
|---|---|---|---|
| `models/acoustic_svm.pkl` | 62 KB | RBF-SVM on 82-dim MFCC-ΔΔ | AUC 0.999 |
| `models/image_probe.pkl` | 49 KB | LogReg on EfficientNet-B3 1536-dim embeddings | — |
| `models/fusion_xgb.pkl` | 40 KB | XGBoost, 10 features (4 modality + 6 cross-pairs) | AUC 1.000 |
| `models/streak_logreg.pkl` | 1 KB | LogReg on 13 HSV features | Acc 0.812 |

---

## Three Novel Contributions

| # | Name | Description |
|---|---|---|
| 1 | MFCC-ΔΔ acoustic fingerprinting | Delta-delta MFCC captures the unique ring decay of genuine gold; fake items (plated base metals, tungsten core) produce a significantly different temporal envelope |
| 2 | Benford's Law density monitor | At branch level, first digits of submerged weight measurements follow Benford's distribution for genuine items; systematic deviations flag organised fraud patterns |
| 3 | Cross-modal contradiction detection | Six correlated modality pairs are scored; when density says genuine but acoustic says fake, the contradiction score is boosted 0.40× into the final verdict — critical for tungsten-core fraud |

---

## Integration Tests

```bash
python scripts/integration_test.py
# Expected: 27/27 passed
```

---

## Environment Variables

Copy `.env.example` → `.env`:

```env
GROQ_API_KEY=        # primary LLM — Groq Llama-3 70B (free tier available)
GOOGLE_API_KEY=      # fallback LLM — Gemini 1.5 Flash
ROBOFLOW_API_KEY=    # DS-4 jewelry images
KAGGLE_USERNAME=     # DS-2, DS-3 download
KAGGLE_API_KEY=      # DS-2, DS-3 download
PORT=8000
```

LLM keys are optional — the system falls back to a rule-based heuristic verdict if none are configured.

---

## Project Structure

```
KANCHAN/
├── app/
│   ├── main.py
│   ├── routers/        analyze.py  benford.py  history.py
│   ├── models/         density  acoustic  image  streak  fusion  contradiction
│   ├── benford/        monitor.py
│   └── llm/            verdict_prompt.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/ Header  AnalysisForm  ResultsPanel  VerdictCard  …
├── scripts/
│   ├── build_and_train.py    full pipeline: download → embed → train
│   ├── download_ds4.py       Roboflow DS-4 downloader
│   └── integration_test.py  27-check test suite
├── models/             trained .pkl files
├── data/
│   ├── raw/            downloaded datasets (git-ignored)
│   └── processed/      embeddings cache (git-ignored)
└── .env.example
```
