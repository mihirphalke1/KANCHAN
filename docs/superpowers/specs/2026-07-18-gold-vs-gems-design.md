# Gold vs Gems — Accurate Colour Separation + Weight Breakdown

**Date:** 2026-07-18
**Status:** Approved design — ready for implementation plan
**Scope:** One feature, one implementation plan.

## 1. Goal

Make the ornament's **gold-vs-gems separation** genuinely accurate, prominent,
and weight-aware. The system already separates gold from gems by colour
(`_gold_gem_map` in `app/utils/xray.py`), but three problems were observed by
running the current pipeline over the demo + saved-case image panel:

1. **Gold bleeds into "other".** `images.jpeg` reads 56% gold / 40% "other";
   the 40% is overwhelmingly gold misclassified under shadow/glare. A single
   median gold colour + one fixed Lab ΔE cutoff cannot hold gold together
   across lighting.
2. **No stone ever reaches "confirmed".** Across every demo/case image,
   `stones_confirmed == 0` — every detected stone is an "uncertain ?". The
   `CONFIDENT_THRESHOLD = 0.72` is effectively never met.
3. **Background separation fails on some photos** (busy backgrounds), which is
   pre-existing and out of scope here except to respect it (feature stays
   inert when `background_removed` is false, as today).

## 2. Non-negotiable constraints

- **Strictly additive. Nothing is deleted.** `_gold_gem_map` keeps its name,
  signature, and return shape. Existing `gold_gem_split` keys
  (`gold_pct`, `gem_pct`, `other_pct`, `method`, `stones_used`) are preserved;
  new keys are added alongside.
- **Fusion, risk scoring, and the loan decision are untouched.** The colour
  split stays visual-only. The carat estimate is advisory/informational and
  does **not** change LTV or risk.
- **Classical and auditable.** Every new constant is documented and
  env-overridable, matching the existing style (`STONE_DELTA_E_MIN`,
  `BACKDROP_DIST_THRESHOLD`, `GOLD_LAB_MATCH_MAX`, ...).
- **Graceful fallback everywhere.** Any new step that fails reverts to today's
  behaviour; the feature can never make output worse than the current version.
- **Existing MobileSAM stone-boundary refinement is kept** (Approach C is the
  hybrid: rebuild the pixel colour classifier, keep SAM for stone boundaries).

## 3. Components

### Component 1 — Robust gold chromaticity manifold classifier

**File:** `app/utils/xray.py`, inside `_gold_gem_map` (and a new private helper,
e.g. `_gold_membership(...)`).

Replace the pixel-level gold test — currently `delta_e <= GOLD_LAB_MATCH_MAX`
(single median + fixed cutoff) OR a fixed HSV gold band — with a manifold model:

- Convert to CIE Lab. Reference gold pixels = non-stone item pixels (as today).
- Fit the gold signature as a **robust ellipse in Lab a*/b* chromaticity only**
  (median centre + robust covariance; Mahalanobis distance as membership).
  **L\* (lightness) is deliberately excluded** so a gold pixel classifies as
  gold whether in deep shadow or a blown-out specular highlight. This is the
  direct fix for the "gold → other" inflation.
- **Specular-highlight rescue:** bright + low-chroma pixels whose hue still
  points toward gold → classified **gold** (specular metal), not gem. A bright
  pixel whose chroma points *away* from gold (a lit ruby/sapphire) stays gem —
  fixes "glare → gem".
- **Adaptive metal hue:** derive the accept band from the item's own metal hue
  (handles yellow / rose / white gold / rhodium) instead of the fixed
  HSV 8–38 band. White-gold / low-saturation metals lean on the tight
  near-neutral chromaticity ellipse.
- `other` = item pixels that are neither gold-manifold nor a detected gem
  (solder, rhodium wear, genuinely unexplained). It shrinks because the gold
  manifold now captures shadow/highlight gold.

**New env-overridable constants** (names indicative):
`GOLD_MANIFOLD_MAHALANOBIS_MAX`, `GOLD_SPECULAR_L_MIN`, `GOLD_SPECULAR_CHROMA_MAX`,
plus retention of `GOLD_LAB_MATCH_MAX` and the HSV band constants as fallbacks.

**Fallback:** too few metal pixels to fit a stable ellipse → revert to today's
single-threshold + HSV-band logic.

**Success criteria:** on the demo panel, "other" drops sharply on gold-dominant
items; no gold pixel is reclassified as *gem* (no gold→gem regression).

### Component 2 — Stone-confidence recalibration

**File:** `app/utils/xray.py` (`_region_confidence` weights / thresholds).

1. Add a diagnostic (temporary script under scratchpad) dumping the five
   per-signal sub-scores (`edge_score`, `local_contrast_score`, `shape_score`,
   `identity_score`, `size_score`) and the final confidence for every candidate
   across the demo + case panel.
2. Identify the signal(s) systematically dragging genuine stones below
   `CONFIDENT_THRESHOLD`.
3. Recalibrate the weights and/or threshold (all env-overridable) so **clear
   stones reach "confirmed"** while preserving the conservative "ties →
   uncertain" bias and the deliberate high bar (a false "confirmed" is worse
   than a false "uncertain").

Data-driven, not guessed. Weights stay env-overridable for later panel
recalibration, exactly as they are today.

**Success criteria:** obvious stones (e.g. the coloured stones in `nose.jpg`,
`img.jpg`) reach "confirmed"; ambiguous/colourless candidates still route to
"uncertain".

### Component 3 — Carat weight from size (novel USP)

**New file:** `app/utils/gem_weight.py`.

- Input per stone: pixel area, `px_per_mm` (from the already-detected fiducial
  card), stone identity/SG, and shape hint (from the region's circularity /
  aspect already computed in `_region_confidence`).
- Convert pixel area → real mm² → **estimated carat** via standard
  gemmological weight estimation (equivalent face-up dimensions + a typical
  cut-depth assumption for the inferred shape) × stone specific gravity ÷
  0.2 g/ct.
- Report each estimate as a **range with an explicit caveat**: "approx,
  assumes typical cut depth — not a substitute for unmounting".
- **No-card behaviour (decided):** when `px_per_mm` is unavailable, show each
  gem's **relative size (% of item area) only** and omit carats. Never
  fabricate a scale.
- Aggregate: total estimated gem carats across N stones, alongside the physics
  gold-grams (`composition_result.gold_mass_g`) for a complete breakdown.
- **Advisory/informational only** — does not alter LTV, fusion, or risk.

**Wiring:** computed in `app/routers/analyze.py` next to `build_grid_stats`
(both already have `stones` + `px_per_mm`). Result attached to the case payload
(e.g. under a new `gem_weight` / extended `gem_grid` key) and to the live
response for the card. Wrapped in try/except; failure → omit, never block.

**SG source:** reuse/extend `STONE_DENSITIES` (in `composition.py`) and the
richer `STONE_COLOR_REFERENCES` names already in `xray.py`.

### Component 4 — Prominence: hero result card

**New file:** `frontend/src/components/GoldVsGemsCard.jsx` (+ CSS module),
placed prominently in `ResultsPanel.jsx`.

Contents:
- The `gold_gem` overlay image as a hero visual.
- Big headline: **gold % vs gems %** (from the improved `gold_gem_split`).
- Weight breakdown: gold ≈ X g (physics), gems ≈ Y ct across N stones (or
  relative-size-only when no card).
- Per-gem chips: name, colour swatch, size (mm when scaled), estimated carat.
- A transparency line naming the method (colour manifold + SAM boundaries +
  physics grams) and the "visual/advisory only" caveat.
- Theme-matched to existing cards (`CompositionCard`, `XRayView`).

The existing "Gold vs Gems" X-ray stage and the `CompositionCard` split stay
exactly where they are — this card gives the result a front seat, it does not
replace anything.

## 4. Data flow

```
_run_pipeline (xray.py)
  └─ _gold_gem_map  → improved gold_gem_split (backward-compatible + new keys)
analyze.py
  ├─ xray_preview(...)            (unchanged entry point)
  ├─ detect_marker(...) px_per_mm (existing)
  ├─ build_grid_stats(...)        (existing)
  └─ estimate_gem_weights(...)    (NEW, app/utils/gem_weight.py) → case.gem_weight
frontend
  └─ GoldVsGemsCard (NEW) renders gold_gem_split + gem_weight + overlay
```

No change to the fusion / verdict / LTV computation path.

## 5. Error handling

- Manifold fit fails / too few metal pixels → fall back to current classifier.
- No fiducial card → relative size only, no carats.
- `background_removed == false` → feature inert, as today.
- Any exception in the new util → caught, feature omitted, analysis proceeds.
- Backward-compatible keys guarantee the existing `XRayView` and
  `CompositionCard` renders keep working untouched.

## 6. Validation

- Diagnostic script over the demo + saved-case panel, before/after:
  `gold_pct` / `gem_pct` / `other_pct`, `stones_confirmed`.
  - Expect: "other" drops on gold-dominant items; obvious stones reach
    "confirmed"; zero gold→gem regressions.
- Visual eyeball of regenerated `gold_gem` overlays for the demo images.
- Confirm existing outputs (material composition, stone counts feeding
  composition/fusion) are unchanged in shape and that the app still runs end
  to end.

## 7. Out of scope

- Improving background separation on busy photos (pre-existing; feature stays
  inert when it fails).
- Training any new ML model (SAM stays frozen/pretrained; classification stays
  classical).
- Changing fusion weights, LTV tiers, or the decision policy.
