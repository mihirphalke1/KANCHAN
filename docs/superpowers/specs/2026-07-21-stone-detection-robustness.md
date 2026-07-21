# Stone-detection robustness — "fails for a few images" fix

**Date:** 2026-07-21
**Files:** `app/llm/gem_vision.py`, `app/utils/stone_fusion.py`,
`app/routers/analyze.py`, `frontend/src/components/XRayView.jsx`
**Tests:** `tests/test_stone_robustness.py` (14, all pure/network-free)
**Diagnostic:** `scripts/diagnose_failing_stones.py`

## Problem

Stone detection is AI-primary (`STONE_AI_ONLY=1`): `gem_vision.py` (Fireworks
Qwen3-VL) is the sole authority on stone count + type; classical CV + MobileSAM
only lend boundaries and act as a fallback. This failed on some images because
of three real gaps:

1. **Single roll of the dice.** One vision call per photo. The model's own
   measured behaviour includes runs that return junk / 0 stones on a pavé ring
   it counts fine the next run. One bad run → the weak ML-only fallback.
2. **Tiny stones unresolvable.** Small catalogue photos put pavé / illusion-set
   stones only a few pixels across; the model simply misses them at native size.
3. **AI "zero" trusted blindly.** When the AI returned an *empty* set, STRICT
   AI-ONLY dropped every CV detection → a real diamond ring reported as bare
   metal.

`GEM_VISION_VOTES`, `GEM_VISION_MIN_SIDE`, `GEM_VISION_OUTER_TIMEOUT_S` were in
`.env` but wired to nothing — the intended robustness layer was never built.

## What was built

### 1. Multi-vote consensus + upscaling (`gem_vision.py`)

- Run `GEM_VISION_VOTES` independent calls concurrently, bounded by
  `GEM_VISION_OUTER_TIMEOUT_S`.
- Upscale the crop's short side to `GEM_VISION_MIN_SIDE` first (cubic). Reply
  coordinates are fractions 0..1, so this only improves recall — it never shifts
  a position. Coordinate mapping still uses the *original* crop dimensions.
- Cluster detections across votes (greedy single-linkage; one detection per vote
  per cluster). The **reported set is the strict-majority set** — stones seen in
  ≥ ⌊n/2⌋+1 votes. This is what stops naive-union count inflation (measured on a
  dense pavé ring: 1 run = 46, naive union of 2 = **126**, strict-majority = 29,
  all double-confirmed; true ≈ 32–36).
- **Recovery fallback:** if the majority set is empty (votes agreed on nothing,
  e.g. one run saw a lone solitaire, the other returned nothing) surface the
  union tagged `low_consensus=True` so a real stone one good run found is
  recovered — but never asserted confirmed.
- `GEM_VISION_VOTES=1` + `GEM_VISION_MIN_SIDE=0` ⇒ byte-identical to the old
  single-call path.

### 2. Vote-consensus gate (`stone_fusion.py`)

An AI stone tagged `low_consensus` is capped at `uncertain` + `needs_review`
regardless of the model's own confidence — recovered for a human, never asserted.

### 3. Graceful degradation on AI-empty (`stone_fusion.py`)

When the AI returns an empty set but the CV pass is confident
(`confidence ≥ STONE_RESCUE_ML_MIN_CONF`), keep those stones as
`uncertain` + `needs_review` (agreement `ml_only_ai_empty`) instead of dropping
to zero. Meta exposes `n_rescued_ai_empty`, `n_needs_review`,
`ai_empty_ml_disagree`. Never fires when the AI actually reported stones (that
set stays authoritative); never fabricates on plain metal (validated: a plain
gold ring stays 0). Gated by `STONE_RESCUE_ML_ON_AI_EMPTY` (default on).

### 4. Surfacing

- `analyze.py`: a `flag`-status audit-trace step when `ai_empty_ml_disagree`.
- `XRayView.jsx`: a `CV found, AI missed` chip and a `— flag for review` suffix;
  the summary line shows the review count.

### 5. Openwork / floated-stone recovery (`stone_fusion.py` + `xray.py`)

Validated against the real failing images: a pavé kite-frame ring reported only
**1** stone even though the AI vision judge consistently found 9–18. Cause: the
reconcile guards validated AI-only stones against `item_bool` (the pixel-precise
metal mask), so a stone floated in an **openwork gap** (the kite's open centre,
a prong-set cluster) sat in a *hole* of that mask and was dropped — reintroducing
the background-separation dependency the AI-primary design exists to avoid. Two
further drops compounded it: SAM over-growing a dense-cluster prompt past the
area cap, and edge pavé clipped by a tight hull.

Fix:
- AI-only stones are validated against an **envelope** = the convex hull of the
  metal mask (ornament silhouette incl. openwork gaps), dilated by
  `STONE_AI_ENVELOPE_MARGIN_FRAC` (0.04) so edge pavé and the model's
  approximate centres aren't clipped.
- The "boxed the whole ornament" area cap uses the **envelope area**, not the
  metal-only area (an openwork piece's metal is a fraction of its extent).
- When the SAM boundary over-grows past the cap, **fall back to the AI's tight
  bbox** instead of dropping the stone; only reject if the bbox *also* spans the
  ornament.

Result on the three failing catalogue photos (`tests/fixtures/failing_stones/`):
kite pavé **1 → 6–11 confirmed**, illusion cluster **→ 7 confirmed**, green
aventurine **1 confirmed**. Residual count variance on 28-stone micro-pavé is
the vision model's own limit (it sees 9–18 run-to-run) and is handled honestly
by consensus — only double-voted stones are confirmed.

## Guarantees

- Every new path is guarded; any failure falls back to the prior behaviour.
- No public signatures changed.
- With voting disabled the pipeline is byte-identical to before.

## Validation

Validated end-to-end on the demo panel and on the three real failing catalogue
photos in `tests/fixtures/failing_stones/` (see §5 for the numbers). Re-run any
time with `python scripts/diagnose_failing_stones.py`.
