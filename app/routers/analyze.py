"""
POST /api/analyze — main analysis endpoint.
"""
import base64
import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from app.auth import require_session
from app.models.acoustic_model import analyze_acoustic
from app.models.acoustic_physics import extract_ring_frequency, ring_frequency_check
from app.models.contradiction import contradiction_summary
from app.models.density_model import analyze_density
from app.models.fusion_model import analyze_fusion
from app.models.hallmark_model import analyze_hallmark_engraving
from app.models.image_model import analyze_image
from app.models.spatial_acoustic import analyze_spatial_taps
from app.models.streak_model import analyze_streak
from app.models.tarnish_model import analyze_tarnish, analyze_uv_pass
from app.models.xray_model import analyze_xray, VISUAL_BLEND_IMAGE, VISUAL_BLEND_XRAY
from app.utils.bis_registry import lookup_huid
from app.utils.composition import analyze_composition
from app.benford.monitor import append_density_reading, run_benford_test
from app.utils.fiducial import detect_marker
from app.utils.gem_grid import build_grid_stats
from app.utils.gem_weight import estimate_gem_weights
from app.utils.ltv import assess_value
from app.utils.phash import check_duplicate, register_photo
from app.utils.stamp import stamp_image
from app.utils.xray import xray_preview, reconcile_stones
from app.llm.verdict_prompt import generate_verdict
from app.llm.gem_vision import detect_gems

logger = logging.getLogger(__name__)
router = APIRouter()

HISTORY_PATH = Path("data/case_history.json")
CASES_DIR    = Path("data/cases")


def _load_history() -> list:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            return []
    return []


def _save_case(case: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = _load_history()
    history.append(case)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


_AUDIO_CONTENT_TYPE_EXT = {
    "audio/webm": "webm", "video/webm": "webm",
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/mp4": "m4a", "audio/x-m4a": "m4a",
    "audio/ogg": "ogg", "audio/opus": "opus",
}
_AUDIO_KNOWN_EXTS = {"webm", "wav", "mp3", "m4a", "ogg", "opus"}


def _audio_extension(upload) -> str:
    """The saved file's extension must match its actual container, or
    neither a browser nor a media player can play it back later — see the
    call site for the bug this fixes. Prefers the uploaded filename's own
    extension, falls back to content-type, then to webm (what the browser's
    live capture path — MediaRecorder — always produces)."""
    filename = (upload.filename or "").lower()
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1]
        if ext in _AUDIO_KNOWN_EXTS:
            return ext
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    return _AUDIO_CONTENT_TYPE_EXT.get(content_type, "webm")


POLICY_PATH = Path("data/decision_policy.json")
_DEFAULT_POLICY = {
    "genuine_high_below":      0.25,
    "genuine_medium_below":    0.45,
    "borderline_medium_below": 0.60,
    "borderline_low_below":    0.75,
    "density_override_at":     0.85,
    "contradiction_boost":     0.40,
}


def load_policy() -> dict:
    """Decision policy: bank risk-appetite dials, declared and auditable in
    data/decision_policy.json rather than buried in code."""
    if POLICY_PATH.exists():
        try:
            return {**_DEFAULT_POLICY, **json.loads(POLICY_PATH.read_text())}
        except Exception as e:
            logger.warning("Could not read decision policy: %s", e)
    return dict(_DEFAULT_POLICY)


def _determine_verdict(
    fusion_risk: float, contra_score: float, density_risk: float = 0.0,
    policy: dict | None = None, extra_boost: float = 0.0,
) -> tuple[str, str, str]:
    """
    Return (risk_level, confidence, loan_action) per the declared policy.
    - Density risk ≥ override threshold → REJECT (physics override)
    - Contradiction boost catches mixed-signal composites
    - extra_boost folds in advisory signals that aren't part of core fusion
      (spatial-acoustic spread, hallmark engraving, tarnish/UV anomalies) —
      same additive-boost mechanism as the contradiction score, so a new
      signal can move the verdict without having to retrain the fusion model
    """
    p = policy or load_policy()
    if density_risk >= p["density_override_at"]:
        return "REJECT", "HIGH", "DECLINE"

    boosted = min(1.0, fusion_risk + contra_score * p["contradiction_boost"] + extra_boost)

    if boosted < p["genuine_high_below"]:
        return "GENUINE", "HIGH", "APPROVE"
    elif boosted < p["genuine_medium_below"]:
        return "GENUINE", "MEDIUM", "APPROVE"
    elif boosted < p["borderline_medium_below"]:
        return "BORDERLINE", "MEDIUM", "HOLD"
    elif boosted < p["borderline_low_below"]:
        return "BORDERLINE", "LOW", "HOLD"
    else:
        return "REJECT", "HIGH", "DECLINE"


def _build_trace(
    density_result, xray_result, xray_score, composition_result,
    acoustic_result, streak_result, image_result, visual_risk,
    contra, fusion, verdict_tuple, boosted_risk, water_temp_c,
) -> list[dict]:
    """
    Officer-facing verification trail: every step of the detection process
    with its inputs, the formula applied, the outputs, and the data source.
    Rendered as a stepper in the UI so each stage can be checked by hand.
    """
    risk_level, confidence, loan_action = verdict_tuple
    steps = []

    steps.append({
        "step": "Weight-in-water measurement",
        "status": "done",
        "summary": f"ρ = {density_result['measured_density']} ± {density_result['sigma']} g/cm³, "
                   f"V = {density_result['volume_cm3']} cm³",
        "formula": "ρ = W_dry · ρ_water(T) / (W_dry − W_sub);  σ_ρ/ρ ≈ √2·σ_scale/(W_dry−W_sub)",
        "details": {
            "W_dry (g)":        density_result["weight_dry"],
            "W_submerged (g)":  density_result["weight_submerged"],
            "water T (°C)":     water_temp_c,
            "ρ_water (g/cm³)":  density_result["rho_water"],
        },
        "source": "CRC Handbook water table; Kell (1975)",
    })

    verdict_text = {
        "IN_RANGE":            "Within the expected range",
        "LOW_DENSITY":         "Below the expected range — too light for the declared karat",
        "HIGH_DENSITY":        "Above the expected range — denser than the declared alloy",
        "TUNGSTEN_BLIND_SPOT": "Matches both 24K gold and tungsten — density alone cannot tell",
        "IDENTIFIED_FROM_PHYSICS": "No karat declared — the physics identified the metal",
        "NOT_GOLD":            "No karat declared — the density matches no gold alloy",
    }.get(density_result["karat_verdict"], density_result["karat_verdict"])
    best = density_result.get("best_match") or {}
    physics_says = f" Physics best match: {best.get('name', '—')}"
    if density_result.get("misdeclared_purity"):
        physics_says += " — likely purity mis-declaration, not fake metal"
    steps.append({
        "step": "Does the density match the declared karat?",
        "status": "flag" if density_result["risk_score"] > 0.5 else "done",
        "summary": f"{verdict_text}; chance it matches: "
                   f"{round(density_result['conformity_probability'] * 100, 1)}%."
                   + physics_says,
        "formula": "risk = P(true ρ outside karat band | measurement), Gaussian CDF",
        "details": {
            "declared band (g/cm³)": f"{density_result['expected_low']} – {density_result['expected_high']}",
            "raw density risk":      density_result["risk_score"],
            "closest fake metal":    density_result["closest_fake"] or "—",
            "tungsten blind spot":   density_result["tungsten_warning"],
        },
        "source": "Band derived from IS 1417:2016 fineness + CRC densities (mixture rule); scoring per JCGM 106",
    })

    if xray_result:
        steps.append({
            "step": "Separating the item from the background",
            "status": "done" if xray_result.get("background_removed") else "flag",
            "summary": (f"Item = {xray_result.get('item_area_pct')}% of frame, backdrop removed"
                        if xray_result.get("background_removed")
                        else ("Backdrop could not be separated, so area-based numbers "
                              "(composition, gold-vs-gems) are omitted — but the AI vision "
                              "judge still read the stones directly; retake on a plain "
                              "background for full measurements"
                              if xray_result.get("gem_regions", 0) > 0
                              else "Backdrop could not be separated — the photo is set aside "
                                   "(no visual evidence used); retake on a plain background")),
            "formula": "Backdrop colour from frame border → weighted HSV distance; border-touching components discarded",
            "details": {"stage_image": "material"},
            "source": "Classical CV (no ML) — thresholds in app/utils/xray.py",
        })
        # The stone-finding and dark-spot checks are only meaningful once the
        # item is separated from its background. When it isn't, every count
        # would be measured on the scene, not the item — so we omit them.
        if xray_result.get("background_removed"):
            stone_mode = xray_result.get("stone_detection_mode", "classical")
            steps.append({
                "step": "Finding the stones in the photo",
                "status": "done",
                "summary": f"{xray_result.get('gem_regions', 0)} coloured "
                           f"({xray_result.get('gem_area_pct', 0)}%), "
                           f"{xray_result.get('colourless_regions', 0)} colourless candidates "
                           f"({xray_result.get('colourless_area_pct', 0)}%) — "
                           + ("boundaries refined by MobileSAM (ML)" if stone_mode == "ml_sam"
                              else "classical colour-threshold detection (ML unavailable this run)"),
                "formula": "Region boundaries: MobileSAM-refined when available, else colour-threshold "
                           "connected-components. Classification: saturated non-gold-hue clusters vs "
                           "bright low-saturation regions, using calibrated saturation-purity to tell a "
                           "genuinely coloured stone from a colourless one flashing colour off its facets",
                "details": {"stage_image": "gems", "detection_mode": stone_mode},
                "source": "Independent CV/ML detection — description text is never trusted",
            })
            ggs = xray_result.get("gold_gem_split") or {}
            if ggs:
                steps.append({
                    "step": "Separating gold metal from gems by colour",
                    "status": "done",
                    "summary": (f"Gold ≈ {ggs.get('gold_pct', 0)}% of item, "
                                f"gems ≈ {ggs.get('gem_pct', 0)}%, "
                                f"other ≈ {ggs.get('other_pct', 0)}% "
                                f"({ggs.get('stones_used', 0)} stone region(s) used)"),
                    "formula": "Gems = kept stone mask; remaining item pixels → gold if Lab ΔE to "
                               "metal reference ≤ GOLD_LAB_MATCH_MAX or HSV in gold band; else other",
                    "details": {"stage_image": "gold_gem", **ggs},
                    "source": "Additive colour segmentation (app/utils/xray.py::_gold_gem_map) — "
                              "visual only, does not change fusion risk",
                })
            steps.append({
                "step": "Checking dark spots against the found stones",
                "status": "flag" if xray_result.get("inclusions_unexplained", 0) > 0 else "done",
                "summary": f"{xray_result.get('inclusions_explained', 0)} dark region(s) explained by detected stones, "
                           f"{xray_result.get('inclusions_unexplained', 0)} unexplained",
                "formula": "Dark region counts as a stone only if ≥25% of it overlaps an independently detected gem",
                "details": {},
                "source": "Two independent CV passes must agree (anti-laundering design)",
            })
        elif xray_result.get("gem_regions", 0) > 0:
            # Backdrop could not be separated, yet the AI vision judge read the
            # ornament directly and still found stones. Area-based numbers stay
            # omitted (they need the item mask), but the stones themselves are
            # real evidence and are surfaced so the trace matches the scan card.
            sa = xray_result.get("stone_agreement") or {}
            steps.append({
                "step": "Finding the stones in the photo (AI vision)",
                "status": "done",
                "summary": f"{xray_result.get('gem_regions', 0)} stone(s) identified by the AI vision "
                           "judge directly from the photo, despite the busy background — "
                           "boundaries refined by MobileSAM where available. "
                           "Area %/composition are omitted (they need a clean item mask).",
                "formula": "AI vision model names + locates each set stone; STONE_AI_ONLY makes the AI "
                           "the sole authority on the stone set, so background separation is not required",
                "details": {"stage_image": "gems", "detection_mode": "ml_ai",
                            "n_ai_only": sa.get("n_ai_only", 0), "n_both": sa.get("n_both", 0)},
                "source": "AI vision (Fireworks/Kimi or Gemini) — description text is never trusted",
            })

    if composition_result:
        if composition_result.get("model_valid") is False:
            comp_summary = ("Density too low for any gold/stone mixture — "
                            "the item is not predominantly gold")
        else:
            comp_summary = (f"Gold ≈ {composition_result['gold_mass_g']} g "
                            f"({round((composition_result['gold_mass_fraction'] or 0) * 100)}%), "
                            f"stones photo {round(composition_result['stone_frac_photo']*100)}% vs "
                            f"physics {round(composition_result['stone_frac_implied']*100)}%, "
                            f"z = {composition_result['consistency_z']}")
        steps.append({
            "step": "How much of the weight is actually gold?",
            "status": "flag" if (composition_result["hidden_volume_flag"]
                                 or composition_result.get("model_valid") is False) else "done",
            "summary": comp_summary,
            "formula": "ρ_bulk = (1−f)·ρ_gold + f·ρ_stone;  f_implied = (ρ_gold−ρ_meas)/(ρ_gold−ρ_stone)",
            "details": {
                "assumed stone ρ (g/cm³)":  composition_result["rho_stone_assumed"],
                "adjusted density risk":    composition_result["adjusted_density_risk"],
                "hidden-volume flag":       composition_result["hidden_volume_flag"],
            },
            "source": "Two-component mixture rule; gem SG per Webster/GIA",
        })

    steps.append({
        "step": "Sound test (ring of the tapped item)",
        "status": "done" if acoustic_result.get("mode") not in ("no_audio",) else "skipped",
        "summary": (f"risk {acoustic_result['risk_score']}"
                    if acoustic_result.get("mode") not in ("no_audio",)
                    else "No tap recording provided — the filled-core check needs it"),
        "formula": "MFCC-ΔΔ features → SVM (trained on DS-1 tap recordings)",
        "details": {"method": acoustic_result.get("mode", "—")},
        "source": "DS-1: Kaggle counterfeit-gold tap dataset",
    })

    ring = acoustic_result.get("ring")
    if ring:
        steps.append({
            "step": "Checking the ring pitch against genuine gold",
            "status": "flag" if ring.get("stiff_core_flag") else (
                "done" if ring.get("status") in ("consistent", "marginal") else "skipped"),
            "summary": ring.get("reason", ring.get("status", "")),
            "formula": "Sound speed v = √(stiffness E / density ρ) — a stiffer core rings higher. "
                       "Measured pitch is compared to a band CALIBRATED from known-genuine recordings",
            "details": {
                "measured pitch (Hz)": ring.get("dominant_freq_hz") or "—",
                "genuine band (Hz)":   " – ".join(map(str, ring.get("genuine_band_hz", []))) or "uncalibrated",
                "recording SNR (dB)":  ring.get("snr_db"),
            },
            "source": "E per CRC/ASM (Au 79 GPa, W 411 GPa); band via scripts/calibrate_acoustic.py",
        })

    if xray_score.get("mode") == "dsip_xray":
        steps.append({
            "step": "Combining the photo checks",
            "status": "done",
            "summary": (
                f"photo risk = material scan risk = {visual_risk}"
                if not xray_score.get("fusion_contribution")
                   or xray_score["fusion_contribution"]["image_weight"] == 0
                else f"photo risk = {xray_score['fusion_contribution']['image_weight']}·CNN({image_result['risk_score']}) "
                     f"+ {xray_score['fusion_contribution']['xray_weight']}·scan({xray_score['risk_score']}) "
                     f"= {visual_risk}"
            ),
            "formula": "Weighted blend; DSIP weighted higher (CNN is proxy-trained)",
            "details": {"DSIP findings": xray_score.get("signals", [])},
            "source": "Blend weights: VISUAL_BLEND_XRAY (env-configurable)",
        })

    steps.append({
        "step": "Combining all tests & checking for disagreement",
        "status": "flag" if contra["flags"] else "done",
        "summary": f"combined risk {fusion['risk_score']}, "
                   f"biggest disagreement {contra['contradiction_score']}",
        "formula": (
            "Σ weightᵢ · ln(pᵢ/(1−pᵢ)) over performed tests — hand-recomputable; "
            "a missing test contributes exactly 0"
            if fusion.get("mode") == "logodds"
            else "XGBoost over 4 modality risks + 6 |risk_A − risk_B| contradiction features"
        ),
        "details": {"method": fusion.get("mode", "—"),
                    "flags": contra["flags"] or ["none"]},
        "source": ("Log-odds evidence combination — recomputable by hand"
                   if fusion.get("mode") == "logodds"
                   else "Trained on leakage-free dataset (scripts/rebuild_fusion.py)"),
    })

    steps.append({
        "step": "Final decision",
        "status": "flag" if risk_level == "REJECT" else "done",
        "summary": f"{risk_level} ({confidence}) → {loan_action}",
        "formula": "overrides first (density, ring-pitch, multi-test mandate); else combined risk + "
                   "contradiction boost, banded per the declared policy. APPROVE requires weight + photo + sound",
        "details": {"boosted risk": boosted_risk},
        "source": "data/decision_policy.json — bank risk-appetite dials, auditable",
    })
    return steps


@router.post("/analyze")
async def analyze(
    response: Response,
    item_description: str = Form(""),
    declared_karat: int = Form(...),
    weight_dry: float = Form(...),
    weight_submerged: float = Form(...),
    water_temp_c: float = Form(25.0),
    branch_id: str = Form("default"),
    customer_name: str = Form(""),
    customer_account: str = Form(""),
    loan_app_no: str = Form(""),
    officer_name: str = Form(""),
    weight_capture_source: str = Form("manual"),
    geo_lat: Optional[float] = Form(None),
    geo_lon: Optional[float] = Form(None),
    huid: str = Form(""),
    hallmark_result: str = Form(""),
    kyc_record: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    audio: Optional[UploadFile] = File(default=None),
    streak_image: Optional[UploadFile] = File(default=None),
    tap_audio: list[UploadFile] = File(default=[]),
    tap_positions: list[str] = Form(default=[]),
    uv_image: Optional[UploadFile] = File(default=None),
    session: dict = Depends(require_session),
):
    case_id = uuid.uuid4().hex[:8]
    response.headers["X-Case-ID"] = case_id
    # officer_name is now resolved from the authenticated evaluator session,
    # not trusted as free text — the Evaluator Integrity Layer's whole point
    # is that every case is traceable to a person who was physically present
    # at login (session selfie captured), not a typed name anyone could enter.
    officer_name = session["name"]
    evaluator_id = session["evaluator_id"]

    if declared_karat not in (0, 14, 18, 22, 24):
        raise HTTPException(status_code=422,
                            detail="declared_karat must be 14, 18, 22, 24, or 0 (not declared)")
    if weight_dry <= 0 or weight_submerged <= 0:
        raise HTTPException(status_code=422, detail="Weights must be positive")
    if weight_submerged >= weight_dry:
        raise HTTPException(status_code=422, detail="Submerged weight must be less than dry weight")
    if not 0.0 < water_temp_c < 45.0:
        raise HTTPException(status_code=422, detail="water_temp_c must be between 0 and 45 °C")

    image_bytes_list = [await img.read() for img in images] if images else []
    audio_bytes      = await audio.read() if audio else None
    streak_bytes     = await streak_image.read() if streak_image else None
    uv_bytes         = await uv_image.read() if uv_image else None
    tap_audio_bytes  = [await t.read() for t in tap_audio] if tap_audio else []

    try:
        hallmark_parsed = json.loads(hallmark_result) if hallmark_result else None
    except Exception:
        hallmark_parsed = None
    try:
        kyc_parsed = json.loads(kyc_record) if kyc_record else None
    except Exception:
        kyc_parsed = None

    # ── Duplicate-photo check (Evaluator Integrity Layer) ──
    # Compared against every photo ever recorded system-wide, BEFORE this
    # case's own photos are registered — a reused image from another case
    # (or an earlier attempt) surfaces here, not after the fact.
    duplicate_flags = []
    for i, img_bytes in enumerate(image_bytes_list):
        match = check_duplicate(img_bytes)
        if match:
            duplicate_flags.append({
                "image_role": f"item_photo_{i}", "matches_case_id": match["case_id"],
                "matched_role": match.get("image_role"), "matched_at": match.get("timestamp"),
                "distance": match["distance"],
            })
    if streak_bytes:
        match = check_duplicate(streak_bytes)
        if match:
            duplicate_flags.append({
                "image_role": "streak_photo", "matches_case_id": match["case_id"],
                "matched_role": match.get("image_role"), "matched_at": match.get("timestamp"),
                "distance": match["distance"],
            })

    # ── Material scan on the primary photograph ──
    # Uploaded photos/audio/streak and the processed X-ray stages are saved
    # under data/cases/<case_id>/ further down (once case_id-scoped data is
    # ready) so the PDF report and the History page can show real evidence,
    # not just the live response the browser already has.
    xray_result = None
    xray_ctx = None
    if image_bytes_list:
        try:
            xray_result, xray_ctx = xray_preview(image_bytes_list[0], _return_ctx=True)
        except Exception as e:
            logger.warning("Material scan failed for case %s: %s", case_id, e)

    # ── AI vision stone detection (Layer B — the PRIMARY stone judge) ─────
    # The AI names + locates every set gemstone directly in the photo; stone
    # fusion then uses the ML/SAM boundary only where the AI also saw a stone
    # (STONE_AI_ONLY). The AI reads the raw ornament, so — unlike the ML/CV
    # pass — it does NOT need the backdrop separated: a busy/leafy background
    # that defeats background removal must NOT suppress a stone the AI can
    # plainly see. So this runs whenever we have the pipeline context, even when
    # background_removed is False. Fully guarded: no API key / toggle off / any
    # failure -> ai_stones is None and reconcile_stones returns an ML-only
    # passthrough, so the result is never worse than the ML-only path. Runs on
    # the illumination-normalised image so the AI's coordinates match the
    # item bbox the pipeline computed.
    if xray_result and xray_ctx:
        try:
            ai_stones = await detect_gems(xray_ctx.get("norm"), xray_ctx.get("item_bbox"))
            stats_patch, stage_patch = reconcile_stones(xray_ctx, ai_stones)
            if stats_patch:
                xray_result.update(stats_patch)
            if stage_patch and isinstance(xray_result.get("stages"), dict):
                xray_result["stages"].update(stage_patch)
            # The AI reliably found stones even though the backdrop could not be
            # separated: expose them for display/weighing. Composition/density
            # stay gated on true background separation (area% needs the item
            # mask), but the stone list, count and gem overlay do not.
            if (ai_stones and not xray_result.get("background_removed")
                    and (stats_patch or {}).get("gem_regions", 0) > 0):
                xray_result["ai_stones_without_bg"] = True
        except Exception as e:
            logger.warning("Stone AI detection failed for case %s: %s", case_id, e)

    # ── Fiducial calibration card: tamper-evidence + real-world scale ──
    # Detected in the same primary photo (SOP: place the card flat beside
    # the item in-frame). Best-effort — absence just means no scale
    # reference for stone sizing, never a hard block.
    fiducial_result = None
    gem_grid_result = None
    gem_weight_result = None
    if image_bytes_list:
        try:
            arr = np.frombuffer(image_bytes_list[0], dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                fiducial_result = detect_marker(bgr, branch_id)
        except Exception as e:
            logger.warning("Fiducial detection failed for case %s: %s", case_id, e)
            fiducial_result = {"detected": False, "px_per_mm": None, "reason": f"error: {e}"}
        if xray_result and xray_result.get("background_removed"):
            gem_grid_result = build_grid_stats(
                item_bbox = xray_result.get("item_bbox"),
                stones    = xray_result.get("stones", []),
                grid_n    = 4,
                px_per_mm = (fiducial_result or {}).get("px_per_mm"),
            )
            try:
                gem_weight_result = estimate_gem_weights(
                    stones         = xray_result.get("stones", []),
                    px_per_mm      = (fiducial_result or {}).get("px_per_mm"),
                    declared_karat = declared_karat,
                )
            except Exception as e:
                logger.warning("Gem weight estimation failed for case %s: %s", case_id, e)
                gem_weight_result = None

    # ── Tarnish/rust + UV-pass fluorescence (assistive, low-weight) ──
    tarnish_result = None
    if image_bytes_list:
        try:
            tarnish_result = analyze_tarnish(image_bytes_list[0], declared_karat or 22)
        except Exception as e:
            logger.warning("Tarnish analysis failed for case %s: %s", case_id, e)
    uv_result = None
    if uv_bytes:
        try:
            uv_result = analyze_uv_pass(uv_bytes)
        except Exception as e:
            logger.warning("UV-pass analysis failed for case %s: %s", case_id, e)

    density_result  = analyze_density(weight_dry, weight_submerged, declared_karat, water_temp_c)

    # CNN probe is off the decision path by default (proxy-trained — no
    # defensible evidence). USE_CNN_PROBE=1 re-enables it as a sub-signal.
    if os.getenv("USE_CNN_PROBE", "0") == "1":
        image_result = analyze_image(image_bytes_list)
    else:
        image_result = {
            "risk_score": 0.5, "confidence": "low",
            "mode": "no_cnn" if image_bytes_list else "no_images",
        }
    acoustic_result = analyze_acoustic(audio_bytes) if audio_bytes else {"risk_score": 0.5, "confidence": "low", "mode": "no_audio"}
    streak_result   = analyze_streak(streak_bytes, declared_karat)

    # ── Ring-frequency physics (stiff-core / tungsten cross-check) ──
    # Calibration-gated: only decides against a measured genuine band.
    ring_check = None
    if audio_bytes:
        try:
            ring = extract_ring_frequency(audio_bytes)
            ring_check = ring_frequency_check(ring, density_result)
            acoustic_result["ring"] = {**ring, **ring_check}
        except Exception as e:
            logger.warning("Ring-frequency check failed for case %s: %s", case_id, e)

    # ── Spatial acoustic mapping — the multi-tap grid ──
    # The primary tap (if any) counts as the first grid point; additional
    # tap_audio/tap_positions extend it. Needs >=2 usable taps to say
    # anything about spatial consistency.
    spatial_taps = []
    if audio_bytes:
        spatial_taps.append({"position": "Primary", "audio_bytes": audio_bytes})
    for i, tb in enumerate(tap_audio_bytes):
        pos = tap_positions[i] if i < len(tap_positions) else f"Point {i+2}"
        spatial_taps.append({"position": pos, "audio_bytes": tb})
    spatial_result = analyze_spatial_taps(spatial_taps) if len(spatial_taps) >= 2 else {
        "risk_score": 0.5, "confidence": "low", "mode": "insufficient_taps", "taps": [], "n_usable": 0,
    }

    xray_result_score = analyze_xray(xray_result, item_description)

    # ── Stone-corrected composition (two-component mixture model) ──
    # Uses the CAMERA-detected stone fraction, never the description. The
    # adjusted density risk replaces the raw one downstream so a genuine
    # stone-set item is not rejected for the stones it visibly carries;
    # readings above the gold band are never softened.
    composition_result = None
    density_risk_effective = density_result["risk_score"]
    # With no declaration, the mixture model runs against the karat the
    # physics identified; if the physics matched no gold alloy there is no
    # gold hypothesis to correct for stones.
    effective_karat = declared_karat or (
        (density_result.get("best_match") or {}).get("karat") or 0
    )
    if (
        effective_karat
        and xray_result
        and xray_result.get("background_removed")
        and (xray_result.get("gem_regions", 0) > 0
             or xray_result.get("colourless_regions", 0) > 0)
    ):
        try:
            # colourless candidates join the stone budget as unconfirmed
            # entries with a wide mid-range density assumption
            all_stones = list(xray_result.get("gems", [])) + [
                {"area_pct": c["area_pct"], "hue_class": "colourless"}
                for c in xray_result.get("colourless", [])
            ]
            composition_result = analyze_composition(
                weight_dry       = weight_dry,
                measured_density = density_result["measured_density"],
                sigma_rho        = density_result["sigma"],
                volume_cm3       = density_result["volume_cm3"],
                declared_karat   = effective_karat,
                gems             = all_stones,
                gem_area_pct     = xray_result.get("gem_area_pct", 0.0)
                                   + xray_result.get("colourless_area_pct", 0.0),
            )
            # The stone-mixture correction may only LOWER density risk when
            # the model is valid (density is explainable by gold + stones).
            # When invalid (density too low/high for any gold-stone mix) the
            # correction is meaningless — keep the raw density verdict, and
            # never let a stone story rescue a non-gold density.
            if composition_result.get("model_valid"):
                density_risk_effective = min(
                    density_result["risk_score"],
                    composition_result["adjusted_density_risk"],
                )
            else:
                density_risk_effective = density_result["risk_score"]
        except Exception as e:
            logger.warning("Composition analysis failed for case %s: %s", case_id, e)

    # Exclude modalities with no real data from contradiction — a missing audio
    # signal shouldn't create a false contradiction with density.
    contra_scores = {"density": density_risk_effective}
    if acoustic_result.get("mode", "no_audio") not in ("no_audio",):
        contra_scores["acoustic"] = acoustic_result["risk_score"]
    if not image_result.get("mode", "no_images").startswith("no_"):
        contra_scores["image"] = image_result["risk_score"]
    if streak_result.get("mode", "no_streak") not in ("no_streak",):
        contra_scores["streak"] = streak_result["risk_score"]
    if xray_result_score.get("mode") == "dsip_xray":
        contra_scores["xray"] = xray_result_score["risk_score"]

    contra = contradiction_summary(contra_scores)

    # Hidden-volume contradiction: physics demands more non-gold volume than
    # the camera can account for as stones — surfaces in flags and boosts the
    # verdict through the existing contradiction mechanism.
    if composition_result and composition_result["hidden_volume_flag"]:
        contra["flags"].append(
            "density↔photo: " + composition_result["note"]
        )
        contra["contradiction_score"] = round(
            max(contra["contradiction_score"], composition_result["hidden_severity"]), 4
        )

    # Blend the DSIP X-ray risk into the visual channel before fusion — the
    # trained XGBoost expects exactly 4 modality inputs, so X-ray influences
    # the model through the image feature rather than as a 5th input.
    visual_risk = image_result["risk_score"]
    if xray_result_score.get("mode") == "dsip_xray" and image_result.get("mode") != "no_images":
        visual_risk = round(
            VISUAL_BLEND_IMAGE * image_result["risk_score"]
            + VISUAL_BLEND_XRAY * xray_result_score["risk_score"], 4)
        xray_result_score["fusion_contribution"] = {
            "image_weight":        VISUAL_BLEND_IMAGE,
            "xray_weight":         VISUAL_BLEND_XRAY,
            "blended_visual_risk": visual_risk,
        }

    fusion = analyze_fusion(
        image_risk    = visual_risk,
        density_risk  = density_risk_effective,
        acoustic_risk = acoustic_result["risk_score"],
        streak_risk   = streak_result["risk_score"],
    )

    # Append first so the current item is included in the Benford test
    try:
        append_density_reading(
            density          = density_result["measured_density"],
            case_id          = case_id,
            branch_id        = branch_id,
            declared_karat   = declared_karat,
            weight_submerged = weight_submerged,
        )
    except Exception as e:
        logger.warning("Failed to append density log: %s", e)

    benford = run_benford_test()

    # ── Assessed value + RBI-tiered LTV ──
    # Net gold weight prefers the stone-corrected composition estimate when
    # the mixture model is valid; falls back to gross dry weight otherwise
    # (plain-metal items, or a composition model that didn't apply).
    net_gold_weight_g = (
        composition_result["gold_mass_g"]
        if composition_result and composition_result.get("model_valid")
           and composition_result.get("gold_mass_g") is not None
        else weight_dry
    )
    ltv_result = assess_value(net_gold_weight_g)

    # ── Advisory-signal boost ──
    # Spatial-acoustic spread, hallmark engraving irregularity, and
    # tarnish/UV anomalies are real evidence but aren't part of the core
    # 4-modality fusion model (adding a 5th input would break the trained
    # XGBoost feature vector) — they fold in as an additive boost, the same
    # mechanism the contradiction score already uses. Each capped so no
    # single advisory signal can carry a verdict on its own.
    aux_boost, aux_notes = 0.0, []
    if spatial_result.get("spatial_inconsistency_flag"):
        b = round(min(0.15, spatial_result["risk_score"] * 0.15), 4)
        aux_boost += b
        aux_notes.append(f"spatial-acoustic (+{b}): {spatial_result['reason']}")
    if hallmark_parsed and (hallmark_parsed.get("engraving") or {}).get("risk_score", 0) > 0.6:
        aux_boost += 0.05
        aux_notes.append("hallmark (+0.05): engraving pattern inconsistent with laser engraving")
    if hallmark_parsed and hallmark_parsed.get("cross_check") and hallmark_parsed["cross_check"].get("match") is False:
        aux_boost += 0.10
        aux_notes.append("hallmark (+0.10): registry purity does not match declared karat")
    if tarnish_result and tarnish_result.get("risk_score", 0) > 0.6:
        aux_boost += 0.05
        aux_notes.append("tarnish (+0.05): localized rust/tarnish anomaly detected")
    if uv_result and uv_result.get("risk_score", 0) > 0.6:
        aux_boost += 0.05
        aux_notes.append("uv (+0.05): unexpected fluorescence detected")
    if duplicate_flags:
        aux_boost += 0.20
        aux_notes.append(f"integrity (+0.20): {len(duplicate_flags)} photo(s) matched evidence already on file for another case")
    aux_boost = round(min(0.5, aux_boost), 4)
    contra["flags"].extend(aux_notes)

    policy = load_policy()
    boosted_risk = round(
        min(1.0, fusion["risk_score"] + contra["contradiction_score"] * policy["contradiction_boost"] + aux_boost), 4)
    risk_level, confidence, loan_action = _determine_verdict(
        fusion["risk_score"],
        contra["contradiction_score"],
        density_risk=density_risk_effective,
        policy=policy,
        extra_boost=aux_boost,
    )

    # Physics override: density says gold, ring pitch says stiffer-than-gold
    # core — the one signature no filler metal can avoid. Calibration-gated.
    override_reason = None
    if ring_check and ring_check.get("stiff_core_flag"):
        risk_level, confidence, loan_action = "REJECT", "HIGH", "DECLINE"
        override_reason = ring_check["reason"]
        contra["flags"].append("density↔ring-pitch: " + ring_check["reason"])

    # Multi-test mandate: an APPROVE must rest on the full core battery
    # (weight + photo + sound). One failing test can reject — a fake needs
    # only one decisive contradiction — but a pass needs all three. Known
    # frauds passed precisely because a single factor was trusted alone.
    core_missing = []
    if xray_result_score.get("mode") != "dsip_xray":
        core_missing.append(
            "usable photo (backdrop not separable — retake on a plain background)"
            if xray_result_score.get("mode") == "dsip_unusable" else "photo"
        )
    if acoustic_result.get("mode") in ("no_audio",):
        core_missing.append("sound test")
    if loan_action == "APPROVE" and core_missing:
        risk_level, confidence, loan_action = "BORDERLINE", "LOW", "HOLD"
        override_reason = (
            "Approval requires all three core tests (weight, photo, sound) — "
            f"missing: {', '.join(core_missing)}. Add the missing test(s) and re-analyse. "
            "No single test can approve an item on its own."
        )

    llm_payload = {
        "item_description": item_description or "gold item (no description provided)",
        "declared_karat":   declared_karat,
        "fusion_risk":      fusion["risk_score"],
        "density_risk":     density_risk_effective,
        "risk_level":       risk_level,
        "loan_action":      loan_action,
        "modality_scores": {
            "image":    image_result,
            "density":  density_result,
            "acoustic": acoustic_result,
            "streak":   streak_result,
            "xray":     xray_result_score,
        },
        "contradiction":   contra,
        "density_details": density_result,
        "composition":     composition_result,
    }
    llm_result = await generate_verdict(llm_payload)

    verification_trace = _build_trace(
        density_result, xray_result, xray_result_score, composition_result,
        acoustic_result, streak_result, image_result, visual_risk,
        contra, fusion, (risk_level, confidence, loan_action), boosted_risk,
        water_temp_c,
    )
    if spatial_result.get("mode") == "spatial_map":
        verification_trace.append({
            "step": "Spatial acoustic mapping (multi-tap grid)",
            "status": "flag" if spatial_result.get("spatial_inconsistency_flag") else "done",
            "summary": spatial_result.get("reason", ""),
            "formula": "Ring frequency compared across tap points; risk scales with (max-min)/mean "
                       "spread beyond a calibrated threshold",
            "details": {"tap points": spatial_result.get("n_usable"), "spread": spatial_result.get("spread_ratio")},
            "source": "Extension of the single-tap ring-frequency physics check (acoustic_physics.py)",
        })
    verification_trace.append({
        "step": "Assessed value & RBI-tiered LTV",
        "status": "done",
        "summary": (
            f"Net gold {ltv_result['net_gold_weight_g']} g × ₹{ltv_result['rate_per_gram_inr']:,.0f}/g "
            f"= ₹{ltv_result['assessed_value_inr']:,.0f}; LTV {ltv_result['ltv_pct']*100:.0f}% "
            f"({ltv_result['tier']}) → max loan ₹{ltv_result['max_loan_inr']:,.0f}"
        ),
        "formula": "assessed_value = net_gold_weight_g × rate_per_gram; max_loan = assessed_value × tiered LTV%",
        "details": {"tier": ltv_result["tier"], "ltv_pct": ltv_result["ltv_pct"]},
        "source": ltv_result["source"],
    })

    # ── Persist the evidence itself under data/cases/<case_id>/ ──
    # Photos, the tap-test audio, the streak photo, and every X-ray stage
    # image are written to disk so a saved case can still show real evidence
    # later — from the History page or the PDF report — not just the
    # in-memory response the browser already has for a live analysis.
    case_dir = CASES_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    analysis_ts = datetime.utcnow()

    def _stamp(img_bytes: bytes, label: str) -> bytes:
        return stamp_image(
            img_bytes, case_id=case_id, customer_name=customer_name,
            evaluator_name=officer_name, evaluator_id=evaluator_id,
            geo_lat=geo_lat, geo_lon=geo_lon, timestamp=analysis_ts, label=label,
        )

    saved_images = []
    for i, img_bytes in enumerate(image_bytes_list):
        p = case_dir / f"img_{i}.jpg"
        p.write_bytes(_stamp(img_bytes, f"Item photo {i+1}"))
        saved_images.append(str(p))
        register_photo(img_bytes, case_id=case_id, image_role=f"item_photo_{i}",
                        timestamp=analysis_ts.isoformat())

    saved_audio = None
    if audio_bytes:
        # Save with the container the browser actually recorded (webm/opus
        # from MediaRecorder, or whatever format an uploaded demo file was)
        # — always writing ".wav" regardless of real content made the saved
        # file unplayable once reopened from History: the bytes are a webm
        # container, but the ".wav" extension makes both the browser and any
        # media player expect PCM/WAV framing and refuse to decode it. The
        # live post-analysis player worked around this by playing the
        # browser's own in-memory Blob instead of the saved file, which hid
        # the bug until the case was reopened later.
        ext = _audio_extension(audio)
        p = case_dir / f"audio.{ext}"
        p.write_bytes(audio_bytes)
        saved_audio = str(p)

    saved_tap_audio = []
    for i, (tb, tap_upload) in enumerate(zip(tap_audio_bytes, tap_audio)):
        ext = _audio_extension(tap_upload)
        p = case_dir / f"tap_{i+2}.{ext}"
        p.write_bytes(tb)
        saved_tap_audio.append(str(p))

    saved_streak = None
    if streak_bytes:
        p = case_dir / "streak.jpg"
        p.write_bytes(_stamp(streak_bytes, "Touchstone streak"))
        saved_streak = str(p)
        register_photo(streak_bytes, case_id=case_id, image_role="streak_photo",
                        timestamp=analysis_ts.isoformat())

    saved_uv = None
    if uv_bytes:
        p = case_dir / "uv.jpg"
        p.write_bytes(_stamp(uv_bytes, "UV-pass"))
        saved_uv = str(p)

    saved_xray_stages = {}
    if xray_result and xray_result.get("stages"):
        xray_dir = case_dir / "xray"
        xray_dir.mkdir(exist_ok=True)
        for stage_name, data_uri in xray_result["stages"].items():
            try:
                header, b64data = data_uri.split(",", 1)
                ext = "png" if "image/png" in header else "jpg"
                p = xray_dir / f"{stage_name}.{ext}"
                p.write_bytes(base64.b64decode(b64data))
                saved_xray_stages[stage_name] = str(p)
            except Exception as e:
                logger.warning("Could not save xray stage '%s' for case %s: %s", stage_name, case_id, e)

    case = {
        "case_id":          case_id,
        "timestamp":        analysis_ts.isoformat() + "Z",
        "item_description": item_description,
        "declared_karat":   declared_karat,
        "branch_id":        branch_id,
        "geo":              {"lat": geo_lat, "lon": geo_lon} if geo_lat is not None else None,
        "weight_capture_source": weight_capture_source,
        "customer": {
            "name":        customer_name,
            "account_no":  customer_account,
            "loan_app_no": loan_app_no,
            "officer_name": officer_name,
        },
        "evaluator": {
            "evaluator_id": session["evaluator_id"],
            "name":         session["name"],
            "branch_id":    session["branch_id"],
            "role":         session["role"],
            "login_time":   session["login_time"],
            "selfie_captured": bool(session.get("selfie_path")),
        },
        "kyc": kyc_parsed,
        "integrity": {
            "duplicate_photo_flags": duplicate_flags,
            "fiducial":              fiducial_result,
        },
        "media": {
            "images_provided": len(image_bytes_list),
            "audio_provided":  audio_bytes is not None,
            "streak_provided": streak_bytes is not None,
            "uv_provided":     uv_bytes is not None,
            "images":          saved_images,
            "audio":           saved_audio,
            "tap_audio":       saved_tap_audio,
            "streak":          saved_streak,
            "uv":              saved_uv,
            "xray":            xray_result,
        },
        "modality_scores": {
            "image":            image_result,
            "density":          density_result,
            "acoustic":         acoustic_result,
            "streak":           streak_result,
            "xray":             xray_result_score,
            "spatial_acoustic": spatial_result,
            "tarnish":          tarnish_result,
            "uv":               uv_result,
        },
        "hallmark":         hallmark_parsed,
        "gem_grid":         gem_grid_result,
        "gem_weight":       gem_weight_result,
        "ltv":              ltv_result,
        "contradiction":    contra,
        "composition":      composition_result,
        "verification_trace": verification_trace,
        "fusion":           fusion,
        "benford":          benford,
        "verdict": {
            "risk_level":    risk_level,
            "confidence":    confidence,
            "loan_action":   loan_action,
            "override_reason": override_reason,
            "fusion_risk":   round(float(fusion["risk_score"]), 4),
            "aux_boost":     aux_boost,
            "aux_notes":     aux_notes,
            "boosted_risk":  boosted_risk,
            "plain_english": llm_result["plain_english"],
            "action":        llm_result["action"],
            "llm_provider":  llm_result["llm_provider"],
        },
    }

    from app.utils.numpy_safe import numpy_safe as _numpy_safe
    safe_case = _numpy_safe(case)

    # The live response embeds X-ray stages as base64 so the just-completed
    # analysis renders instantly with no extra round trip. The persisted copy
    # swaps that for the on-disk paths saved above, so case_history.json
    # stays small while a saved case still points at real image files.
    persist_case = json.loads(json.dumps(safe_case))
    # media["xray"] is present-but-None when no X-ray was captured, so a plain
    # .get("xray", {}) returns None (not the default) and would blow up below.
    if ((persist_case.get("media") or {}).get("xray") or {}).get("stages") is not None:
        persist_case["media"]["xray"]["stages"] = saved_xray_stages
    _save_case(persist_case)
    return JSONResponse(content=safe_case)
