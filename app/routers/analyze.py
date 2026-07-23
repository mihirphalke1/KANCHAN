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
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from app.auth import require_session
from app.rate_limit import ANALYZE_RATE_LIMIT, limiter
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
from app.utils.approval import build_approval
from app.utils.fraud_scenario import map_scenarios
from app.utils.bis_registry import lookup_huid
from app.utils.composition import analyze_composition
from app.benford.monitor import append_density_reading, run_benford_test
from app.utils.fiducial import detect_marker, CARD_SIDE_MM
from app.utils.card_calibration import (
    classify_photos, white_balance_gains, apply_white_balance,
)
from app.utils.gem_grid import build_grid_stats
from app.utils.gem_weight import estimate_gem_weights
from app.utils.hashchain import append_with_chain
from app.utils.ltv import assess_value, compute_net_gold_weight
from app.utils.pledging import check_pledging_cap
from app.utils.phash import check_duplicate, register_photo
from app.utils.stamp import stamp_image
from app.utils.xray import xray_preview, reconcile_stones, rebuild_gem_summaries
from app.utils.stone_fusion import review_closeup_photos
from app.utils.mesh3d import initial_mesh3d_status, start_mesh3d_job
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
    # Stamp the case into the tamper-evident hash chain (prev_hash + entry_hash)
    # before it is persisted, so the ledger is verifiable from this point on.
    append_with_chain(history, case)
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

    # Status reflects the discrete verdict bucket (does the point estimate sit
    # in the expected band?), NOT the raw Gaussian-CDF risk_score directly.
    # risk_score is P(true density outside the band | measurement uncertainty)
    # — it can exceed 0.5 purely from a large sigma (a light item on the same
    # balance has proportionally more measurement noise) even when the
    # measured density sits squarely inside the band, which used to show this
    # step as "flag" (amber) despite karat_verdict == IN_RANGE. That
    # uncertainty is still real and worth surfacing, so it's now a caution
    # note in the summary rather than a false "this looks wrong" flag.
    _density_ok = density_result["karat_verdict"] in ("IN_RANGE", "IDENTIFIED_FROM_PHYSICS")
    _low_confidence = _density_ok and not density_result.get("measurement_adequate", True)
    confidence_note = (
        " Note: measurement precision is limited for an item this light — the "
        "reading passes, but a heavier sample or repeat weighing would be more conclusive."
        if _low_confidence else ""
    )
    steps.append({
        "step": "Does the density match the declared karat?",
        "status": "done" if _density_ok else "flag",
        "summary": f"{verdict_text}; chance it matches: "
                   f"{round(density_result['conformity_probability'] * 100, 1)}%."
                   + physics_says + confidence_note,
        "formula": "risk = P(true ρ outside karat band | measurement), Gaussian CDF",
        "details": {
            "declared band (g/cm³)": f"{density_result['expected_low']} – {density_result['expected_high']}",
            "raw density risk":      density_result["risk_score"],
            "closest fake metal":    density_result["closest_fake"] or "—",
            "tungsten blind spot":   density_result["tungsten_warning"],
            "measurement adequate":  density_result.get("measurement_adequate", True),
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

        # Two independent detectors disagreed hard: the AI vision judge saw NO
        # stones, yet the CV/ML pass was confident it found some. Rather than
        # silently trust the AI's zero (which is how a small/low-contrast pavé
        # ring gets mis-reported as bare metal), those stones were recovered as
        # review-flagged evidence — surface that to the officer explicitly.
        sa = xray_result.get("stone_agreement") or {}
        if sa.get("ai_empty_ml_disagree"):
            steps.append({
                "step": "Detector disagreement — stones flagged for review",
                "status": "flag",
                "summary": f"AI vision found no stones, but the CV/ML pass confidently detected "
                           f"{sa.get('n_rescued_ai_empty', 0)} — kept as UNCERTAIN and flagged for a "
                           "human check rather than dropped. Common on tiny pavé / illusion settings "
                           "the vision model under-reads.",
                "formula": "AI-empty + CV confidence ≥ STONE_RESCUE_ML_MIN_CONF → recover as "
                           "needs_review (never asserted as confirmed)",
                "details": {"stage_image": "gems",
                            "n_rescued": sa.get("n_rescued_ai_empty", 0)},
                "source": "Graceful degradation (app/utils/stone_fusion.py) — recovers, never invents",
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

    _ac_mode = acoustic_result.get("mode", "—")
    if _ac_mode == "svm":
        _ac_formula = ("122-dim MFCC + Δ (velocity) + ΔΔ (acceleration) fingerprint → RBF-SVM. "
                       "ΔΔ captures the acceleration of spectral decay that separates a solid "
                       "ring from a damped composite thud.")
    elif _ac_mode == "svm_data_limited":
        _ac_formula = ("122-dim MFCC + Δ + ΔΔ fingerprint computed, but the classifier is "
                       "data-limited (DS-1 absent) — reported for information only; the ring-pitch "
                       "physics below carries the acoustic verdict.")
    else:
        _ac_formula = ("122-dim MFCC + Δ + ΔΔ fingerprint → heuristic fallback "
                       "(RMS-decay ratio + zero-crossing rate).")
    _ac_details = {"method": _ac_mode}
    if acoustic_result.get("svm_raw_risk") is not None:
        _ac_details["svm raw risk (not fused)"] = acoustic_result["svm_raw_risk"]
        _ac_details["svm LOOCV"] = acoustic_result.get("svm_loocv")
    steps.append({
        "step": "Sound test (ring of the tapped item)",
        "status": "done" if _ac_mode not in ("no_audio",) else "skipped",
        "summary": (f"risk {acoustic_result['risk_score']}"
                    if _ac_mode not in ("no_audio",)
                    else "No tap recording provided — the filled-core check needs it"),
        "formula": _ac_formula,
        "details": _ac_details,
        "source": "MFCC-ΔΔ per librosa; classifier via scripts/retrain_acoustic_svm.py",
    })

    ring = acoustic_result.get("ring")
    if ring:
        _vcc = ring.get("velocity_cross_check") or {}
        _ring_details = {
            "measured pitch (Hz)": ring.get("dominant_freq_hz") or "—",
            "genuine band (Hz)":   " – ".join(map(str, ring.get("genuine_band_hz", []))) or "uncalibrated",
            "recording SNR (dB)":  ring.get("snr_db"),
        }
        # Noise-cleanup trail (0a): show what cleaning bought, for auditability.
        if ring.get("snr_pre_db") is not None:
            _ring_details["SNR before / after cleanup (dB)"] = f"{ring.get('snr_pre_db')} → {ring.get('snr_post_db')}"
            _ring_details["noise cleanup"] = "on" if ring.get("denoise_applied") else "off"
        # Length-adaptive velocity cross-check (0c).
        if _vcc.get("available"):
            if _vcc.get("gold_ref_velocity_ms"):
                _ring_details["velocity v=2·L·f (m/s)"] = f"{_vcc.get('velocity_ms')} vs genuine {_vcc.get('gold_ref_velocity_ms')} (ratio {_vcc.get('velocity_ratio')})"
            else:
                _ring_details["velocity v=2·L·f (m/s)"] = _vcc.get("velocity_ms")
            _ring_details["item length (mm)"] = _vcc.get("item_length_mm")
        steps.append({
            "step": "Checking the ring pitch against genuine gold",
            "status": "flag" if ring.get("stiff_core_flag") else (
                "done" if ring.get("status") in ("consistent", "marginal") else "skipped"),
            "summary": ring.get("reason", ring.get("status", "")),
            "formula": "Sound speed v = √(stiffness E / density ρ) — a stiffer core rings higher. "
                       "Pitch is compared to a CALIBRATED genuine band, and cross-checked against the "
                       "length-adaptive expectation v = 2·L·f when the calibration card gives item size.",
            "details": _ring_details,
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
@limiter.limit(ANALYZE_RATE_LIMIT)
async def analyze(
    request: Request,
    response: Response,
    item_description: str = Form(""),
    declared_karat: int = Form(...),
    weight_dry: float = Form(...),
    weight_submerged: float = Form(...),
    water_temp_c: float = Form(25.0),
    hollow_item: bool = Form(False),
    borrower_present: bool = Form(False),
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
    image_sources: list[str] = Form(default=[]),
    audio: Optional[UploadFile] = File(default=None),
    streak_image: Optional[UploadFile] = File(default=None),
    streak_source: str = Form("camera"),
    tap_audio: list[UploadFile] = File(default=[]),
    tap_positions: list[str] = Form(default=[]),
    uv_image: Optional[UploadFile] = File(default=None),
    uv_source: str = Form("camera"),
    session: dict = Depends(require_session),
):
    case_id = uuid.uuid4().hex[:8]
    response.headers["X-Case-ID"] = case_id

    # Evidence capture policy (P2): in production (ALLOW_UPLOAD_EVIDENCE=0) only
    # live-camera captures are accepted as photographic evidence — a gallery /
    # file-picker upload could be an old or someone else's photo, defeating the
    # whole live-capture chain. We reject such a submission outright rather than
    # silently trusting client-side labelling. Kept permissive by default so the
    # demo/dev upload escape hatch still works; the source is recorded either way.
    _evidence_sources = [s for s in ([*image_sources, streak_source, uv_source]) if s]
    _uploaded = [s for s in _evidence_sources if s == "upload"]
    if _uploaded and os.getenv("ALLOW_UPLOAD_EVIDENCE", "1").strip() not in ("1", "true", "True"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(_uploaded)} evidence image(s) were gallery-uploaded, not live-captured. "
                "This deployment accepts only live-camera evidence — retake the photo(s) with the camera."
            ),
        )
    # officer_name is now resolved from the authenticated evaluator session,
    # not trusted as free text — the Evaluator Integrity Layer's whole point
    # is that every case is traceable to a person who was physically present
    # at login (session selfie captured), not a typed name anyone could enter.
    officer_name = session["name"]
    evaluator_id = session["evaluator_id"]

    if declared_karat not in (0, 14, 18, 22, 23, 24):
        raise HTTPException(status_code=422,
                            detail="declared_karat must be 14, 18, 22, 23, 24, or 0 (not declared)")
    if weight_dry <= 0 or weight_submerged <= 0:
        raise HTTPException(status_code=422, detail="Weights must be positive")
    if weight_submerged >= weight_dry:
        raise HTTPException(status_code=422,
                            detail="Water displaced must be greater than zero — check the reading and that the scale was tared")
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

    # ── Card-free item photo for all CV; card photo for calibration only ──
    # The calibration card must never be counted as part of the item. So we
    # pick the CARD-FREE photo as the frame every CV stage runs on (material
    # scan, stone detection, gold-vs-gems, tarnish, 3D), and use the CARD photo
    # for two things only: real-world scale (px-per-mm) and colour quality
    # (a white-balance reference sampled from the card's neutral body, applied
    # to the item photo so gold/gem colours read true). Falls back cleanly to
    # single-photo behaviour when there's no separate card/item pair.
    photo_split = classify_photos(image_bytes_list) if image_bytes_list else {
        "item_index": None, "card_index": None, "other_index": [], "has_card": []
    }
    item_index = photo_split["item_index"]
    card_index = photo_split["card_index"]

    # Colour calibration: neutralise the item photo's colour cast using the
    # card's white body as a reference (only when card and item are separate
    # frames — a card in the item frame is handled by masking downstream).
    wb_applied = False
    wb_gains = None
    cv_image_bytes = image_bytes_list[item_index] if item_index is not None else None
    if cv_image_bytes is not None and card_index is not None and card_index != item_index:
        try:
            gains = white_balance_gains(image_bytes_list[card_index])
            if gains is not None:
                cv_image_bytes = apply_white_balance(cv_image_bytes, gains)
                wb_applied = True
                wb_gains = [round(float(x), 3) for x in gains]  # BGR
        except Exception as e:
            logger.warning("Colour calibration failed for case %s: %s", case_id, e)

    # ── Material scan on the CARD-FREE item photograph ──
    # Uploaded photos/audio/streak and the processed X-ray stages are saved
    # under data/cases/<case_id>/ further down (once case_id-scoped data is
    # ready) so the PDF report and the History page can show real evidence,
    # not just the live response the browser already has.
    xray_result = None
    xray_ctx = None
    if cv_image_bytes is not None:
        try:
            xray_result, xray_ctx = xray_preview(cv_image_bytes, _return_ctx=True)
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
            ai_stones = await detect_gems(
                xray_ctx.get("norm"), xray_ctx.get("item_bbox"),
                source_bgr=xray_ctx.get("source_bgr"),
                source_to_norm=xray_ctx.get("source_to_norm", 1.0),
            )
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

    # ── Close-up stone photos (angle 2+) ──────────────────────────────────
    # A ring or bangle often can't show both the calibration card AND small
    # set stones clearly in one frame, so the officer may add extra photos
    # that are just a close-up of the stones. Photo 0 stays the ONLY frame
    # with a scale reference (fiducial card) and therefore stays authoritative
    # for stone size/weight; extra photos can only sharpen stone COUNT/TYPE —
    # see stone_fusion.review_closeup_photos for exactly what they're allowed
    # to change. Stashed here; folded into contra["flags"] once contra exists.
    closeup_review = None
    closeup_flags: list[str] = []
    if xray_result and photo_split["other_index"]:
        # Close-ups are the extra photos that are neither the item frame nor
        # the calibration-card frame — the card photo is calibration-only and
        # is never read for stone content.
        closeup_passes = []
        for i in photo_split["other_index"]:
            try:
                arr = np.frombuffer(image_bytes_list[i], dtype=np.uint8)
                extra_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if extra_bgr is None:
                    continue
                extra_gems = await detect_gems(extra_bgr, item_bbox=None)
                if extra_gems is not None:
                    closeup_passes.append({"photo_index": i, "gems": extra_gems})
            except Exception as e:
                logger.warning("Close-up stone detection failed for case %s photo %d: %s", case_id, i, e)
        if closeup_passes:
            try:
                closeup_review = review_closeup_photos(xray_result.get("stones", []), closeup_passes)
                xray_result["stones"] = closeup_review["stones"]
                xray_result.update(rebuild_gem_summaries(closeup_review["stones"]))
                closeup_flags = closeup_review["flags"]
            except Exception as e:
                logger.warning("Close-up stone review failed for case %s: %s", case_id, e)
                closeup_review = None

    # ── Fiducial calibration card: tamper-evidence + real-world scale ──
    # Read from the CARD photo (the calibration frame), not the item frame —
    # the card contributes scale (px-per-mm) and tamper-evidence only, and is
    # never part of the analysed item. Best-effort — absence just means no
    # scale reference for stone sizing, never a hard block.
    fiducial_result = None
    gem_grid_result = None
    gem_weight_result = None
    calib_index = card_index if card_index is not None else item_index
    if calib_index is not None and image_bytes_list:
        try:
            arr = np.frombuffer(image_bytes_list[calib_index], dtype=np.uint8)
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
    # Runs on the card-free, colour-calibrated item frame like every other CV
    # stage, so the calibration card can't be read as tarnish/rust.
    tarnish_result = None
    if cv_image_bytes is not None:
        try:
            tarnish_result = analyze_tarnish(cv_image_bytes, declared_karat or 22)
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
    # Item longest dimension in metres, from the calibration-card scale — feeds
    # the length-adaptive v = 2·L·f velocity cross-check. None when the card
    # isn't in frame (then the velocity check simply abstains).
    item_length_m = None
    try:
        _ppm = (fiducial_result or {}).get("px_per_mm")
        _bbox = (xray_result or {}).get("item_bbox")
        if _ppm and _bbox and len(_bbox) == 4:
            longest_px = max(_bbox[2] - _bbox[0], _bbox[3] - _bbox[1])
            if longest_px > 0:
                item_length_m = longest_px / _ppm / 1000.0
    except Exception as e:
        logger.warning("Item-length derivation failed for case %s: %s", case_id, e)

    ring_check = None
    if audio_bytes:
        try:
            ring = extract_ring_frequency(audio_bytes)
            ring_check = ring_frequency_check(ring, density_result, item_length_m=item_length_m)
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
    if closeup_flags:
        contra["flags"].extend(closeup_flags)

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
            evaluator_id     = evaluator_id,
            declared_karat   = declared_karat,
            weight_submerged = weight_submerged,
        )
    except Exception as e:
        logger.warning("Failed to append density log: %s", e)

    # Branch-level monitor (organised ring across the branch) plus a
    # per-evaluator slice (localises a systematic anomaly to a single corrupt
    # officer — P3-11). The per-evaluator result only fires once that officer
    # has accrued enough attributed readings; until then it reports
    # insufficient_data and never blocks a case.
    #
    # Benford is a cross-case pattern signal about the EVALUATOR's history,
    # not evidence about THIS item's purity — it must stay out of contra
    # ("flags"), which feeds the contradiction score, the "combining all
    # tests" trace status, and the LLM verdict prompt. It is surfaced only
    # via the dedicated benford/benford_evaluator response fields (rendered
    # by BenfordStatus / the admin dashboard), never mixed into the
    # authenticity verdict for this case.
    benford = run_benford_test()
    benford_evaluator = run_benford_test(evaluator_id=evaluator_id)

    # ── Assessed value + RBI-tiered LTV ──
    # Net gold weight = total measured weight − estimated weight of the set
    # stones, so LTV is charged on the gold only. Prefers the direct per-stone
    # size→carat→grams deduction (needs the calibration card), falling back to
    # the density mixture model, then gross weight. See ltv.compute_net_gold_weight.
    net_gold = compute_net_gold_weight(weight_dry, gem_weight_result, composition_result, gem_grid_result)
    net_gold_weight_g = net_gold["net_gold_weight_g"]
    _density_best_match = density_result.get("best_match") or {}
    _matched_karat = _density_best_match.get("karat") if _density_best_match.get("kind") == "karat" else None
    ltv_result = assess_value(net_gold_weight_g, declared_karat=declared_karat or None, matched_karat=_matched_karat)
    ltv_result["net_gold_breakdown"] = net_gold

    # Per-borrower aggregate pledging cap (RBI Para 16, P4-15): a breach blocks
    # sanction. Coins vs ornaments inferred from the item description.
    _item_type = "coin" if "coin" in (item_description or "").lower() else "ornament"
    pledging = check_pledging_cap(customer_account, net_gold_weight_g, item_type=_item_type,
                                  exclude_case_id=case_id)
    ltv_result["pledging_cap"] = pledging

    # Borrower-present attestation gate (P4-16): valuation/LTV is only FINAL when
    # the borrower is attested present at valuation; otherwise it is provisional.
    ltv_result["borrower_present"] = bool(borrower_present)
    ltv_result["status"] = "final" if borrower_present else "provisional"

    # ── Advisory-signal boost ──
    # Spatial-acoustic spread, hallmark engraving irregularity, and
    # tarnish/UV anomalies are real evidence but aren't part of the core
    # 4-modality fusion model (adding a 5th input would break the trained
    # XGBoost feature vector) — they fold in as an additive boost, the same
    # mechanism the contradiction score already uses. Each capped so no
    # single advisory signal can carry a verdict on its own.
    aux_boost, aux_notes = 0.0, []
    # spatial_acoustic is experimental and unvalidated for smartphone-mic
    # jewellery acoustics, so by default it must NOT move a real verdict — it is
    # still computed and displayed for information, but only contributes to
    # aux_boost when explicitly enabled (ENABLE_SPATIAL_ACOUSTIC=1).
    _spatial_enabled = os.getenv("ENABLE_SPATIAL_ACOUSTIC", "0").strip() in ("1", "true", "True")
    if _spatial_enabled and spatial_result.get("spatial_inconsistency_flag"):
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

    # Hollow-item density correction (P1-1). A genuinely hollow piece (bangle,
    # jhumka dome) should read LIGHTER than solid gold. If the officer marks the
    # item hollow (or the photo shows filigree openwork) yet the density sits in
    # a solid-gold karat band, that is the dense-core-fill signature — the
    # density "looks like gold" precisely because a dense filler replaced the
    # air. Density's reassurance is inverted, and the acoustic ring pitch is
    # promoted to the decisive signal for this case.
    xray_filigree = (xray_result or {}).get("filigree") or {}
    is_openwork = bool(xray_filigree.get("is_filigree"))
    treat_hollow = bool(hollow_item or is_openwork)
    hollow_anomaly = False
    if treat_hollow:
        best = (density_result or {}).get("best_match") or {}
        if best.get("kind") == "karat":
            hollow_anomaly = True
            aux_boost += 0.15
            src = "declared hollow" if hollow_item else "photo shows filigree openwork"
            aux_notes.append(
                f"hollow-item (+0.15): {src}, but density is consistent with SOLID gold "
                f"({best.get('label', 'a gold karat band')}) — a hollow piece should read lighter. "
                "Possible dense-core fill; ring-pitch acoustic is the decisive check."
            )
        # Promote acoustic: for a hollow/openwork item, an above-band ring pitch
        # (not only the stricter stiff_core_flag) is treated as decisive later.

    if (xray_result or {}).get("multiple_items", {}).get("multiple_items_detected"):
        mi = xray_result["multiple_items"]
        aux_notes.append(
            f"multiple items: {mi.get('component_count')} separate pieces in frame"
            + (" (likely a pair)" if mi.get("likely_pair") else "")
            + " — value and weigh each separately; a single blended reading is unreliable."
        )

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
    elif treat_hollow and hollow_anomaly and ring_check and ring_check.get("status") == "above_genuine_band":
        # Acoustic promoted to primary for a hollow/openwork item: a solid-gold
        # density on a piece that should be hollow, plus a ring pitch above the
        # genuine band, is the dense-core-fill signature even short of the
        # stricter stiff_core_flag threshold.
        risk_level, confidence, loan_action = "REJECT", "HIGH", "DECLINE"
        override_reason = (
            "Item is hollow/openwork yet density reads as solid gold and the ring pitch is "
            "above the genuine band — dense-core-fill signature (acoustic promoted to primary)."
        )
        contra["flags"].append("hollow↔ring-pitch: " + override_reason)

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

    # RBI Para 16 aggregate pledging cap (P4-15): a breach blocks sanction
    # regardless of the authenticity verdict — the gold may be genuine, but the
    # borrower has hit their pledging ceiling.
    if pledging.get("exceeds"):
        contra["flags"].append("pledging-cap: " + pledging["reason"])
        if loan_action != "DECLINE":
            risk_level, confidence, loan_action = "BORDERLINE", confidence, "HOLD"
            override_reason = pledging["reason"]

    # Borrower-present attestation (P4-16): the valuation/LTV cannot be finalised
    # until the borrower is attested present. This does not change the
    # authenticity verdict — it holds the LOAN from being disbursed on an
    # unattested valuation.
    if loan_action == "APPROVE" and not borrower_present:
        contra["flags"].append(
            "borrower-present: attestation not recorded — valuation is provisional; "
            "confirm the borrower is present before finalising/disbursing.")

    # Translate the internal verdict + contradiction pattern into the bank's
    # 8-scenario spurious-gold vocabulary (P3-13) for the response and the PDF.
    _gross_for_frac = float(net_gold.get("gross_weight_g") or weight_dry or 0.0)
    _stone_frac = (float(net_gold.get("stone_weight_g") or 0.0) / _gross_for_frac) if _gross_for_frac > 0 else 0.0
    fraud_scenarios = map_scenarios({
        "density":              density_result,
        "contradiction_flags":  contra.get("flags"),
        "stiff_core_flag":      bool(ring_check and ring_check.get("stiff_core_flag")),
        "hollow_anomaly":       bool(hollow_anomaly),
        "hollow_item":          bool(hollow_item),
        "pledging_exceeds":     bool(pledging.get("exceeds")),
        "hallmark":             hallmark_parsed,
        "n_stones":             net_gold.get("n_stones", 0),
        "stone_weight_fraction": _stone_frac,
    })

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
    if closeup_review:
        _n_seen = [r["n_detected"] for r in closeup_review["reviews"]]
        verification_trace.append({
            "step": "Close-up stone photo review",
            "status": "flag" if closeup_review["flags"] else "done",
            "summary": (
                f"{len(closeup_review['reviews'])} close-up photo(s) reviewed, "
                f"{_n_seen} stone(s) seen per photo"
                + (f" — {len(closeup_review['flags'])} count mismatch flag(s)" if closeup_review["flags"] else "")
            ),
            "formula": "Close-ups have no scale reference so they never change stone size/weight; "
                       "they may only raise a count-mismatch flag or, when unambiguous, upgrade a "
                       "vague 'colourless' stone to a confirmed type.",
            "details": {"reviews": closeup_review["reviews"], "flags": closeup_review["flags"]},
            "source": "app/utils/stone_fusion.py::review_closeup_photos",
        })
    _dedn = (
        f"Total {net_gold['gross_weight_g']} g − stones ≈ {net_gold['stone_weight_g']} g "
        f"({net_gold.get('n_stones', 0)} stone(s), ~{net_gold.get('stone_carat_total', 0)} ct) "
        f"= net gold {net_gold['net_gold_weight_g']} g. "
        if net_gold["method"] == "stone_weight_deduction"
        else f"Net gold {net_gold['net_gold_weight_g']} g ({net_gold['method']}). "
    )
    _rate_note = f"₹{ltv_result['rate_per_gram_inr']:,.0f}/g at {ltv_result['valuation_karat']}K"
    if ltv_result.get("rate_per_gram_calculated_karat") and ltv_result["valuation_karat"] != _matched_karat:
        _rate_note += (f" (physics matched {_matched_karat}K ≈ "
                       f"₹{ltv_result['rate_per_gram_calculated_karat']:,.0f}/g, for comparison only)")
    verification_trace.append({
        "step": "Net gold weight, assessed value & RBI-tiered LTV",
        "status": "done",
        "summary": (
            _dedn
            + f"{_rate_note} → value ₹{ltv_result['assessed_value_inr']:,.0f}; "
            f"LTV {ltv_result['ltv_pct']*100:.0f}% ({ltv_result['tier']}) "
            f"→ max loan ₹{ltv_result['max_loan_inr']:,.0f}"
        ),
        "formula": ("net_gold = total_weight − Σ(stone size→carat→grams);  "
                    "rate_per_gram = rate_24k_999 × BIS fineness(karat);  "
                    "assessed_value = net_gold × rate_per_gram;  max_loan = assessed_value × tiered LTV%"),
        "details": {"tier": ltv_result["tier"], "ltv_pct": ltv_result["ltv_pct"],
                    "method": net_gold["method"], **net_gold},
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

    # The calibrated frame: the card-free item photo every CV stage ran on,
    # white-balanced from the card. Saved so the officer can inspect exactly
    # what the analysis saw, with true colours.
    saved_calibrated = None
    if cv_image_bytes is not None:
        cp = case_dir / "calibrated.jpg"
        cp.write_bytes(_stamp(cv_image_bytes, "Calibrated (card-free, colour-corrected)"))
        saved_calibrated = str(cp)

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

    # Async TRELLIS 3D — status only here; generation starts after persist.
    mesh3d_meta = initial_mesh3d_status() if image_bytes_list else {
        "status": "disabled",
        "glb_url": None,
        "error": "No item photo — 3D generation skipped.",
        "updated_at": None,
    }
    if isinstance(xray_result, dict):
        xray_result["mesh3d"] = mesh3d_meta

    case = {
        "case_id":          case_id,
        "timestamp":        analysis_ts.isoformat() + "Z",
        "item_description": item_description,
        "declared_karat":   declared_karat,
        "hollow_item":      bool(hollow_item),
        "hollow_anomaly":   bool(hollow_anomaly),
        "borrower_present": bool(borrower_present),
        "branch_id":        branch_id,
        "geo":              {"lat": geo_lat, "lon": geo_lon} if geo_lat is not None else None,
        "weight_capture_source": weight_capture_source,
        "evidence_capture": {
            "image_sources": list(image_sources),
            "streak_source": streak_source if streak_image is not None else None,
            "uv_source":     uv_source if uv_image is not None else None,
            "any_uploaded":  bool(_uploaded),
            "policy":        "live_only" if os.getenv("ALLOW_UPLOAD_EVIDENCE", "1").strip() not in ("1", "true", "True") else "upload_allowed",
        },
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
            "photo_roles": {
                "item_photo_index":  item_index,
                "card_photo_index":  card_index,
                "closeup_indexes":   photo_split["other_index"],
                "colour_calibrated": wb_applied,
                "note": ("All image analysis ran on the card-free item photo; "
                         "the calibration card photo was used only for scale"
                         + (" and colour white-balance." if wb_applied else ".")),
            },
        },
        "media": {
            "images_provided": len(image_bytes_list),
            "audio_provided":  audio_bytes is not None,
            "streak_provided": streak_bytes is not None,
            "uv_provided":     uv_bytes is not None,
            "images":          saved_images,
            "calibrated":      saved_calibrated,
            "audio":           saved_audio,
            "tap_audio":       saved_tap_audio,
            "streak":          saved_streak,
            "uv":              saved_uv,
            "xray":            xray_result,
            "mesh3d":          mesh3d_meta,
        },
        "calibration": {
            # The Calibrated section shows the CALIBRATION-CARD photo — the
            # reference frame the scale (px/mm) and colour white-balance are
            # derived from. Every other stage uses the card-free item frame;
            # this is the one place the card itself is shown.
            "image": (
                saved_images[card_index]
                if (card_index is not None and card_index < len(saved_images))
                else saved_calibrated
            ),
            "calibrated_item_image": saved_calibrated,
            "item_photo_index":   item_index,
            "card_photo_index":   card_index,
            "card_detected":      bool((fiducial_result or {}).get("detected")),
            "px_per_mm":          (fiducial_result or {}).get("px_per_mm"),
            "card_side_mm":       CARD_SIDE_MM,
            "checksum_valid":     (fiducial_result or {}).get("checksum_valid"),
            "colour_calibrated":  wb_applied,
            "white_balance_gains_bgr": wb_gains,
            "scale_note": (
                f"1 mm ≈ {round((fiducial_result or {}).get('px_per_mm'), 1)} px "
                f"(from the {CARD_SIDE_MM:.0f} mm calibration card)"
                if (fiducial_result or {}).get("px_per_mm") else
                "No calibration card detected — scale reference unavailable"
            ),
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
        "stone_closeup_review": closeup_review["reviews"] if closeup_review else None,
        "ltv":              ltv_result,
        "contradiction":    contra,
        "composition":      composition_result,
        "verification_trace": verification_trace,
        "fusion":           fusion,
        "benford":          benford,
        "benford_evaluator": benford_evaluator,
        "fraud_scenarios":  fraud_scenarios,
        "approval": build_approval(
            risk_level, loan_action,
            maker_id=session["evaluator_id"], maker_name=session["name"],
        ),
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

    # Kick off TRELLIS 3D in a daemon thread AFTER the case is saved so the
    # HTTP response is not blocked. Soft-fails if token/Space unavailable.
    if image_bytes_list and mesh3d_meta.get("status") == "pending":
        try:
            # Pass every photo: the job prefers a card-free shot and masks the
            # calibration card out, so the card never enters the 3D model.
            start_mesh3d_job(case_id, image_bytes_list)
        except Exception as e:
            logger.warning("Could not start mesh3d job for case %s: %s", case_id, e)

    return JSONResponse(content=safe_case)
