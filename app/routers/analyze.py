"""
POST /api/analyze — main analysis endpoint.
"""
import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from app.models.acoustic_model import analyze_acoustic
from app.models.acoustic_physics import extract_ring_frequency, ring_frequency_check
from app.models.contradiction import contradiction_summary
from app.models.density_model import analyze_density
from app.models.fusion_model import analyze_fusion
from app.models.image_model import analyze_image
from app.models.streak_model import analyze_streak
from app.models.xray_model import analyze_xray, VISUAL_BLEND_IMAGE, VISUAL_BLEND_XRAY
from app.utils.composition import analyze_composition
from app.benford.monitor import append_density_reading, run_benford_test
from app.utils.xray import xray_preview
from app.llm.verdict_prompt import generate_verdict

logger = logging.getLogger(__name__)
router = APIRouter()

HISTORY_PATH = Path("data/case_history.json")


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
    policy: dict | None = None,
) -> tuple[str, str, str]:
    """
    Return (risk_level, confidence, loan_action) per the declared policy.
    - Density risk ≥ override threshold → REJECT (physics override)
    - Contradiction boost catches mixed-signal composites
    """
    p = policy or load_policy()
    if density_risk >= p["density_override_at"]:
        return "REJECT", "HIGH", "DECLINE"

    boosted = min(1.0, fusion_risk + contra_score * p["contradiction_boost"])

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
    }.get(density_result["karat_verdict"], density_result["karat_verdict"])
    steps.append({
        "step": "Does the density match the declared karat?",
        "status": "flag" if density_result["risk_score"] > 0.5 else "done",
        "summary": f"{verdict_text}; chance it matches: "
                   f"{round(density_result['conformity_probability'] * 100, 1)}%",
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
                        else "Backdrop NOT separable — stats include background"),
            "formula": "Backdrop colour from frame border → weighted HSV distance; border-touching components discarded",
            "details": {"stage_image": "material"},
            "source": "Classical CV (no ML) — thresholds in app/utils/xray.py",
        })
        steps.append({
            "step": "Finding the stones in the photo",
            "status": "done",
            "summary": f"{xray_result.get('gem_regions', 0)} coloured "
                       f"({xray_result.get('gem_area_pct', 0)}%), "
                       f"{xray_result.get('colourless_regions', 0)} colourless candidates "
                       f"({xray_result.get('colourless_area_pct', 0)}%)",
            "formula": "Coloured: saturated non-gold-hue clusters. Colourless: bright low-sat regions, "
                       "classified round+smooth (pearl-like) vs micro-edge sparkle (faceted)",
            "details": {"stage_image": "gems"},
            "source": "Independent CV detection — description text is never trusted",
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
        "formula": "density/physics overrides first; else combined risk + contradiction boost, banded per the declared policy",
        "details": {"boosted risk": boosted_risk},
        "source": "data/decision_policy.json — bank risk-appetite dials, auditable",
    })
    return steps


@router.post("/analyze")
async def analyze(
    response: Response,
    item_description: str = Form(...),
    declared_karat: int = Form(...),
    weight_dry: float = Form(...),
    weight_submerged: float = Form(...),
    water_temp_c: float = Form(25.0),
    branch_id: str = Form("default"),
    customer_name: str = Form(""),
    customer_account: str = Form(""),
    loan_app_no: str = Form(""),
    officer_name: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    audio: Optional[UploadFile] = File(default=None),
    streak_image: Optional[UploadFile] = File(default=None),
):
    case_id = uuid.uuid4().hex[:8]
    response.headers["X-Case-ID"] = case_id

    if declared_karat not in (14, 18, 22, 24):
        raise HTTPException(status_code=422, detail="declared_karat must be 14, 18, 22, or 24")
    if weight_dry <= 0 or weight_submerged <= 0:
        raise HTTPException(status_code=422, detail="Weights must be positive")
    if weight_submerged >= weight_dry:
        raise HTTPException(status_code=422, detail="Submerged weight must be less than dry weight")
    if not 0.0 < water_temp_c < 45.0:
        raise HTTPException(status_code=422, detail="water_temp_c must be between 0 and 45 °C")

    image_bytes_list = [await img.read() for img in images] if images else []
    audio_bytes      = await audio.read() if audio else None
    streak_bytes     = await streak_image.read() if streak_image else None

    # ── Material scan on the primary photograph ──
    # No media is written to disk: the processed stages travel inside the
    # response as embedded images; the browser shows the uploaded photos
    # from its own copies. The PDF report degrades gracefully without files.
    xray_result = None
    if image_bytes_list:
        try:
            xray_result = xray_preview(image_bytes_list[0])
        except Exception as e:
            logger.warning("Material scan failed for case %s: %s", case_id, e)

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
    streak_result   = analyze_streak(streak_bytes)

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
    xray_result_score = analyze_xray(xray_result, item_description)

    # ── Stone-corrected composition (two-component mixture model) ──
    # Uses the CAMERA-detected stone fraction, never the description. The
    # adjusted density risk replaces the raw one downstream so a genuine
    # stone-set item is not rejected for the stones it visibly carries;
    # readings above the gold band are never softened.
    composition_result = None
    density_risk_effective = density_result["risk_score"]
    if (
        xray_result
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
                declared_karat   = declared_karat,
                gems             = all_stones,
                gem_area_pct     = xray_result.get("gem_area_pct", 0.0)
                                   + xray_result.get("colourless_area_pct", 0.0),
            )
            density_risk_effective = composition_result["adjusted_density_risk"]
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

    policy = load_policy()
    boosted_risk = round(
        min(1.0, fusion["risk_score"] + contra["contradiction_score"] * policy["contradiction_boost"]), 4)
    risk_level, confidence, loan_action = _determine_verdict(
        fusion["risk_score"],
        contra["contradiction_score"],
        density_risk=density_risk_effective,
        policy=policy,
    )

    # Physics override: density says gold, ring pitch says stiffer-than-gold
    # core — the one signature no filler metal can avoid. Calibration-gated.
    override_reason = None
    if ring_check and ring_check.get("stiff_core_flag"):
        risk_level, confidence, loan_action = "REJECT", "HIGH", "DECLINE"
        override_reason = ring_check["reason"]
        contra["flags"].append("density↔ring-pitch: " + ring_check["reason"])

    llm_payload = {
        "item_description": item_description,
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

    case = {
        "case_id":          case_id,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "item_description": item_description,
        "declared_karat":   declared_karat,
        "branch_id":        branch_id,
        "customer": {
            "name":        customer_name,
            "account_no":  customer_account,
            "loan_app_no": loan_app_no,
            "officer_name": officer_name,
        },
        "media": {
            "images_provided": len(image_bytes_list),
            "audio_provided":  audio_bytes is not None,
            "streak_provided": streak_bytes is not None,
            "xray":            xray_result,
        },
        "modality_scores": {
            "image":    image_result,
            "density":  density_result,
            "acoustic": acoustic_result,
            "streak":   streak_result,
            "xray":     xray_result_score,
        },
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
            "boosted_risk":  boosted_risk,
            "plain_english": llm_result["plain_english"],
            "action":        llm_result["action"],
            "llm_provider":  llm_result["llm_provider"],
        },
    }

    from app.main import _numpy_safe
    safe_case = _numpy_safe(case)

    # History keeps numbers only — the embedded stage images stay in the
    # response and are never persisted (no local media storage).
    persist_case = json.loads(json.dumps(safe_case))
    if persist_case.get("media", {}).get("xray"):
        persist_case["media"]["xray"] = {
            k: v for k, v in persist_case["media"]["xray"].items() if k != "stages"
        }
    _save_case(persist_case)
    return JSONResponse(content=safe_case)
