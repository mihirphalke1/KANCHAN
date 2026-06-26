"""
POST /api/analyze — main analysis endpoint.
"""
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from app.models.acoustic_model import analyze_acoustic
from app.models.contradiction import contradiction_summary
from app.models.density_model import analyze_density
from app.models.fusion_model import analyze_fusion
from app.models.image_model import analyze_image
from app.models.streak_model import analyze_streak
from app.benford.monitor import append_density_reading, run_benford_test
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


def _determine_verdict(
    fusion_risk: float, contra_score: float, density_risk: float = 0.0
) -> tuple[str, str, str]:
    """
    Return (risk_level, confidence, loan_action).
    Rules:
    - Density risk ≥ 0.85 always → REJECT (density is the most reliable signal)
    - Contradiction boost 0.40× catches tungsten-core
    """
    if density_risk >= 0.85:
        return "REJECT", "HIGH", "DECLINE"

    boosted = min(1.0, fusion_risk + contra_score * 0.40)

    if boosted < 0.25:
        return "GENUINE", "HIGH", "APPROVE"
    elif boosted < 0.45:
        return "GENUINE", "MEDIUM", "APPROVE"
    elif boosted < 0.60:
        return "BORDERLINE", "MEDIUM", "HOLD"
    elif boosted < 0.75:
        return "BORDERLINE", "LOW", "HOLD"
    else:
        return "REJECT", "HIGH", "DECLINE"


@router.post("/analyze")
async def analyze(
    response: Response,
    item_description: str = Form(...),
    declared_karat: int = Form(...),
    weight_dry: float = Form(...),
    weight_submerged: float = Form(...),
    branch_id: str = Form("default"),
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

    image_bytes_list = [await img.read() for img in images] if images else []
    audio_bytes      = await audio.read() if audio else None
    streak_bytes     = await streak_image.read() if streak_image else None

    density_result  = analyze_density(weight_dry, weight_submerged, declared_karat)
    image_result    = analyze_image(image_bytes_list)
    acoustic_result = analyze_acoustic(audio_bytes) if audio_bytes else {"risk_score": 0.5, "confidence": "low", "mode": "no_audio"}
    streak_result   = analyze_streak(streak_bytes)

    # Exclude modalities with no real data from contradiction — a missing audio
    # signal shouldn't create a false contradiction with density.
    contra_scores = {"density": density_result["risk_score"]}
    if acoustic_result.get("mode", "no_audio") not in ("no_audio",):
        contra_scores["acoustic"] = acoustic_result["risk_score"]
    if image_result.get("mode", "no_images") not in ("no_images",):
        contra_scores["image"] = image_result["risk_score"]
    if streak_result.get("mode", "no_streak") not in ("no_streak",):
        contra_scores["streak"] = streak_result["risk_score"]

    contra = contradiction_summary(contra_scores)

    fusion = analyze_fusion(
        image_risk    = image_result["risk_score"],
        density_risk  = density_result["risk_score"],
        acoustic_risk = acoustic_result["risk_score"],
        streak_risk   = streak_result["risk_score"],
    )

    benford = run_benford_test()

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

    risk_level, confidence, loan_action = _determine_verdict(
        fusion["risk_score"],
        contra["contradiction_score"],
        density_risk=density_result["risk_score"],
    )

    llm_payload = {
        "item_description": item_description,
        "declared_karat":   declared_karat,
        "fusion_risk":      fusion["risk_score"],
        "modality_scores": {
            "image":    image_result,
            "density":  density_result,
            "acoustic": acoustic_result,
            "streak":   streak_result,
        },
        "contradiction":   contra,
        "density_details": density_result,
    }
    llm_result = await generate_verdict(llm_payload)

    case = {
        "case_id":          case_id,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "item_description": item_description,
        "declared_karat":   declared_karat,
        "branch_id":        branch_id,
        "modality_scores": {
            "image":    image_result,
            "density":  density_result,
            "acoustic": acoustic_result,
            "streak":   streak_result,
        },
        "contradiction":    contra,
        "fusion":           fusion,
        "benford":          benford,
        "verdict": {
            "risk_level":    risk_level,
            "confidence":    confidence,
            "loan_action":   loan_action,
            "fusion_risk":   round(float(fusion["risk_score"]), 4),
            "plain_english": llm_result["plain_english"],
            "action":        llm_result["action"],
            "llm_provider":  llm_result["llm_provider"],
        },
    }

    from app.main import _numpy_safe
    safe_case = _numpy_safe(case)
    _save_case(safe_case)
    return JSONResponse(content=safe_case)
