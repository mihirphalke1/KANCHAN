"""
POST /api/hallmark/verify — hallmark engraving check + mock BIS HUID lookup.
SIMULATED — see app/utils/bis_registry.py for why.
"""
import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.auth import require_login
from app.models.hallmark_model import analyze_hallmark_engraving
from app.utils.bis_registry import lookup_huid

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/hallmark/verify")
async def verify_hallmark(
    huid: str = Form(""),
    declared_karat: int = Form(0),
    hallmark_image: UploadFile = File(...),
    session: dict = Depends(require_login),
):
    image_bytes = await hallmark_image.read()
    engraving = analyze_hallmark_engraving(image_bytes)

    registry_record = lookup_huid(huid) if huid else None
    cross_check = None
    if registry_record and declared_karat:
        cross_check = {
            "registry_purity_karat": registry_record["purity_karat"],
            "declared_karat":        declared_karat,
            "match":                 registry_record["purity_karat"] == declared_karat,
        }

    return JSONResponse(content={
        "huid":            huid or None,
        "simulated":       True,
        "registry_found":  registry_record is not None,
        "registry_record": registry_record,
        "engraving":       engraving,
        "cross_check":     cross_check,
        "note": (
            "BIS has no public developer API — this lookup runs against a local "
            "simulated registry reproducing the real data contract, pending a BIS "
            "integration/MOU. The HUID is never trusted alone: cross-check against "
            "the physics-measured karat (density test) before relying on it."
        ),
    })
