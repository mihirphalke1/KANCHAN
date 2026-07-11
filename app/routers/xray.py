"""
POST /api/xray — standalone DSIP pseudo X-ray with adjustable thresholds.

Lets the frontend re-run the pipeline on an image with custom T1/T2/T3
without creating a case. Stages come back as base64 data URIs.
"""
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.utils.xray import xray_preview

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/xray")
async def xray(
    image: UploadFile = File(...),
    t1: Optional[int] = Form(None),
    t2: Optional[int] = Form(None),
    t3: Optional[int] = Form(None),
):
    raw = await image.read()
    try:
        result = xray_preview(raw, t1, t2, t3)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("X-ray pipeline failed")
        raise HTTPException(status_code=500, detail=f"X-ray pipeline failed: {e}")
    return JSONResponse(content=result)
