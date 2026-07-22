"""
GET  /api/mesh3d/{case_id}       — poll async TRELLIS 3D job status
POST /api/mesh3d/{case_id}/retry — re-queue generation from saved img_0
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.auth import require_login
from app.utils.mesh3d import (
    find_case_image_bytes,
    get_mesh3d_status,
    start_mesh3d_job,
    write_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/mesh3d/{case_id}")
async def mesh3d_status(case_id: str, session: dict = Depends(require_login)):
    status = get_mesh3d_status(case_id)
    return JSONResponse(content=status)


@router.post("/mesh3d/{case_id}/retry")
async def mesh3d_retry(case_id: str, session: dict = Depends(require_login)):
    current = get_mesh3d_status(case_id)
    if current.get("status") in ("pending", "generating"):
        return JSONResponse(content={
            **current,
            "message": "Generation is already in progress.",
        })

    image_bytes = find_case_image_bytes(case_id)
    if not image_bytes:
        raise HTTPException(
            status_code=404,
            detail=f"No saved item photo found for case {case_id} — cannot retry 3D generation.",
        )

    write_status(case_id, {
        "status": "pending",
        "glb_url": None,
        "error": None,
        "message": "Retry queued.",
        "provider": current.get("provider"),
    })
    status = start_mesh3d_job(case_id, image_bytes)
    return JSONResponse(content=status)
