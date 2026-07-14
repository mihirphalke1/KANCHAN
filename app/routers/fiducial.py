"""
GET /api/fiducial/today — printable daily calibration card for a branch.
"""
from datetime import date

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from app.auth import require_login
from app.utils.fiducial import generate_marker_png

router = APIRouter()


@router.get("/fiducial/today")
async def fiducial_today(branch_id: str = "BLR-001", session: dict = Depends(require_login)):
    png_bytes = generate_marker_png(branch_id, date.today())
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="kanchan-calibration-{branch_id}-{date.today().isoformat()}.png"'},
    )
