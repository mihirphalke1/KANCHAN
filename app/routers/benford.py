from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.benford.monitor import run_benford_test
from app.main import _numpy_safe

router = APIRouter()


@router.get("/benford")
async def benford_status(branch_id: str = None, min_samples: int = 30):
    return JSONResponse(content=_numpy_safe(run_benford_test(branch_id=branch_id, min_samples=min_samples)))
