from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.benford.monitor import run_benford_test
from app.utils.numpy_safe import numpy_safe as _numpy_safe

router = APIRouter()


@router.get("/benford")
async def benford_status(
    branch_id: str = None,
    evaluator_id: str = None,
    min_samples: int = 30,
):
    """Benford first-digit monitor on the branch density log.

    Pass `evaluator_id` to slice the test to a single officer's appraisals —
    the fraud-relevant view that localises a systematic anomaly to one corrupt
    evaluator instead of diluting it across the whole branch.
    """
    return JSONResponse(content=_numpy_safe(
        run_benford_test(branch_id=branch_id, evaluator_id=evaluator_id, min_samples=min_samples)
    ))
