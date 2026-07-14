"""
GET /api/kyc/lookup — mock core-banking KYC lookup. SIMULATED.

Neither UIDAI's Aadhaar eKYC API nor NSDL/PAN verification is open to
third-party developers without AUA/KUA or comparable institutional
licensing. The realistic production path for a bank pilot is not KANCHAN
verifying Aadhaar/PAN itself — it's pulling an ALREADY-KYC'd customer record
from Canara's own core banking system, which holds it already. This mocks
that lookup contract against a small local dataset.
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.auth import require_login

logger = logging.getLogger(__name__)
router = APIRouter()

MOCK_PATH = Path("data/mock_core_banking.json")


def _load() -> dict:
    if MOCK_PATH.exists():
        try:
            return json.loads(MOCK_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read mock core banking data: %s", e)
    return {}


@router.get("/kyc/lookup")
async def kyc_lookup(account_no: str, session: dict = Depends(require_login)):
    record = _load().get(account_no.strip())
    if not record:
        return JSONResponse(status_code=404, content={
            "found": False, "simulated": True,
            "note": "No matching record in the simulated core-banking dataset.",
        })
    return JSONResponse(content={
        "found": True, "simulated": True, "record": record,
        "note": (
            "SIMULATED — pending Canara core-banking integration. Aadhaar/PAN "
            "verification is not independently performed by KANCHAN; this "
            "reproduces the lookup contract for an already-KYC'd customer "
            "record from the bank's own system."
        ),
    })
