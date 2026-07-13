"""
KANCHAN-AI - Spurious Gold Intelligence System
FastAPI application entry point.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

import json
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def _numpy_safe(data):
    """Round-trip through JSON to convert all numpy scalars to Python primitives."""
    return json.loads(json.dumps(data, cls=_NumpyEncoder))

from app.routers import analyze, benford, history, report, xray

app = FastAPI(
    title="KANCHAN-AI",
    description="Spurious Gold Intelligence System — Canara Bank / SuRaksha Hackathon 2.0",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router,  prefix="/api")
app.include_router(benford.router,  prefix="/api")
app.include_router(history.router,  prefix="/api")
app.include_router(report.router,   prefix="/api")
app.include_router(xray.router,     prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "kanchan-ai"}


# Saved case evidence (photos, tap-test audio, X-ray stage images) lives
# under data/cases/<case_id>/ — served at /cases so History/PDF can show it.
CASES_DIR = Path("data/cases")
CASES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/cases", StaticFiles(directory=str(CASES_DIR)), name="case_media")

FRONTEND_DIST = Path("frontend/dist")
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
