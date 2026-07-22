"""
KANCHAN-AI - Spurious Gold Intelligence System
FastAPI application entry point.
"""
import logging
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# Python's mimetypes module maps .webm -> video/webm by default, even for an
# audio-only recording (what the browser's MediaRecorder produces for the
# tap test) — some browsers refuse to play an <audio> element served with a
# video/* Content-Type. Registered before the /cases StaticFiles mount below
# picks it up.
mimetypes.add_type("audio/webm", ".webm")

from app.utils.numpy_safe import numpy_safe as _numpy_safe
from app.rate_limit import limiter

from app.routers import admin, analyze, auth, benford, fiducial, gold_rate, hallmark, history, kyc, mesh3d, report, xray

app = FastAPI(
    title="KANCHAN-AI",
    description="Spurious Gold Intelligence System — Canara Bank / SuRaksha Hackathon 2.0",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router,     prefix="/api")
app.include_router(analyze.router,   prefix="/api")
app.include_router(auth.router,      prefix="/api")
app.include_router(benford.router,   prefix="/api")
app.include_router(fiducial.router,  prefix="/api")
app.include_router(gold_rate.router, prefix="/api")
app.include_router(hallmark.router,  prefix="/api")
app.include_router(history.router,   prefix="/api")
app.include_router(kyc.router,       prefix="/api")
app.include_router(mesh3d.router,    prefix="/api")
app.include_router(report.router,    prefix="/api")
app.include_router(xray.router,      prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "kanchan-ai"}


# Saved case evidence (photos, tap-test audio, X-ray stage images) lives
# under data/cases/<case_id>/ — served at /cases so History/PDF can show it.
CASES_DIR = Path("data/cases")
CASES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/cases", StaticFiles(directory=str(CASES_DIR)), name="case_media")

class SPAStaticFiles(StaticFiles):
    """Serve the Vite build, falling back to index.html for any path that
    isn't a real file — react-router routes (/login, /dashboard, /admin, ...)
    only exist client-side, so a direct navigation, refresh, or PWA launch
    (manifest start_url) needs the SPA shell returned instead of a 404.
    Starlette's StaticFiles raises HTTPException(404) rather than returning
    a 404 response, so the fallback has to catch it, not check a status code."""
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


FRONTEND_DIST = Path("frontend/dist")
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
