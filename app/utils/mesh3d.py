"""
Async image→3D mesh generation via public Hugging Face Gradio Spaces.

Primary provider: Tencent Hunyuan3D-2 (reliable public Space, returns GLB).
Fallbacks: Microsoft TRELLIS.2, trellis-community/TRELLIS (when available).

Additive / non-decision path: never feeds fusion or loan risk. Failures are
soft — analysis still succeeds if the Space is cold, queued, or unreachable.

Job status lives in data/cases/<case_id>/mesh3d/status.json so polling does
not thrash the hash chain; when the job reaches a terminal state we also
patch case_history.json (with restamp_from) so History can reopen a ready GLB.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CASES_DIR = Path("data/cases")
HISTORY_PATH = Path("data/case_history.json")

USE_MESH3D = os.getenv("USE_MESH3D", "1").strip() not in ("0", "false", "False")
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
# Comma-separated Space IDs tried in order. Hunyuan3D-2 is primary because the
# public TRELLIS Spaces frequently raise opaque ZeroGPU / CONFIG errors.
MESH3D_SPACES = [
    s.strip()
    for s in os.getenv(
        "MESH3D_SPACES",
        os.getenv("TRELLIS_SPACE", "")  # single-space override still supported
        or "tencent/Hunyuan3D-2,Microsoft/TRELLIS.2,trellis-community/TRELLIS",
    ).split(",")
    if s.strip()
]
MESH3D_TIMEOUT_S = float(os.getenv("MESH3D_TIMEOUT_S", "600"))
# Displayed provider label (updated when a Space succeeds).
DEFAULT_PROVIDER = MESH3D_SPACES[0] if MESH3D_SPACES else "tencent/Hunyuan3D-2"

# In-process lock so two retries don't double-hit the Space for one case.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _case_mesh_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id / "mesh3d"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status_path(case_id: str) -> Path:
    return _case_mesh_dir(case_id) / "status.json"


def _glb_path(case_id: str) -> Path:
    return _case_mesh_dir(case_id) / "model.glb"


def _glb_url(case_id: str) -> str:
    return f"/cases/{case_id}/mesh3d/model.glb"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _case_lock(case_id: str) -> threading.Lock:
    with _locks_guard:
        if case_id not in _locks:
            _locks[case_id] = threading.Lock()
        return _locks[case_id]


def _decode_bgr(image_bytes: bytes):
    import cv2
    import numpy as np
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _select_mesh_image(images: list[bytes]) -> bytes:
    """Pick the best photo to reconstruct in 3D. The calibration card must
    never become part of the model, so a photo WITHOUT a card (a clean item
    shot) is always preferred; only if every photo carries the card do we fall
    back to the first and mask the card out in _prepare_mesh_image."""
    if not images:
        raise ValueError("no images")
    try:
        from app.utils.fiducial import locate_card
        for b in images:
            try:
                bgr = _decode_bgr(b)
                if bgr is not None and locate_card(bgr) is None:
                    return b            # first card-free photo wins
            except Exception:
                continue
    except Exception as e:
        logger.info("mesh image selection skipped (%s)", e)
    return images[0]


def _prepare_mesh_image(image_bytes: bytes) -> bytes:
    """Remove the calibration card from the frame so the Space reconstructs
    only the jewellery. The card is detected geometrically (its finder
    squares), its region is painted over with the background colour, and the
    frame is cropped to the remaining item so it fills the view. Soft — returns
    the original bytes unchanged on any miss (no card, decode failure, etc.)."""
    try:
        import cv2
        import numpy as np
        from app.utils.fiducial import locate_card

        bgr = _decode_bgr(image_bytes)
        if bgr is None:
            return image_bytes
        H, W = bgr.shape[:2]

        # Strip the forensic caption bar burned onto saved evidence photos
        # (a dark full-width band at the bottom) so it isn't reconstructed as a
        # slab. Live-analysis images have no stamp, so this is a no-op there.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark_frac = (gray < 60).mean(axis=1)
        cut = H
        for y in range(H - 1, int(H * 0.80), -1):
            if dark_frac[y] > 0.6:
                cut = y
            else:
                break
        if cut < H - 2:
            bgr = bgr[:cut, :, :]
            H = bgr.shape[0]

        card = locate_card(bgr)
        if not card or not card.get("bbox"):
            # No card: still crop to the item so it fills the frame (helps
            # small items reconstruct in more detail). Reuses the crop below.
            card_bbox = None
        else:
            card_bbox = [int(v) for v in card["bbox"]]

        # Background colour from a thin frame border (away from the item).
        border = np.concatenate([
            bgr[:6, :, :].reshape(-1, 3), bgr[-6:, :, :].reshape(-1, 3),
            bgr[:, :6, :].reshape(-1, 3), bgr[:, -6:, :].reshape(-1, 3),
        ])
        bg = np.median(border, axis=0).astype(np.uint8)

        if card_bbox is not None:
            x0, y0, x1, y1 = card_bbox
            x0, y0 = max(0, x0), max(0, min(H, y0))
            x1, y1 = min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                bgr[y0:y1, x0:x1] = bg           # erase the card

        # Crop to the item: the largest non-background, non-card blob.
        diff = np.linalg.norm(bgr.astype(np.int16) - bg.astype(np.int16), axis=2)
        fg = (diff > 32).astype(np.uint8)
        if card_bbox is not None:
            fg[max(0, card_bbox[1]):min(H, card_bbox[3]),
               max(0, card_bbox[0]):min(W, card_bbox[2])] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            # largest component (excluding background label 0) = the item
            areas = stats[1:, cv2.CC_STAT_AREA]
            k = 1 + int(np.argmax(areas))
            if stats[k, cv2.CC_STAT_AREA] > 200:
                bx = stats[k, cv2.CC_STAT_LEFT]; by = stats[k, cv2.CC_STAT_TOP]
                bw = stats[k, cv2.CC_STAT_WIDTH]; bh = stats[k, cv2.CC_STAT_HEIGHT]
                pad = int(0.15 * max(bw, bh))
                cx0, cy0 = max(0, bx - pad), max(0, by - pad)
                cx1, cy1 = min(W, bx + bw + pad), min(H, by + bh + pad)
                if cx1 - cx0 > 20 and cy1 - cy0 > 20:
                    bgr = bgr[cy0:cy1, cx0:cx1]

        ok, buf = cv2.imencode(".png", bgr)
        return buf.tobytes() if ok else image_bytes
    except Exception as e:
        logger.warning("mesh image preparation failed (%s) — using original", e)
        return image_bytes


def _case_stones(case_id: str) -> list:
    """Detected set stones for a case, from the saved case record — used to
    colour the generated mesh (gem colour + presence). Empty on any miss."""
    for case in _load_history():
        if case.get("case_id") != case_id:
            continue
        media = case.get("media") or {}
        xray = media.get("xray") or {}
        stones = xray.get("stones") or xray.get("gems") or []
        return stones if isinstance(stones, list) else []
    return []


def _colorize_ready_glb(case_id: str, glb_dest: Path) -> Optional[dict]:
    """Colour the freshly generated white GLB in place: gold metal body + the
    detected gem, sampled from the case photo. Soft — on any failure the
    original white GLB is left untouched and None is returned. Never raises."""
    try:
        from app.utils.mesh_color import colorize_glb

        case_dir = CASES_DIR / case_id
        image_path = None
        for name in ("img_0.jpg", "img_0.png", "img_0.jpeg"):
            p = case_dir / name
            if p.exists():
                image_path = p
                break

        tmp = glb_dest.with_suffix(".colored.glb")
        meta = colorize_glb(
            glb_dest, tmp,
            stones=_case_stones(case_id),
            image_path=image_path,
        )
        # Keep the untouched white mesh for retry/debug, then swap colour in.
        raw = glb_dest.with_name("model_raw.glb")
        if not raw.exists():
            shutil.copy2(glb_dest, raw)
        shutil.move(str(tmp), str(glb_dest))
        logger.info("mesh3d coloured for case %s: %s", case_id, meta)
        return meta
    except Exception as e:
        logger.warning("mesh3d colouring failed for case %s (%s) — keeping white model", case_id, e)
        return None


def initial_mesh3d_status() -> dict:
    """Status blob attached to the analyze response / persisted case."""
    if not USE_MESH3D:
        return {
            "status": "disabled",
            "glb_url": None,
            "error": "3D model generation is turned off (USE_MESH3D=0).",
            "updated_at": _now_iso(),
            "provider": DEFAULT_PROVIDER,
        }
    if not _runtime_token():
        return {
            "status": "disabled",
            "glb_url": None,
            "error": "HF_TOKEN is not set — add a Hugging Face token to enable 3D generation.",
            "updated_at": _now_iso(),
            "provider": DEFAULT_PROVIDER,
        }
    return {
        "status": "pending",
        "glb_url": None,
        "error": None,
        "updated_at": _now_iso(),
        "provider": DEFAULT_PROVIDER,
    }


def write_status(case_id: str, payload: dict) -> dict:
    payload = {**payload, "updated_at": _now_iso(), "case_id": case_id}
    path = _status_path(case_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_status(case_id: str) -> dict:
    path = _status_path(case_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # If GLB exists on disk but status lagged, promote to ready.
                glb = _glb_path(case_id)
                if glb.exists() and data.get("status") != "ready":
                    data["status"] = "ready"
                    data["glb_url"] = _glb_url(case_id)
                    data["error"] = None
                    write_status(case_id, data)
                return data
        except Exception as e:
            logger.warning("Could not read mesh3d status for %s: %s", case_id, e)

    # Fall back to case_history media.xray.mesh3d / media.mesh3d
    hist = _load_history()
    for case in hist:
        if case.get("case_id") != case_id:
            continue
        media = case.get("media") or {}
        xray = media.get("xray") or {}
        mesh = xray.get("mesh3d") or media.get("mesh3d")
        if isinstance(mesh, dict):
            return {**mesh, "case_id": case_id}
        break

    return {
        "case_id": case_id,
        "status": "disabled",
        "glb_url": None,
        "error": "No 3D job found for this case.",
        "updated_at": _now_iso(),
    }


def get_mesh3d_status(case_id: str) -> dict:
    return read_status(case_id)


def _load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _patch_case_history(case_id: str, mesh3d: dict) -> None:
    """Write terminal mesh3d status into the saved case and restamp the chain."""
    try:
        from app.utils.hashchain import restamp_from
    except Exception as e:
        logger.warning("hashchain import failed while patching mesh3d: %s", e)
        return

    history = _load_history()
    for idx, case in enumerate(history):
        if case.get("case_id") != case_id:
            continue
        media = case.setdefault("media", {})
        xray = media.get("xray")
        if not isinstance(xray, dict):
            xray = {}
            media["xray"] = xray
        xray["mesh3d"] = {
            "status": mesh3d.get("status"),
            "glb_url": mesh3d.get("glb_url"),
            "error": mesh3d.get("error"),
            "updated_at": mesh3d.get("updated_at"),
            "provider": mesh3d.get("provider", DEFAULT_PROVIDER),
        }
        media["mesh3d"] = xray["mesh3d"]
        try:
            restamp_from(history, idx)
            HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not restamp history after mesh3d update for %s: %s", case_id, e)
        return


def _runtime_spaces() -> list[str]:
    """Re-read Space list each job so .env changes apply without code edits."""
    raw = os.getenv("MESH3D_SPACES") or os.getenv("TRELLIS_SPACE") or ""
    spaces = [s.strip() for s in raw.split(",") if s.strip()]
    return spaces or [
        "tencent/Hunyuan3D-2",
        "Microsoft/TRELLIS.2",
        "trellis-community/TRELLIS",
    ]


def _runtime_token() -> str:
    return (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or HF_TOKEN or "").strip()


def _call_trellis_space(image_bytes: bytes, work_dir: Path) -> tuple[Path, str]:
    """
    Try configured HF Spaces in order until one returns a GLB.
    Returns (local_glb_path, provider_id).
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError as e:
        raise RuntimeError(
            "gradio_client is not installed — run: pip install gradio_client"
        ) from e

    img_path = work_dir / "input.png"
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        # Cap long side so uploads stay snappy on public Spaces.
        max_side = 768
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        im.save(img_path, format="PNG")
    except Exception:
        img_path = work_dir / "input.jpg"
        img_path.write_bytes(image_bytes)

    errors: list[str] = []
    spaces = _runtime_spaces()
    token = _runtime_token()
    for space in spaces:
        try:
            logger.info("mesh3d trying Space %s", space)
            glb = _generate_on_space(space, img_path, work_dir, Client, handle_file, token)
            return glb, space
        except Exception as e:
            msg = f"{space}: {type(e).__name__}: {e}"
            logger.warning("mesh3d Space failed — %s", msg)
            errors.append(msg)

    raise RuntimeError(
        "All image→3D Spaces failed. "
        + " | ".join(errors[:3])
    )


def _generate_on_space(space: str, img_path: Path, work_dir: Path, Client, handle_file, token: str) -> Path:
    client = Client(
        space,
        token=token or None,
        httpx_kwargs={"timeout": max(120.0, MESH3D_TIMEOUT_S)},
    )
    api_names: set[str] = set()
    try:
        info = client.view_api(return_format="dict")
        api_names = set(((info or {}).get("named_endpoints") or {}).keys())
    except Exception as e:
        logger.info("Could not list API for %s: %s", space, e)

    def has(name: str) -> bool:
        return (not api_names) or (name in api_names)

    file_ref = handle_file(str(img_path))

    # ── Hunyuan3D-2 (primary — verified working) ──────────────────────────
    if has("/shape_generation"):
        result = client.predict(
            None,                 # caption
            file_ref,             # image
            None, None, None, None,
            20,                   # steps
            5.0,                  # guidance
            1234,                 # seed
            128,                  # octree_resolution (fast)
            True,                 # rembg
            8000,                 # chunks
            False,                # randomize_seed
            api_name="/shape_generation",
        )
        return _pick_glb(result, work_dir)

    if has("/generation_all"):
        result = client.predict(
            None, file_ref, None, None, None, None,
            20, 5.0, 1234, 128, True, 8000, False,
            api_name="/generation_all",
        )
        return _pick_glb(result, work_dir)

    # ── TRELLIS.2 / community TRELLIS ─────────────────────────────────────
    if has("/start_session"):
        try:
            client.predict(api_name="/start_session")
        except Exception:
            pass

    processed = file_ref
    if has("/preprocess_image"):
        try:
            pre = client.predict(file_ref, api_name="/preprocess_image")
            if pre is not None:
                processed = pre
        except Exception as e:
            logger.info("preprocess skipped on %s: %s", space, e)

    if has("/generate_and_extract_glb"):
        result = client.predict(
            processed, [], 0, 7.5, 12, 3.0, 12, "stochastic", 0.95, 1024,
            api_name="/generate_and_extract_glb",
        )
        return _pick_glb(result, work_dir)

    # Microsoft TRELLIS.2: image_to_3d (preview) then extract_glb
    if has("/image_to_3d") and has("/extract_glb"):
        # TRELLIS.2 signature (no multiimage args)
        try:
            client.predict(
                processed, 0, "512",
                7.5, 0.7, 12, 5.0,
                7.5, 0.5, 12, 3.0,
                1.0, 0.0, 12, 3.0,
                api_name="/image_to_3d",
            )
        except Exception:
            # Older TRELLIS multi-arg signature
            client.predict(
                processed, None, False, 0, 7.5, 12, 3.0, 12, "stochastic",
                api_name="/image_to_3d",
            )
        try:
            glb_result = client.predict(100000, 1024, api_name="/extract_glb")
        except Exception:
            glb_result = client.predict(0.95, 1024, api_name="/extract_glb")
        return _pick_glb(glb_result, work_dir)

    if has("/handle_image_to_3d"):
        result = client.predict(
            processed, 0, True, "512",
            7.5, 0.7, 12, 5.0,
            7.5, 0.5, 12, 3.0,
            1.0, 0.0, 12, 3.0,
            100000, 1024,
            api_name="/handle_image_to_3d",
        )
        return _pick_glb(result, work_dir)

    raise RuntimeError(
        f"Space '{space}' has no known image→GLB endpoint "
        f"(saw: {sorted(api_names) or 'unknown'})."
    )


def _pick_glb(result: Any, work_dir: Path) -> Path:
    """Walk a Gradio result (tuple/list/dict/str) and materialize the first GLB."""
    items = list(result) if isinstance(result, (list, tuple)) else [result]
    # Prefer later items (download buttons often come last).
    last_err: Optional[Exception] = None
    for candidate in reversed(items):
        try:
            return _materialize_glb(candidate, work_dir)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"No GLB in Space response ({last_err})"
    )


def _materialize_glb(glb_file: Any, work_dir: Path) -> Path:
    """Normalize gradio_client file result into a real .glb path under work_dir."""
    path: Optional[Path] = None
    if isinstance(glb_file, str) and glb_file:
        # HTML iframe responses are not meshes.
        if "<iframe" in glb_file or glb_file.strip().startswith("<"):
            raise RuntimeError("Got HTML preview, not a GLB file")
        path = Path(glb_file)
    elif isinstance(glb_file, dict):
        # Hunyuan returns {'value': '.../white_mesh.glb', '__type__': 'update'}
        p = (
            glb_file.get("path")
            or glb_file.get("name")
            or glb_file.get("value")
            or (glb_file.get("video") if isinstance(glb_file.get("video"), str) else None)
        )
        if isinstance(p, dict):
            p = p.get("path") or p.get("value")
        if p:
            path = Path(str(p))
    elif glb_file is not None and hasattr(glb_file, "name"):
        path = Path(getattr(glb_file, "name"))

    if path is None:
        raise RuntimeError(f"No path in result ({type(glb_file)!r})")
    if not path.exists():
        raise RuntimeError(f"GLB path does not exist: {path}")
    if path.suffix.lower() not in (".glb", ".gltf", ".obj", ".ply", ".stl"):
        # Still accept if content looks like glb later; otherwise reject videos.
        if path.suffix.lower() in (".mp4", ".webm", ".html"):
            raise RuntimeError(f"Not a mesh file: {path.name}")

    dest = work_dir / ("sample" + (path.suffix.lower() or ".glb"))
    if path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
    # Prefer .glb extension for the viewer even if source used another name.
    if dest.suffix.lower() != ".glb" and dest.suffix.lower() in (".gltf",):
        pass
    final = work_dir / "sample.glb"
    if dest != final:
        shutil.copy2(dest, final)
        return final
    return dest


def run_mesh3d_job(case_id: str, image_bytes: bytes) -> dict:
    """Blocking worker — call from a background thread only."""
    lock = _case_lock(case_id)
    if not lock.acquire(blocking=False):
        return read_status(case_id)

    try:
        if not USE_MESH3D:
            status = initial_mesh3d_status()
            write_status(case_id, status)
            return status

        # ── Cache first: a near-identical photo reuses a stored coloured GLB
        # with no Space call. Works even without a token — a cached model can
        # still be served — so this runs before the token gate.
        try:
            from app.utils.mesh_cache import lookup as _cache_lookup
            hit = _cache_lookup(image_bytes)
        except Exception as e:
            logger.warning("mesh cache lookup failed for %s: %s", case_id, e)
            hit = None
        if hit:
            dest = _glb_path(case_id)
            shutil.copy2(hit["_path"], dest)
            status = write_status(case_id, {
                "status": "ready",
                "glb_url": _glb_url(case_id),
                "error": None,
                "provider": hit.get("provider", DEFAULT_PROVIDER),
                "color": hit.get("color"),
                "cached": True,
                "elapsed_s": 0.0,
                "message": "3D model loaded from cache (matched a previous photo).",
            })
            _patch_case_history(case_id, status)
            logger.info("mesh3d served from cache for case %s", case_id)
            return status

        if not _runtime_token():
            status = initial_mesh3d_status()
            write_status(case_id, status)
            _patch_case_history(case_id, status)
            return status

        status = write_status(case_id, {
            "status": "generating",
            "glb_url": None,
            "error": None,
            "provider": DEFAULT_PROVIDER,
            "message": "Generating 3D model on Hugging Face — usually under a minute.",
        })

        work_dir = Path(tempfile.mkdtemp(prefix=f"kanchan_mesh3d_{case_id}_"))
        try:
            started = time.time()
            # Run Space call with a soft wall-clock timeout via a nested thread.
            result: dict[str, Any] = {"glb": None, "provider": DEFAULT_PROVIDER, "error": None}

            def _inner():
                try:
                    glb_path, provider = _call_trellis_space(image_bytes, work_dir)
                    result["glb"] = glb_path
                    result["provider"] = provider
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_inner, daemon=True)
            t.start()
            t.join(timeout=MESH3D_TIMEOUT_S)
            if t.is_alive():
                raise TimeoutError(
                    f"3D generation exceeded {int(MESH3D_TIMEOUT_S)}s "
                    "(Space queue or cold start). Try Retry later."
                )
            if result["error"] is not None:
                raise result["error"]
            glb_src: Path = result["glb"]
            provider = result.get("provider") or DEFAULT_PROVIDER
            if glb_src is None or not glb_src.exists():
                raise RuntimeError("3D generation finished but no GLB was produced.")

            dest = _glb_path(case_id)
            shutil.copy2(glb_src, dest)
            # Colour the white mesh to match the item (gold body + set gem),
            # sampled from the case photo. Additive/cosmetic — never blocks a
            # ready model if it fails.
            color_meta = _colorize_ready_glb(case_id, dest)
            # Cache the finished coloured model against the photo so an
            # identical re-upload / retry skips the Space next time.
            try:
                from app.utils.mesh_cache import store as _cache_store
                _cache_store(image_bytes, dest, color_meta, provider)
            except Exception as e:
                logger.warning("mesh cache store failed for %s: %s", case_id, e)
            elapsed = round(time.time() - started, 1)
            status = write_status(case_id, {
                "status": "ready",
                "glb_url": _glb_url(case_id),
                "error": None,
                "provider": provider,
                "elapsed_s": elapsed,
                "color": color_meta,
                "message": "3D model ready.",
            })
            _patch_case_history(case_id, status)
            logger.info("mesh3d ready for case %s via %s in %ss", case_id, provider, elapsed)
            return status
        except Exception as e:
            logger.warning("mesh3d failed for case %s: %s", case_id, e, exc_info=True)
            status = write_status(case_id, {
                "status": "failed",
                "glb_url": None,
                "error": str(e),
                "provider": DEFAULT_PROVIDER,
                "message": "3D generation failed — you can retry from the 3D Model tab.",
            })
            _patch_case_history(case_id, status)
            return status
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        lock.release()


def start_mesh3d_job(case_id: str, images) -> dict:
    """
    Mark job pending and spawn a daemon thread. Safe to call from analyze
    after the HTTP response path — never raises into the caller.

    `images` may be a single bytes object or a list of item photos. When a list
    is given, the card-free photo is preferred and the calibration card is
    masked out of whichever photo is used, so the card never becomes part of
    the 3D model (it is still used elsewhere for scale/tamper-evidence).
    """
    try:
        status = initial_mesh3d_status()
        write_status(case_id, status)
        if status["status"] == "disabled":
            return status

        image_list = images if isinstance(images, (list, tuple)) else [images]
        image_list = [b for b in image_list if b]
        if not image_list:
            return status
        chosen = _select_mesh_image(image_list)
        chosen = _prepare_mesh_image(chosen)

        # Copy bytes so caller can discard its buffer.
        payload = bytes(chosen)

        def _runner():
            try:
                run_mesh3d_job(case_id, payload)
            except Exception as e:
                logger.error("mesh3d background runner crashed for %s: %s", case_id, e, exc_info=True)
                write_status(case_id, {
                    "status": "failed",
                    "glb_url": None,
                    "error": str(e),
                    "provider": DEFAULT_PROVIDER,
                })

        threading.Thread(target=_runner, name=f"mesh3d-{case_id}", daemon=True).start()
        return status
    except Exception as e:
        logger.error("start_mesh3d_job failed for %s: %s", case_id, e, exc_info=True)
        return {
            "status": "failed",
            "glb_url": None,
            "error": str(e),
            "updated_at": _now_iso(),
            "provider": DEFAULT_PROVIDER,
        }


def find_case_image_bytes(case_id: str) -> Optional[bytes]:
    """Load the primary item photo for retry."""
    case_dir = CASES_DIR / case_id
    for name in ("img_0.jpg", "img_0.png", "img_0.jpeg"):
        p = case_dir / name
        if p.exists():
            return p.read_bytes()
    # Any img_* fallback
    if case_dir.exists():
        for p in sorted(case_dir.glob("img_*")):
            if p.is_file():
                return p.read_bytes()
    return None


def find_all_case_images(case_id: str) -> list[bytes]:
    """All saved item photos for a case, in order — so a retry can still
    prefer a card-free shot for 3D, same as the original analysis."""
    case_dir = CASES_DIR / case_id
    if not case_dir.exists():
        return []
    out = []
    for p in sorted(case_dir.glob("img_*")):
        if p.is_file():
            try:
                out.append(p.read_bytes())
            except Exception:
                continue
    return out
