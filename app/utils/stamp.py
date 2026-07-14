"""
Server-side evidence stamping — burns timestamp, GPS coordinates, customer
name and evaluator identity onto every stored evidence photo, the way a
medical X-ray carries a patient/date burn-in.

Trust boundary: the stamp is drawn AFTER analysis, on the copy written to
disk — never on the bytes the CV pipeline scores (a caption bar in-frame
would contaminate the material scan), and never client-side (anything drawn
in the browser could be edited before upload; this cannot be).
"""
import io
import logging
from datetime import datetime, timezone, timedelta

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def stamp_image(
    image_bytes: bytes,
    *,
    case_id: str,
    customer_name: str = "",
    evaluator_name: str = "",
    evaluator_id: str = "",
    geo_lat: float | None = None,
    geo_lon: float | None = None,
    timestamp: datetime | None = None,
    label: str = "",
) -> bytes:
    """Return JPEG bytes with a bottom caption bar burned in. Falls back to
    the original bytes unchanged if the image can't be decoded — stamping
    must never be the reason evidence capture fails."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        logger.warning("Could not open image for stamping: %s", e)
        return image_bytes

    ts = (timestamp or datetime.now(IST)).astimezone(IST)
    ts_text = ts.strftime("%d %b %Y, %H:%M:%S IST")
    geo_text = (
        f"{geo_lat:.5f}, {geo_lon:.5f}"
        if geo_lat is not None and geo_lon is not None
        else "GPS unavailable"
    )

    w, h = img.size
    font_size = max(11, int(round(h * 0.022)))
    font = _load_font(font_size)
    line_h = int(font_size * 1.45)

    lines = [
        f"KANCHAN-AI  ·  Case #{case_id}" + (f"  ·  {label}" if label else ""),
        f"{ts_text}  ·  {geo_text}",
        f"Customer: {customer_name or '—'}   Evaluator: {evaluator_name or '—'} ({evaluator_id or '—'})",
    ]
    bar_h = line_h * len(lines) + 16

    try:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle([0, h - bar_h, w, h], fill=(10, 14, 24, 195))
        y = h - bar_h + 8
        for line in lines:
            odraw.text((11, y + 1), line, font=font, fill=(0, 0, 0, 160))   # shadow
            odraw.text((10, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_h
        stamped = Image.alpha_composite(img, overlay).convert("RGB")
    except Exception as e:
        logger.warning("Stamping failed, saving unstamped image: %s", e)
        stamped = img.convert("RGB")

    buf = io.BytesIO()
    stamped.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
