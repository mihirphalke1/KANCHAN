"""
GET /api/history/{case_id}/report
Professional 3-page bank-quality gold purity assessment report.
"""
import io
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from xml.sax.saxutils import escape as _xml_escape
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.platypus import Image as RLImage

router = APIRouter()

# ── Unicode-capable fonts ────────────────────────────────────────────────
# reportlab's built-in Helvetica/Courier are the 14 base PDF fonts and only
# cover WinAnsi (cp1252) — they have NO glyph for ₹, ✓, ✗, ⚠, Δ, ρ, σ, √, →,
# ≥, etc., all of which this report uses (currency, status marks, physics
# formulas in the verification trace). Under Helvetica those silently render
# as missing/garbled glyphs — the "broken text" in the certificate. DejaVu
# Sans covers the full set used across the codebase (verified against every
# non-ASCII character in app/**/*.py) and is vendored in data/fonts/ (public
# domain / Bitstream-Vera-licensed, redistributable) so PDF generation never
# depends on matplotlib happening to be installed at a particular version.
_FONT_DIR = Path("data/fonts")
FONT_REGULAR = "DejaVuSans"
FONT_BOLD    = "DejaVuSans-Bold"
FONT_MONO    = "DejaVuSansMono"

_FONT_CMAP = frozenset()
try:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(_FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD,    str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_MONO,    str(_FONT_DIR / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR, bold=FONT_BOLD)
    from fontTools.ttLib import TTFont as _FTFont
    _FONT_CMAP = frozenset(_FTFont(str(_FONT_DIR / "DejaVuSans.ttf")).getBestCmap().keys())
except Exception as _e:  # pragma: no cover - defensive; falls back to base-14 fonts
    logging.getLogger(__name__).warning(
        "Could not register DejaVu Sans (%s) — PDF falls back to Helvetica/Courier, "
        "which cannot render ₹/✓/✗/⚠/Δ and similar glyphs", _e)
    FONT_REGULAR, FONT_BOLD, FONT_MONO = "Helvetica", "Helvetica-Bold", "Courier"

# ── Colours ────────────────────────────────────────────────────────────
BLUE        = colors.HexColor('#019EEC')
BLUE_DARK   = colors.HexColor('#0181C2')
BLUE_DARKER = colors.HexColor('#01538A')
BLUE_LIGHT  = colors.HexColor('#EBF8FF')
GOLD        = colors.HexColor('#FFB600')
GREEN       = colors.HexColor('#16A34A')
GREEN_LIGHT = colors.HexColor('#F0FDF4')
AMBER       = colors.HexColor('#B45309')
AMBER_LIGHT = colors.HexColor('#FFFBEB')
RED         = colors.HexColor('#DC2626')
RED_LIGHT   = colors.HexColor('#FEF2F2')
GREY_50     = colors.HexColor('#F9FAFB')
GREY_100    = colors.HexColor('#F3F4F6')
GREY_200    = colors.HexColor('#E5E7EB')
GREY_300    = colors.HexColor('#D1D5DB')
GREY_500    = colors.HexColor('#6B7280')
GREY_700    = colors.HexColor('#374151')
GREY_900    = colors.HexColor('#111827')
WHITE       = colors.white

LOGO_PATH    = Path("data/canara_logo.png")   # drop the PNG here — activates automatically
HISTORY_PATH = Path("data/case_history.json")
W, H  = A4
MARGIN = 18 * mm


# ── Text safety ──────────────────────────────────────────────────────────
# Case data isn't all authored by us: customer names/descriptions are free
# text an officer typed, and verdict.plain_english/action are LLM output —
# either can contain "&"/"<" (breaks reportlab's Paragraph mini-XML parser,
# previously unescaped in several places) or a character the chosen font has
# no glyph for (previously EVERY non-ASCII symbol under Helvetica). _esc()
# is the single funnel both problems go through before reaching a Paragraph.
_EMOJI_REPLACEMENTS = {
    '\U0001F6A8': '⚠',   # 🚨 has no text-font glyph; ⚠ (used elsewhere here) does
    '️':     '',    # variation selector-16 (emoji presentation) — drop, invisible
}


def _pdf_safe(text: str) -> str:
    """Replace/strip characters the registered PDF font can't render, so a
    stray glyph never shows as a blank box or corrupts layout."""
    if not text:
        return text
    for bad, good in _EMOJI_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Supplementary-plane codepoints (U+10000+: emoji, rare scripts) are
    # stripped unconditionally rather than trusting the cmap lookup below —
    # fontTools' getBestCmap() can report a codepoint as "present" via a
    # format-12 subtable entry that doesn't correspond to a sane glyph for
    # it (observed: an emoji rendering as an unrelated garbled character).
    # DejaVu Sans has no legitimate glyphs up there for this report anyway.
    text = ''.join(c for c in text if ord(c) < 0x10000)
    if _FONT_CMAP:
        text = ''.join(c for c in text if ord(c) < 128 or ord(c) in _FONT_CMAP)
    return text


def _esc(txt) -> str:
    """XML-escape (for reportlab's Paragraph markup) + font-safe sanitize.
    Use for EVERY piece of case-derived text placed in a Paragraph — the
    single place both classes of "broken text" bug are fixed at once."""
    return _xml_escape(_pdf_safe(str(txt)))


# ── Drawing flowable (module-level so helpers can use it) ──────────────
class _Drawing(Flowable):
    def __init__(self, d: Drawing):
        super().__init__()
        self.d      = d
        self.width  = d.width
        self.height = d.height

    def draw(self):
        renderPDF.draw(self.d, self.canv, 0, 0)


# ── Paragraph styles ────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, parent=base['Normal'], **kw)
    return {
        'h2':         s('h2',  fontName=FONT_BOLD, fontSize=14, textColor=GREY_900,  leading=20),
        'h3':         s('h3',  fontName=FONT_BOLD, fontSize=11, textColor=GREY_700,  leading=16),
        'body':       s('bd',  fontName=FONT_REGULAR,      fontSize=9,  textColor=GREY_700,  leading=14),
        'small':      s('sm',  fontName=FONT_REGULAR,      fontSize=8,  textColor=GREY_500,  leading=12),
        'mono':       s('mo',  fontName=FONT_MONO,        fontSize=8,  textColor=GREY_700,  leading=12),
        'label':      s('lb',  fontName=FONT_BOLD, fontSize=7,  textColor=GREY_500,  leading=11, spaceAfter=1),
        'value':      s('vl',  fontName=FONT_REGULAR,      fontSize=9,  textColor=GREY_900,  leading=13),
        'value_bold': s('vb',  fontName=FONT_BOLD, fontSize=9,  textColor=GREY_900,  leading=13),
        'verdict_g':  s('vg',  fontName=FONT_BOLD, fontSize=20, textColor=GREEN,     leading=26, alignment=TA_CENTER),
        'verdict_b':  s('va',  fontName=FONT_BOLD, fontSize=20, textColor=AMBER,     leading=26, alignment=TA_CENTER),
        'verdict_r':  s('vr',  fontName=FONT_BOLD, fontSize=20, textColor=RED,       leading=26, alignment=TA_CENTER),
        'center_sm':  s('cs',  fontName=FONT_REGULAR,      fontSize=8,  textColor=GREY_500,  leading=12, alignment=TA_CENTER),
    }


# ── Drawing helpers ─────────────────────────────────────────────────────
def _risk_gauge(risk: float, width: int, height: int = 12) -> Drawing:
    label_h = 10
    d = Drawing(width, height + label_h)
    d.add(Rect(0, label_h, width, height, fillColor=GREY_200, strokeColor=None))
    fill_w = width * min(max(risk, 0), 1)
    fc = GREEN if risk < 0.35 else (GOLD if risk < 0.65 else RED)
    d.add(Rect(0, label_h, fill_w, height, fillColor=fc, strokeColor=None))
    for xp, lbl, anc in [(0, '0', 'start'), (width/2, '50%', 'middle'), (width, '100%', 'end')]:
        d.add(String(xp, 1, lbl, fontName=FONT_REGULAR, fontSize=6,
                     fillColor=GREY_500, textAnchor=anc))
    return d


def _bar_row(value: float, width: int, height: int = 9) -> Drawing:
    d  = Drawing(width, height)
    fc = GREEN if value < 0.35 else (GOLD if value < 0.65 else RED)
    d.add(Rect(0, 0, width,         height, fillColor=GREY_200, strokeColor=None))
    d.add(Rect(0, 0, width * value, height, fillColor=fc,       strokeColor=None))
    return d


def _benford_chart(observed, expected, width: int, height: int = 90) -> Drawing:
    label_h = 12
    chart_h = height - label_h
    d   = Drawing(width, height)
    n   = 9
    bar = (width - 20) / (n * 2.5)
    gap = bar * 1.5
    x   = 10
    for i, (obs, exp) in enumerate(zip(observed, expected)):
        obs_h = max(obs * chart_h * 2.8, 1)
        exp_h = max(exp * chart_h * 2.8, 1)
        d.add(Rect(x,          label_h, bar,       obs_h, fillColor=BLUE,    strokeColor=None))
        d.add(Rect(x+bar+2,    label_h, bar * 0.6, exp_h, fillColor=GREY_300, strokeColor=None))
        d.add(String(x+bar/2,  2, str(i+1), fontName=FONT_REGULAR, fontSize=7,
                     fillColor=GREY_700, textAnchor='middle'))
        x += gap + bar
    return d


def _legend_swatch(fc, w=12, h=8) -> Drawing:
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=fc, strokeColor=None))
    return d


def _waveform(audio_path: str, width: int, height: int = 70) -> Optional[Drawing]:
    try:
        import librosa, numpy as np
        y, _ = librosa.load(audio_path, duration=4.0, mono=True, sr=22050)
        if not len(y): return None
        n   = width
        hop = max(1, len(y) // n)
        env = np.array([np.abs(y[i:i+hop]).max() for i in range(0, len(y)-hop, hop)])[:n]
        if env.max() > 0: env /= env.max()
        d   = Drawing(width, height)
        mid = height / 2
        bw  = max(1.0, width / len(env))
        d.add(Rect(0, 0, width, height, fillColor=colors.HexColor('#F0F8FE'), strokeColor=None))
        d.add(Line(0, mid, width, mid, strokeColor=GREY_300, strokeWidth=0.5))
        for i, amp in enumerate(env):
            bh = max(amp * mid * 0.90, 0.5)
            xp = i * bw
            fc = BLUE if amp < 0.65 else RED
            d.add(Rect(xp, mid,      bw*0.72, bh,  fillColor=fc, strokeColor=None))
            d.add(Rect(xp, mid - bh, bw*0.72, bh,  fillColor=fc, strokeColor=None))
        for xp, lbl, anc in [(2, '0s', 'start'), (width/2, '2s', 'middle'), (width-2, '4s', 'end')]:
            d.add(String(xp, 2, lbl, fontName=FONT_REGULAR, fontSize=6,
                         fillColor=GREY_500, textAnchor=anc))
        return d
    except Exception:
        return None


def _no_waveform(width: int, height: int = 70) -> Drawing:
    d   = Drawing(width, height)
    mid = height / 2
    d.add(Rect(0, 0, width, height, fillColor=GREY_50, strokeColor=GREY_200, strokeWidth=0.5))
    dash, x = 8, 4
    while x < width - 4:
        d.add(Line(x, mid, min(x+dash, width-4), mid, strokeColor=GREY_300, strokeWidth=1))
        x += dash * 2
    d.add(String(width/2, mid+5, 'No audio recording provided — acoustic test not available',
                 fontName=FONT_REGULAR, fontSize=7, fillColor=GREY_500, textAnchor='middle'))
    return d


# ── Page header / footer ────────────────────────────────────────────────
class _DocCanvas:
    def __init__(self, case_id, timestamp):
        self.case_id   = case_id
        self.timestamp = timestamp

    def draw_header(self, canvas, doc):
        canvas.saveState()
        # Blue banner
        canvas.setFillColor(BLUE_DARKER)
        canvas.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)

        # ── Logo (PNG) or text fallback ──
        if LOGO_PATH.exists():
            try:
                logo_h = 16*mm
                logo_w = logo_h * 3.0          # will be clamped by aspect ratio
                canvas.drawImage(
                    str(LOGO_PATH),
                    MARGIN, H - 24*mm,
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask='auto'
                )
                text_x = MARGIN + logo_w + 5*mm
            except Exception:
                text_x = MARGIN
                canvas.setFont(FONT_BOLD, 15)
                canvas.setFillColor(WHITE)
                canvas.drawString(MARGIN, H - 16*mm, 'CANARA BANK')
                canvas.setFont(FONT_REGULAR, 8)
                canvas.setFillColor(colors.HexColor('#BEE8FB'))
                canvas.drawString(MARGIN, H - 22*mm, 'Gold Loan Division  ·  AI-Assisted Purity Assessment')
        else:
            text_x = MARGIN
            canvas.setFont(FONT_BOLD, 15)
            canvas.setFillColor(WHITE)
            canvas.drawString(MARGIN, H - 16*mm, 'CANARA BANK')
            canvas.setFont(FONT_REGULAR, 8)
            canvas.setFillColor(colors.HexColor('#BEE8FB'))
            canvas.drawString(MARGIN, H - 22*mm, 'Gold Loan Division  ·  AI-Assisted Purity Assessment')

        # Right: system name
        canvas.setFont(FONT_BOLD, 10)
        canvas.setFillColor(GOLD)
        canvas.drawRightString(W - MARGIN, H - 16*mm, 'KANCHAN-AI')
        canvas.setFont(FONT_REGULAR, 8)
        canvas.setFillColor(colors.HexColor('#BEE8FB'))
        canvas.drawRightString(W - MARGIN, H - 22*mm, 'Gold Fraud Detection System')

        # Gold rule
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, H - 28*mm + 2, W - MARGIN, H - 28*mm + 2)
        canvas.restoreState()

    def draw_footer(self, canvas, doc):
        canvas.saveState()
        y = 12*mm
        canvas.setStrokeColor(GREY_200)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, y + 8, W - MARGIN, y + 8)
        canvas.setFont(FONT_REGULAR, 7)
        canvas.setFillColor(GREY_500)
        canvas.drawString(MARGIN, y,
            f'CONFIDENTIAL  ·  Case #{self.case_id}  ·  {self.timestamp}')
        canvas.drawString(MARGIN, y - 9,
            'This report is generated by KANCHAN-AI for Canara Bank Gold Loan Division. '
            'AI verdict is advisory — final lending decision rests with the authorised branch officer.')
        canvas.drawRightString(W - MARGIN, y, f'Page {doc.page}')
        if doc.page == 1:
            canvas.saveState()
            canvas.setFillColor(colors.HexColor('#F0F7FF'))
            canvas.setFont(FONT_BOLD, 52)
            canvas.translate(W/2, H/2)
            canvas.rotate(35)
            canvas.drawCentredString(0, 0, 'CONFIDENTIAL')
            canvas.restoreState()
        canvas.restoreState()


# ── Cell paragraph helper ────────────────────────────────────────────────
def _cell(txt, bold=False, color=GREY_700, size=8):
    return Paragraph(_esc(txt), ParagraphStyle(
        'tc', fontName=FONT_BOLD if bold else FONT_REGULAR,
        fontSize=size, textColor=color, leading=size + 4))


_TS_BASE = [
    ('TOPPADDING',    (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ('RIGHTPADDING',  (0,0), (-1,-1), 8),
]


# ── Main builder ────────────────────────────────────────────────────────
def build_report(case: dict) -> bytes:
    S   = _styles()
    buf = io.BytesIO()

    raw_ts = case.get('timestamp', '')
    IST = timezone(timedelta(hours=5, minutes=30))
    try:
        dt     = datetime.fromisoformat(raw_ts.replace('Z', '+00:00')).astimezone(IST)
        ts_display = dt.strftime('%d %B %Y, %H:%M IST')
        ts_short   = dt.strftime('%d %b %Y %H:%M IST')
    except Exception:
        ts_display = raw_ts
        ts_short   = raw_ts

    case_id = case.get('case_id', 'UNKNOWN').upper()
    dc      = _DocCanvas(case_id, ts_short)

    def on_page(canvas, doc):
        dc.draw_header(canvas, doc)
        dc.draw_footer(canvas, doc)

    frame = Frame(MARGIN, 18*mm, W - 2*MARGIN, H - 50*mm, id='main')
    tpl   = PageTemplate(id='main', frames=[frame], onPage=on_page)
    doc   = BaseDocTemplate(buf, pagesize=A4, pageTemplates=[tpl],
                            title=f'Gold Purity Report — {case_id}',
                            author='KANCHAN-AI / Canara Bank')

    story = []
    bw    = W - 2*MARGIN

    # ── Extract ──────────────────────────────────────────────────────
    verdict  = case.get('verdict', {})
    ms       = case.get('modality_scores', {})
    density  = ms.get('density', {})
    acoustic = ms.get('acoustic', {})
    image_m  = ms.get('image', {})
    streak   = ms.get('streak', {})
    xray_m   = ms.get('xray', {}) or {}

    # The live decision path blends the classical-CV material scan (DSIP)
    # into the visual channel — the CNN probe (image_m) is off by default and
    # stuck at a neutral 50%. Show the score that actually drove the verdict.
    if xray_m.get('fusion_contribution'):
        visual_display = {
            'risk_score': xray_m['fusion_contribution']['blended_visual_risk'],
            'mode':       'blended_visual',
        }
    elif xray_m.get('mode') == 'dsip_xray':
        visual_display = {'risk_score': xray_m.get('risk_score', 0.5), 'mode': 'dsip_xray'}
    elif xray_m.get('mode') == 'dsip_unusable':
        visual_display = {'risk_score': 0.5, 'mode': 'dsip_unusable'}
    else:
        visual_display = image_m
    contra   = case.get('contradiction', {})
    fusion   = case.get('fusion', {})
    benford  = case.get('benford', {})
    ltv      = case.get('ltv', {}) or {}
    customer = case.get('customer', {})
    media    = case.get('media', {})

    rl          = verdict.get('risk_level',  'BORDERLINE')
    loan_action = verdict.get('loan_action', 'HOLD')
    confidence  = verdict.get('confidence',  'LOW')
    frisk       = verdict.get('fusion_risk', 0.5)

    vmap = {
        'GENUINE':    (GREEN_LIGHT, GREEN, GREEN,  'verdict_g', '✓  GENUINE'),
        'BORDERLINE': (AMBER_LIGHT, AMBER, AMBER,  'verdict_b', '⚠  BORDERLINE'),
        'REJECT':     (RED_LIGHT,   RED,   RED,    'verdict_r', '✗  REJECT'),
    }
    v_bg, v_border, v_ink, v_skey, v_txt = vmap.get(rl, (GREY_100, GREY_700, GREY_700, 'verdict_b', rl))

    lmap = {
        'APPROVE': (GREEN, 'APPROVE LOAN'),
        'HOLD':    (AMBER, 'HOLD FOR REVIEW'),
        'DECLINE': (RED,   'DECLINE LOAN'),
    }
    l_ink, l_txt = lmap.get(loan_action, (GREY_700, loan_action))

    cname    = customer.get('name',         '—')
    cacct    = customer.get('account_no',   '—')
    cloan    = customer.get('loan_app_no',  '—')
    cofficer = customer.get('officer_name', '—')

    def _info_block(rows, col_w):
        t = Table(rows, colWidths=[col_w*0.42, col_w*0.58])
        t.setStyle(TableStyle(_TS_BASE + [
            ('SPAN',         (0,0), (1,0)),
            ('BACKGROUND',   (0,0), (1,0),  GREY_100),
            ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
            ('LINEBELOW',    (0,0), (1,0),   0.5, GREY_200),
            ('GRID',         (0,1), (-1,-1), 0.3, GREY_200),
            ('TOPPADDING',   (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ]))
        return t

    def _section_tbl(rows, col_w=None):
        t = Table(rows, colWidths=col_w or [bw])
        t.setStyle(TableStyle(_TS_BASE + [
            ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
            ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
            ('LINEBELOW',    (0,0), (-1,-1), 0.3, GREY_200),
            ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ]))
        return t

    # ════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER + SUMMARY
    # ════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 4*mm))

    # Title
    tbl_title = Table([[Paragraph('GOLD PURITY ASSESSMENT REPORT', S['h2'])]], colWidths=[bw])
    tbl_title.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,-1), BLUE_LIGHT),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('TOPPADDING',   (0,0), (-1,-1), 10),
        ('BOTTOMPADDING',(0,0), (-1,-1), 10),
    ]))
    story.append(tbl_title)
    story.append(Spacer(1, 5*mm))

    # Case metadata row
    meta = Table([
        [_cell('CASE ID',      bold=True, color=GREY_500),
         _cell('REPORT DATE',  bold=True, color=GREY_500),
         _cell('BRANCH',       bold=True, color=GREY_500),
         _cell('STATUS',       bold=True, color=GREY_500)],
        [Paragraph(f'#{case_id}', ParagraphStyle('mid', fontName=FONT_MONO, fontSize=8,
                                                  textColor=GREY_700, leading=12)),
         _cell(ts_display),
         _cell(case.get('branch_id', '—')),
         _cell('Completed', bold=True)],
    ], colWidths=[bw/4]*4)
    meta.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_50),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.5, GREY_200),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]))
    story.append(meta)
    story.append(Spacer(1, 5*mm))

    # Customer + Item info (side by side)
    half = (bw - 8) / 2
    cust_rows = [
        [Paragraph('CUSTOMER INFORMATION', S['label']), ''],
        [_cell('Customer Name',      color=GREY_500), _cell(cname,    bold=True)],
        [_cell('Account Number',     color=GREY_500), _cell(cacct)],
        [_cell('Loan Application',   color=GREY_500), _cell(cloan)],
        [_cell('Assessment Officer', color=GREY_500), _cell(cofficer)],
    ]
    item_rows = [
        [Paragraph('GOLD ITEM', S['label']), ''],
        [_cell('Description',    color=GREY_500), _cell(case.get('item_description', '—'), bold=True)],
        [_cell('Declared Karat', color=GREY_500), _cell(f"{case.get('declared_karat','—')}K gold")],
        [_cell('Dry Weight',     color=GREY_500), _cell(f"{density.get('weight_dry','—')} g")],
        [_cell('Submerged Wt.',  color=GREY_500), _cell(f"{density.get('weight_submerged','—')} g")],
    ]
    info_row = Table([[_info_block(cust_rows, half), _info_block(item_rows, half)]],
                     colWidths=[half, half])
    info_row.setStyle(TableStyle([
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 4),
        ('TOPPADDING',   (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
    ]))
    story.append(info_row)
    story.append(Spacer(1, 6*mm))

    # Verdict box
    v_ps  = S[v_skey]
    conf_ps = ParagraphStyle('cp', parent=v_ps, fontSize=10, leading=14)
    loan_ps = ParagraphStyle('lp', fontName=FONT_BOLD, fontSize=11,
                              textColor=l_ink, leading=16, alignment=TA_CENTER)
    vt = Table([
        [[Paragraph(v_txt, v_ps), Paragraph(f'{confidence} CONFIDENCE · {l_txt}', loan_ps)]],
    ], colWidths=[bw])
    vt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), v_bg),
        ('BOX',           (0,0),(-1,-1), 1.5, v_border),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
    ]))
    story.append(KeepTogether([vt]))
    story.append(Spacer(1, 5*mm))

    # AI explanation
    plain = verdict.get('plain_english', '')
    if plain:
        et = _section_tbl([
            [_cell('ASSESSMENT', bold=True, color=GREY_500)],
            [Paragraph(_esc(plain), S['body'])],
        ])
        story.append(et)
        story.append(Spacer(1, 3*mm))

    # Maker-checker sign-off status (P3-12): a BORDERLINE/HELD case cannot close
    # until a second, different officer signs off. Surface that gate on the face
    # of the certificate so a reader knows whether the case is closable.
    approval = case.get('approval') or {}
    if approval.get('maker_checker_required'):
        st = approval.get('status')
        if st == 'approved':
            ap_txt = (f"DUAL SIGN-OFF COMPLETE — approved by checker "
                      f"{approval.get('checker_name') or approval.get('checker_id') or '—'}"
                      f" on {(approval.get('signed_at') or '')[:19].replace('T', ' ')}. Case closable.")
            ap_bg, ap_ink = GREEN_LIGHT if 'GREEN_LIGHT' in globals() else BLUE_LIGHT, GREY_900
        elif st == 'rejected':
            ap_txt = (f"DUAL SIGN-OFF — REJECTED by checker "
                      f"{approval.get('checker_name') or approval.get('checker_id') or '—'}. Loan not to be disbursed.")
            ap_bg, ap_ink = AMBER_LIGHT, GREY_900
        else:
            ap_txt = ("MAKER-CHECKER REQUIRED — this borderline case is PENDING a second "
                      "authorised officer's sign-off and cannot be closed on the maker alone.")
            ap_bg, ap_ink = AMBER_LIGHT, GREY_900
        apt = _section_tbl([
            [_cell('MAKER-CHECKER SIGN-OFF', bold=True, color=GREY_500)],
            [Paragraph(_esc(ap_txt), ParagraphStyle('mc', fontName=FONT_BOLD,
                                              fontSize=8.5, textColor=ap_ink, leading=13))],
        ])
        apt.setStyle(TableStyle(_TS_BASE + [
            ('BACKGROUND',  (0,0), (-1,0),  GREY_100),
            ('BACKGROUND',  (0,1), (-1,1),  ap_bg),
            ('BOX',         (0,0), (-1,-1), 0.5, AMBER),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(apt)
        story.append(Spacer(1, 3*mm))

    # Recommended action
    action = verdict.get('action', '')
    if action:
        at = _section_tbl([
            [_cell('RECOMMENDED ACTION', bold=True, color=GREY_500)],
            [Paragraph(_esc(action), ParagraphStyle('ac', fontName=FONT_BOLD,
                                               fontSize=9, textColor=GREY_900, leading=14))],
        ])
        at.setStyle(TableStyle(_TS_BASE + [
            ('BACKGROUND',    (0,0), (-1,0),  GREY_100),
            ('BACKGROUND',    (0,1), (-1,1),  BLUE_LIGHT),
            ('BOX',           (0,0), (-1,-1), 0.5, GREY_200),
            ('LINEBELOW',     (0,0), (-1,-1), 0.3, GREY_200),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ]))
        story.append(at)

    # ════════════════════════════════════════════════════════════════
    # PAGE 2 — TECHNICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('TECHNICAL ANALYSIS', S['h2']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_200, spaceAfter=5*mm))

    # ── 1. Density ───────────────────────────────────────────────────
    drisk = density.get('risk_score', 0)
    meas  = density.get('measured_density', 0)
    elo   = density.get('expected_low', 0)
    ehi   = density.get('expected_high', 0)

    def _status_cell(r):
        if r < 0.35: return _cell('✓ Within range', bold=True, color=GREEN)
        if r < 0.75: return _cell('! Marginal',     bold=True, color=AMBER)
        return           _cell('✗ ANOMALOUS',       bold=True, color=RED)

    best_match     = density.get('best_match') or {}
    closest_fake   = density.get('closest_fake')
    tungsten_warn  = density.get('tungsten_warning')
    misdeclared    = density.get('misdeclared_purity')

    d_body = [[_cell('PARAMETER'), _cell('VALUE'), _cell('EXPECTED'), _cell('STATUS')]]
    d_body += [
        [_cell('Declared Karat'),   _cell(f"{case.get('declared_karat','—')}K"), _cell('—'), _cell('—')],
        [_cell('Dry Weight'),       _cell(f"{density.get('weight_dry','—')} g"),  _cell('—'), _cell('—')],
        [_cell('Submerged Weight'), _cell(f"{density.get('weight_submerged','—')} g"), _cell('—'), _cell('—')],
        [_cell('Measured Density'),
         _cell(f'{meas:.3f} g/cm³' if isinstance(meas, float) else '—', bold=True),
         _cell(f'{elo}–{ehi} g/cm³'), _status_cell(drisk)],
        [_cell('Risk Score'),
         _cell(f'{drisk:.2%}' if isinstance(drisk, float) else '—', bold=True),
         _cell('< 35% = pass'), _cell('')],
    ]
    if best_match.get('name'):
        d_body.append([
            _cell('What the physics says it is'),
            _cell(best_match['name'].title() +
                  (f" ({round((best_match.get('probability') or 0) * 100)}% match)"
                   if best_match.get('kind') == 'karat' else ''), bold=True),
            _cell('—'),
            _cell('! Over-stated' if misdeclared else '✓ Consistent',
                  bold=True, color=AMBER if misdeclared else GREEN),
        ])
    if closest_fake and best_match.get('kind') != 'karat':
        d_body.append([_cell('Closest Fake Metal'), _cell(closest_fake.title(), bold=True, color=RED),
                        _cell('—'), _cell('✗ Check', bold=True, color=RED)])
    if tungsten_warn:
        d_body.append([_cell('Tungsten Blind-Spot'),
                        _cell('Density matches tungsten — indistinguishable from 24K by density alone',
                              bold=True, color=AMBER),
                        _cell('—'), _cell('! Review', bold=True, color=AMBER)])
    dt = Table(d_body, colWidths=[bw*0.30, bw*0.22, bw*0.27, bw*0.21])
    dt.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  FONT_BOLD),
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
    ]))
    story.append(KeepTogether([
        Paragraph('1. Archimedes Density Test', S['h3']),
        Spacer(1, 2*mm),
        dt,
    ]))
    story.append(Spacer(1, 2*mm))

    gauge_lbl = Table([[Paragraph('Density risk', S['label']),
                        Paragraph(f'{drisk:.1%}', S['small'])]],
                      colWidths=[bw/2]*2)
    gauge_lbl.setStyle(TableStyle([('ALIGN', (1,0),(1,0), 'RIGHT')]))
    story.append(gauge_lbl)
    story.append(_Drawing(_risk_gauge(drisk, width=int(bw), height=12)))
    story.append(Spacer(1, 6*mm))

    # ── 2. Composition — gold vs stones ───────────────────────────────
    comp = case.get('composition')
    if comp:
        model_valid = comp.get('model_valid')
        c_body = [[_cell('PARAMETER'), _cell('VALUE'), _cell('NOTE')]]
        if model_valid and comp.get('gold_mass_g') is not None:
            goldpct = round((comp.get('gold_mass_fraction') or 0) * 100)
            c_body.append([_cell('Gold Mass'), _cell(f"{comp['gold_mass_g']} g ({goldpct}%)", bold=True),
                            _cell(f"of {density.get('weight_dry','—')} g total")])
            c_body.append([_cell('Stone/Other Mass'), _cell(f"{comp.get('stone_mass_g','—')} g", bold=True), _cell('')])
        c_body.append([_cell('Stone Volume (photo)'), _cell(f"{comp.get('stone_frac_photo',0)*100:.0f}%"),
                        _cell('Camera-detected stone area — a lower bound')])
        c_body.append([_cell('Stone Volume (physics)'), _cell(f"{comp.get('stone_frac_implied',0)*100:.0f}%"),
                        _cell('Non-gold volume implied by measured density')])
        c_body.append([_cell('Predicted Bulk Density'), _cell(f"{comp.get('rho_predicted','—')} g/cm³"), _cell('')])
        z = comp.get('consistency_z', 0)
        c_body.append([_cell('Consistency (z)'),
                        _cell(str(z), bold=True, color=AMBER if z and z > 2 else GREY_900),
                        _cell('< 2 consistent · > 3 anomalous')])
        adj = comp.get('adjusted_density_risk')
        if adj is not None:
            c_body.append([_cell('Stone-Corrected Density Risk'),
                            _cell(f"{adj*100:.0f}%", bold=True, color=AMBER if adj > 0.5 else GREEN),
                            _cell('Feeds the final decision for stone-set items')])
        ct = Table(c_body, colWidths=[bw*0.32, bw*0.24, bw*0.44])
        ct.setStyle(TableStyle(_TS_BASE + [
            ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
            ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
            ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
            ('FONTNAME',     (0,0), (-1,0),  FONT_BOLD),
            ('FONTSIZE',     (0,0), (-1,0),  7),
            ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
        ]))
        gems = comp.get('gems') or []
        def _gem_label(g):
            name = g.get('stone_name')
            if name and name != 'unidentified':
                return f"{name} ({round((g.get('match_confidence') or 0) * 100)}% colour match)"
            return g.get('hue_class', '—')
        gem_txt = ('Detected stones: ' + ', '.join(
            f"#{i+1} {_gem_label(g)} ({g.get('area_pct',0)}%)" for i, g in enumerate(gems)
        )) if gems else 'No stones detected — plain-metal density analysis applies.'
        story.append(KeepTogether([
            Paragraph('2. Composition — Gold vs Stones', S['h3']),
            Spacer(1, 2*mm),
            ct,
            Spacer(1, 2*mm),
            Paragraph(_esc(comp.get('note') or gem_txt), S['small']),
        ]))
        story.append(Spacer(1, 6*mm))

    # ── 3. Signal breakdown ──────────────────────────────────────────

    mode_labels = {
        'computed':          'Archimedes physics',
        'efficientnet':      'EfficientNet-B3',
        'svm':               'MFCC-ΔΔ SVM (122-dim)',
        'svm_data_limited':  'MFCC-ΔΔ SVM — data-limited, ring physics carries verdict',
        'heuristic':         'HSV Heuristic',
        'heuristic:heuristic': 'ZCR+RMS heuristic',
        'heuristic:silent_audio': 'Silent audio (50% default)',
        'heuristic:empty_audio':  'Empty audio (50% default)',
        'heuristic:error':   'Audio error (50% default)',
        'logreg':            'Logistic Regression',
        'hsv_bands':         'HSV hue-band physics',
        'no_audio':          'No audio (50% default)',
        'no_images':         'No images (50% default)',
        'no_streak':         'No streak (50% default)',
        'dsip_xray':         'Classical CV material scan (DSIP)',
        'dsip_unusable':     'Photo unusable — background not separable (50% default)',
        'blended_visual':    'Material scan (DSIP), blended',
        'no_cnn':            'CNN probe disabled (50% default)',
    }
    def _rl(r):
        if r < 0.35: return ('LOW',      GREEN)
        if r < 0.65: return ('MODERATE', AMBER)
        return           ('HIGH',        RED)

    bar_w = int(bw * 0.28)
    mod_rows = [[_cell('MODALITY'), _cell('RISK'), _cell('RISK BAR'),
                 _cell('LEVEL'), _cell('METHOD')]]
    for name, m in [('Density',density), ('Visual',visual_display), ('Acoustic',acoustic), ('Streak',streak)]:
        r      = m.get('risk_score', 0.5)
        lvl, lc = _rl(r)
        mod_rows.append([
            _cell(name, bold=True),
            _cell(f'{r:.1%}'),
            _Drawing(_bar_row(r, width=bar_w, height=8)),
            _cell(lvl, bold=True, color=lc),
            _cell(mode_labels.get(m.get('mode',''), m.get('mode',''))),
        ])
    mt = Table(mod_rows, colWidths=[bw*0.15, bw*0.10, bw*0.30, bw*0.13, bw*0.32])
    mt.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  FONT_BOLD),
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(KeepTogether([
        Paragraph('3. Signal Breakdown — All Modalities', S['h3']),
        Spacer(1, 2*mm),
        mt,
    ]))
    story.append(Spacer(1, 6*mm))

    # ── 3. Fusion ────────────────────────────────────────────────────
    fmode       = fusion.get('mode', 'heuristic')
    boosted     = verdict.get('boosted_risk')
    contra_sc   = contra.get('contradiction_score', 0)
    # Reconstruct boosted_risk for older seeded cases that predate this field
    if boosted is None:
        boosted = round(min(1.0, frisk + contra_sc * 0.40), 4)
    # Describe the fusion engine that ACTUALLY produced this number (default is
    # transparent log-odds with per-modality reliability weights; XGBoost is an
    # opt-in baseline via FUSION_MODE) — never a hardcoded XGBoost/35-30-25-10
    # description next to a log-odds figure.
    if fmode == 'xgboost':
        _fusion_title = '4. XGBoost Fusion Analysis'
        _fusion_risk_desc = 'Trained XGBoost meta-classifier (10-dim: 4 risks + 6 pairwise contradictions)'
        _fusion_model_desc = 'XGBoost (10-dim) with SHAP attributions'
    elif fmode == 'logodds':
        _fusion_title = '4. Log-Odds Evidence Fusion'
        _fusion_risk_desc = ('Transparent log-odds sum with reliability weights '
                             '(density 1.0 · acoustic 0.6 · visual 0.5 · streak 0.3) and blind-spot floors')
        _fusion_model_desc = 'Log-odds evidence sum (hand-recomputable; reliability-weighted)'
    else:
        _fusion_title = '4. Weighted-Heuristic Fusion Analysis'
        _fusion_risk_desc = 'Weighted average: density 35% · acoustic 30% · visual 25% · streak 10%'
        _fusion_model_desc = 'Weighted heuristic (density 35%, acoustic 30%, visual 25%, streak 10%)'
    f_body = [[_cell('METRIC'), _cell('VALUE'), _cell('DESCRIPTION')]]
    for row in [
        ['Fusion Risk Score',   f'{frisk:.2%}', _fusion_risk_desc],
        ['Contradiction Score', f'{contra_sc:.2%}',
         'Max pairwise disagreement across modalities that actually ran'],
        ['Boosted Risk Score',  f'{boosted:.2%}',
         'Fusion + contradiction×0.40 — THIS is the number that determines the verdict'],
        ['Fusion Model', fmode.upper(), _fusion_model_desc],
        ['Confidence',          confidence, ''],
        ['Loan Recommendation', loan_action, ''],
    ]:
        f_body.append([_cell(row[0]), _cell(row[1], bold=True, color=GREY_900), _cell(row[2])])
    ft = Table(f_body, colWidths=[bw*0.28, bw*0.18, bw*0.54])
    ft.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BACKGROUND',   (0,3), (-1,3),  BLUE_LIGHT),   # boosted risk row highlighted
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  FONT_BOLD),
        ('FONTNAME',     (0,3), (-1,3),  FONT_BOLD),  # boosted row bold
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
    ]))
    story.append(KeepTogether([
        Paragraph(_fusion_title, S['h3']),
        Spacer(1, 2*mm),
        ft,
    ]))
    story.append(Spacer(1, 4*mm))

    # ── Valuation & loan eligibility ─────────────────────────────────
    # The whole point of the appraisal for the branch: net gold weight, the
    # rate applied, assessed value, the RBI-tiered LTV, and the resulting
    # maximum loan — none of which the report carried before.
    if ltv:
        brk = ltv.get('net_gold_breakdown', {}) or {}
        rate_src = (case.get('gold_rate_source')
                    or ('IBJA live feed' if ltv.get('rate_source') == 'ibja_api' else 'admin-configured'))
        rate_date = ltv.get('rate_updated_at') or '—'

        # Defects noted — pulled from the signals that actually fired.
        defects = []
        if case.get('hollow_item'):
            defects.append('declared hollow' + (' — density anomaly (possible core fill)'
                                                 if case.get('hollow_anomaly') else ''))
        _xr = (case.get('media', {}).get('xray') or {})
        if (_xr.get('filigree') or {}).get('is_filigree'):
            defects.append('filigree/openwork')
        if (_xr.get('multiple_items') or {}).get('multiple_items_detected'):
            defects.append(f"{_xr['multiple_items'].get('component_count')} separate items in frame")
        _tar = (ms.get('tarnish') or {}).get('features', {})
        if _tar.get('discoloration_type') == 'contamination_tarnish':
            defects.append('contamination tarnish')
        elif _tar.get('discoloration_type') == 'intentional_patina':
            defects.append('antique patina (intentional)')
        if brk.get('n_stones'):
            defects.append(f"{brk['n_stones']} set stone(s) deducted")
        defects_txt = '; '.join(defects) if defects else 'None noted'

        valuation_karat = ltv.get('valuation_karat')
        rate_declared_txt = f"₹ {ltv.get('rate_per_gram_inr', 0):,.0f} / g" + (
            f" at {valuation_karat}K" if valuation_karat else "")

        val_rows = [
            ['METRIC', 'VALUE', 'BASIS'],
            ['Gross weight (measured)', f"{brk.get('gross_weight_g', density.get('weight_dry', '—'))} g",
             'Dry weight on the branch scale'],
            ['Less: set-stone weight', f"− {brk.get('stone_weight_g', 0)} g",
             brk.get('method', '—').replace('_', ' ')],
            ['Net gold weight', f"{ltv.get('net_gold_weight_g', '—')} g",
             'Gold charged for — stones are not lent against at the gold rate'],
            ['Gold rate applied', rate_declared_txt,
             f'{rate_src} · as of {rate_date} · 999-fine rate × BIS fineness for the karat'],
        ]
        # Physics-matched karat rate shown for comparison ONLY when it differs
        # from the declared karat — never substituted into the valuation itself
        # (a mismatch is a misdeclared-purity flag elsewhere, handled by
        # revaluation, not a silent rate swap here).
        matched_karat = ltv.get('matched_karat')
        if matched_karat and matched_karat != valuation_karat:
            val_rows.append([
                'Rate at physics-matched karat', f"₹ {ltv['rate_per_gram_calculated_karat']:,.0f} / g at {matched_karat}K",
                'For comparison only — not used in this valuation',
            ])
        val_rows += [
            ['Assessed value', f"₹ {ltv.get('assessed_value_inr', 0):,.0f}",
             'Net gold weight × rate'],
            ['Maximum LTV', f"{ltv.get('ltv_pct', 0)*100:.0f}%",
             f"RBI tier: {ltv.get('tier', '—')}"],
            ['Maximum eligible loan', f"₹ {ltv.get('max_loan_inr', 0):,.0f}",
             'Assessed value × LTV — RBI Gold Collateral Directions 2025'],
            ['Defects noted', defects_txt, 'From the analysis signals that fired'],
        ]
        vt_body = [[_cell(c) for c in val_rows[0]]]
        for r in val_rows[1:]:
            vt_body.append([_cell(r[0], color=GREY_500), _cell(r[1], bold=True, color=GREY_900), _cell(r[2])])
        _loan_row = len(val_rows) - 2   # "Maximum eligible loan" is always 2nd-to-last
        vt = Table(vt_body, colWidths=[bw*0.30, bw*0.24, bw*0.46])
        vt.setStyle(TableStyle(_TS_BASE + [
            ('BACKGROUND', (0,0), (-1,0), GREY_100),
            ('BACKGROUND', (0,_loan_row), (-1,_loan_row), BLUE_LIGHT),
            ('BOX',        (0,0), (-1,-1), 0.5, GREY_200),
            ('GRID',       (0,0), (-1,-1), 0.3, GREY_200),
            ('FONTNAME',   (0,0), (-1,0), FONT_BOLD),
            ('FONTNAME',   (0,_loan_row), (-1,_loan_row), FONT_BOLD),
            ('FONTSIZE',   (0,0), (-1,0), 7),
            ('TEXTCOLOR',  (0,0), (-1,0), GREY_500),
        ]))
        is_final = case.get('borrower_present') or ltv.get('status') == 'final'
        borrower_note = (
            'Borrower attested present at valuation — this valuation is FINAL.'
            if is_final else
            'INDICATIVE VALUATION ONLY — borrower-present attestation was not recorded. '
            'This assessed value, LTV, and maximum loan figure are PROVISIONAL and must be '
            're-confirmed as FINAL, with the borrower attested present, before disbursal.'
        )
        note_ps = ParagraphStyle('bn', fontName=FONT_BOLD if not is_final else FONT_REGULAR,
                                  fontSize=8.5, textColor=GREY_900 if not is_final else GREY_700, leading=13)
        if is_final:
            story.append(KeepTogether([
                Paragraph('5. Valuation & Loan Eligibility', S['h3']),
                Spacer(1, 2*mm),
                vt,
                Spacer(1, 2*mm),
                Paragraph(borrower_note, note_ps),
            ]))
        else:
            note_tbl = Table([[Paragraph(borrower_note, note_ps)]], colWidths=[bw])
            note_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), AMBER_LIGHT),
                ('BOX',           (0,0),(-1,-1), 0.5, AMBER),
                ('TOPPADDING',    (0,0),(-1,-1), 7),
                ('BOTTOMPADDING', (0,0),(-1,-1), 7),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ]))
            story.append(KeepTogether([
                Paragraph('5. Valuation & Loan Eligibility', S['h3']),
                Spacer(1, 2*mm),
                vt,
                Spacer(1, 2*mm),
                note_tbl,
            ]))
        story.append(Spacer(1, 4*mm))

    # ── 4. Contradiction flags ───────────────────────────────────────
    flags = contra.get('flags', [])
    if flags:
        story.append(Paragraph('6. Cross-Modal Contradiction Flags', S['h3']))
        story.append(Spacer(1, 2*mm))
        for flag in flags:
            ft2 = Table([[_cell(f'⚠  {flag}', bold=True, color=AMBER)]], colWidths=[bw])
            ft2.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), AMBER_LIGHT),
                ('BOX',           (0,0),(-1,-1), 0.5, AMBER),
                ('TOPPADDING',    (0,0),(-1,-1), 7),
                ('BOTTOMPADDING', (0,0),(-1,-1), 7),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ]))
            story.append(ft2)
            story.append(Spacer(1, 2*mm))

    # ── Fraud-scenario mapping (bank Sl.1–Sl.8 vocabulary, P3-13) ─────
    fraud = case.get('fraud_scenarios') or {}
    fraud_matched = fraud.get('matched') or []
    if fraud_matched:
        story.append(Paragraph('6a. Bank Fraud-Scenario Classification', S['h3']))
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(
            'The verdict and contradiction pattern above, translated into the bank’s '
            'standard spurious-gold scenario vocabulary:', S['small']))
        story.append(Spacer(1, 2*mm))
        fs_rows = [[_cell('Sl.', bold=True), _cell('Scenario', bold=True), _cell('Evidence', bold=True)]]
        for m in fraud_matched:
            fs_rows.append([
                _cell(m.get('sl', ''), bold=True, color=AMBER),
                _cell(m.get('title', '')),
                _cell(m.get('evidence', '') or '—', size=7),
            ])
        fs_tbl = Table(fs_rows, colWidths=[bw*0.08, bw*0.42, bw*0.50])
        fs_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0), AMBER_LIGHT),
            ('BOX',           (0,0),(-1,-1), 0.5, AMBER),
            ('INNERGRID',     (0,0),(-1,-1), 0.3, GREY_200),
            ('TOPPADDING',    (0,0),(-1,-1), 5),
            ('BOTTOMPADDING', (0,0),(-1,-1), 5),
            ('LEFTPADDING',   (0,0),(-1,-1), 6),
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
        ]))
        story.append(fs_tbl)
        story.append(Spacer(1, 5*mm))

    # ── 6. Item photographs + material scan ───────────────────────────
    # Every evidence photo captured for the case — not just the first few —
    # each shown as a small thumbnail (fixed column count, independent of how
    # many photos exist) so the certificate never renders a single photo at
    # near-full-page size.
    img_paths   = [p for p in (media.get('images') or []) if p and Path(p).exists()]
    streak_path = (media.get('streak') or '')
    streak_path = streak_path if streak_path and Path(streak_path).exists() else None
    uv_path     = (media.get('uv') or '')
    uv_path     = uv_path if uv_path and Path(uv_path).exists() else None
    audio_path  = (media.get('audio') or '')
    audio_path  = audio_path  if audio_path  and Path(audio_path).exists()  else None

    xray_stages     = (media.get('xray') or {}).get('stages') or {}
    material_path   = xray_stages.get('material')
    material_path   = material_path if material_path and Path(material_path).exists() else None
    gems_path       = xray_stages.get('gems')
    gems_path       = gems_path if gems_path and Path(gems_path).exists() else None
    gold_gem_path   = xray_stages.get('gold_gem')
    gold_gem_path   = gold_gem_path if gold_gem_path and Path(gold_gem_path).exists() else None
    stones_grid_path = xray_stages.get('stones_grid')
    stones_grid_path = stones_grid_path if stones_grid_path and Path(stones_grid_path).exists() else None

    # (path, label) pairs, not a path-keyed dict — a dict would silently
    # collapse two entries that happen to share the same file path (e.g. a
    # xray stage reused across roles) down to whichever label was set last.
    all_photos = (
        [(p, f'Photo {idx + 1}') for idx, p in enumerate(img_paths)]
        + [(p, lbl) for p, lbl in (
            (streak_path, 'Touchstone streak'),
            (uv_path, 'UV-pass'),
            (material_path, 'Material map'),
            (gems_path, 'Stones found'),
            (gold_gem_path, 'Gold / gem split'),
            (stones_grid_path, 'Stone grid overlay'),
        ) if p]
    )

    if all_photos:
        story.append(Paragraph('7. Item Photographs & Material Scan', S['h3']))
        story.append(Spacer(1, 2*mm))
        # (photos grid is large enough that KeepTogether would force a page break — leave as-is)

        # Fixed column count -> consistently small thumbnails whether there's
        # 1 photo or 12; extra photos simply wrap onto further rows.
        n_cols  = min(len(all_photos), 4)
        img_w   = (bw - (n_cols - 1) * 5) / n_cols
        img_h   = img_w * 0.72

        cells, row_buf = [], []
        for path, lbl in all_photos:
            try:
                img_f = RLImage(path, width=img_w, height=img_h, kind='bound')
                cell  = Table([[img_f], [_cell(lbl, color=GREY_500, size=6.5)]],
                              colWidths=[img_w])
                cell.setStyle(TableStyle([
                    ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
                    ('BOX',          (0,0),(-1,-1), 0.5, GREY_200),
                    ('TOPPADDING',   (0,0),(-1,-1), 3),
                    ('BOTTOMPADDING',(0,0),(-1,-1), 3),
                    ('BACKGROUND',   (0,1),(-1, 1), GREY_50),
                ]))
                row_buf.append(cell)
            except Exception:
                row_buf.append(Spacer(img_w, img_h))
            if len(row_buf) == n_cols:
                cells.append(row_buf)
                row_buf = []
        if row_buf:
            while len(row_buf) < n_cols:
                row_buf.append(Spacer(img_w, img_h))
            cells.append(row_buf)

        if cells:
            pg = Table(cells, colWidths=[img_w]*n_cols, hAlign='LEFT')
            pg.setStyle(TableStyle([
                ('TOPPADDING',   (0,0),(-1,-1), 3),
                ('BOTTOMPADDING',(0,0),(-1,-1), 3),
                ('LEFTPADDING',  (0,0),(-1,-1), 0),
                ('RIGHTPADDING', (0,0),(-1,-1), 5),
                ('VALIGN',       (0,0),(-1,-1), 'TOP'),
            ]))
            story.append(pg)
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            f'{len(img_paths)} item photograph(s)' +
            (' + touchstone streak' if streak_path else '') +
            (' + UV-pass image' if uv_path else '') +
            (' + material scan' if material_path else '') +
            (' + stone-detection overlay' if gems_path else '') +
            (' + gold/gem split map' if gold_gem_path else '') +
            (' + stone grid overlay' if stones_grid_path else '') +
            ' — all evidence captured at time of assessment.',
            S['small']))
        story.append(Spacer(1, 5*mm))

    # ── 7. Acoustic waveform ─────────────────────────────────────────
    if audio_path:
        wf = _waveform(audio_path, width=int(bw), height=72)
        if wf:
            a_r = acoustic.get('risk_score', 0.5)
            _, ac = _rl(a_r)
            ac_row = Table([[
                _cell('Acoustic Risk Score', color=GREY_500),
                _cell(f'{a_r:.1%}', bold=True, color=ac),
                _cell(mode_labels.get(acoustic.get('mode', ''), acoustic.get('mode', '—')), color=GREY_500),
            ]], colWidths=[bw*0.28, bw*0.14, bw*0.58])
            ac_row.setStyle(TableStyle(_TS_BASE + [
                ('BOX',       (0,0),(-1,-1), 0.5, GREY_200),
                ('BACKGROUND',(0,0),(-1,-1), GREY_50),
                ('VALIGN',    (0,0),(-1,-1), 'MIDDLE'),
            ]))
            story.append(KeepTogether([
                Paragraph('8. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
                Spacer(1, 2*mm),
                Paragraph(
                    'Amplitude envelope of the recorded ring sound. '
                    'Blue bars = normal amplitude · Red bars = elevated / anomalous.',
                    S['small']),
                Spacer(1, 2*mm),
                _Drawing(wf),
                Spacer(1, 2*mm),
                ac_row,
            ]))
        else:
            story.append(KeepTogether([
                Paragraph('8. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
                Spacer(1, 2*mm),
                _Drawing(_no_waveform(int(bw), 72)),
            ]))
    else:
        story.append(KeepTogether([
            Paragraph('8. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
            Spacer(1, 2*mm),
            _Drawing(_no_waveform(int(bw), 72)),
            Spacer(1, 2*mm),
            Paragraph(
                'No audio recording was submitted with this case. '
                'The acoustic modality defaulted to a neutral 50% risk score.',
                S['small']),
        ]))

    # ════════════════════════════════════════════════════════════════
    # PAGE 3 — VERIFICATION TRACE
    # ════════════════════════════════════════════════════════════════
    trace = case.get('verification_trace') or []
    if trace:
        story.append(PageBreak())
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph('HOW THIS RESULT WAS REACHED — STEP BY STEP', S['h2']))
        story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_200, spaceAfter=5*mm))
        story.append(Paragraph(
            'Every step of the detection pipeline — inputs, formula, outputs, and data source — '
            'kept for officer verification and audit.', S['body']))
        story.append(Spacer(1, 4*mm))

        STATUS_META_PDF = {
            'done':    ('OK',     GREEN),
            'flag':    ('FLAG',   AMBER),
            'skipped': ('SKIPPED', GREY_500),
        }
        for i, step in enumerate(trace, 1):
            tag, tag_c = STATUS_META_PDF.get(step.get('status'), ('OK', GREY_700))
            head = Table([[
                _cell(str(i), bold=True, color=GREY_500),
                _cell(_esc(step.get('step', '')), bold=True, color=GREY_900),
                _cell(tag, bold=True, color=tag_c),
            ]], colWidths=[bw*0.05, bw*0.75, bw*0.20])
            head.setStyle(TableStyle(_TS_BASE + [
                ('BACKGROUND', (0,0), (-1,-1), AMBER_LIGHT if step.get('status') == 'flag' else GREY_50),
                ('BOX',        (0,0), (-1,-1), 0.5, GREY_200),
                ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN',      (2,0), (2,0),   'RIGHT'),
            ]))
            block = [head]
            if step.get('summary'):
                block.append(Spacer(1, 1.5*mm))
                block.append(Paragraph(_esc(step['summary']), S['body']))
            if step.get('formula'):
                block.append(Spacer(1, 1*mm))
                block.append(Paragraph(_esc(step['formula']), S['mono']))
            details = {k: v for k, v in (step.get('details') or {}).items() if k != 'stage_image'}
            if details:
                dt_txt = '  ·  '.join(f'{_esc(str(k))}: {_esc(str(v))}' for k, v in details.items())
                block.append(Spacer(1, 1*mm))
                block.append(Paragraph(dt_txt, S['small']))
            if step.get('source'):
                block.append(Spacer(1, 0.5*mm))
                block.append(Paragraph(f"Source: {_esc(step['source'])}", S['small']))
            block.append(Spacer(1, 3.5*mm))
            story.append(KeepTogether(block))

    # ════════════════════════════════════════════════════════════════
    # PAGE 4 — BENFORD + DECLARATION
    # ════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("BENFORD'S LAW MONITOR & DECLARATION", S['h2']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_200, spaceAfter=5*mm))

    # Benford section
    b_n     = benford.get('n_samples', 0)
    b_p     = benford.get('p_value')
    b_alert = benford.get('alert', False)
    b_obs   = benford.get('digit_observed', [])
    b_exp   = benford.get('digit_expected', [])

    # Per-evaluator slice (P3-11): localises an anomaly to a single corrupt
    # officer rather than diluting it across the whole branch.
    benford_ev = case.get('benford_evaluator', {}) or {}
    be_alert   = benford_ev.get('alert', False)
    be_n       = benford_ev.get('n_samples', 0)
    be_p       = benford_ev.get('p_value')
    evaluator_id = (case.get('evaluator') or {}).get('evaluator_id', '—')

    bb = [[_cell('PARAMETER'), _cell('RESULT'), _cell('NOTE')]]
    _be_result = (
        ('🚨 ANOMALY' if be_alert else '✓ Normal') + (f' (p={be_p:.4f}, n={be_n})' if be_p is not None else f' (n={be_n})')
    ) if be_n else 'Insufficient attributed data'
    for row in [
        ['Samples Analysed', str(b_n),                                   'Min 30 needed for reliable test'],
        ['Chi-Squared Test', f'p = {b_p:.4f}' if b_p else '—',          'p < 0.05 triggers alert'],
        ['Alert Status',     '🚨 ANOMALY DETECTED' if b_alert else '✓ Normal distribution', ''],
        ['Branch',           case.get('branch_id', '—'),                  ''],
        ['Per-evaluator (' + str(evaluator_id) + ')', _be_result, 'Localises anomaly to one officer'],
    ]:
        clr = RED if (row[0]=='Alert Status' and b_alert) else \
              GREEN if row[0]=='Alert Status' else \
              (RED if be_alert else GREEN) if row[0].startswith('Per-evaluator') and be_n else GREY_900
        bb.append([_cell(row[0]), _cell(row[1], bold=True, color=clr), _cell(row[2])])
    bt = Table(bb, colWidths=[bw*0.28, bw*0.28, bw*0.44])
    bt.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  FONT_BOLD),
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
    ]))
    story.append(KeepTogether([
        Paragraph("9. Benford's Law Population Monitor", S['h3']),
        Spacer(1, 2*mm),
        Paragraph(
            "The Benford's Law Monitor analyses the distribution of first significant digits in "
            "submerged weight measurements across all branch appraisals. For genuine items, first "
            "digits follow Benford's natural logarithmic distribution (P[d] = log₁₀(1 + 1/d)). "
            "Systematic deviation (p < 0.05 on chi-squared test) indicates possible organised fraud "
            "at branch level.", S['body']),
        Spacer(1, 3*mm),
        bt,
    ]))

    if b_obs and b_exp and len(b_obs) == 9:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph('First-Digit Distribution Chart', S['label']))
        story.append(Spacer(1, 2*mm))
        chart_w = int(bw)
        story.append(_Drawing(_benford_chart(b_obs, b_exp, width=chart_w, height=90)))
        story.append(Spacer(1, 2*mm))
        leg = Table([[
            _Drawing(_legend_swatch(BLUE)),   _cell('  Observed distribution'),
            _Drawing(_legend_swatch(GREY_300)), _cell('  Benford expected'),
        ]], colWidths=[14, 110, 14, 110])
        leg.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
                                 ('TOPPADDING', (0,0),(-1,-1), 0),
                                 ('BOTTOMPADDING', (0,0),(-1,-1), 0)]))
        story.append(leg)

    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY_200))
    story.append(Spacer(1, 5*mm))

    # Declaration
    story.append(KeepTogether([
        Paragraph('DECLARATION & AUTHORISATION', S['h3']),
        Spacer(1, 3*mm),
        Paragraph(
            'This Gold Purity Assessment has been conducted using the KANCHAN-AI system (Version 1.0), '
            'a multi-modal AI platform developed for Canara Bank\'s Gold Loan Division. '
            'The system employs independent analysis modalities — Archimedes density physics, '
            'acoustic MFCC-ΔΔ fingerprinting with ring-frequency cross-check, a classical computer-vision '
            'material scan (DSIP) for surface/stone analysis, and HSV touchstone streak physics — '
            'combined via a transparent, hand-recomputable evidence-fusion model. '
            'Population-level fraud detection is provided by a Benford\'s Law monitor on submerged '
            'weight measurements. AI-generated verdicts are advisory in nature; the final lending '
            'decision remains the sole responsibility of the authorised branch officer.',
            S['body']),
    ]))
    story.append(Spacer(1, 8*mm))

    # Signatures
    def _sig_block(title, name, date_str):
        return Table([
            [_cell(title, color=GREY_500)],
            [Spacer(1, 14)],
            [HRFlowable(width='90%', thickness=0.5, color=GREY_300)],
            [_cell(name if name != '—' else '________________________________',
                   bold=bool(name and name != '—'))],
            [_cell('Name & Employee ID', color=GREY_500)],
            [Spacer(1, 3)],
            [_cell(date_str if date_str else ts_short)],
            [_cell('Date & Time', color=GREY_500)],
        ], colWidths=[bw*0.45])

    sig_tbl = Table([[
        _sig_block('Assessment Officer', cofficer, ts_short),
        Spacer(1, 1),
        _sig_block('Branch Manager', '—', '___ / ___ / ______'),
    ]], colWidths=[bw*0.47, bw*0.06, bw*0.47])
    sig_tbl.setStyle(TableStyle([
        ('VALIGN',      (0,0),(-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0),(-1,-1), 0),
        ('RIGHTPADDING',(0,0),(-1,-1), 0),
        ('TOPPADDING',  (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 6*mm))

    # Audit trail
    a_body = [[_cell('AUDIT TRAIL', bold=True, color=BLUE_DARKER), '', '', '']]
    for row in [
        ['Case ID',      f'#{case_id}',                    'Analysis Timestamp', ts_display],
        ['AI System',    'KANCHAN-AI v1.0',                'LLM Provider',       verdict.get('llm_provider','heuristic')],
        ['Fusion Model', {'logodds': 'Log-odds', 'xgboost': 'XGBoost', 'heuristic': 'Weighted-heuristic'}.get(fmode, fmode).title() + f" ({fmode})", 'Benford Samples', str(b_n)],
        ['Branch',       case.get('branch_id','—'),        'Report Generated',   datetime.now(IST).strftime('%d %b %Y %H:%M IST')],
    ]:
        a_body.append([_cell(row[0], color=GREY_500), _cell(row[1], bold=True),
                       _cell(row[2], color=GREY_500), _cell(row[3], bold=True)])
    at = Table(a_body, colWidths=[bw*0.22, bw*0.28, bw*0.22, bw*0.28])
    at.setStyle(TableStyle(_TS_BASE + [
        ('SPAN',         (0,0), (3,0)),
        ('BACKGROUND',   (0,0), (3,0),  BLUE_LIGHT),
        ('BACKGROUND',   (0,1), (3,-1), GREY_50),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
    ]))
    story.append(at)

    doc.build(story)
    return buf.getvalue()


# ── Endpoint ────────────────────────────────────────────────────────────
@router.get('/history/{case_id}/report')
async def download_report(case_id: str):
    if not HISTORY_PATH.exists():
        raise HTTPException(status_code=404, detail='No case history found')
    try:
        history = json.loads(HISTORY_PATH.read_text())
    except Exception:
        raise HTTPException(status_code=500, detail='Failed to read case history')
    case = next((c for c in history if c.get('case_id') == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail=f'Case {case_id} not found')
    try:
        pdf_bytes = build_report(case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'PDF generation failed: {e}')
    IST   = timezone(timedelta(hours=5, minutes=30))
    fname = f'KANCHAN_{case_id}_{datetime.now(IST).strftime("%Y%m%d")}.pdf'
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )
