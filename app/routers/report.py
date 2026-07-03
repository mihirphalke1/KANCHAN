"""
GET /api/history/{case_id}/report
Professional 3-page bank-quality gold purity assessment report.
"""
import io
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.platypus import Image as RLImage

router = APIRouter()

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
        'h2':         s('h2',  fontName='Helvetica-Bold', fontSize=14, textColor=GREY_900,  leading=20),
        'h3':         s('h3',  fontName='Helvetica-Bold', fontSize=11, textColor=GREY_700,  leading=16),
        'body':       s('bd',  fontName='Helvetica',      fontSize=9,  textColor=GREY_700,  leading=14),
        'small':      s('sm',  fontName='Helvetica',      fontSize=8,  textColor=GREY_500,  leading=12),
        'mono':       s('mo',  fontName='Courier',        fontSize=8,  textColor=GREY_700,  leading=12),
        'label':      s('lb',  fontName='Helvetica-Bold', fontSize=7,  textColor=GREY_500,  leading=11, spaceAfter=1),
        'value':      s('vl',  fontName='Helvetica',      fontSize=9,  textColor=GREY_900,  leading=13),
        'value_bold': s('vb',  fontName='Helvetica-Bold', fontSize=9,  textColor=GREY_900,  leading=13),
        'verdict_g':  s('vg',  fontName='Helvetica-Bold', fontSize=20, textColor=GREEN,     leading=26, alignment=TA_CENTER),
        'verdict_b':  s('va',  fontName='Helvetica-Bold', fontSize=20, textColor=AMBER,     leading=26, alignment=TA_CENTER),
        'verdict_r':  s('vr',  fontName='Helvetica-Bold', fontSize=20, textColor=RED,       leading=26, alignment=TA_CENTER),
        'center_sm':  s('cs',  fontName='Helvetica',      fontSize=8,  textColor=GREY_500,  leading=12, alignment=TA_CENTER),
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
        d.add(String(xp, 1, lbl, fontName='Helvetica', fontSize=6,
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
        d.add(String(x+bar/2,  2, str(i+1), fontName='Helvetica', fontSize=7,
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
            d.add(String(xp, 2, lbl, fontName='Helvetica', fontSize=6,
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
                 fontName='Helvetica', fontSize=7, fillColor=GREY_500, textAnchor='middle'))
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
                canvas.setFont('Helvetica-Bold', 15)
                canvas.setFillColor(WHITE)
                canvas.drawString(MARGIN, H - 16*mm, 'CANARA BANK')
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(colors.HexColor('#BEE8FB'))
                canvas.drawString(MARGIN, H - 22*mm, 'Gold Loan Division  ·  AI-Assisted Purity Assessment')
        else:
            text_x = MARGIN
            canvas.setFont('Helvetica-Bold', 15)
            canvas.setFillColor(WHITE)
            canvas.drawString(MARGIN, H - 16*mm, 'CANARA BANK')
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#BEE8FB'))
            canvas.drawString(MARGIN, H - 22*mm, 'Gold Loan Division  ·  AI-Assisted Purity Assessment')

        # Right: system name
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(GOLD)
        canvas.drawRightString(W - MARGIN, H - 16*mm, 'KANCHAN-AI')
        canvas.setFont('Helvetica', 8)
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
        canvas.setFont('Helvetica', 7)
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
            canvas.setFont('Helvetica-Bold', 52)
            canvas.translate(W/2, H/2)
            canvas.rotate(35)
            canvas.drawCentredString(0, 0, 'CONFIDENTIAL')
            canvas.restoreState()
        canvas.restoreState()


# ── Cell paragraph helper ────────────────────────────────────────────────
def _cell(txt, bold=False, color=GREY_700, size=8):
    return Paragraph(str(txt), ParagraphStyle(
        'tc', fontName='Helvetica-Bold' if bold else 'Helvetica',
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
    contra   = case.get('contradiction', {})
    fusion   = case.get('fusion', {})
    benford  = case.get('benford', {})
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
        [Paragraph(f'#{case_id}', ParagraphStyle('mid', fontName='Courier', fontSize=8,
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
    loan_ps = ParagraphStyle('lp', fontName='Helvetica-Bold', fontSize=11,
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
            [Paragraph(plain, S['body'])],
        ])
        story.append(et)
        story.append(Spacer(1, 3*mm))

    # Recommended action
    action = verdict.get('action', '')
    if action:
        at = _section_tbl([
            [_cell('RECOMMENDED ACTION', bold=True, color=GREY_500)],
            [Paragraph(action, ParagraphStyle('ac', fontName='Helvetica-Bold',
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
    dt = Table(d_body, colWidths=[bw*0.30, bw*0.22, bw*0.27, bw*0.21])
    dt.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
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

    # ── 2. Signal breakdown ──────────────────────────────────────────

    mode_labels = {
        'computed':          'Archimedes physics',
        'efficientnet':      'EfficientNet-B3',
        'svm':               'MFCC-ΔΔ SVM',
        'heuristic':         'HSV Heuristic',
        'heuristic:heuristic': 'ZCR+RMS heuristic',
        'heuristic:silent_audio': 'Silent audio (50% default)',
        'heuristic:empty_audio':  'Empty audio (50% default)',
        'heuristic:error':   'Audio error (50% default)',
        'logreg':            'Logistic Regression',
        'no_audio':          'No audio (50% default)',
        'no_images':         'No images (50% default)',
        'no_streak':         'No streak (50% default)',
    }
    def _rl(r):
        if r < 0.35: return ('LOW',      GREEN)
        if r < 0.65: return ('MODERATE', AMBER)
        return           ('HIGH',        RED)

    bar_w = int(bw * 0.28)
    mod_rows = [[_cell('MODALITY'), _cell('RISK'), _cell('RISK BAR'),
                 _cell('LEVEL'), _cell('METHOD')]]
    for name, m in [('Density',density), ('Visual',image_m), ('Acoustic',acoustic), ('Streak',streak)]:
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
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(KeepTogether([
        Paragraph('2. Signal Breakdown — All Modalities', S['h3']),
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
    f_body = [[_cell('METRIC'), _cell('VALUE'), _cell('DESCRIPTION')]]
    for row in [
        ['Fusion Risk Score',   f'{frisk:.2%}',
         'Weighted average: density 35% · acoustic 30% · visual 25% · streak 10%'],
        ['Contradiction Score', f'{contra_sc:.2%}',
         'Max pairwise disagreement across modalities that actually ran'],
        ['Boosted Risk Score',  f'{boosted:.2%}',
         'Fusion + contradiction×0.40 — THIS is the number that determines the verdict'],
        ['Fusion Model', fmode.upper(),
         'XGBoost (10-dim) with SHAP' if fmode == 'xgboost' else
         'Weighted heuristic (density 35%, acoustic 30%, visual 25%, streak 10%)'],
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
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',     (0,3), (-1,3),  'Helvetica-Bold'),  # boosted row bold
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
    ]))
    story.append(KeepTogether([
        Paragraph('3. XGBoost Fusion Analysis', S['h3']),
        Spacer(1, 2*mm),
        ft,
    ]))
    story.append(Spacer(1, 4*mm))

    # ── 4. Contradiction flags ───────────────────────────────────────
    flags = contra.get('flags', [])
    if flags:
        story.append(Paragraph('4. Cross-Modal Contradiction Flags', S['h3']))
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

    # ── 5. Item photographs ──────────────────────────────────────────
    img_paths   = [p for p in (media.get('images') or []) if p and Path(p).exists()]
    streak_path = (media.get('streak') or '')
    streak_path = streak_path if streak_path and Path(streak_path).exists() else None
    audio_path  = (media.get('audio') or '')
    audio_path  = audio_path  if audio_path  and Path(audio_path).exists()  else None

    all_photos = img_paths[:4] + ([streak_path] if streak_path else [])

    if all_photos:
        story.append(Paragraph('5. Item Photographs', S['h3']))
        story.append(Spacer(1, 2*mm))
        # (photos grid is large enough that KeepTogether would force a page break — leave as-is)

        n_cols  = min(len(all_photos), 3)
        img_w   = (bw - (n_cols - 1) * 6) / n_cols
        img_h   = img_w * 0.72

        cells, row_buf = [], []
        for i, path in enumerate(all_photos):
            lbl = 'Touchstone streak' if path == streak_path else f'Photo {i+1}'
            try:
                img_f = RLImage(path, width=img_w, height=img_h, kind='bound')
                cell  = Table([[img_f], [_cell(lbl, color=GREY_500, size=7)]],
                              colWidths=[img_w])
                cell.setStyle(TableStyle([
                    ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
                    ('BOX',          (0,0),(-1,-1), 0.5, GREY_200),
                    ('TOPPADDING',   (0,0),(-1,-1), 4),
                    ('BOTTOMPADDING',(0,0),(-1,-1), 4),
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
            (' + 1 touchstone streak image' if streak_path else '') +
            ' captured at time of assessment.',
            S['small']))
        story.append(Spacer(1, 5*mm))

    # ── 6. Acoustic waveform ─────────────────────────────────────────
    if audio_path:
        wf = _waveform(audio_path, width=int(bw), height=72)
        if wf:
            a_r = acoustic.get('risk_score', 0.5)
            _, ac = _rl(a_r)
            ac_row = Table([[
                _cell('Acoustic Risk Score', color=GREY_500),
                _cell(f'{a_r:.1%}', bold=True, color=ac),
                _cell('MFCC-ΔΔ feature extraction (82-dim) + RBF-SVM classifier', color=GREY_500),
            ]], colWidths=[bw*0.28, bw*0.14, bw*0.58])
            ac_row.setStyle(TableStyle(_TS_BASE + [
                ('BOX',       (0,0),(-1,-1), 0.5, GREY_200),
                ('BACKGROUND',(0,0),(-1,-1), GREY_50),
                ('VALIGN',    (0,0),(-1,-1), 'MIDDLE'),
            ]))
            story.append(KeepTogether([
                Paragraph('6. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
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
                Paragraph('6. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
                Spacer(1, 2*mm),
                _Drawing(_no_waveform(int(bw), 72)),
            ]))
    else:
        story.append(KeepTogether([
            Paragraph('6. Acoustic Fingerprint — Ring Test Waveform', S['h3']),
            Spacer(1, 2*mm),
            _Drawing(_no_waveform(int(bw), 72)),
            Spacer(1, 2*mm),
            Paragraph(
                'No audio recording was submitted with this case. '
                'The acoustic modality defaulted to a neutral 50% risk score.',
                S['small']),
        ]))

    # ════════════════════════════════════════════════════════════════
    # PAGE 3 — BENFORD + DECLARATION
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

    bb = [[_cell('PARAMETER'), _cell('RESULT'), _cell('NOTE')]]
    for row in [
        ['Samples Analysed', str(b_n),                                   'Min 30 needed for reliable test'],
        ['Chi-Squared Test', f'p = {b_p:.4f}' if b_p else '—',          'p < 0.05 triggers alert'],
        ['Alert Status',     '🚨 ANOMALY DETECTED' if b_alert else '✓ Normal distribution', ''],
        ['Branch',           case.get('branch_id', '—'),                  ''],
    ]:
        clr = RED if (row[0]=='Alert Status' and b_alert) else \
              GREEN if row[0]=='Alert Status' else GREY_900
        bb.append([_cell(row[0]), _cell(row[1], bold=True, color=clr), _cell(row[2])])
    bt = Table(bb, colWidths=[bw*0.28, bw*0.28, bw*0.44])
    bt.setStyle(TableStyle(_TS_BASE + [
        ('BACKGROUND',   (0,0), (-1,0),  GREY_100),
        ('BOX',          (0,0), (-1,-1), 0.5, GREY_200),
        ('GRID',         (0,0), (-1,-1), 0.3, GREY_200),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,0),  7),
        ('TEXTCOLOR',    (0,0), (-1,0),  GREY_500),
    ]))
    story.append(KeepTogether([
        Paragraph("7. Benford's Law Population Monitor", S['h3']),
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
            'The system employs four independent analysis modalities — Archimedes density, '
            'acoustic MFCC-ΔΔ fingerprinting, EfficientNet-B3 visual analysis, and touchstone streak '
            'analysis — fused via an XGBoost ensemble model with SHAP explainability. '
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
        ['Fusion Model', f"XGBoost ({fmode})",             'Benford Samples',    str(b_n)],
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
