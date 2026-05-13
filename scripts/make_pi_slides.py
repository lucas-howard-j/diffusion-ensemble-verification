"""
PI Meeting draft slide deck — DDPM SSH S2S Forecast, Phase 1 Results
Run: conda run -n ssh_ml python make_pi_slides.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.oxml.ns import qn
from lxml import etree
import copy

OUT = "figures/pi_meeting_draft.pptx"

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x06, 0x1C, 0x30)   # title / dark bg
BLUE    = RGBColor(0x06, 0x5A, 0x82)   # primary
TEAL    = RGBColor(0x1C, 0x72, 0x93)   # secondary
SEAFOAM = RGBColor(0x02, 0xC3, 0x9A)   # positive / accent
AMBER   = RGBColor(0xE6, 0x7E, 0x22)   # warning / negative
LGRAY   = RGBColor(0xF0, 0xF7, 0xFA)   # light bg
MID     = RGBColor(0x44, 0x6A, 0x7C)   # muted text
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
BLACK   = RGBColor(0x1A, 0x1A, 0x2E)

W, H = Inches(13.33), Inches(7.5)   # LAYOUT_WIDE


# ── Helpers ───────────────────────────────────────────────────────────────────
def prs_new():
    p = Presentation()
    p.slide_width  = W
    p.slide_height = H
    return p


def add_rect(slide, x, y, w, h, fill=None, line=None):
    from pptx.util import Inches
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.line.fill.background()
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, x, y, w, h,
             size=16, bold=False, italic=False,
             color=BLACK, align=PP_ALIGN.LEFT,
             font="Calibri", margin=None):
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    if margin is not None:
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = margin
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font
    return txb


def add_multiline(slide, lines, x, y, w, h,
                  size=14, color=BLACK, font="Calibri",
                  bold_first=False, bullet=False, line_spacing=None):
    """lines: list of str or (str, dict) tuples with per-line overrides."""
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(lines):
        if isinstance(line, tuple):
            txt, opts = line
        else:
            txt, opts = line, {}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", PP_ALIGN.LEFT)
        if line_spacing:
            from pptx.util import Pt as PT2
            p.line_spacing = PT2(line_spacing)
        run = p.add_run()
        run.text = txt
        run.font.size   = Pt(opts.get("size", size))
        run.font.bold   = opts.get("bold", bold_first and i == 0)
        run.font.italic = opts.get("italic", False)
        run.font.color.rgb = opts.get("color", color)
        run.font.name   = opts.get("font", font)
    return txb


def bg_solid(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_title(prs):
    """Slide 1 — Title (dark navy)"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg_solid(sl, NAVY)

    # Left accent bar
    add_rect(sl, 0, 0, 0.18, 7.5, fill=SEAFOAM)

    # Kicker
    add_text(sl, "BARNES LAB  ·  PHASE 1  ·  PI MEETING",
             0.5, 1.0, 11, 0.5, size=11, color=MID, font="Calibri",
             align=PP_ALIGN.LEFT)

    # Main title
    add_multiline(sl, [
        ("DDPM S2S SSH Forecast", {"size": 42, "bold": True, "color": WHITE}),
        ("Phase 1 Results: Marginal Skill, Energy Score, Variogram Score",
         {"size": 22, "color": RGBColor(0xA8, 0xD4, 0xE8)}),
    ], 0.5, 1.65, 12, 2.2, line_spacing=32)

    # Divider line
    add_rect(sl, 0.5, 3.65, 10, 0.03, fill=TEAL)

    # Sub-details
    add_multiline(sl, [
        ("Test set locked: 2022–2024  ·  132 initializations  ·  8 US East Coast gauges",
         {"size": 14, "color": RGBColor(0x7E, 0xB8, 0xD0)}),
        ("Model: gauge-space DDPM  ·  Conditioning: SSH+SST EOFs, NAO, ENSO, lead  ·  ~165k params",
         {"size": 12, "color": MID}),
    ], 0.5, 3.85, 12, 0.9, line_spacing=22)

    # Bottom right label
    add_text(sl, "2026-04-28", 11.5, 6.9, 1.5, 0.4,
             size=10, color=MID, align=PP_ALIGN.RIGHT)

    return sl


def slide_architecture(prs):
    """Slide 2 — Model Architecture"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, LGRAY)
    add_rect(sl, 0, 0, 13.33, 0.75, fill=BLUE)

    # Title
    add_text(sl, "Model Architecture", 0.4, 0.1, 10, 0.6,
             size=26, bold=True, color=WHITE, font="Calibri")

    # ── Three boxes: Inputs → DDPM → Outputs ──
    box_y, box_h = 1.1, 4.8
    col_w = 3.5

    # Input box
    add_rect(sl, 0.4, box_y, col_w, box_h, fill=WHITE, line=TEAL)
    add_rect(sl, 0.4, box_y, col_w, 0.45, fill=BLUE)
    add_text(sl, "INPUTS", 0.4, box_y + 0.03, col_w, 0.42,
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    add_multiline(sl, [
        ("SSH anomaly PCs",      {"bold": True, "size": 13, "color": BLUE}),
        ("15 modes × 3 lags  (GLORYS12V1)", {"size": 11, "color": MID}),
        ("",                     {}),
        ("SST anomaly PCs",      {"bold": True, "size": 13, "color": BLUE}),
        ("15 modes × 3 lags  (GLORYS12V1)", {"size": 11, "color": MID}),
        ("",                     {}),
        ("Wind stress PCs",      {"bold": True, "size": 13, "color": BLUE}),
        ("10 modes  (ERA5)",     {"size": 11, "color": MID}),
        ("",                     {}),
        ("NAO index",            {"bold": True, "size": 13, "color": BLUE}),
        ("(ERA5-derived)",       {"size": 11, "color": MID}),
        ("",                     {}),
        ("Niño3.4",              {"bold": True, "size": 13, "color": BLUE}),
        ("(GLORYS SST)",         {"size": 11, "color": MID}),
        ("",                     {}),
        ("Lead time",            {"bold": True, "size": 13, "color": BLUE}),
        ("2 / 4 / 8 / 12 weeks", {"size": 11, "color": MID}),
    ], 0.6, box_y + 0.6, col_w - 0.3, box_h - 0.7, line_spacing=16)

    # Arrow 1
    add_text(sl, "→", 4.05, 3.1, 0.6, 0.5, size=28, bold=True, color=TEAL,
             align=PP_ALIGN.CENTER)

    # DDPM box
    ddpm_x = 4.7
    add_rect(sl, ddpm_x, box_y, col_w, box_h, fill=WHITE, line=SEAFOAM)
    add_rect(sl, ddpm_x, box_y, col_w, 0.45, fill=SEAFOAM)
    add_text(sl, "DDPM", ddpm_x, box_y + 0.03, col_w, 0.42,
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    add_multiline(sl, [
        ("Denoising Diffusion\nProbabilistic Model", {"bold": True, "size": 15, "color": BLACK, "align": PP_ALIGN.CENTER}),
        ("", {}),
        ("Architecture", {"bold": True, "size": 12, "color": BLUE}),
        ("MLP:  256 → 256 → 128 → 8", {"size": 11, "color": MID}),
        ("cond_dim = 131", {"size": 11, "color": MID}),
        ("~165k parameters", {"size": 11, "color": MID}),
        ("", {}),
        ("Training", {"bold": True, "size": 12, "color": BLUE}),
        ("GLORYS SSH anomaly at gauges", {"size": 11, "color": MID}),
        ("1993–2018  (TRAIN)", {"size": 11, "color": MID}),
        ("Val loss: 0.1592  (ep ~156)", {"size": 11, "color": MID}),
        ("", {}),
        ("Generates 50-member", {"size": 11, "color": MID}),
        ("SSH anomaly ensembles", {"size": 11, "color": MID}),
    ], ddpm_x + 0.2, box_y + 0.6, col_w - 0.35, box_h - 0.7, line_spacing=16)

    # Arrow 2
    add_text(sl, "→", 8.35, 3.1, 0.6, 0.5, size=28, bold=True, color=TEAL,
             align=PP_ALIGN.CENTER)

    # Outputs box
    out_x = 9.0
    add_rect(sl, out_x, box_y, col_w, box_h, fill=WHITE, line=BLUE)
    add_rect(sl, out_x, box_y, col_w, 0.45, fill=BLUE)
    add_text(sl, "OUTPUTS", out_x, box_y + 0.03, col_w, 0.42,
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    gauges = ["Eastport ME", "Boston MA", "Atlantic City NJ",
              "Battery NY", "Sewells Point VA",
              "Charleston SC", "Mayport FL", "Key West FL"]
    lines = [("8 NOAA Tide Gauges", {"bold": True, "size": 13, "color": BLUE}), ("US East Coast", {"size": 11, "color": MID}), ("", {})]
    for g in gauges:
        lines.append((f"▸  {g}", {"size": 11, "color": BLACK}))
    lines += [("", {}), ("SSH anomaly (cm)", {"size": 11, "color": MID}),
              ("GLORYS-component", {"size": 11, "color": MID})]
    add_multiline(sl, lines, out_x + 0.2, box_y + 0.55, col_w - 0.35, box_h - 0.65, line_spacing=16)

    # Bottom note
    add_text(sl, "* Model predicts GLORYS-component of coastal SSH — NOAA gauges used for independent evaluation",
             0.4, 6.3, 12.5, 0.45, size=10, italic=True, color=MID)

    return sl


def slide_crps(prs):
    """Slide 3 — Marginal CRPS Skill"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, LGRAY)
    add_rect(sl, 0, 0, 13.33, 0.75, fill=SEAFOAM)
    add_text(sl, "Marginal Skill: CRPS vs Seasonal Climatology", 0.4, 0.1, 11, 0.6,
             size=26, bold=True, color=WHITE, font="Calibri")
    add_text(sl, "TEST SET  ·  2022–2024  ·  132 inits",
             10.5, 0.22, 2.7, 0.4, size=10, color=WHITE, align=PP_ALIGN.RIGHT)

    # ── Bar chart: mean CRPS skill by lead ──
    chart_data = ChartData()
    chart_data.categories = ["2 wk", "4 wk", "8 wk", "12 wk"]
    chart_data.add_series("Mean skill", (0.075, 0.104, 0.050, 0.025))

    from pptx.util import Inches as IN
    chart = sl.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        IN(0.4), IN(0.9), IN(5.5), IN(3.7),
        chart_data
    ).chart

    chart.chart_title.text_frame.text = "Mean CRPS Skill (all 8 stations)"
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True

    chart.plots[0].series[0].format.fill.solid()
    chart.plots[0].series[0].format.fill.fore_color.rgb = SEAFOAM

    plot = chart.plots[0]
    chart.has_legend = False

    vs = chart.value_axis
    vs.minimum_scale = 0.0
    vs.maximum_scale = 0.12
    vs.major_gridlines.format.line.color.rgb = RGBColor(0xCC, 0xDD, 0xE5)

    # Callout boxes: mean skill headline
    cx, cy = 0.4, 4.75
    means = [("2 wk", "+0.075", SEAFOAM), ("4 wk", "+0.104", SEAFOAM),
             ("8 wk", "+0.050", SEAFOAM), ("12 wk", "+0.025", SEAFOAM)]
    bw = 1.28
    for i, (lbl, val, col) in enumerate(means):
        bx = cx + i * (bw + 0.06)
        add_rect(sl, bx, cy, bw, 0.9, fill=WHITE, line=TEAL)
        add_rect(sl, bx, cy, bw, 0.28, fill=col)
        add_text(sl, lbl, bx, cy + 0.02, bw, 0.26,
                 size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)
        add_text(sl, val, bx, cy + 0.3, bw, 0.5,
                 size=20, bold=True, color=col, align=PP_ALIGN.CENTER)

    # ── Per-station table ──
    stations = ["East", "Boston", "Atl City", "Battery", "Sewells", "Charleston", "Mayport", "Key West"]
    data = [
        ["+.011", "+.055", "+.087", "+.044", "+.092", "+.065", "+.066", "+.178"],
        ["+.063", "+.111", "+.132", "+.089", "+.142", "+.073", "+.075", "+.149"],
        ["−.002", "+.026", "+.039", "+.013", "+.057", "+.063", "+.067", "+.134"],
        ["+.004", "+.031", "+.036", "−.005", "+.014", "+.006", "−.011", "+.121"],
    ]
    leads = ["2 wk", "4 wk", "8 wk", "12 wk"]

    tx = 6.1; ty = 0.9; tw = 6.9; col_w2 = tw / (len(stations) + 1)
    row_h = 0.38
    header_bg = BLUE

    # Header row
    add_rect(sl, tx, ty, col_w2, row_h, fill=header_bg)
    add_text(sl, "Lead", tx, ty + 0.03, col_w2, row_h - 0.06,
             size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)
    for j, s in enumerate(stations):
        bx = tx + (j + 1) * col_w2
        add_rect(sl, bx, ty, col_w2, row_h, fill=header_bg)
        add_text(sl, s, bx, ty + 0.03, col_w2, row_h - 0.06,
                 size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    for i, (lead, row) in enumerate(zip(leads, data)):
        ry = ty + (i + 1) * row_h
        bg = WHITE if i % 2 == 0 else LGRAY
        add_rect(sl, tx, ry, col_w2, row_h, fill=header_bg)
        add_text(sl, lead, tx, ry + 0.03, col_w2, row_h - 0.06,
                 size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)
        for j, val in enumerate(row):
            bx = tx + (j + 1) * col_w2
            cell_color = RGBColor(0xD5, 0xF5, 0xE3) if not val.startswith("−") else RGBColor(0xFD, 0xED, 0xEC)
            add_rect(sl, bx, ry, col_w2, row_h, fill=cell_color, line=LGRAY)
            txt_color = RGBColor(0x1A, 0x73, 0x48) if not val.startswith("−") else RGBColor(0xC0, 0x39, 0x2B)
            add_text(sl, val, bx, ry + 0.04, col_w2, row_h - 0.08,
                     size=10, bold=False, color=txt_color, align=PP_ALIGN.CENTER, margin=0)

    # Bottom message
    add_rect(sl, 0.4, 5.85, 12.5, 0.55, fill=RGBColor(0xE8, 0xF8, 0xF5), line=SEAFOAM)
    add_text(sl, "✓  Positive skill at 30/32 station × lead combinations  ·  Mean +0.064 overall  ·  Strongest: Key West 2wk (+0.178), 4wk mean (+0.104)",
             0.6, 5.9, 12.1, 0.45, size=12, bold=True, color=RGBColor(0x1A, 0x73, 0x48))

    return sl


def slide_es_vs(prs):
    """Slide 4 — ES vs VS: Central Finding"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, LGRAY)
    add_rect(sl, 0, 0, 13.33, 0.75, fill=NAVY)
    add_text(sl, "Central Finding: Energy Score vs Variogram Score", 0.4, 0.1, 10, 0.6,
             size=26, bold=True, color=WHITE)
    add_text(sl, "TEST SET  ·  2022–2024  ·  132 inits",
             10.5, 0.22, 2.7, 0.4, size=10, color=MID, align=PP_ALIGN.RIGHT)

    # ── Left panel: ES chart ──
    es_data = ChartData()
    es_data.categories = ["2 wk", "4 wk", "8 wk", "12 wk"]
    es_data.add_series("ES skill", (0.075, 0.095, 0.055, 0.021))

    es_chart = sl.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.35), Inches(1.1), Inches(5.8), Inches(3.1),
        es_data
    ).chart

    es_chart.chart_title.text_frame.text = "Energy Score Skill vs Lead"
    es_chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    es_chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    es_chart.has_legend = False
    es_chart.plots[0].series[0].format.fill.solid()
    es_chart.plots[0].series[0].format.fill.fore_color.rgb = SEAFOAM
    es_chart.value_axis.minimum_scale = 0.0
    es_chart.value_axis.maximum_scale = 0.13

    # ES annotation box
    add_rect(sl, 0.35, 4.35, 5.8, 1.1, fill=RGBColor(0xE8, 0xF8, 0xF5), line=SEAFOAM)
    add_multiline(sl, [
        ("ES verdict: marginal skill is real  ✓", {"bold": True, "size": 13, "color": RGBColor(0x1A, 0x73, 0x48)}),
        ("Δ_ddpm ≈ Δ_clim → joint structure adds ~0 beyond climatology", {"size": 11, "color": MID}),
        ("ES is marginal-dominated  (Pinson & Tastu 2013)", {"size": 11, "color": MID, "italic": True}),
    ], 0.55, 4.4, 5.4, 1.0, line_spacing=18)

    # ── Right panel: VS table ──
    add_rect(sl, 6.5, 1.0, 6.5, 0.4, fill=NAVY)
    add_text(sl, "Variogram Score Skill  (p = 0.5)", 6.5, 1.0, 6.5, 0.42,
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    vs_headers = ["Lead", "DDPM", "DDPM shuf", "GLORYS shuf"]
    vs_data_rows = [
        ("2 wk",  "−0.149", "−1.144", "−0.837"),
        ("4 wk",  "−0.207", "−1.237", "−0.858"),
        ("8 wk",  "−0.161", "−1.258", "−0.887"),
        ("12 wk", "−0.189", "−1.215", "−0.834"),
    ]
    tx2, ty2, tw2 = 6.5, 1.42, 6.5
    cw = tw2 / 4
    rh = 0.42

    for j, hdr in enumerate(vs_headers):
        add_rect(sl, tx2 + j*cw, ty2, cw, rh, fill=TEAL)
        add_text(sl, hdr, tx2 + j*cw, ty2 + 0.04, cw, rh - 0.08,
                 size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    for i, row in enumerate(vs_data_rows):
        ry = ty2 + (i+1) * rh
        bg = WHITE if i % 2 == 0 else LGRAY
        for j, val in enumerate(row):
            if j == 0:
                cell_bg = NAVY
                txt_col = WHITE
                bld = True
            elif j == 1:
                cell_bg = RGBColor(0xFD, 0xED, 0xEC)
                txt_col = RGBColor(0xC0, 0x39, 0x2B)
                bld = True
            else:
                cell_bg = bg
                txt_col = MID
                bld = False
            add_rect(sl, tx2 + j*cw, ry, cw, rh, fill=cell_bg, line=LGRAY)
            add_text(sl, val, tx2 + j*cw, ry + 0.05, cw, rh - 0.1,
                     size=11, bold=bld, color=txt_col, align=PP_ALIGN.CENTER, margin=0)

    # VS annotation box
    add_rect(sl, 6.5, 4.12, 6.5, 1.33, fill=RGBColor(0xFD, 0xED, 0xEC), line=AMBER)
    add_multiline(sl, [
        ("VS verdict: joint structure is present but over-amplified  ✗", {"bold": True, "size": 13, "color": RGBColor(0xC0, 0x39, 0x2B)}),
        ("DDPM is 1.13–1.21× worse than climatology in absolute VS", {"size": 11, "color": MID}),
        ("VS penalizes pair-difference variance → detects covariance inflation", {"size": 11, "color": MID}),
        ("  (Scheuerer & Hamill 2015)", {"size": 11, "color": MID, "italic": True}),
        ("ES cannot distinguish marginal inflation from joint failure", {"size": 11, "color": MID}),
    ], 6.7, 4.17, 6.1, 1.28, line_spacing=18)

    # Bottom punchline
    add_rect(sl, 0.35, 5.65, 12.65, 0.7, fill=NAVY)
    add_text(sl,
             '"DDPM learns the right covariance shape but the wrong scale  —  '
             'ES and VS disagree because ES is marginal-dominated; '
             'VS reveals the over-amplified joint structure"',
             0.55, 5.68, 12.25, 0.65,
             size=13, italic=True, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    return sl


def slide_mechanism(prs):
    """Slide 5 — Calibration Mechanism"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, LGRAY)
    add_rect(sl, 0, 0, 13.33, 0.75, fill=AMBER)
    add_text(sl, "Mechanism: Over-Amplified Covariance Matrix", 0.4, 0.1, 10, 0.6,
             size=26, bold=True, color=WHITE)
    add_text(sl, "TEST SET  ·  2022–2024", 10.5, 0.22, 2.7, 0.4,
             size=10, color=WHITE, align=PP_ALIGN.RIGHT)

    # ── Left: inflation table ──
    add_rect(sl, 0.35, 0.85, 6.2, 0.42, fill=NAVY)
    add_text(sl, "Marginal & Off-Diagonal Inflation vs Observed",
             0.35, 0.85, 6.2, 0.42,
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    inf_hdrs = ["Lead", "Marg. ratio", "Off-diag ratio", "Implied corr ×"]
    inf_rows = [
        ("2 wk",  "1.39×", "2.06×", "1.07×"),
        ("4 wk",  "1.37×", "2.03×", "1.08×"),
        ("8 wk",  "1.37×", "1.99×", "1.06×"),
        ("12 wk", "1.40×", "2.10×", "1.07×"),
    ]
    tx, ty, tw = 0.35, 1.27, 6.2
    cw = tw / 4
    rh = 0.42

    for j, h in enumerate(inf_hdrs):
        add_rect(sl, tx + j*cw, ty, cw, rh, fill=TEAL)
        add_text(sl, h, tx + j*cw, ty + 0.04, cw, rh - 0.08,
                 size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    for i, row in enumerate(inf_rows):
        ry = ty + (i+1) * rh
        bg = WHITE if i % 2 == 0 else LGRAY
        for j, val in enumerate(row):
            if j == 0:
                cell_bg, txt_col, bld = NAVY, WHITE, True
            elif j == 2:
                cell_bg = RGBColor(0xFD, 0xED, 0xEC)
                txt_col = RGBColor(0xC0, 0x39, 0x2B)
                bld = True
            elif j == 1:
                cell_bg = RGBColor(0xFE, 0xF9, 0xE7)
                txt_col = RGBColor(0x9A, 0x5B, 0x00)
                bld = True
            else:
                cell_bg, txt_col, bld = bg, MID, False
            add_rect(sl, tx + j*cw, ry, cw, rh, fill=cell_bg, line=LGRAY)
            add_text(sl, val, tx + j*cw, ry + 0.05, cw, rh - 0.1,
                     size=11, bold=bld, color=txt_col, align=PP_ALIGN.CENTER, margin=0)

    # Inflation diagram (text)
    add_rect(sl, 0.35, 3.65, 6.2, 2.1, fill=WHITE, line=AMBER)
    add_multiline(sl, [
        ("\"Right shape, wrong scale\"", {"bold": True, "size": 15, "color": AMBER}),
        ("", {}),
        ("Marginal variance:   1.37–1.40× observed  (overdispersive)", {"size": 12, "color": BLACK}),
        ("Off-diagonal cov:    2.0–2.1× observed  (over-coherent)", {"size": 12, "color": RGBColor(0xC0, 0x39, 0x2B)}),
        ("Implied corr:         1.06–1.08× obs  (≈ correct shape)", {"size": 12, "color": MID}),
        ("", {}),
        ("→  Both diagonal & off-diagonal inflated together", {"size": 12, "color": NAVY, "bold": True}),
        ("→  Potentially fixable with post-hoc variance rescaling", {"size": 11, "color": MID, "italic": True}),
        ("→  Consistent with Slivinski 2025 / Tian 2024: ML covariances", {"size": 11, "color": MID}),
        ('   "too strong, wrong amplitude"', {"size": 11, "color": MID, "italic": True}),
    ], 0.55, 3.75, 5.85, 1.95, line_spacing=18)

    # ── Right: per-pair VS ──
    add_rect(sl, 6.85, 0.85, 6.1, 0.42, fill=NAVY)
    add_text(sl, "Per-Pair VS Skill — Worst Pairs Drive Deficit",
             6.85, 0.85, 6.1, 0.42,
             size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

    pair_hdrs = ["Station Pair", "2 wk", "4 wk", "8 wk", "12 wk"]
    pair_rows = [
        ("Atlantic – Battery",     "−0.797", "−0.787", "−0.684", "−0.565"),
        ("E-Coast clustered pairs","  high deficit", "", "", ""),
        ("Charleston – Key West",  "≈ 0", "≈ 0", "≈ 0", "≈ 0"),
        ("Eastport – Charleston",  "+ skill", "+ skill", "+ skill", "+ skill"),
    ]

    tx2, ty2, tw2 = 6.85, 1.27, 6.1
    cw2 = [2.3, 0.95, 0.95, 0.95, 0.95]
    rh2 = 0.42

    cx = tx2
    for j, h in enumerate(pair_hdrs):
        add_rect(sl, cx, ty2, cw2[j], rh2, fill=TEAL)
        add_text(sl, h, cx, ty2 + 0.04, cw2[j], rh2 - 0.08,
                 size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)
        cx += cw2[j]

    for i, row in enumerate(pair_rows):
        ry2 = ty2 + (i+1) * rh2
        bg2 = WHITE if i % 2 == 0 else LGRAY
        cx = tx2
        for j, val in enumerate(row):
            if j == 0:
                cell_bg, txt_col, bld = NAVY, WHITE, True
            elif i == 0 and j > 0:
                cell_bg = RGBColor(0xFD, 0xED, 0xEC)
                txt_col = RGBColor(0xC0, 0x39, 0x2B)
                bld = True
            elif i == 3 and j > 0:
                cell_bg = RGBColor(0xD5, 0xF5, 0xE3)
                txt_col = RGBColor(0x1A, 0x73, 0x48)
                bld = True
            else:
                cell_bg, txt_col, bld = bg2, MID, False
            if val:
                add_rect(sl, cx, ry2, cw2[j], rh2, fill=cell_bg, line=LGRAY)
                add_text(sl, val, cx, ry2 + 0.05, cw2[j], rh2 - 0.1,
                         size=10, bold=bld, color=txt_col, align=PP_ALIGN.CENTER, margin=0)
            cx += cw2[j]

    # Interpretation
    add_rect(sl, 6.85, 3.65, 6.1, 2.1, fill=WHITE, line=TEAL)
    add_multiline(sl, [
        ("Key pattern:", {"bold": True, "size": 14, "color": NAVY}),
        ("", {}),
        ("Closest / most-correlated pairs worst:", {"bold": True, "size": 12, "color": BLACK}),
        ("Atlantic–Battery (≈200km, highest r) dominates VS deficit", {"size": 11, "color": MID}),
        ("", {}),
        ("Distant / low-correlation pairs neutral or positive:", {"bold": True, "size": 12, "color": BLACK}),
        ("Model correctly handles uncorrelated, far-apart stations", {"size": 11, "color": MID}),
        ("", {}),
        ("→  Over-amplification concentrated in closely-coupled pairs", {"size": 12, "color": AMBER, "bold": True}),
        ("→  Fig 4b (per-pair VS vs obs corr) will show this gradient", {"size": 11, "color": MID, "italic": True}),
    ], 7.0, 3.72, 5.8, 1.98, line_spacing=17)

    # N→S gradient note
    add_rect(sl, 0.35, 5.85, 12.65, 0.55, fill=NAVY)
    add_text(sl,
             "N→S dispersivity gradient: Eastport & Boston near-calibrated (1.01–1.15×)  "
             "·  Southern stations overdispersive (Charleston 1.88×, Mayport 1.75×)"
             "  ·  diag_frac ≈ 0.20",
             0.55, 5.88, 12.2, 0.5, size=11, color=WHITE, align=PP_ALIGN.CENTER)

    return sl


def slide_decisions(prs):
    """Slide 6 — Open Decisions for PI"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, LGRAY)
    add_rect(sl, 0, 0, 13.33, 0.75, fill=BLUE)
    add_text(sl, "Open Decisions  —  PI Input Needed", 0.4, 0.1, 10, 0.6,
             size=26, bold=True, color=WHITE)
    add_text(sl, "Split: LOCKED ✓", 11.0, 0.2, 2.1, 0.4,
             size=12, bold=True, color=SEAFOAM, align=PP_ALIGN.RIGHT)

    items = [
        {
            "num": "1",
            "title": "NOAA Verification at Weekly Resolution",
            "body": [
                "Currently: NOAA observations are monthly → can't score CRPS/ES/VS against real gauges",
                "Option A  —  Keep GLORYS-at-gauge as primary truth; NOAA monthly as supplementary  (0 days work)",
                "Option C  —  Re-download NOAA hourly via CO-OPS API, resample to weekly  (~1–2 days pipeline rework)",
                "Recommendation: Option C gives a cleaner paper; Option A is defensible for draft",
            ],
            "q": "Decision: proceed with Option A or C?",
        },
        {
            "num": "2",
            "title": "Training Target Framing",
            "body": [
                "Model target = GLORYS-at-gauge SSH anomaly  (not NOAA directly)",
                "Proposed framing: \"model predicts the GLORYS-component of coastal SSH  "
                "at points where independent in-situ verification is available\"",
                "NOAA serves as evaluation-only — quantifies real-world skill against the coastal residual",
                "GLORYS–NOAA residual largely unpredictable from current conditioning → not a training target",
            ],
            "q": "Decision: confirm this framing for Methods section?",
        },
        {
            "num": "3",
            "title": "GLORYS DA Independence Check  (~30 min to verify)",
            "body": [
                "Question: does GLORYS12V1 (GLOBAL_ANALYSIS_PHYSICS_001_024) assimilate tide gauge obs?",
                "If YES → need explicit methods caveat about indirect leakage into training signal",
                "If NO  → clean independence claim; NOAA verification is genuinely held-out",
                "This determines strength of the verification independence argument in the paper",
            ],
            "q": "Decision: who runs this check before submission?",
        },
    ]

    for i, item in enumerate(items):
        iy = 0.9 + i * 2.05
        ih = 1.9
        add_rect(sl, 0.35, iy, 12.65, ih, fill=WHITE,
                 line=TEAL if i < 2 else AMBER)

        # Number circle
        add_rect(sl, 0.35, iy, 0.55, ih, fill=BLUE if i < 2 else AMBER)
        add_text(sl, item["num"], 0.35, iy + ih/2 - 0.28, 0.55, 0.56,
                 size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER, margin=0)

        # Title
        add_text(sl, item["title"], 1.05, iy + 0.07, 11.7, 0.38,
                 size=14, bold=True, color=NAVY)

        # Body lines
        for j, line in enumerate(item["body"]):
            add_text(sl, f"▸  {line}", 1.1, iy + 0.48 + j * 0.28, 11.6, 0.3,
                     size=10, color=MID)

        # Question bar
        add_rect(sl, 1.0, iy + ih - 0.35, 11.95, 0.3,
                 fill=RGBColor(0xEA, 0xF2, 0xF8), line=BLUE if i < 2 else AMBER)
        add_text(sl, item["q"], 1.1, iy + ih - 0.33, 11.7, 0.28,
                 size=10, bold=True, color=BLUE if i < 2 else AMBER)

    # Bottom
    add_rect(sl, 0.35, 7.0, 12.65, 0.35, fill=RGBColor(0xD6, 0xEA, 0xF8))
    add_text(sl,
             "Non-blocking (proceed in parallel): Fig 4b per-pair VS code (~3–5h)  ·  "
             "AGU AI-disclosure check  ·  Ablation runs  ·  System diagram update",
             0.55, 7.02, 12.2, 0.3, size=10, color=NAVY)

    return sl


def slide_summary(prs):
    """Slide 7 — Summary / Next Steps"""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_solid(sl, NAVY)
    add_rect(sl, 0, 0, 0.18, 7.5, fill=SEAFOAM)

    add_text(sl, "Summary", 0.55, 0.5, 11, 0.7,
             size=36, bold=True, color=WHITE, font="Calibri")
    add_rect(sl, 0.55, 1.2, 10, 0.04, fill=TEAL)

    left_lines = [
        ("What works", {"bold": True, "size": 16, "color": SEAFOAM}),
        ("✓  CRPS skill positive at 30/32 station × lead", {"size": 13, "color": WHITE}),
        ("✓  Mean skill +0.064 across all leads & stations", {"size": 13, "color": WHITE}),
        ("✓  Key West strongest  (+0.178 at 2wk)", {"size": 13, "color": WHITE}),
        ("✓  Coherence declines with lead (physically realistic)", {"size": 13, "color": WHITE}),
        ("✓  ES skill positive at every lead", {"size": 13, "color": WHITE}),
        ("", {}),
        ("What needs fixing", {"bold": True, "size": 16, "color": AMBER}),
        ("✗  Joint structure over-amplified  (off-diag 2.0–2.1×)", {"size": 13, "color": RGBColor(0xF0, 0xAA, 0x60)}),
        ("✗  Marginal overdispersive  (1.37–1.40×; N→S gradient)", {"size": 13, "color": RGBColor(0xF0, 0xAA, 0x60)}),
        ("✗  VS worse than climatology in absolute terms", {"size": 13, "color": RGBColor(0xF0, 0xAA, 0x60)}),
    ]
    add_multiline(sl, left_lines, 0.55, 1.35, 6.0, 4.5, line_spacing=22)

    right_lines = [
        ("Critical path to GRL draft", {"bold": True, "size": 16, "color": SEAFOAM}),
        ("", {}),
        ("1.  PI decisions on NOAA verification (→ today)", {"size": 13, "color": WHITE, "bold": True}),
        ("2.  GLORYS DA independence check  (~30 min)", {"size": 13, "color": WHITE}),
        ("3.  Fig 4b: per-pair VS code  (~3–5h)", {"size": 13, "color": WHITE}),
        ("4.  Ablation runs (Full / No-LS / NAO-only)", {"size": 13, "color": WHITE}),
        ("5.  VS sanity checks (p=1, baseline convention)", {"size": 13, "color": WHITE}),
        ("6.  gauge_correlation.py rerun on new split", {"size": 13, "color": WHITE}),
        ("", {}),
        ("Central claim (GRL)", {"bold": True, "size": 14, "color": SEAFOAM}),
        ('"ES and VS disagree: ES is marginal-dominated;", ', {"size": 12, "color": RGBColor(0xA8, 0xD4, 0xE8), "italic": True}),
        ('"VS reveals over-amplified joint structure invisible to ES."', {"size": 12, "color": RGBColor(0xA8, 0xD4, 0xE8), "italic": True}),
    ]
    add_multiline(sl, right_lines, 6.9, 1.35, 6.0, 4.5, line_spacing=22)

    add_rect(sl, 0.55, 5.95, 12.3, 0.03, fill=TEAL)
    add_text(sl, "Slivinski 2025 · Tian 2024 · Pinson & Tastu 2013 · Scheuerer & Hamill 2015",
             0.55, 6.05, 12.3, 0.4, size=11, color=MID, align=PP_ALIGN.CENTER)

    return sl


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    prs = prs_new()
    slide_title(prs)
    slide_architecture(prs)
    slide_crps(prs)
    slide_es_vs(prs)
    slide_mechanism(prs)
    slide_decisions(prs)
    slide_summary(prs)
    prs.save(OUT)
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()
