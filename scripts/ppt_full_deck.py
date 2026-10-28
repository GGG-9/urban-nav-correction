"""Merge every contributor's part into one deck on the green course template.

    python scripts/ppt_full_deck.py

Source parts (E:\\大作业\\新建文件夹):
    1-4....pptx                     cover, agenda, chapter 01 divider, 1.1
    Part_II_..._English_7_Slides    2.1 mechanisation ... 2.6 trajectory
    Part_III_GPU_Acceleration       3.1-3.3 GPU (renumbered 2.6-2.8 here)
    part4(1).pptx                   4.1-4.4 learned module (renumbered 3.1-3.4)
    课程汇报绿色模板_我的部分-v2      3.5/3.6 gating + chapter 04 map prior

The other parts numbered the same material differently (Part III called the GPU
pages 03, part four called the learned module 04, while the agenda in the cover
deck already grouped filtering + GPU as one chapter).  This file re-numbers
everything to the agenda's four-chapter scheme and the order of the experiment:

    01 overview & background   1.1 architecture  1.2 corpus  1.3 drift
    02 filtering & GPU         2.1 mechanisation 2.2 ESKF  2.3 RTS
                               2.4-2.5 results   2.6-2.8 GPU
    03 foundation model        3.1 residual 3.2 20-channel 3.3 GPT4TS 3.4 LoRA
                               3.5 training & gating 3.6 what it bought
    04 spatial information     4.1 road prior 4.2 lane map 4.3 three numbers
    conclusion + thank you

Build order: the seven already-styled pages of the green deck are the base; the
twenty-one new pages are inserted around them with the template's own layouts,
so every page shares one band, one footer, one type scale (微软雅黑).
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageOps
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import ppt_green_deck as g
from ppt_green_deck import (  # noqa: F401
    BAND_TXT, F, F_SERIF, G_DARK, G_LIGHT, G_MAIN, G_PALE, G_TEAL, INK, LINE,
    MUTE, WASH, WHITE, chevron, footer, highlight, pill, quote, rect, rule,
    stat, strip_placeholders, text,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
GREEN_DECK = pathlib.Path(r"E:\大作业\课程汇报绿色模板_我的部分-v2.pptx")
SRC = ROOT / "results" / "ppt" / "_extract"          # images lifted from the parts
GFX = ROOT / "results" / "ppt" / "green"             # figures in the green palette
OUT = pathlib.Path(r"E:\大作业\新建文件夹\GPU-Parallel_Navigation_Correction_Full_Deck.pptx")

T1 = "PART ONE · OVERVIEW & BACKGROUND"
T2 = "PART TWO · FILTERING & GPU BATCHING"
T3 = "PART THREE · FOUNDATION MODEL AND LoRA"
T4 = "PART FOUR · SPATIAL INFORMATION"


# ------------------------------------------------------------------ layout
def new_slide(prs, layout, idx):
    """Add a slide and move it to ``idx`` (python-pptx always appends)."""
    s = prs.slides.add_slide(layout)
    lst = prs.slides._sldIdLst
    sid = lst[-1]
    lst.remove(sid)
    lst.insert(idx, sid)
    strip_placeholders(s)
    return s


def band(prs, layout, idx, title, tag):
    s = new_slide(prs, layout, idx)
    text(s, 0.97, 0.13, 8.35, 0.60, title, size=22, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.45, 0.13, 3.63, 0.60, tag, size=9, color=BAND_TXT,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, spc=40)
    footer(s)
    return s


def pic(s, path, x, y, w, h, frame=False, card=False):
    """Fit an image inside the box, centred, keeping its aspect ratio."""
    path = pathlib.Path(path)
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    if card:
        rect(s, x, y, w, h, fill=WHITE, line=LINE)
    s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                         Inches(pw), Inches(ph))
    if frame and not card:
        rect(s, x, y, w, h, fill=None, line=LINE)


def crop_white(path):
    """Trim the near-white margins off a diagram once, so it fills its box."""
    path = pathlib.Path(path)
    out = path.with_name(path.stem + "_crop" + path.suffix)
    if out.exists():
        return out
    im = Image.open(path).convert("RGB")
    mask = ImageOps.invert(im.convert("L")).point(lambda v: 255 if v > 6 else 0)
    box = mask.getbbox()
    im.crop(box).save(out) if box else im.save(out)
    return out


def kicker(s, x, y, w, label, color=G_TEAL):
    text(s, x, y, w, 0.26, label, size=9.2, bold=True, color=color, spc=80)


def kpi(s, x, y, w, value, label, val_size=30, sub=None):
    text(s, x, y, w, 0.58, value, size=val_size, bold=True, color=G_DARK)
    text(s, x, y + 0.60, w, 0.44, label, size=9.4, color=MUTE, spacing=1.08)
    if sub:
        text(s, x, y + 1.02, w, 0.30, sub, size=8.8, color=MUTE)


def numbered_step(s, x, y, w, num, head, body, size=10.4, num_size=22):
    text(s, x, y - 0.05, 0.55, 0.40, num, size=num_size, bold=True, color=G_LIGHT)
    text(s, x + 0.55, y, w - 0.55, 0.32, head, size=12.2, bold=True, color=INK)
    if body:
        text(s, x + 0.55, y + 0.40, w - 0.55, 0.80, body, size=size, color=MUTE,
             spacing=1.12)


def divider(prs, layout, idx, num, part, title, sub, bullets, sub_w=7.0):
    s = new_slide(prs, layout, idx)
    text(s, 0.20, 2.83, 3.40, 1.90,
         [[(num, {"size": 60, "bold": True})],
          [(part, {"size": 24, "font": F_SERIF, "spc": 400})]],
         color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.15)
    text(s, 4.94, 1.90, 7.86, 1.24, title, size=40, bold=True, color=INK,
         anchor=MSO_ANCHOR.MIDDLE)
    box = text(s, 5.66, 3.22, sub_w, 0.46, "", size=19, font=F_SERIF, color=INK)
    p = box.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = sub
    g._style_run(r, 19, INK, False, F_SERIF, False, None)
    highlight(r)
    if len(bullets) > 5:
        y0, gap, size = 3.92, 0.45, 16.0
    elif len(bullets) > 3:
        y0, gap, size = 3.96, 0.50, 17.0
    else:
        y0, gap, size = 4.02, 0.60, 19.0
    for i, b in enumerate(bullets):
        y = y0 + i * gap
        tri = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(5.72),
                                 Inches(y + 0.07), Inches(0.16), Inches(0.13))
        tri.rotation = 90
        tri.shadow.inherit = False
        tri.fill.solid()
        tri.fill.fore_color.rgb = G_MAIN
        tri.line.fill.background()
        text(s, 6.08, y, 6.65, 0.40, b, size=size, color=INK)
    return s


# -------------------------------------------------------------------- pages
def p_cover(prs, lay, i):
    s = new_slide(prs, lay, i)
    rect(s, 0, 0, 4.75, 7.5, fill=G_DARK)
    rect(s, 0.20, 0.26, 4.35, 6.98, fill=None, line=WHITE, lw=1.0)
    pic(s, SRC / "seal.png", 1.52, 0.82, 1.70, 1.70)

    text(s, 0.60, 3.12, 3.55, 0.32, "MS022161", size=12, color=G_PALE,
         align=PP_ALIGN.CENTER, spc=300)
    text(s, 0.55, 3.50, 3.65, 0.66, "Engineering Big-Data Analysis", size=12.5,
         color=WHITE, align=PP_ALIGN.CENTER, spacing=1.05)
    rule(s, 1.78, 4.36, 1.20, 0.012, color=G_LIGHT)
    text(s, 0.55, 4.56, 3.65, 0.34, "Southeast University", size=12.5,
         font=F_SERIF, color=G_PALE, align=PP_ALIGN.CENTER)
    text(s, 0.55, 6.55, 3.65, 0.28, "October 2026", size=10, color=G_PALE,
         align=PP_ALIGN.CENTER)

    text(s, 5.42, 1.24, 7.30, 0.28, "COURSE PROJECT · GROUP PRESENTATION",
         size=10.5, bold=True, color=G_MAIN, spc=120)
    text(s, 5.40, 1.66, 7.30, 2.10,
         "GPU-Parallel Batched Kalman Smoothing and a LoRA-Tuned Time-Series "
         "Foundation Model", size=29, bold=True, color=INK, spacing=1.06)
    text(s, 5.42, 3.90, 7.20, 0.80, "for Urban Vehicular Integrated-Navigation "
         "Correction", size=20, color=G_TEAL, spacing=1.05)
    rule(s, 5.45, 4.88, 1.55, 0.030, color=G_MAIN)
    text(s, 5.45, 5.06, 7.10, 0.74,
         "Physics baseline · GPU acceleration · gated residual learning · "
         "road-centreline prior", size=12.5, font=F_SERIF, color=MUTE, spacing=1.12)
    text(s, 5.45, 6.30, 7.20, 0.30,
         "Guo Haisu  ·  MS022161  ·  Southeast University", size=11, color=INK)
    return s


def p_agenda(prs, lay, i):
    s = band(prs, lay, i, "Contents", "MS022161 · ENGINEERING BIG-DATA ANALYSIS")
    rows = [
        ("01", "Overview & Background",
         "Three-tier architecture, the 8.64 M-row sensor corpus, and the O(t²) "
         "mechanics of inertial drift."),
        ("02", "Filtering & GPU Batching",
         "15-state error-state Kalman filter, full-state RTS smoothing, "
         "192-window CUDA parallelism, and the runtime/accuracy trade-off."),
        ("03", "Foundation Model & LoRA",
         "20-channel sensor tensors, a GPT4TS-style patched DistilGPT2, LoRA "
         "fine-tuning, then training and safety gating."),
        ("04", "Spatial Information & Conclusion",
         "Continuity-constrained road projection, the lane-level map result, and "
         "what the four tiers actually bought."),
    ]
    for k, (num, title, desc) in enumerate(rows):
        y = 1.52 + k * 1.30
        text(s, 0.62, y - 0.10, 0.95, 0.70, num, size=34, bold=True, color=G_LIGHT)
        text(s, 1.62, y - 0.04, 10.9, 0.40, title, size=17, bold=True, color=INK)
        text(s, 1.62, y + 0.44, 10.6, 0.66, desc, size=10.6, color=MUTE, spacing=1.12)
        if k < 3:
            rule(s, 0.62, y + 1.14, 12.10)
    return s


def p_11(prs, lay, i):
    s = band(prs, lay, i, "1.1  Why a Three-Tier Architecture?", T1)
    quote(s, 1.02, "Each tier removes what the one before cannot — physics, then "
                   "compute, then learned and spatial information.")
    pic(s, SRC / "one4_s04_01.png", 0.62, 1.90, 4.50, 4.40, frame=True)
    text(s, 0.62, 6.42, 4.55, 0.42,
         "MEMS IMU / AHRS and GNSS antenna — the sensor class this pipeline targets.",
         size=8.8, color=MUTE, spacing=1.05)

    tiers = [
        ("TIER 1", "Physical filtering & smoothing (ESKF + RTS)",
         "A 15-state filter fuses 100 Hz IMU with 1 Hz GNSS; an RTS pass then "
         "uses the post-outage fixes. Outage error 104.65 m → 9.93 m (−90.5 %)."),
        ("TIER 2", "GPU window-batched parallelism",
         "192 windows from 48 routes in one batch on a single RTX 4090. "
         "1 M IMU rows in 46.2 s instead of 2,245 s; the whole 8.64 M-row "
         "corpus in 17 minutes."),
        ("TIER 3", "Learned residual and a map prior",
         "A LoRA-tuned DistilGPT2 reads 20-channel tensors and is admitted only "
         "through a one-standard-error gate. A road-centreline prior then takes "
         "B3 from 9.93 to 6.30 m."),
    ]
    for k, (tag, head, body) in enumerate(tiers):
        y = 1.90 + k * 1.57
        pill(s, 5.55, y + 0.02, 0.98, 0.34, tag, size=8.4)
        text(s, 6.66, y - 0.02, 6.00, 0.40, head, size=13, bold=True, color=INK)
        text(s, 5.55, y + 0.50, 7.05, 0.90, body, size=10.4, color=MUTE, spacing=1.12)
    return s


def p_12(prs, lay, i):
    s = band(prs, lay, i, "1.2  The 8.64 M-Row Sensor Corpus", T1)
    quote(s, 1.02, "Forty-eight routes of 1,800 s at 100 Hz — 8.64 M IMU rows, "
                   "86,400 GNSS fixes, and 48 injected 60 s outages.")

    for k, (v, lab) in enumerate([("48", "routes, each 1,800 s"),
                                  ("8.64 M", "IMU rows at 100 Hz"),
                                  ("86,400", "GNSS fixes at 1 Hz"),
                                  ("60 s", "per injected outage")]):
        kpi(s, 0.62 + k * 3.06, 1.86, 2.85, v, lab, val_size=30)
    rule(s, 0.62, 2.92, 12.10)

    kicker(s, 0.62, 3.06, 5.6, "ROUTE SPLIT")
    split = [("Train", "35 routes", "9,310 windows"),
             ("Validation", "5 routes", "1,330 windows"),
             ("Held-out test", "8 routes", "2,128 windows")]
    for k, (name, r, w) in enumerate(split):
        y = 3.50 + k * 0.78
        text(s, 0.62, y, 1.95, 0.34, name, size=11.5, bold=True, color=INK)
        text(s, 2.60, y + 0.02, 1.55, 0.32, r, size=10.4, color=MUTE)
        text(s, 4.05, y + 0.02, 1.75, 0.32, w, size=10.4, color=MUTE)
    text(s, 0.62, 5.62, 5.30, 0.62,
         "Windows never cross a route boundary, and no test route contributes a "
         "normalisation statistic.", size=9.2, color=MUTE, spacing=1.12)

    rect(s, 6.22, 3.06, 0.012, 3.04, fill=LINE)
    kicker(s, 6.55, 3.06, 5.9, "TWENTY CHANNELS PER STEP (10 Hz)")
    groups = [("6", "IMU", "gyro (3) and accelerometer (3), bias-corrected"),
              ("5", "context", "GNSS innovation (3), outage flag, time since fix"),
              ("9", "filter state", "velocity (3), accelerometer bias (3), gyro bias (3)")]
    for k, (n, name, body) in enumerate(groups):
        y = 3.50 + k * 0.92
        text(s, 6.55, y - 0.06, 0.60, 0.44, n, size=19, bold=True, color=G_TEAL)
        text(s, 7.22, y, 1.55, 0.32, name, size=11, bold=True, color=INK)
        text(s, 8.85, y + 0.02, 3.90, 0.60, body, size=9.8, color=MUTE, spacing=1.1)

    rect(s, 0.62, 6.32, 12.10, 0.48, fill=WASH)
    text(s, 0.86, 6.32, 11.6, 0.48,
         [[("SIMULATED CORPUS", {"bold": True, "color": G_TEAL, "spc": 60}),
           ("    A self-written analytic IMU/GNSS simulator (configs/imu_gnss.yaml); "
            "every number in this deck is measured on it.", {"color": MUTE})]],
         size=8.8, anchor=MSO_ANCHOR.MIDDLE)
    return s


def p_13(prs, lay, i):
    s = band(prs, lay, i, "1.3  Why Physics Alone Drifts", T1)
    quote(s, 1.02, "A constant accelerometer bias is integrated twice — without a "
                   "fix, position error grows as O(t²).")
    pic(s, SRC / "p2_s02_01.png", 0.62, 1.92, 4.60, 3.30)
    text(s, 0.62, 5.28, 4.60, 0.52,
         "Idealised bias-only growth (b_a = 0.01 m s⁻²): 18 m after 60 s.",
         size=8.8, color=MUTE, spacing=1.05)

    pic(s, SRC / "p2_s02_02.png", 5.60, 1.92, 4.55, 0.50)
    points = [
        ("01", "IMU errors are integrated",
         "Acceleration bias enters velocity once and position twice; attitude "
         "error additionally misaligns gravity."),
        ("02", "An outage removes the only absolute fix",
         "The filter can only predict: the covariance keeps growing and nothing "
         "pulls the trajectory back."),
        ("03", "Measured on the corpus",
         "Dead reckoning (B1) over the 8 test routes: 104.65 m mean outage error, "
         "413 m worst single outage."),
    ]
    for k, (n, head, body) in enumerate(points):
        y = 2.62 + k * 1.14
        text(s, 5.60, y, 0.55, 0.34, n, size=13, bold=True, color=G_LIGHT)
        text(s, 6.15, y - 0.02, 6.55, 0.32, head, size=11.5, bold=True, color=INK)
        text(s, 6.15, y + 0.36, 6.50, 0.62, body, size=10.2, color=MUTE, spacing=1.1)

    rect(s, 0.62, 6.14, 12.10, 0.62, fill=WASH)
    text(s, 0.90, 6.14, 11.6, 0.62,
         [[("WHY THE CURVE IS NOT THE WHOLE STORY   ",
            {"bold": True, "color": G_TEAL, "spc": 60}),
           ("the plotted case is bias only; attitude misalignment and scale "
            "factor make up the rest of the measured 104.65 m.", {"color": MUTE})]],
         size=8.8, anchor=MSO_ANCHOR.MIDDLE)
    return s


def p_21(prs, lay, i):
    s = band(prs, lay, i, "2.1  IMU Mechanization", T2)
    quote(s, 1.02, "Mechanization turns bias-corrected IMU measurements into "
                   "attitude, velocity and position.")

    rect(s, 0.62, 1.90, 4.30, 4.74, fill=WHITE, line=LINE)
    kicker(s, 0.88, 2.04, 3.8, "INPUTS")
    rule(s, 0.88, 2.34, 3.80)
    text(s, 0.88, 2.46, 3.80, 0.30, "Bias-corrected IMU", size=10.8, bold=True, color=INK)
    pic(s, SRC / "p2_s03_01.png", 0.88, 2.80, 2.55, 0.60)
    pic(s, SRC / "p2_s03_02.png", 0.88, 3.48, 2.85, 0.44)
    text(s, 0.88, 4.00, 3.80, 0.30, "Local ENU gravity", size=10.8, bold=True, color=INK)
    pic(s, SRC / "p2_s03_03.png", 0.82, 4.32, 3.35, 0.40)
    rule(s, 0.88, 4.86, 3.80)
    text(s, 0.88, 4.98, 3.80, 0.52,
         "Subtract the biases before integration; normalise the quaternion after "
         "every attitude update.", size=9.6, color=MUTE, spacing=1.14)
    rule(s, 0.88, 5.62, 3.80)
    kicker(s, 0.88, 5.72, 3.8, "OUTPUTS")
    text(s, 0.88, 5.98, 3.80, 0.56,
         "Attitude q, velocity v, position p at 100 Hz — the nominal state the "
         "error filter later corrects.", size=9.6, color=MUTE, spacing=1.14)

    steps = [
        ("1", "Update attitude from the angular rate",
         "Quaternion exponential of the bias-corrected gyro, normalised at every step.",
         "p2_s03_04.png", 3.40, 0.56),
        ("2", "Rotate specific force and update velocity",
         "Body-frame specific force rotated into ENU, gravity removed, then integrated.",
         "p2_s03_05.png", 2.95, 0.50),
        ("3", "Integrate velocity to position",
         "Trapezoidal integration at 100 Hz.",
         "p2_s03_07.png", 3.60, 0.50),
    ]
    ys = [1.86, 3.40, 5.44]
    for (n, head, body, img, iw, ih), y in zip(steps, ys):
        numbered_step(s, 5.45, y, 7.25, n, head, body)
        pic(s, SRC / img, 5.95, y + 0.78, iw, ih)
    pic(s, SRC / "p2_s03_06.png", 5.95, 4.72, 2.60, 0.46)
    return s


def p_22(prs, lay, i):
    s = band(prs, lay, i, "2.2  15-State Error-State Kalman Filter", T2)
    quote(s, 1.02, "The filter estimates the error, injects it into the nominal "
                   "state, and resets — the state itself is never linearised twice.")

    pic(s, SRC / "p2_s04_01.png", 3.55, 1.88, 4.90, 0.62)
    text(s, 0.62, 2.02, 2.72, 0.28, "15 error states, three each:", size=9.2,
         color=MUTE)
    text(s, 0.62, 2.30, 2.72, 0.30, "δp · δv · ψ · δb_a · δb_g", size=9.2, color=MUTE)

    cols = [
        (0.62, 3.75, "1 · Predict the covariance", ["p2_s04_02.png"], [3.25, 0.46]),
        (4.42, 3.75, "2 · Form the GNSS correction",
         ["p2_s04_03.png", "p2_s04_04.png"], [3.55, 0.27]),
        (8.28, 4.15, "3 · Inject the correction and reset",
         ["p2_s04_05.png", "p2_s04_06.png", "p2_s04_07.png", "p2_s04_08.png"],
         [3.45, 0.30]),
    ]
    for x, w, head, imgs, size0 in cols:
        kicker(s, x, 2.78, w, head)
        rule(s, x, 3.06, w - 0.20)
        yy = 3.18
        for img in imgs:
            pic(s, SRC / img, x, yy, w - 0.30, size0[1] + 0.12)
            yy += size0[1] + 0.30

    text(s, 0.62, 4.22, 3.10, 0.96,
         "Φ is rebuilt every step from the mechanization Jacobian; Q comes from "
         "the IMU noise model.", size=9.0, color=MUTE, spacing=1.14)
    text(s, 4.42, 4.36, 3.30, 0.96,
         "H selects position only, so S is 3×3 and the gain needs no 15×15 "
         "inverse.", size=9.0, color=MUTE, spacing=1.14)
    text(s, 0.62, 5.26, 7.60, 0.34,
         "Two solves and two products per step — which is why one Kalman step is "
         "launch-bound on the GPU (page 2.6).", size=9.0, color=G_TEAL)

    rule(s, 0.62, 5.66, 12.10)
    text(s, 0.62, 5.78, 4.60, 0.30, "Joseph covariance update", size=10.8,
         bold=True, color=INK)
    pic(s, SRC / "p2_s04_09.png", 4.55, 5.70, 5.30, 0.44)
    text(s, 0.62, 6.30, 6.9, 0.50,
         "Kept in Joseph form so the covariance stays symmetric positive definite "
         "over 180,000 steps.", size=9.4, color=MUTE, spacing=1.1)
    return s


def p_23(prs, lay, i):
    s = band(prs, lay, i, "2.3  RTS Smoothing", T2)
    quote(s, 1.02, "A forward filter keeps the past; the RTS pass adds the future — "
                   "the fixes after an outage reach backwards into it.")

    kicker(s, 0.62, 1.92, 6.1, "RTS RECURSION IN CONSISTENT LOCAL COORDINATES")
    rule(s, 0.62, 2.20, 5.95)
    for k, (img, h) in enumerate([("p2_s05_01.png", 0.50), ("p2_s05_02.png", 0.56),
                                  ("p2_s05_03.png", 0.50), ("p2_s05_04.png", 0.56)]):
        pic(s, SRC / img, 0.62, 2.34 + k * 0.86, 5.75, h)
    text(s, 0.62, 5.86, 5.95, 0.34, "All five state blocks are corrected — position, "
         "velocity, attitude and both biases.", size=9.6, color=MUTE, spacing=1.1)

    steps = [("1", "Forward ESKF",
              "Store the filtered and predicted states, covariances and transition "
              "matrices for every step."),
             ("2", "Backward recursion",
              "Start from the final filtered estimate; the smoother gain carries "
              "each future correction backwards."),
             ("3", "Recover the full state",
              "Attitude by quaternion multiplication, then renormalise.")]
    for k, (n, head, body) in enumerate(steps):
        y = 1.92 + k * 1.55
        numbered_step(s, 6.95, y, 5.75, n, head, body)
    pic(s, SRC / "p2_s05_05.png", 7.55, 4.42, 3.05, 0.30)
    return s


def p_24(prs, lay, i):
    s = band(prs, lay, i, "2.4  Results I — Position Error", T2)
    quote(s, 1.02, "RTS pulls the mean outage error from 51.44 m to 9.93 m "
                   "(−80.7 %) and the worst single outage from 413 m to 28.3 m.")
    for k, (v, lab) in enumerate([("104.65 m", "B1 · dead reckoning"),
                                  ("51.44 m", "B2 · forward ESKF"),
                                  ("9.93 m", "B3 · ESKF + RTS"),
                                  ("−80.7 %", "B2 → B3, 8 test routes")]):
        kpi(s, 0.62 + k * 3.06, 1.86, 2.85, v, lab, val_size=28)
    pic(s, GFX / "fig4_error.png", 0.62, 2.92, 12.10, 3.42, card=True)
    text(s, 0.62, 6.42, 12.10, 0.42,
         "Grey bands are the injected GNSS outages. The forward filter bends away "
         "and returns; the smoother stays with the ground truth.", size=8.8,
         color=MUTE, spacing=1.05)
    return s


def p_25(prs, lay, i):
    s = band(prs, lay, i, "2.5  Results II — Trajectory", T2)
    quote(s, 1.02, "Inside the outage the forward filter leaves the road; the "
                   "smoother restores the path geometry.")
    pic(s, GFX / "fig3_trajectory.png", 0.62, 1.90, 8.30, 4.30, card=True)
    text(s, 0.62, 6.28, 8.30, 0.42,
         "Route 6, the self-intersecting ring. Windowed GPU RTS follows the "
         "full-route smoother to within a metre in this view.", size=8.6, color=MUTE)

    blocks = [("ESKF DRIFTS INSIDE THE OUTAGE",
               "The faint dotted path leaves the black reference along the S-bend: "
               "with no fixes the forward pass can only accumulate error."),
              ("RTS RESTORES THE GEOMETRY",
               "The solid green path stays on the reference through the same bend — "
               "the returning fixes are projected backwards along the stored "
               "transition matrices.")]
    for k, (head, body) in enumerate(blocks):
        y = 1.95 + k * 2.20
        text(s, 9.15, y, 3.55, 0.30, head, size=9.2, bold=True, color=G_TEAL, spc=50)
        rule(s, 9.15, y + 0.36, 3.55)
        text(s, 9.15, y + 0.48, 3.50, 1.30, body, size=10.2, color=INK, spacing=1.14)
    return s


def p_26(prs, lay, i):
    s = band(prs, lay, i, "2.6  Parallelism Across Windows", T2)
    quote(s, 1.02, "Each window is an independent Kalman task; the batch dimension "
                   "is the parallel axis.")

    # the batch, drawn as a matrix: one row per window, one column per step
    kicker(s, 0.66, 1.94, 5.4, "THE BATCH, AS A MATRIX")
    x0, y0, cw, gap = 0.66, 2.36, 0.155, 0.045
    rows, cols = 11, 27
    for r in range(rows):
        y = y0 + r * (cw + gap)
        live = r in (2, 6, 9)
        rect(s, x0, y, cols * cw, cw, fill=G_MAIN if live else WASH, line=LINE, lw=0.5)
    text(s, x0 + cols * cw + 0.14, y0 + 0.02, 1.05, 0.50,
         [[("B = 192", {"bold": True})], [("windows", {})]],
         size=8.5, color=MUTE, spacing=1.05)
    text(s, x0, y0 + rows * (cw + gap) + 0.06, cols * cw, 0.24,
         "T = 60,000 time steps, left to right →", size=8, color=MUTE)
    text(s, x0 - 0.04, y0 + rows * (cw + gap) + 0.32, 5.25, 0.52,
         "Inside a row the filter is strictly serial; across rows everything is "
         "independent — that is the only parallel axis, and it is enough.",
         size=8.8, color=MUTE, spacing=1.12)

    items = [("01  Independent tasks",
              "Split every log into overlapping windows; windows from different "
              "routes and vehicles share one batch. Nothing inside a window is "
              "parallel — everything across windows is."),
             ("02  Batched tensors",
              "One batch of B windows of T samples: IMU [B, T, 3], error state "
              "[B, 15], covariance [B, 15, 15]. RTS keeps the whole state history "
              "in the same layout."),
             ("03  GPU matrix operations",
              "At each step all B windows update together with batched products "
              "and linear solves — forward ESKF first, backward RTS after.")]
    for k, (head, body) in enumerate(items):
        y = 1.96 + k * 1.36
        text(s, 6.42, y, 6.10, 0.30, head, size=12, bold=True, color=INK)
        text(s, 6.42, y + 0.36, 6.10, 0.86, body, size=10.2, color=MUTE, spacing=1.14)

    rect(s, 0.62, 5.96, 12.10, 0.72, fill=WASH)
    text(s, 0.90, 5.96, 11.6, 0.72,
         [[("WHY IT PAYS   ", {"bold": True, "color": G_TEAL, "spc": 60}),
           ("one Kalman step is a chain of 15×15 products — microseconds of work "
            "with almost no data reuse. A single step costs about 4 ms whatever the "
            "batch size: the kernel is launch-bound, not arithmetic-bound.",
            {"color": MUTE})]],
         size=8.8, anchor=MSO_ANCHOR.MIDDLE)
    return s


def p_27(prs, lay, i):
    s = band(prs, lay, i, "2.7  Window Batching and Stitching", T2)
    quote(s, 1.02, "600 s windows with 90 s margins: the margin must cover the "
                   "60 s outage, and the overlap must hide the seam.")

    steps = [("1", "Parallel initialization",
              "Initialize every window from nearby GNSS and run a first parallel pass."),
             ("2", "Two-pass warm start",
              "Re-run each window from its predecessor's final estimate."),
             ("3", "Bidirectional context",
              "Use past and future GNSS, then discard the burn-in region from the output."),
             ("4", "Overlap-add fusion",
              "Blend overlapping outputs with normalised tent weights.")]
    for k, (n, head, body) in enumerate(steps):
        x = 0.62 + k * 3.05
        rect(s, x, 1.86, 2.85, 1.78, fill=WHITE, line=LINE, shape=MSO_SHAPE.FOLDED_CORNER)
        text(s, x + 0.22, 1.96, 0.5, 0.4, n, size=20, bold=True, color=G_LIGHT)
        text(s, x + 0.62, 2.00, 2.05, 0.34, head, size=10.6, bold=True, color=INK)
        text(s, x + 0.22, 2.46, 2.45, 1.10, body, size=9.4, color=MUTE, spacing=1.12)

    kicker(s, 0.62, 3.84, 8.0, "ONE WINDOW  ·  600 s AT 100 Hz")
    total, m = 11.30, 1.695
    rect(s, 1.0, 4.14, total, 0.66, fill=WASH, line=LINE)
    rect(s, 1.0, 4.14, m, 0.66, fill=G_LIGHT)
    rect(s, 1.0 + m, 4.14, total - 2 * m, 0.66, fill=G_MAIN)
    rect(s, 1.0 + total - m, 4.14, m, 0.66, fill=G_LIGHT)
    text(s, 1.0, 4.14, m, 0.66, "90 s", size=10, color=INK, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.0 + total - m, 4.14, m, 0.66, "90 s", size=10, color=INK,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.0 + m, 4.14, total - 2 * m, 0.66,
         "burn-in  ·  420 s of output per window  ·  burn-out", size=10.5,
         color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    text(s, 1.0, 4.92, 11.30, 0.50,
         "The 90 s margin is a requirement, not a knob: at 30 s, route 43's outage "
         "starts before the output boundary and the error jumps to 133 m.",
         size=9.4, color=MUTE, spacing=1.1)

    # five consecutive windows along one route
    kicker(s, 0.66, 5.52, 7.6, "ALONG A ROUTE  ·  FIVE CONSECUTIVE WINDOWS")
    bw, step = 4.30, 0.86
    for k in range(5):
        x, y = 0.66 + k * step, 5.84 + k * 0.19
        rect(s, x, y, bw, 0.15, fill=G_LIGHT)
        rect(s, x + 0.645, y, bw - 1.29, 0.15, fill=G_MAIN)
    text(s, 9.10, 5.84, 3.60, 0.98,
         "Only the central 420 s of each window is kept; the 90 s arms are burn-in "
         "and burn-out, supplied again by the neighbouring window.", size=8.8,
         color=MUTE, spacing=1.14)
    return s


def p_28(prs, lay, i):
    s = band(prs, lay, i, "2.8  Runtime Gains and Accuracy", T2)
    quote(s, 1.02, "One RTX 4090: 48.6× at one million rows, the full corpus in "
                   "17 minutes — for 8.2 % of the accuracy.")
    pic(s, GFX / "fig1_speedup.png", 0.62, 1.92, 12.10, 3.40, card=True)
    for k, (v, lab) in enumerate([("48.6×", "1 M IMU rows: 2,245 s CPU vs 46.2 s GPU"),
                                  ("17 min", "the full 8.64 M-row corpus on one GPU"),
                                  ("+8.2 %", "outage error: 9.93 → 10.75 m, the price of windows")]):
        kpi(s, 0.62 + k * 4.06, 5.52, 3.80, v, lab, val_size=26)
    text(s, 0.62, 6.62, 12.10, 0.32,
         "At chunk 500 the same benchmark reaches 177.6× — the kernel is "
         "launch-bound until the batch is large.", size=8.6, color=MUTE)
    return s


def p_31(prs, lay, i):
    s = band(prs, lay, i, "3.1  Residual Learning", T3)
    quote(s, 1.02, "Smoothing leaves about ten metres. The question is whether "
                   "that residual is predictable from the sensors.")

    cards = [
        ("BASELINE", "9.93 m", 30,
         "Mean outage error after full-route RTS on the 8 held-out routes — the "
         "number the learned corrector has to beat."),
        ("TARGET", "δp = p_GT − p_KF", 20,
         "Three components, labelled by ground truth at every step. Never an input."),
        ("INTERFACE", "p = p_KF + α · δp", 20,
         "δp here is the model's prediction; α is chosen on validation and may be "
         "zero — the test set never sees it."),
        ("HYPOTHESIS", "the channels may carry the missing state", 13,
         "Motion, GNSS availability and the filter's own bias estimates: the "
         "residual depends on all three, so it might be identifiable."),
    ]
    for k, (lab, val, val_size, body) in enumerate(cards):
        x = 0.62 + (k % 2) * 5.94
        y = 1.86 + (k // 2) * 2.16
        rect(s, x, y, 5.72, 2.00, fill=WHITE, line=LINE, shape=MSO_SHAPE.FOLDED_CORNER)
        text(s, x + 0.28, y + 0.20, 5.10, 0.26, lab, size=8.6, bold=True,
             color=G_TEAL, spc=60)
        text(s, x + 0.28, y + 0.54, 5.16, 0.70, val, size=val_size, bold=True,
             color=G_DARK, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.28, y + 1.32, 5.16, 0.60, body, size=9.8, color=INK, spacing=1.12)

    rect(s, 0.62, 6.18, 12.10, 0.60, fill=WASH)
    text(s, 0.90, 6.18, 11.6, 0.60,
         [[("THE RULE   ", {"bold": True, "color": G_TEAL, "spc": 60}),
           ("a corrector that cannot beat predict-zero is not allowed to ship — the "
            "gate is part of the product, not an afterthought.", {"color": MUTE})]],
         size=8.8, anchor=MSO_ANCHOR.MIDDLE)
    return s


def p_32(prs, lay, i):
    s = band(prs, lay, i, "3.2  Twenty-Channel Input", T3)
    quote(s, 1.02, "What the vehicle did, whether GNSS was there, and what the "
                   "filter believes — ten times per second.")

    rows = [("6", "IMU", "gyro (3)  ·  accelerometer (3)", "raw motion at 10 Hz"),
            ("5", "GNSS context", "innovation (3)  ·  outage flag  ·  time since fix",
             "the reacquisition spike is kept, not averaged"),
            ("9", "filter state", "velocity (3)  ·  accel bias (3)  ·  gyro bias (3)",
             "the quantity being integrated during the outage")]
    for k, (n, name, chans, note) in enumerate(rows):
        y = 1.92 + k * 1.18
        rect(s, 0.62, y, 7.55, 1.02, fill=WHITE, line=LINE)
        text(s, 0.88, y + 0.06, 0.60, 0.44, n, size=22, bold=True, color=G_TEAL,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 1.62, y + 0.14, 2.05, 0.30, name, size=11.5, bold=True, color=INK)
        text(s, 1.62, y + 0.52, 4.35, 0.34, chans, size=9.6, color=MUTE)
        text(s, 6.10, y + 0.14, 1.95, 0.74, note, size=8.6, color=MUTE, spacing=1.06)

    stats3 = [("10 Hz", "feature rate"), ("1024", "input steps · 102.4 s"),
              ("128", "output steps · 12.8 s")]
    for k, (v, lab) in enumerate(stats3):
        x = 8.62 + (k % 2) * 0
        y = 1.92 + k * 1.18
        text(s, 8.62, y + 0.04, 4.10, 0.46, v, size=24, bold=True, color=G_DARK)
        text(s, 8.62, y + 0.52, 4.05, 0.30, lab, size=9.2, color=MUTE)

    rect(s, 8.62, 5.42, 4.10, 1.10, fill=WASH)
    text(s, 8.86, 5.54, 3.66, 0.90,
         "The window is symmetric: the whole pipeline is offline, so the "
         "post-outage fix on the right is allowed to explain the drift on the left.",
         size=9.0, color=INK, spacing=1.12)
    return s


def p_33(prs, lay, i):
    s = band(prs, lay, i, "3.3  GPT4TS-Style Adaptation", T3)
    quote(s, 1.02, "Patch the sensor stream into tokens, embed them, and adapt a "
                   "frozen language model to regression.")
    pic(s, crop_white(SRC / "p4_s04_01.png"), 0.62, 1.88, 12.10, 3.95, card=True)
    notes = [("16 patches", "1024 steps × 20 channels → 16 tokens of 160 values"),
             ("82.8 M frozen", "distilgpt2 backbone, only LoRA adapters train"),
             ("128 × 3", "regression head returns the central 12.8 s residual")]
    for k, (v, lab) in enumerate(notes):
        x = 0.62 + k * 4.06
        text(s, x, 6.02, 3.80, 0.36, v, size=15, bold=True, color=G_DARK)
        text(s, x, 6.40, 3.80, 0.40, lab, size=9.2, color=MUTE, spacing=1.05)
    return s


def p_34(prs, lay, i):
    s = band(prs, lay, i, "3.4  LoRA Fine-Tuning", T3)
    quote(s, 1.02, "LoRA learns a low-rank update and leaves the 82.8 M pretrained "
                   "weights exactly where they are.")
    rect(s, 0.62, 1.88, 6.05, 4.55, fill=WHITE, line=LINE)
    pic(s, SRC / "p4_s05_01.png", 0.78, 1.96, 5.72, 4.39)

    text(s, 7.00, 1.92, 5.60, 0.60, "1.06 %", size=40, bold=True, color=G_DARK)
    text(s, 7.00, 2.62, 5.60, 0.34, "of the parameters are trainable — 0.87 M of 82.8 M",
         size=10.2, color=MUTE)
    rule(s, 7.00, 3.18, 5.60)
    for k, (head, body) in enumerate([
            ("Why not full fine-tuning",
             "A noisy three-channel residual does not justify 82.8 M weights; "
             "LoRA keeps the language prior intact and the update cheap."),
            ("What is trained",
             "Rank-8 adapters on the attention projections, added to the frozen "
             "backbone and merged at inference."),
            ("A naming warning",
             "The LoRA scaling α is unrelated to the trajectory correction "
             "strength α used on the next pages.")]):
        y = 3.34 + k * 1.00
        text(s, 7.00, y, 5.55, 0.30, head, size=11, bold=True, color=INK)
        text(s, 7.00, y + 0.34, 5.55, 0.60, body, size=9.8, color=MUTE, spacing=1.12)
    return s


def p_conclusion(prs, lay, i):
    s = band(prs, lay, i, "Conclusion  ·  Four Numbers, One Thread",
             "CONCLUSION · MS022161")
    quote(s, 1.02, "Each tier removes a different kind of error. What is left after "
                   "the last one is information the outage removed.")

    rows = [("90.5 %", "PHYSICS",
             "of the outage error removed by ESKF + RTS: 104.65 → 9.93 m."),
            ("48.6×", "COMPUTE",
             "faster than serial CPU at one million rows, for 8.2 % of accuracy."),
            ("α = 0", "LEARNING",
             "the gate returns zero on all 8 held-out routes — the channels do not "
             "carry the missing state."),
            ("36.5 %", "SPACE",
             "further reduction from a road-centreline prior: 9.93 → 6.30 m, with "
             "86.2 % of mapped points inside the 0.5 m tolerance.")]
    for k, (v, lab, body) in enumerate(rows):
        y = 1.84 + k * 1.00
        text(s, 0.62, y, 1.95, 0.52, v, size=27, bold=True, color=G_DARK)
        text(s, 2.62, y + 0.10, 1.55, 0.30, lab, size=9.2, bold=True, color=G_TEAL, spc=60)
        text(s, 4.05, y + 0.02, 8.65, 0.64, body, size=11.5, color=INK, spacing=1.1)
        if k < 3:
            rule(s, 0.62, y + 0.90, 12.10)

    rect(s, 0.62, 6.12, 12.10, 0.62, fill=G_DARK)
    text(s, 0.90, 6.12, 11.6, 0.62,
         "Next: replay a real IMU / GNSS log, widen the window, and give the "
         "corrector the one thing the outage took away.", size=10, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    return s


# ------------------------------------------------------------------- notes
def add_notes(prs):
    path = ROOT / "results" / "ppt" / "speaker_notes.txt"
    chunks = path.read_text(encoding="utf-8").split("--- slide ")[1:]
    notes = {int(c.split(" ---\n", 1)[0]): c.split(" ---\n", 1)[1].rstrip("\n")
             for c in chunks}
    # final index -> speaker_notes block; pages whose block is absent get a
    # short note written here rather than a mismatched one
    amap = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 8: 8, 9: 9,
            12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 18: 15, 19: 16}
    own = {
        7: "Mechanization is the part that never changes. Subtract the estimated biases, "
           "rotate the specific force into ENU, remove gravity, integrate twice. Everything "
           "after this page is about correcting the two integrations.",
        10: "This is the whole accuracy story on one axis. Dead reckoning is over a hundred "
            "metres; the forward filter halves it; the smoother takes it to ten. Everything "
            "that follows — GPU windows, the learned module, the map — is measured against "
            "these three numbers.",
        11: "The spatial view matters because the mean can hide a systematic failure. In the "
            "ring route the forward filter leaves the road entirely during the outage and comes "
            "back; the smoother never leaves it. That is what a map prior will later exploit.",
        17: "Twenty channels, three groups. The important design choice is the last group: the "
            "filter's own velocity and bias estimates. The residual during an outage is, to first "
            "order, the error of those biases integrated twice — without them the mapping is not "
            "identifiable, however large the model.",
        26: "Four tiers, four different limits. Physics removes nine tenths; compute makes it "
            "fast enough to matter at fleet scale; learning honestly reports that the channels "
            "do not carry the missing state; and the map supplies what only space can. The next "
            "step is a real log, not a bigger model.",
    }
    for idx, src in amap.items():
        prs.slides[idx].notes_slide.notes_text_frame.text = notes[src]
    for idx, txt in own.items():
        prs.slides[idx].notes_slide.notes_text_frame.text = txt


def renumber_green(prs):
    """The inherited green pages keep their design but move to the new numbering."""
    fixes = {
        "3.4  Training and Safety Gating": "3.5  Training and Safety Gating",
        "3.5  What It Bought — and Where It Stops": "3.6  What It Bought — and Where It Stops",
        "CHAPTER 03 · FOUNDATION MODEL AND LoRA": T3,
        "CHAPTER 04 · SPATIAL INFORMATION": T4,
    }
    sizes = {"Spatial Information": 40}     # match the rebuilt dividers
    for slide in prs.slides:
        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            full = sh.text_frame.text.strip()
            if full in fixes:
                first = sh.text_frame.paragraphs[0]
                runs = first.runs
                if runs:
                    runs[0].text = fixes[full]
                    for r in runs[1:]:
                        r.text = ""
            if full in sizes:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(sizes[full])


def main() -> None:
    prs = Presentation(GREEN_DECK)
    lay_band = prs.slide_masters[1].slide_layouts[2]     # 节标题: band + seal + tab
    lay_div = prs.slide_masters[1].slide_layouts[1]      # 标题和内容: green panel
    lay_blank = prs.slide_masters[1].slide_layouts[6]    # 空白: cover

    # the seven inherited green pages sit at 0..6 and end up at 20..25 + thanks
    p_cover(prs, lay_blank, 0)
    p_agenda(prs, lay_band, 1)
    divider(prs, lay_div, 2, "01", "PART ONE", "Overview & Background",
            "Why a Progressive Three-Tier Correction Architecture?",
            ["1.1  Three-tier architecture", "1.2  The 8.64 M-row corpus",
             "1.3  Why physics alone drifts"])
    p_11(prs, lay_band, 3)
    p_12(prs, lay_band, 4)
    p_13(prs, lay_band, 5)
    divider(prs, lay_div, 6, "02", "PART TWO", "Filtering & GPU Batching",
            "ESKF, RTS Smoothing and Window-Parallel CUDA",
            ["2.1  IMU mechanization", "2.2  15-state ESKF", "2.3  RTS smoothing",
             "2.4–2.5  Results: error and trajectory",
             "2.6–2.8  GPU windows, stitching, throughput"], sub_w=6.6)
    p_21(prs, lay_band, 7)
    p_22(prs, lay_band, 8)
    p_23(prs, lay_band, 9)
    p_24(prs, lay_band, 10)
    p_25(prs, lay_band, 11)
    p_26(prs, lay_band, 12)
    p_27(prs, lay_band, 13)
    p_28(prs, lay_band, 14)
    divider(prs, lay_div, 15, "03", "PART THREE", "Foundation Model & LoRA",
            "Learning the Residual the Filter Leaves Behind",
            ["3.1  Residual learning", "3.2  Twenty-channel input",
             "3.3  GPT4TS-style model", "3.4  LoRA fine-tuning",
             "3.5  Training and safety gating", "3.6  What it bought"], sub_w=6.4)
    p_31(prs, lay_band, 16)
    p_32(prs, lay_band, 17)
    p_33(prs, lay_band, 18)
    p_34(prs, lay_band, 19)
    p_conclusion(prs, lay_band, 26)

    renumber_green(prs)
    add_notes(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"{OUT.name}: {len(prs.slides._sldIdLst)} slides")

    dump = ROOT / "results" / "ppt" / "full_deck_text.txt"
    with open(dump, "w", encoding="utf-8") as fh:
        for i, slide in enumerate(prs.slides, 1):
            fh.write(f"\n===== page {i} ({slide.slide_layout.name}) =====\n")
            for sh in slide.shapes:
                if sh.has_text_frame and sh.text_frame.text.strip():
                    fh.write(sh.text_frame.text.replace("\n", " / ") + "\n")
    print(f"text dump -> {dump}")


if __name__ == "__main__":
    main()
