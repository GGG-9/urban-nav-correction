"""Rebuild slide 17 of the deck (3.4 TRAINING & SAFETY GATING) from scratch.

    python scripts/ppt_slide17_redesign.py [deck.pptx]

The old page was three tinted cards holding three paragraphs. This one is a
new editorial layout -- argument on the left, evidence on the right:

    left   01  a 72 pt  x22  anchor + a proportional stat stack + a note
    right  02  training curves (7.27 in wide)       + one-line read
           03  the gate's validation/test split    + one-line read
    bottom     black strip with the shrinkage gate and the equation

The slide is built new (blank layout), moved into slot 17, and the old slide is
deleted -- nothing from the old page is reused except its two header strings.
"""
from __future__ import annotations

import copy
import pathlib
import shutil
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "ppt"
DECK = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    r"E:\大作业\GPU-Parallel Batched Kalman Smoothing and LoRA Fine-Tuned "
    r"Time-Series Foundation Model.pptx")

SLOT = 16                      # 0-based index of the 3.4 page
FONT = "DM Sans"
INK = RGBColor(0x18, 0x18, 0x1B)
MUTE = RGBColor(0x57, 0x53, 0x4E)
FAINT = RGBColor(0x71, 0x71, 0x7A)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
GREY = RGBColor(0xA8, 0xA2, 0x9E)
TRACK = RGBColor(0xF5, 0xF5, 0xF4)
HAIR = RGBColor(0xE7, 0xE5, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DIM = RGBColor(0xA1, 0xA1, 0xAA)

LX, LW = 0.62, 4.45            # left column
RX, RW = 5.45, 7.27            # right column


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def text(slide, x, y, w, h, parts, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         line=None, space_before=0):
    """``parts`` = [(string, {size, color, bold, spc}), ...] in one paragraph."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if line is not None:
        p.line_spacing = line
    if space_before:
        p.space_before = Pt(space_before)
    for s, kw in parts:
        r = p.add_run()
        r.text = s
        f = r.font
        f.name = FONT
        f.size = Pt(kw.get("size", 11))
        f.bold = kw.get("bold", False)
        f.color.rgb = kw.get("color", MUTE)
        if kw.get("spc"):
            r._r.get_or_add_rPr().set("spc", str(kw["spc"]))
    return tb


def rect(slide, x, y, w, h, color, rounded=False, radius=0.5):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def caption(slide, x, y, w, h, s):
    return text(slide, x, y, w, h, [(s, {"size": 10.5, "color": MUTE})], line=1.18)


# --------------------------------------------------------------------------- #
def main() -> None:
    prs = Presentation(DECK)
    old = prs.slides[SLOT]
    head = [(sh.name, sh.text_frame.text) for sh in old.shapes
            if sh.has_text_frame and sh.name in ("TextBox 6", "TextBox 7")]
    eyebrow = dict(head)["TextBox 6"]
    title = dict(head)["TextBox 7"]
    bg = old._element.find(qn("p:cSld")).find(qn("p:bg"))
    print(f"replacing slide {SLOT + 1}: {title[:60]}")

    # 1. build the replacement first -- the part name is handed out by scanning
    #    the package, so the old slide must still be there when it is chosen
    lst = prs.slides._sldIdLst
    old_el = list(lst)[SLOT]
    old_rId = old_el.get(qn("r:id"))

    blank = next(lay for lay in prs.slide_layouts if lay.name == "Blank")
    new = prs.slides.add_slide(blank)
    el = list(lst)[-1]
    lst.remove(el)
    lst.insert(SLOT, el)
    lst.remove(old_el)
    prs.part.drop_rel(old_rId)
    assert len(new.shapes) == 0, "blank layout carried placeholders"
    if bg is not None:
        new._element.find(qn("p:cSld")).insert(0, copy.deepcopy(bg))

    S = new

    # 2. header -- same strings and type as every other chapter slide
    text(S, 0.62, 0.46, 12.71, 0.19,
         [(eyebrow, {"size": 10.5, "bold": True, "color": ORANGE})])
    text(S, 0.62, 0.71, 12.71, 0.44,
         [(title, {"size": 24, "bold": True, "color": INK})])

    # 3. left column: the loss decision
    text(S, LX, 1.40, LW, 0.20,
         [("01", {"size": 10.5, "bold": True, "color": GREY, "spc": 120}),
          ("   LOSS DESIGN", {"size": 10.5, "bold": True, "color": ORANGE, "spc": 120})])
    text(S, LX, 1.56, LW, 1.10, [("×22", {"size": 72, "bold": True, "color": INK})],
         anchor=MSO_ANCHOR.MIDDLE)
    text(S, LX, 2.66, LW, 0.30,
         [("on every outage sample", {"size": 12.5, "color": MUTE})])

    rows = [("of the samples", 7.0, GREY),
            ("of the squared residual", 17.1, RGBColor(0xFB, 0x92, 0x3C)),
            ("of the gradient", 81.7, ORANGE)]
    for i, (lab, pct, col) in enumerate(rows):
        y = 3.06 + 0.40 * i
        text(S, LX, y, 1.70, 0.30, [(lab, {"size": 9.5, "color": MUTE})],
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        rect(S, 2.44, y + 0.07, 1.72, 0.16, TRACK)
        rect(S, 2.44, y + 0.07, max(0.05, 1.72 * pct / 100.0), 0.16, col)
        text(S, 4.24, y, 0.83, 0.30,
             [(f"{pct:.1f} %", {"size": 12, "bold": True, "color": INK})],
             anchor=MSO_ANCHOR.MIDDLE)

    rect(S, LX, 4.40, LW, 0.012, HAIR)
    text(S, LX, 4.54, LW, 0.90,
         [("Windows get ×2, outage steps ×11 — so 82 % of the gradient lands on the "
           "7 % of the timeline where the filter is flying blind.",
           {"size": 11.5, "color": MUTE})], line=1.22)

    text(S, LX, 5.66, LW, 0.18,
         [("IMPLEMENTATION", {"size": 8.5, "bold": True, "color": GREY, "spc": 160})])
    text(S, LX, 5.86, LW, 0.55,
         [("FP16 autocast forward; loss, unscale and gradient clipping in FP32; "
           "AdamW 1e-4 + OneCycleLR.", {"size": 9.5, "color": FAINT})], line=1.2)

    # 4. right column: the two pieces of evidence
    text(S, RX, 1.40, RW, 0.20,
         [("02", {"size": 10.5, "bold": True, "color": GREY, "spc": 120}),
          ("   TRAINING OUTCOME", {"size": 10.5, "bold": True, "color": ORANGE, "spc": 120})])
    S.shapes.add_picture(str(FIG / "b_training_wide.png"), Inches(RX), Inches(1.62),
                         width=Inches(RW), height=Inches(1.95))
    caption(S, RX, 3.62, RW, 0.44,
            "Train loss 5.30 → 4.40 (LSTM) and 4.98 → 1.94 (GPT4TS). On unseen "
            "routes the residual stays at the 11.48 m² of predicting zero "
            "(best 11.79 / 11.68) — which is exactly what the gate below is for.")

    text(S, RX, 4.16, RW, 0.20,
         [("03", {"size": 10.5, "bold": True, "color": GREY, "spc": 120}),
          ("   SAFETY FALLBACK", {"size": 10.5, "bold": True, "color": ORANGE, "spc": 120})])
    S.shapes.add_picture(str(FIG / "c_gate_wide.png"), Inches(RX), Inches(4.38),
                         width=Inches(RW), height=Inches(1.82))
    caption(S, RX, 6.22, RW, 0.40,
            "Unguarded α* = 2.00 / 1.90 buys +2.7 % / +2.9 % on validation but "
            "−2.2 % / −4.9 % on test; one standard error keeps α = 0.00.")

    # 5. closing strip
    rect(S, 0.62, 6.68, 12.08, 0.44, INK, rounded=True, radius=0.5)
    text(S, 0.92, 6.76, 7.30, 0.30,
         [("VALIDATION SHRINKAGE GATE", {"size": 10.5, "bold": True, "color": WHITE, "spc": 120}),
          ("    one standard error decides whether α ships",
           {"size": 10, "color": DIM})], anchor=MSO_ANCHOR.MIDDLE)
    S.shapes.add_picture(str(FIG / "formula.png"), Inches(8.55), Inches(6.75),
                         width=Inches(3.94), height=Inches(0.30))

    # 6. save
    backup = DECK.with_name(DECK.stem + ".bak.pptx")
    shutil.copy2(DECK, backup)
    try:
        prs.save(DECK)
        out = DECK
    except PermissionError:
        out = DECK.with_name(DECK.stem + "-v3.pptx")
        prs.save(out)
    print(f"slides now: {len(prs.slides._sldIdLst)}")
    print(f"backup -> {backup.name}\nsaved  -> {out.name}")


if __name__ == "__main__":
    main()
