"""Export slides to PNG through the running PowerPoint (COM).

    python scripts/ppt_export.py DECK.pptx 18 19 20 --out results/ppt --prefix tail_
    python scripts/ppt_export.py DECK.pptx 18 --jobs 1        # sequential

PowerPoint is opened on a *temp copy* so the deck you are editing is not
touched, and the application is never quit (the user usually has it open).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import tempfile

import win32com.client as win32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("slides", nargs="+", type=int, help="1-based slide numbers")
    ap.add_argument("--out", default="results/ppt")
    ap.add_argument("--prefix", default="slide")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    a = ap.parse_args()

    deck = pathlib.Path(a.deck).resolve()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp()) / deck.name
    shutil.copy2(deck, tmp)

    app = win32.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(tmp), WithWindow=False)
    try:
        for n in a.slides:
            png = (out / f"{a.prefix}{n}.png").resolve()
            pres.Slides(n).Export(str(png), "PNG", a.w, a.h)
            print(f"  -> {png.name}")
    finally:
        pres.Close()
    shutil.rmtree(tmp.parent, ignore_errors=True)
    print("ok")


if __name__ == "__main__":
    main()
