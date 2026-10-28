"""Figure for the 3.5 page: the error ladder B1 -> GPU on a log axis.

    python scripts/ppt_figs_tail.py
    -> results/ppt/ladder.png   5.43 x 3.10 in, transparent, deck palette

Numbers are read back from results/summary.md's source data so the bars and the
slide copy cannot drift apart.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import RESULTS_DIR, RUNS_DIR  # noqa: E402

GREEN = os.environ.get("FIG_PALETTE") == "green"   # green course-report template
OUT = pathlib.Path(os.environ.get("FIG_OUT_DIR") or (RESULTS_DIR / "ppt"))
INK, MUTE = ("#1F2328", "#5D6B5D") if GREEN else ("#18181B", "#57534E")
# the ladder goes light -> dark as the error falls, matching the template's greens
C = ({"red": "#A9AFA6", "orange": "#9DC48C", "sky": "#587558", "violet": "#31481D"}
     if GREEN else
     {"red": "#DC2626", "orange": "#F97316", "sky": "#0284C7", "violet": "#7C3AED"})

W, H = 5.43, 3.10


def _save(fig, name, w_in, h_in):
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp.png")
    fig.savefig(tmp, transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    im = Image.open(tmp).convert("RGBA")
    cw, ch, m = round(w_in * 300), round(h_in * 300), 8
    k = min((cw - 2 * m) / im.width, (ch - 2 * m) / im.height, 1.0)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    canvas.save(OUT / name)
    tmp.unlink()
    print(f"  -> {name}  {cw}x{ch}")


def ladder():
    # the eight GPU test routes are exactly the keys of metrics_gpu.json
    gpu_m = json.loads((RUNS_DIR / "kf_gpu" / "metrics_gpu.json").read_text())
    ref = json.loads((RUNS_DIR / "kf" / "metrics_all.json").read_text())
    keys = sorted(gpu_m)
    mean = lambda f: float(np.mean([f(k) for k in keys]))          # noqa: E731
    rows = [
        ("B1  dead reckoning", mean(lambda k: ref[k]["B1"]["outage_mean"]), C["red"]),
        ("B2  forward ESKF", mean(lambda k: ref[k]["B2"]["outage_mean"]), C["orange"]),
        ("B3  ESKF + RTS", mean(lambda k: ref[k]["B3"]["outage_mean"]), C["sky"]),
        ("GPU  windowed RTS", mean(lambda k: gpu_m[k]["outage_mean"]), C["violet"]),
    ]
    for lab, v, _ in rows:
        print(f"   {lab:20s} {v:7.2f} m")

    fig, ax = plt.subplots(figsize=(W, H))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, c) in zip(ys, rows):
        ax.barh(y, v, height=0.52, color=c, zorder=3)
        ax.text(v * 1.06, y, f"{v:.2f} m", va="center", ha="left",
                fontsize=11.5, fontweight="bold", color=INK, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(7.5, 320)
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels(["10 m", "30 m", "100 m", "300 m"], fontsize=9)
    ax.set_xlabel("outage mean 3-D error  [m]", fontsize=9.5, labelpad=2)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTE)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    fig.tight_layout(pad=0.05)
    _save(fig, "ladder.png", W, H)


if __name__ == "__main__":
    ladder()
