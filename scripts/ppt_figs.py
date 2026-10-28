"""Three figure panels for the "training / gating" slide of the deck.

    python scripts/ppt_figs.py

Writes results/ppt/{a_imbalance,b_training,c_gate}.png at exactly the size the
deck places them (3.60 x 2.32 in, 300 dpi, transparent background, deck
palette) and prints every number the slide quotes, so the copy and the figures
cannot drift apart.

Panel A  where the residual lives, and what the loss does about it
Panel B  what training actually bought on held-out routes
Panel C  the case the one-standard-error gate exists for
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import DATASET_DIR, RESULTS_DIR, RUNS_DIR, load_config  # noqa: E402

OUT = RESULTS_DIR / "ppt"
# deck palette (slide 16): card tints FFF7ED / F0F9FF / F0FDFA, accents below
INK, MUTE = "#18181B", "#57534E"
ACC = {"orange": "#EA580C", "sky": "#0284C7", "teal": "#0F766E", "red": "#DC2626",
       "violet": "#7C3AED", "grey": "#A8A29E"}

W, H = 3.60, 2.32           # inches -- the box each panel gets on the slide

# measured by `python scripts/alpha_scan.py --model {lstm,gpt4ts}` (README):
# unguarded grid optimum on validation, and what shipping it costs on test.
GATE = [("LSTM", ACC["sky"], 2.00, +2.7, -2.2),
        ("GPT4TS", ACC["violet"], 1.90, +2.9, -4.9)]


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8.8,
        "font.family": "Segoe UI",
        "axes.edgecolor": MUTE,
        "axes.labelcolor": MUTE,
        "axes.linewidth": 0.7,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.transparent": True,
    })


def _save(fig, name: str):
    """Save on the exact 3.60 x 2.32 in canvas the slide reserves: render tight
    (so nothing is clipped), then centre the result on a transparent canvas."""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp.png")
    fig.savefig(tmp, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    im = Image.open(tmp).convert("RGBA")
    cw, ch, m = 1080, 696, 6
    k = min((cw - 2 * m) / im.width, (ch - 2 * m) / im.height, 1.0)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                   Image.LANCZOS)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    canvas.save(OUT / name)
    tmp.unlink()
    print(f"  -> {OUT / name}  {im.width}x{im.height} on {cw}x{ch}")


def _spine_off(ax, keep=()):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# A: 7 % of the samples, 17 % of the residual, 82 % of the loss
# --------------------------------------------------------------------------- #
def _shares(path):
    d = np.load(path)
    y = d["train_y"].astype(np.float64)
    og = d["train_og"] > 0
    w = d["train_w"].astype(np.float64)[:, None, None]
    wgt = w * (1.0 + 10.0 * og[:, :, None])          # 2.0 window x (1 + 10) loss
    sq = y ** 2
    return {"steps": 100 * og.mean(),
            "energy": 100 * sq[og].sum() / sq.sum(),
            "loss": 100 * (wgt * sq)[og].sum() / (wgt * sq).sum()}


def panel_a():
    s = _shares(DATASET_DIR / "dataset.npz")          # smoother base (B3)
    s2 = _shares(DATASET_DIR / "dataset_b2.npz")      # causal base (online variant)
    print(f"A  steps {s['steps']:.1f} %  energy {s['energy']:.1f} %  "
          f"weighted loss {s['loss']:.1f} %   (causal base energy {s2['energy']:.1f} %)")

    rows = [("share of the samples", s["steps"], ACC["grey"]),
            ("share of the squared residual", s["energy"], ACC["orange"]),
            ("share of the loss the model sees", s["loss"], ACC["orange"])]
    fig, ax = plt.subplots(figsize=(W, H))
    for i, (lab, val, c) in enumerate(rows):
        y = 2 - i
        ax.text(0, y + 0.30, lab, fontsize=8.8, color=MUTE, va="bottom", ha="left")
        ax.barh(y, 100, height=0.36, color="#000000", alpha=0.05, zorder=1)
        ax.barh(y, val, height=0.36, color=c, zorder=2)
        ax.text(val + 2.5, y, f"{val:.1f} %", va="center", ha="left",
                fontsize=10.6, fontweight="bold", color=INK)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.42, 2.50)
    ax.set_xticks([])
    ax.set_yticks([])
    _spine_off(ax)
    ax.grid(False)
    fig.tight_layout(pad=0.05)
    _save(fig, "a_imbalance.png")


# --------------------------------------------------------------------------- #
# B: converged on the training routes, at the floor on held-out ones
# --------------------------------------------------------------------------- #
def panel_b():
    d = np.load(DATASET_DIR / "dataset.npz")
    pz = float(np.mean(d["val_y"].astype(np.float64) ** 2))
    fig, ax = plt.subplots(figsize=(W, H))
    txt = []
    for name, col, lab, mk in [("lstm", ACC["sky"], "LSTM", "-o"),
                               ("gpt4ts", ACC["violet"], "GPT4TS", "-o")]:
        h = json.loads((RUNS_DIR / "models" / f"{name}_history.json").read_text())
        ep = [r["epoch"] for r in h]
        val = [r["val_mse"] for r in h]
        ax.plot(ep, val, mk, ms=2.8, lw=1.4, color=col, label=lab)
        i = int(np.argmin(val))
        ax.plot(ep[i], val[i], "o", ms=5.5, mfc="none", mec=col, mew=1.0)
        txt.append((lab, val[i], i + 1))
    print(f"B  predict zero {pz:.3f} m^2; " +
          "  ".join(f"{n} best {v:.3f} ({100 * (v / pz - 1):+.1f} %)" for n, v, _ in txt))

    ax.axhline(pz, color=INK, lw=0.9, ls="--")
    ax.text(9.35, pz + 0.09, f"predict zero  {pz:.2f} m²", fontsize=7.8,
            color=INK, va="bottom", ha="right")
    ax.set_ylim(pz - 0.30, 15.1)
    ax.set_xlim(0.6, 9.4)
    ax.set_xlabel("epoch", fontsize=8.2, labelpad=1)
    ax.set_ylabel("val MSE  [m²]", fontsize=8.2, labelpad=2)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", borderpad=0.1,
              handlelength=1.4, labelspacing=0.25)
    _spine_off(ax, keep=("left", "bottom"))
    fig.tight_layout(pad=0.05)
    _save(fig, "b_training.png")


# --------------------------------------------------------------------------- #
# C: what an unguarded fit would have shipped
# --------------------------------------------------------------------------- #
def panel_c():
    print("C  unguarded: " + "; ".join(
        f"{n} alpha*={a:.2f} val {v:+.1f} % test {t:+.1f} %"
        for n, _, a, v, t in GATE) + "   gated alpha=0 -> 10.75 m")
    fig, ax = plt.subplots(figsize=(W, H))
    x = np.arange(len(GATE))
    wid = 0.30
    for off, key, lab, col in [(-wid / 2 - 0.02, 3, "validation", ACC["teal"]),
                               (+wid / 2 + 0.02, 4, "held-out test", ACC["red"])]:
        vals = [g[key] for g in GATE]
        ax.bar(x + off, vals, width=wid, color=col, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + (0.22 if v > 0 else -0.22), f"{v:+.1f} %".replace("-", "−"),
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=9.0, fontweight="bold", color=col)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g[0]}\n$\\alpha^*$ = {g[2]:.2f}" for g in GATE], fontsize=8.2)
    ax.set_ylim(-7.2, 4.6)
    ax.set_ylabel("error change vs $\\alpha$ = 0  [%]", fontsize=7.8, labelpad=2)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left", ncol=1,
              borderpad=0.1, handlelength=1.1, labelspacing=0.25)
    _spine_off(ax, keep=("left", "bottom"))
    fig.tight_layout(pad=0.05)
    _save(fig, "c_gate.png")


if __name__ == "__main__":
    _style()
    print("writing slide panels to results/ppt/")
    panel_a()
    panel_b()
    panel_c()
