"""Wide panels for the redesigned "3.4 training & safety gating" slide.

    python scripts/ppt_figs_wide.py

The redesign gives the two *evidence* charts the full right-hand column
(7.27 in wide), so they are drawn at that aspect instead of the 3.60 x 2.32 in
box the old three-card layout used:

    results/ppt/b_training_wide.png   7.27 x 1.95 in
    results/ppt/c_gate_wide.png       7.27 x 1.82 in

The imbalance numbers are no longer a PNG -- the new slide draws them as a
typographic stat stack -- but the shares are printed here so the copy on the
slide keeps matching the data.
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
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import DATASET_DIR, RESULTS_DIR, RUNS_DIR  # noqa: E402

GREEN = os.environ.get("FIG_PALETTE") == "green"   # green course-report template
OUT = pathlib.Path(os.environ.get("FIG_OUT_DIR") or (RESULTS_DIR / "ppt"))
INK, MUTE = ("#1F2328", "#5D6B5D") if GREEN else ("#18181B", "#57534E")
ACC = ({"orange": "#9C6B3F", "sky": "#6E9A6B", "teal": "#587558", "red": "#A63D2F",
        "violet": "#31481D", "grey": "#9AA39A"} if GREEN else
       {"orange": "#EA580C", "sky": "#0284C7", "teal": "#0F766E", "red": "#DC2626",
        "violet": "#7C3AED", "grey": "#A8A29E"})

W_B, H_B = 7.27, 1.95
W_C, H_C = 7.27, 1.82

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
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.transparent": True,
    })


def _save(fig, name: str, w_in: float, h_in: float):
    """Exact-canvas render: tight crop, then centre on the declared box."""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp.png")
    fig.savefig(tmp, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    im = Image.open(tmp).convert("RGBA")
    cw, ch, m = round(w_in * 300), round(h_in * 300), 8
    k = min((cw - 2 * m) / im.width, (ch - 2 * m) / im.height, 1.0)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                   Image.LANCZOS)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((cw - im.width) // 2, (ch - im.height) // 2))
    canvas.save(OUT / name)
    tmp.unlink()
    print(f"  -> {name}  {cw}x{ch}")


def _spine_off(ax, keep=()):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


def shares():
    """7.05 % of samples / 17.1 % of the squared residual / 81.7 % of the loss."""
    d = np.load(DATASET_DIR / "dataset.npz")
    y = d["train_y"].astype(np.float64)
    og = d["train_og"] > 0
    w = d["train_w"].astype(np.float64)[:, None, None]
    wgt = w * (1.0 + 10.0 * og[:, :, None])
    sq = y ** 2
    return {"samples": 100 * og.mean(),
            "residual": 100 * sq[og].sum() / sq.sum(),
            "gradient": 100 * (wgt * sq)[og].sum() / (wgt * sq).sum()}


def panel_b():
    """Two panes, because the two curves are not on one scale.

    Left: what the optimiser saw -- the outage-weighted training loss, in m^2.
    Right: what the gate saw -- the unweighted validation MSE against the
    predict-zero floor.  The weights (1 + 10 x outage) put the training loss
    several times above validation, so plotting both on one axis would make the
    training curve look like a failure instead of the loss design working.
    """
    d = np.load(DATASET_DIR / "dataset.npz")
    pz = float(np.mean(d["val_y"].astype(np.float64) ** 2))
    fig, (ax_t, ax_v) = plt.subplots(
        1, 2, figsize=(W_B, H_B),
        gridspec_kw={"width_ratios": [1.0, 1.42], "wspace": 0.40})
    best = {}
    for name, col, lab in [("lstm", ACC["sky"], "LSTM"),
                           ("gpt4ts", ACC["violet"], "GPT4TS")]:
        h = json.loads((RUNS_DIR / "models" / f"{name}_history.json").read_text())
        ep = [r["epoch"] for r in h]
        tr = [r["train_mse"] for r in h]
        val = [r["val_mse"] for r in h]

        ax_t.plot(ep, tr, "-o", ms=2.5, lw=1.35, color=col, solid_capstyle="round")
        ax_t.annotate(f"{tr[0]:.1f} → {tr[-1]:.1f}", (ep[-1], tr[-1]),
                      xytext=(7, 0), textcoords="offset points", va="center",
                      fontsize=7.6, fontweight="bold", color=col)

        ax_v.plot(ep, val, "-o", ms=3.0, lw=1.6, color=col,
                  label=f"{lab}  best {min(val):.2f}", solid_capstyle="round")
        i = int(np.argmin(val))
        ax_v.plot(ep[i], val[i], "o", ms=7.0, mfc="none", mec=col, mew=1.1, zorder=5)
        best[lab] = (min(val), ep[i], tr[0], tr[-1])
    print(f"B  predict zero {pz:.3f} m^2   " + "   ".join(
        f"{k} best {v:.3f} at ep{e} ({100 * (v / pz - 1):+.1f} %), "
        f"train {t0:.1f} -> {t1:.1f} ({100 * (t1 / t0 - 1):+.0f} %)"
        for k, (v, e, t0, t1) in best.items()))

    # left: training loss (weighted, so a different scale from validation)
    ax_t.set_yscale("log")
    ax_t.set_ylim(21, 96)
    ax_t.set_yticks([25, 35, 50, 70], minor=False)
    ax_t.set_yticklabels(["25", "35", "50", "70"])
    ax_t.yaxis.set_minor_locator(NullLocator())
    ax_t.yaxis.set_minor_formatter(NullFormatter())
    ax_t.set_xlim(0.4, 11.3)
    ax_t.set_xticks([1, 3, 5, 7, 9])
    ax_t.set_title("train loss", fontsize=8.2, color=INK, loc="left", pad=3)
    ax_t.set_xlabel("epoch", fontsize=8.2, labelpad=1)
    ax_t.set_ylabel("[m², outage-weighted]", fontsize=8.0, labelpad=2)
    _spine_off(ax_t, keep=("left", "bottom"))

    # right: validation against the floor
    ax_v.axhline(pz, color=INK, lw=0.9, ls="--", zorder=1)
    ax_v.text(9.42, pz + 0.07, f"predict zero  {pz:.2f} m²", fontsize=8.2,
              color=INK, va="bottom", ha="right")
    ax_v.set_ylim(pz - 0.26, 15.05)
    ax_v.set_xlim(0.5, 9.5)
    ax_v.set_xticks(range(1, 10))
    ax_v.set_yticks([12, 13, 14])
    ax_v.set_title("validation MSE", fontsize=8.2, color=INK, loc="left", pad=3)
    ax_v.set_xlabel("epoch", fontsize=8.2, labelpad=1)
    ax_v.set_ylabel("[m²]", fontsize=8.0, labelpad=2)
    ax_v.legend(frameon=False, fontsize=7.8, loc="upper right", borderpad=0.12,
                handlelength=1.4, labelspacing=0.25, ncol=1)
    _spine_off(ax_v, keep=("left", "bottom"))

    fig.tight_layout(pad=0.05)
    _save(fig, "b_training_wide.png", W_B, H_B)


def panel_c():
    print("C  unguarded: " + "; ".join(
        f"{n} alpha*={a:.2f} val {v:+.1f} % test {t:+.1f} %"
        for n, _, a, v, t in GATE))
    fig, ax = plt.subplots(figsize=(W_C, H_C))
    ys = np.arange(len(GATE))[::-1]                     # LSTM on top
    hgt = 0.30
    for off, key, lab, col in [(+0.17, 3, "validation", ACC["teal"]),
                               (-0.17, 4, "held-out test", ACC["red"])]:
        vals = [g[key] for g in GATE]
        ax.barh(ys + off, vals, height=hgt, color=col, label=lab, zorder=3)
        for yi, v in zip(ys + off, vals):
            ax.text(v + (0.14 if v > 0 else -0.14), yi,
                    f"{v:+.1f} %".replace("-", "−"),
                    ha="left" if v > 0 else "right", va="center",
                    fontsize=9.0, fontweight="bold", color=col, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g[0]}\n$\\alpha^*$ = {g[2]:.2f}" for g in GATE],
                       fontsize=8.4, linespacing=1.45)
    ax.set_xlim(-6.6, 4.6)
    ax.set_ylim(-0.62, 1.62)
    ax.set_xlabel("error change vs $\\alpha$ = 0  [%]", fontsize=8.2, labelpad=1)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", ncol=2,
              borderpad=0.1, handlelength=1.1, handletextpad=0.5,
              columnspacing=1.4)
    _spine_off(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.05)
    _save(fig, "c_gate_wide.png", W_C, H_C)


if __name__ == "__main__":
    _style()
    s = shares()
    print(f"A  samples {s['samples']:.1f} %  residual {s['residual']:.1f} %  "
          f"gradient {s['gradient']:.1f} %")
    panel_b()
    panel_c()
