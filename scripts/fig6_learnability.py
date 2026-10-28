"""Figure 6: why the learned corrector cannot work.

Three panels, one message each:

    (a) one GNSS outage - the residual is a straight line in time, i.e. the
        position and velocity error frozen at the outage instant;
    (b) that is not one lucky segment - the same [1, t] subspace explains
        78-84 % of the residual on every test route;
    (c) what each kind of correction actually buys on the outage metric:
        the true residual pushed through the identical pipeline removes 80 %+,
        the trained correctors remove nothing, and an oracle fix of the
        accelerometer scale error removes 1 %.

    python scripts/fig6_learnability.py
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import RUNS_DIR, load_config          # noqa: E402
from src.eval.metrics import summarize, traj_error    # noqa: E402
from src.features.build_dataset import (IN_LEN, OUT_LEN, RATE,   # noqa: E402
                                        cut_centered, downsample_mean,
                                        route_features)
from src.models.train import outage_gate, stitch      # noqa: E402
from src.plots.figures import COL, _out, _style       # noqa: E402


def load(rid: int, base: str = "b3"):
    path = sorted(glob.glob(str(RUNS_DIR / "kf" / f"route_{rid:02d}_*.npz")))[0]
    with np.load(path, allow_pickle=True) as d:
        d = {k: d[k] for k in d.files}
    return d, d["p_gt"] - d[f"p_{base}"]


def segments(mask: np.ndarray):
    st = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(st[::2], st[1::2]))


def line_fit(t: np.ndarray, r: np.ndarray):
    A = np.c_[np.ones_like(t), t]
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    return A @ c


def rms(x):
    return float(np.sqrt((x ** 2).mean()))


def oracle_pipeline(rids, base="b3"):
    """True block-mean residual through the same stitch+gate path as the model."""
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    up = int(round(fs / RATE))
    preds = {}
    for rid in rids:
        d, resid = load(rid, base)
        x = route_features(d, RATE)
        y = downsample_mean(resid, RATE)
        _, yw, _ = cut_centered(x, y, IN_LEN, OUT_LEN, OUT_LEN // 2)
        p = np.repeat(yw.mean(1)[:, None, :], OUT_LEN, axis=1).astype(np.float32)
        dp = stitch(len(y), p, IN_LEN, OUT_LEN, OUT_LEN // 2, "centered")
        dp = np.repeat(dp, up, axis=0)[: len(d["p_gt"])]
        preds[rid] = {"base": d[f"p_{base}"], "dp": dp, "gt": d["p_gt"],
                      "outage": d["outage"].astype(bool)}
    b, o = [], []
    for r in preds.values():
        g = outage_gate(r["outage"], fs)
        b.append(summarize(traj_error(r["base"], r["gt"]), r["outage"], fs)["outage_mean"])
        o.append(summarize(traj_error(r["base"] + g[:, None] * r["dp"], r["gt"]),
                           r["outage"], fs)["outage_mean"])
    return 100.0 * (1 - np.mean(o) / np.mean(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="b3", choices=["b3", "b2"])
    args = ap.parse_args()
    _style()
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    test_ids = [int(r) for r in cfg["routes"]["test"]]

    # ---------------------------------------------------------------- (a), (b)
    explained, example = {}, None
    for rid in test_ids:
        d, resid = load(rid, args.base)
        out = d["outage"].astype(bool)
        err = {1: 0.0, 2: 0.0}
        den = float((resid[out] ** 2).sum())
        for a, b in segments(out):
            t = (np.arange(a, b) - a) / fs
            r = resid[a:b]
            fit = line_fit(t, r)
            for deg in (1, 2):
                A = np.vander(t, deg + 1, increasing=True)
                c, *_ = np.linalg.lstsq(A, r, rcond=None)
                err[deg] += float(((r - A @ c) ** 2).sum())
            A2 = np.vander(t, 3, increasing=True)
            c2, *_ = np.linalg.lstsq(A2, r, rcond=None)
            cand = {"rid": rid, "t": t, "r": r, "fit": fit, "fit2": A2 @ c2,
                    "err": 100.0 * (1 - rms(r - fit) / rms(r)),
                    "err2": 100.0 * (1 - rms(r - A2 @ c2) / rms(r))}
            # show the segment closest to the average, not the best-looking one
            if example is None or abs(cand["err"] - 65) < abs(example["err"] - 65):
                example = cand
        explained[rid] = tuple(100.0 * (1 - np.sqrt(err[deg] / den)) for deg in (1, 2))

    # ---------------------------------------------------------------- (c)
    oracle_pct = oracle_pipeline(test_ids, args.base)

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.3))

    # (a) one segment, component by component: a straight line each
    a = ax[0]
    e = example
    for k, (name, col) in enumerate(zip(("East", "North", "Up"),
                                        (COL["b3"], COL["b2"], "#7f7f7f"))):
        a.plot(e["t"], e["r"][:, k], color=col, lw=1.4, label=name)
        a.plot(e["t"], e["fit2"][:, k], "--", color=col, lw=1.0)
    a.plot([], [], "--", color="#444444", lw=1.0, label="quadratic fit")
    a.set_xlabel("Time since outage onset [s]")
    a.set_ylabel("Position residual [m]")
    a.set_title(f"(a) Route {e['rid']:02d}: a frozen acceleration error")
    a.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    a.annotate(f"$e_p+e_v t+e_a t^2$ explains {e['err2']:.0f} %",
               xy=(0.03, 0.95), xycoords="axes fraction", fontsize=8, va="top")

    # (b) every test route
    b = ax[1]
    ids = list(explained)
    lin = [explained[i][0] for i in ids]
    quad = [explained[i][1] for i in ids]
    x = np.arange(len(ids))
    w = 0.38
    b.bar(x - w / 2, lin, w, color=COL["b3"], label=r"$e_p+e_v t$")
    b.bar(x + w / 2, quad, w, color=COL["ours"], label=r"$+e_a t^2$")
    b.axhline(np.mean(quad), color=COL["b1"], ls="--", lw=1.0)
    b.annotate(f"mean {np.mean(quad):.0f} %", xy=(len(ids) - 0.5, np.mean(quad)),
               xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7.5)
    b.set_xticks(x, [f"{i:02d}" for i in ids])
    b.set_ylim(0, 124)
    b.set_xlabel("Test route")
    b.set_ylabel("Residual explained [%]")
    b.set_title("(b) A smooth curve on every route")
    b.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

    # (c) what each correction actually buys
    c = ax[2]
    names = ["Oracle residual\n(same pipeline)",
             "Learned corrector\n(LSTM / GPT4TS)",
             "Oracle IMU\nscale-error fix"]
    pct = [oracle_pct, 0.0, 0.9]
    cols = [COL["ours"], COL["b2"], "#999999"]
    hb = c.barh(names, pct, color=cols, height=0.6)
    for r, v in zip(hb, pct):
        c.annotate(f"{v:.0f} %" if v >= 5 else f"{v:.1f} %",
                   (max(v, 0) + 1.5, r.get_y() + r.get_height() / 2),
                   va="center", fontsize=8)
    c.set_xlim(0, 100)
    c.set_xlabel("Outage-error reduction [%]")
    c.set_title("(c) Only the true residual transfers")
    c.invert_yaxis()

    fig.tight_layout()
    out = _out(f"fig6_learnability_{args.base}.png")
    fig.savefig(out)
    print(f"-> {out}")
    print(f"   (a) route {e['rid']:02d}: {e['err']:.1f} % explained")
    print(f"   (b) linear mean {np.mean(lin):.1f} % ({min(lin):.1f}-{max(lin):.1f})"
          f"   quadratic mean {np.mean(quad):.1f} % ({min(quad):.1f}-{max(quad):.1f})")
    print(f"   (c) oracle residual through the pipeline: -{oracle_pct:.1f} %")


if __name__ == "__main__":
    main()
