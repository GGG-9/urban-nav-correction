"""Residual dataset for the learned corrector.

Features (20 channels, ``rate`` Hz):
    0-2   gyro  [rad/s]
    3-5   accel [m/s^2]
    6-8   GNSS innovation [m]   (zero-order hold between fixes, max-|.| per block)
    9     outage flag (0/1)
    10    time since the last GNSS fix [s]
    11-13 filter velocity  v_b2 [m/s]
    14-16 filter accel bias b_a [m/s^2]
    17-19 filter gyro  bias b_g [rad/s]

The last nine channels are the filter's own state.  They are what makes the
residual learnable: the drift accumulated during an outage is (to first order)
the *error* of b_a and b_g integrated twice over the time since the last fix,
so the network needs both the elapsed time and the quantity being integrated.
Without them the mapping from a 100 s sensor window to a 10 m position error is
not identifiable, no matter how large the model.

Channel 6-8 keeps the *largest-magnitude* innovation inside each block instead of
the block mean: the reacquisition fix after an outage is a single 0.1 s spike
whose amplitude measures how far the forward filter has drifted, and averaging
would dilute it by the block length.  That spike is the one signal that makes
the outage residual identifiable at all.

Label:
    delta_p = p_gt - p_KF    (3 channels), the residual of the GPU ESKF+RTS
    trajectory that the sequence model has to learn.

Windows (``--centered``, default):
    input  x[c - half : c + half]        symmetric context around the centre
    output y[c - out/2 : c + out/2]      the residual at the centre
    The context is symmetric because the whole pipeline is offline: the RTS
    smoother already uses future data, and the post-outage GNSS reacquisition is
    what makes the drift accumulated during an outage identifiable.  A causal
    window would hide exactly that information.

    ``--causal`` builds the classic [0, in_len) -> [in_len, in_len + out_len)
    layout instead (used by the ablation "causal vs. symmetric context").

Down-sampling to ``--rate`` Hz is a block mean over the IMU / innovation channels
and a max over the outage flag.  Windows never cross a route boundary.
Normalisation statistics come from the training routes only.

Usage:
    python -m src.features.build_dataset                      # 10 Hz, centered
    python -m src.features.build_dataset --rate 100 --causal  # full-rate ablation
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..common import DATASET_DIR, RUNS_DIR, ensure_dirs, load_config

IN_LEN = 1024      # 102.4 s of context at 10 Hz
OUT_LEN = 128      # 12.8 s of residual
RATE = 10
IN_CH = 20
OUT_CH = 3


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
def route_run(stem: str) -> dict:
    path = RUNS_DIR / "kf" / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.nav.pipeline` first")
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def route_position(d: dict, stem: str, source: str = "auto") -> np.ndarray:
    """Base trajectory the corrector refines.

    ``b2`` selects the causal forward filter: the online case, where a smoother
    over the whole route does not exist yet.  ``auto`` / ``kf_gpu`` prefer the
    GPU windowed output and fall back to the numpy B3.
    """
    if source == "b2":
        return d["p_b2"]
    path = RUNS_DIR / "kf_gpu" / f"{stem}.npz"
    if source in ("auto", "kf_gpu") and path.exists():
        with np.load(path) as g:
            p = g["p_smooth"]
        if len(p) == len(d["p_gt"]):
            return p
        if source == "kf_gpu":
            raise ValueError(f"{path} has {len(p)} samples, route has {len(d['p_gt'])}")
    return d["p_b3"]


def route_features(d: dict, rate: float) -> np.ndarray:
    """[n, 11] feature matrix in physical units, at ``rate`` Hz."""
    innov = d["innovation"].astype(np.float64).copy()
    # zero-order hold the last innovation through the non-fix samples
    nz = np.any(innov != 0.0, axis=1)
    if nz.any():
        idx = np.where(nz, np.arange(len(nz)), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = 0
        innov = innov[idx]
    # Compress before normalisation: the innovation is ~4 m during normal
    # tracking but jumps to 100-300 m at the reacquisition fix.  A plain z-score
    # would scale the informative 4 m down to nothing.
    innov = np.sign(innov) * np.log1p(np.abs(innov))
    # time since the last GNSS fix
    fix = ~d["outage"].astype(bool)
    t = np.zeros(len(fix), np.float64)
    last = -1
    for i in range(len(fix)):
        if fix[i]:
            last = i
        t[i] = (i - last) / 100.0 if last >= 0 else i / 100.0
    parts = [d["gyro"], d["accel"], innov,
             d["outage"].astype(np.float64)[:, None], t[:, None]]
    for key in ("v_b2", "ba_b2", "bg_b2"):
        if key in d:
            parts.append(d[key])
    feat = np.concatenate(parts, axis=1)
    return downsample_mean(feat, rate, flags=[9], max_abs=[6, 7, 8])


def downsample_mean(x: np.ndarray, rate: float, flags=(), max_abs=()) -> np.ndarray:
    """Block-aggregate rows of ``x`` from fs_imu Hz to ``rate`` Hz."""
    fs = 100.0
    if abs(rate - fs) < 1e-9:
        return x.astype(np.float32)
    k = int(round(fs / rate))
    n = (len(x) // k) * k
    m = n // k
    y = x[:n].reshape(m, k, x.shape[1]).mean(axis=1)
    for f in flags:                       # flags take the max, not the mean
        y[:, f] = x[:n, f].reshape(m, k).max(axis=1)
    for f in max_abs:                     # keep the signed extreme value
        blk = x[:n, f].reshape(m, k)
        y[:, f] = blk[np.arange(m), np.abs(blk).argmax(axis=1)]
    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def cut_centered(x, y, in_len, out_len, stride, offset=0):
    """Symmetric context: x[c-in_len/2 : c+in_len/2], y[c-out/2 : c+out/2]."""
    n = len(x)
    half, oh = in_len // 2, out_len // 2
    cs = np.arange(half + offset, n - half + 1, stride)
    if cs.size == 0:
        return (np.zeros((0, in_len, x.shape[1]), np.float32),
                np.zeros((0, out_len, y.shape[1]), np.float32),
                np.zeros(0, np.int64))
    xs = np.stack([x[c - half: c + half] for c in cs])
    ys = np.stack([y[c - oh: c + oh] for c in cs])
    return xs.astype(np.float32), ys.astype(np.float32), cs


def cut_causal(x, y, in_len, out_len, stride):
    """Classic causal layout: input then output, no overlap."""
    n = len(x)
    ss = np.arange(0, n - in_len - out_len + 1, stride)
    if ss.size == 0:
        return (np.zeros((0, in_len, x.shape[1]), np.float32),
                np.zeros((0, out_len, y.shape[1]), np.float32),
                np.zeros(0, np.int64))
    xs = np.stack([x[s: s + in_len] for s in ss])
    ys = np.stack([y[s + in_len: s + in_len + out_len] for s in ss])
    return xs.astype(np.float32), ys.astype(np.float32), ss + in_len + out_len // 2


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=RATE, help="Hz of the feature series")
    ap.add_argument("--in-len", type=int, default=IN_LEN)
    ap.add_argument("--out-len", type=int, default=OUT_LEN)
    ap.add_argument("--stride", type=int, default=OUT_LEN // 2)
    ap.add_argument("--causal", action="store_true", help="causal instead of symmetric windows")
    ap.add_argument("--pos", choices=["auto", "kf", "kf_gpu", "b2"], default="auto",
                    help="base trajectory: numpy B3, the GPU windowed output, or the "
                         "causal forward filter (b2 -- the online case)")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": cfg["routes"]["train"], "val": cfg["routes"]["val"],
              "test": cfg["routes"]["test"]}
    cut = cut_causal if args.causal else cut_centered

    per_route = {}
    for path in sorted((RUNS_DIR / "kf").glob("route_*.npz")):
        rid = int(path.stem.split("_")[1])
        d = route_run(path.stem)
        x = route_features(d, args.rate)
        y = downsample_mean(d["p_gt"] - route_position(d, path.stem, args.pos), args.rate)
        xw, yw, cs = cut(x, y, args.in_len, args.out_len, args.stride)
        out_flag = x[:, 9]
        oh = args.out_len // 2
        cs_out = np.clip(cs - oh, 0, len(out_flag) - 1)
        ogate = np.stack([out_flag[c: c + args.out_len] for c in cs_out]).astype(np.float32)
        w = np.ones(len(xw), np.float32)
        # up-weight windows whose output block overlaps an outage
        w[ogate.mean(axis=1) > 0.0] = 2.0
        per_route[rid] = {"x": xw, "y": yw, "w": w, "og": ogate}
        print(f"{path.stem:<28s} windows={len(xw):6d}  outage-weighted={int((w > 1).sum()):5d}")

    train_ids = [i for i in splits["train"] if i in per_route]
    x_tr = np.concatenate([per_route[i]["x"] for i in train_ids])
    mu = x_tr.mean(axis=(0, 1))
    sd = x_tr.std(axis=(0, 1))
    sd[sd < 1e-8] = 1.0

    out = {}
    for split, ids in splits.items():
        ids = [i for i in ids if i in per_route]
        out[f"{split}_x"] = np.concatenate([(per_route[i]["x"] - mu) / sd for i in ids])
        out[f"{split}_y"] = np.concatenate([per_route[i]["y"] for i in ids])
        out[f"{split}_w"] = np.concatenate([per_route[i]["w"] for i in ids])
        out[f"{split}_og"] = np.concatenate([per_route[i]["og"] for i in ids])
        out[f"{split}_route"] = np.concatenate(
            [np.full(len(per_route[i]["x"]), i, np.int16) for i in ids])
        print(f"{split:<5s} x={out[f'{split}_x'].shape} y={out[f'{split}_y'].shape}")
    out["mu"], out["sd"] = mu.astype(np.float32), sd.astype(np.float32)
    out["in_len"], out["out_len"] = args.in_len, args.out_len
    # window geometry: the consumer needs it to place every predicted block back
    # on the route (see src/models/train.py:stitch)
    out["stride"] = args.stride
    out["rate"] = args.rate
    out["mode"] = "causal" if args.causal else "centered"

    tag = "_causal" if args.causal else ""
    if args.pos == "b2":
        tag += "_b2"          # keep the offline dataset intact
    path = DATASET_DIR / f"dataset{tag}.npz"
    np.savez_compressed(path, **out)
    with open(DATASET_DIR / f"stats{tag}.json", "w", encoding="utf-8") as fh:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "rate": args.rate,
                   "in_len": args.in_len, "out_len": args.out_len,
                   "mode": out["mode"]}, fh, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
