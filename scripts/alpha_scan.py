"""Where the shrinkage guard binds: scan alpha on the validation routes.

    python scripts/alpha_scan.py --model lstm
    python scripts/alpha_scan.py --model lstm --pos b2 --data dataset_b2.npz

Loads a trained checkpoint, predicts on the validation routes and prints the
validation outage error for every alpha on ``fit_alpha``'s grid.  The unguarded
optimum is the grid argmin (what a naive fit would ship); the guarded value is
what ``fit_alpha`` returns, i.e. what the reported metrics use.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import RUNS_DIR, load_config          # noqa: E402
from src.eval.metrics import summarize, traj_error    # noqa: E402
from src.models.correctors import build_model         # noqa: E402
from src.models.train import load_dataset, outage_gate, route_predictions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gpt4ts"])
    ap.add_argument("--pos", default="auto")
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--at", type=float, nargs="*", default=None,
                    help="also report the error at these fixed alphas")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config()
    ds = load_dataset(args.data)
    ds["y_scale"] = float(np.sqrt(np.mean(np.square(
        ds["train_y"].astype(np.float64))))) or 1.0
    device = torch.device(args.device)

    stem = args.model + ("_b2" if args.pos == "b2" else "")
    ckpt = torch.load(RUNS_DIR / "models" / f"{stem}.pt", map_location=device,
                      weights_only=False)
    model = build_model(args.model, cfg, in_len=ds["in_len"], out_len=ds["out_len"]).to(device)
    model.load_state_dict(ckpt["state"])
    preds = route_predictions(model, ds, args.split, cfg, types.SimpleNamespace(amp=False),
                              device, args.pos)
    fs = float(cfg["fs_imu"])

    def score(a):
        vals = []
        for r in preds.values():
            g = outage_gate(r["outage"], fs) if args.gate else np.ones(len(r["gt"]))
            e = traj_error(r["base"] + a * g[:, None] * r["dp"], r["gt"])
            vals.append(summarize(e, r["outage"], fs)["outage_mean"])
        return float(np.mean(vals))

    print(f"{stem} (pos={args.pos}): alpha scan on {len(preds)} {args.split} routes")
    zero, best_a, best_s = score(0.0), 0.0, float("inf")
    for a in np.linspace(0.0, 2.0, 41):
        m = score(float(a))
        if m < best_s:
            best_a, best_s = float(a), m
    print(f"  alpha = 0.00 (predict zero)  -> {zero:8.3f} m")
    print(f"  unguarded optimum alpha = {best_a:.2f} -> {best_s:8.3f} m "
          f"({100 * (1 - best_s / zero):+.1f} % on {args.split})")
    for a in args.at or []:
        s = score(float(a))
        print(f"  alpha = {a:.2f}                 -> {s:8.3f} m "
              f"({100 * (1 - s / zero):+.1f} % on {args.split})")


if __name__ == "__main__":
    main()
