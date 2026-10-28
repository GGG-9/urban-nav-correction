"""Summary of the online variant: causal forward filter (B2) + learned corrector.

The offline pipeline corrects the RTS smoother (B3), which is only available
after the whole route has been seen.  A vehicle that has to decide *now* can
only run the forward filter, so this variant trains the same sequence models on
the residual of B2 and reports whether the correction still pays off.

    python scripts/summary_online.py

Reads:
    runs/kf/route_*.json              B1 / B2 / B3 per route (numpy reference)
    results/online/metrics_*.json     B2 -> Ours per route (after training)
Writes results/online/summary.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import RESULTS_DIR, RUNS_DIR, load_config  # noqa: E402

ONLINE = RESULTS_DIR / "online"


def kf_rows(ids):
    rows = {}
    for rid in ids:
        p = sorted((RUNS_DIR / "kf").glob(f"route_{rid:02d}_*.json"))
        if p:
            with open(p[0], encoding="utf-8") as fh:
                rows[rid] = json.load(fh)
    return rows


def ours_rows():
    """route id -> (model, per-route metrics, alpha), best model per route."""
    out = {}
    for p in sorted(ONLINE.glob("metrics_*.json")):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("base") not in (None, "b2"):
            continue
        for rid, r in m.get("routes", {}).items():
            if "ours" not in r:
                continue
            rid = int(rid)
            cur = out.get(rid)
            if cur is None or r["ours"]["outage_mean"] < cur[1]["ours"]["outage_mean"]:
                out[rid] = (m.get("model", "?"), r, m.get("alpha"))
    return out


def main() -> None:
    cfg = load_config()
    ids = list(cfg["routes"]["test"])
    kf, ours = kf_rows(ids), ours_rows()
    if not ours:
        print(f"nothing to summarise: no results/online/metrics_*.json")
        return

    lines = ["# Online variant: learned correction of the causal filter", "",
             "GNSS-outage 3-D position error, mean / max in metres "
             f"({len(kf)} simulated test routes).", "",
             "| route | B1 DR | B2 ESKF | Ours (B2 + corrector) |", "|---|---|---|---|"]
    acc, mx, base_m, base_x, alphas = [], [], [], [], []
    for rid in ids:
        if rid not in kf or rid not in ours:
            continue
        r, (_, rr, a) = kf[rid], ours[rid]
        lines.append(f"| {rid:02d} | {r['B1']['outage_mean']:.1f} / {r['B1']['outage_max']:.0f} "
                     f"| {r['B2']['outage_mean']:.1f} / {r['B2']['outage_max']:.0f} "
                     f"| **{rr['ours']['outage_mean']:.2f} / {rr['ours']['outage_max']:.1f}** |")
        acc.append(rr["ours"]["outage_mean"])
        mx.append(rr["ours"]["outage_max"])
        base_m.append(rr["b3"]["outage_mean"])
        base_x.append(rr["b3"]["outage_max"])
        alphas.append(a)

    n = len(acc)
    lines.append(f"| **test mean** | - | **{np.mean(base_m):.2f}** / {np.max(base_x):.0f} "
                 f"| **{np.mean(acc):.2f}** / {np.max(mx):.0f} |")
    a = alphas[0] if len(set(alphas)) == 1 else f"{min(alphas):.2f}-{max(alphas):.2f}"
    lines += ["", f"Shrinkage coefficient fitted on the validation routes: alpha = {a}; "
                  f"the base column is B2, the causal filter, not the smoother.",
              "", "## Headline", ""]
    lines.append(f"* outage mean, causal filter -> corrected: **{np.mean(base_m):.2f} -> "
                 f"{np.mean(acc):.2f} m ({100 * (1 - np.mean(acc) / np.mean(base_m)):+.1f} %)**")
    lines.append(f"* worst single outage: **{np.max(base_x):.0f} -> {np.max(mx):.1f} m "
                 f"({100 * (1 - np.max(mx) / np.max(base_x)):+.1f} %)**")
    lines.append(f"* routes evaluated: {n}/{len(ids)}")

    text = "\n".join(lines) + "\n"
    ONLINE.mkdir(parents=True, exist_ok=True)
    (ONLINE / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {ONLINE / 'summary.md'}")


if __name__ == "__main__":
    main()
