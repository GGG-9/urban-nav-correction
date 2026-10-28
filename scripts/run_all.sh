#!/usr/bin/env bash
# End-to-end run: data -> numpy reference -> GPU filter -> parity -> benchmark
#                -> residual dataset -> learned correctors -> figures.
# Usage:  bash scripts/run_all.sh [cuda|cpu] [--skip-llm] [--skip-ref]
#   --skip-llm  stop before the learned correctors (GPU filter deliverable only)
#   --skip-ref  reuse an uploaded runs/kf instead of re-running the ~30 min
#               numpy reference (it must already contain metrics_all.json)
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=${1:-cuda}
SKIP_LLM=0
SKIP_REF=0
for a in "$@"; do
  [[ "$a" == "--skip-llm" ]] && SKIP_LLM=1
  [[ "$a" == "--skip-ref" ]] && SKIP_REF=1
done
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=== [1/10] simulate routes ==============================================="
python -m src.data.make_routes

echo "=== [2/10] numpy reference baselines B1/B2/B3 ==========================="
if [[ $SKIP_REF -eq 1 && -f runs/kf/metrics_all.json ]]; then
  echo "skipped: reusing runs/kf ($(ls runs/kf/*.npz | wc -l) routes)"
else
  python -m src.nav.pipeline
fi

echo "=== [3/10] GPU/CPU parity checks ========================================"
python -m src.hpc.test_parity --device "$DEV" --route 6 --n 60000

echo "=== [4/10] window-parallel GPU filter on every route ===================="
python -m src.hpc.batch_kf --device "$DEV"

echo "=== [5/10] CPU vs GPU benchmark ========================================="
python -m src.hpc.benchmark --device "$DEV"

echo "=== [6/10] residual dataset ============================================="
python -m src.features.build_dataset

if [[ $SKIP_LLM -eq 0 ]]; then
  echo "=== [7/10] B4  LSTM corrector =========================================="
  python -m src.models.train --model lstm --epochs 30 --batch 64 --device "$DEV" \
    || echo "!! stage 7 failed -- see the traceback above; the figures below will not include Ours"

  echo "=== [8/10] B6  GPT4TS + LoRA ==========================================="
  python -m src.models.train --model gpt4ts --epochs 15 --batch 32 --lr 1e-4 \
                             --amp --device "$DEV" \
    || echo "!! stage 8 failed -- see the traceback above (B4 may still have succeeded)"
else
  echo "=== [7/10][8/10] learned correctors skipped ============================="
fi

echo "=== [9/10] figures ====================================================="
python -m src.plots.figures

echo "=== [10/10] results/summary.md ========================================="
python scripts/make_summary.py || echo "!! summary table failed -- the figures above are unaffected"

echo
echo "done. results in ./results"
