#!/usr/bin/env bash
# Reshape pair-mode Incytr output → unified viewer.
#
# Reads `bench/incytr_pair_19/output/ma_<age>_<geno>_ma_<age>_WTyp_incytr_output.parquet`
# (9 expected; --strict to enforce), reshapes into the long-form
# `outputs/reports/incytr_factorial/receiver_cache/receiver=*/` layout, then
# rebuilds the unified viewer (which picks up the SiK_score column and NULLs
# out the per-node FC columns pair-mode does not emit).
#
# Resets receiver_cache/ so old partitions (e.g. from the prior factorial run)
# don't leak into the new build. The replaced cache lives at the path the
# viewer already reads — no flags needed downstream.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${PAIR_MODE_INPUT_DIR:-bench/incytr_pair_19/output}"
STRICT_FLAG=""
if [[ "${PAIR_MODE_STRICT:-0}" == "1" ]]; then
  STRICT_FLAG="--strict"
fi

echo "=== $(date -Is) reshape pair-mode → receiver_cache ==="
pixi run python alz/integration/pair_to_receiver_cache.py \
  --input-dir "$INPUT_DIR" \
  $STRICT_FLAG

echo "=== $(date -Is) rebuild unified viewer ==="
pixi run viewer

echo "=== $(date -Is) done ==="
ls -lh outputs/reports/unified_viewer/index.html
