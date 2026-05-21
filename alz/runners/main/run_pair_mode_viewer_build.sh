#!/usr/bin/env bash
# Reshape pair-mode Incytr output → unified viewer.
#
# Reads `outputs/reports/incytr_pair_mode/wide/ma_<age>_<geno>_ma_<age>_WTyp_incytr_output.parquet`
# (9 expected; --strict to enforce), reshapes into the long-form
# `outputs/reports/incytr_pair_mode/receiver_cache/receiver=*/` layout, then
# rebuilds the unified viewer.
#
# Resets receiver_cache/ each run so stale partitions don't leak into the build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${PAIR_MODE_INPUT_DIR:-outputs/reports/incytr_pair_mode/wide}"
STRICT_FLAG=""
if [[ "${PAIR_MODE_STRICT:-0}" == "1" ]]; then
  STRICT_FLAG="--strict"
fi

export INCYTR_PAIR_MODE_INPUT_DIR="$INPUT_DIR"

echo "=== $(date -Is) reshape pair-mode → receiver_cache ==="
pixi run python alz/integration/pair_to_receiver_cache.py \
  --input-dir "$INPUT_DIR" \
  $STRICT_FLAG

echo "=== $(date -Is) rebuild unified viewer ==="
pixi run viewer

echo "=== $(date -Is) done ==="
ls -lh outputs/reports/unified_viewer/index.html
