#!/usr/bin/env bash
# End-to-end smoke run for the Levy-19 per-cluster proportional decomposition pivot.
# See docs/incytr_deconvolution_pivot.md — Step 13.
#
# Sequential, no background polling. Fails loud on any stage error.
#
# Usage:
#   bash alz/runners/main/run_pivot_smoke.sh [--skip-normalize] [--skip-incytr]
#
# Stages:
#   1. Stage 1 (bulk normalize) — only if total_proteome_normalized.csv missing
#   2. Stage 5 — snRNA per-cluster proportions
#   3. Stage 6 — proportional decomposition (st + py)
#   4. Stage 7 — per-cluster MEA enrichment (st + py)
#   5. Stage 8a — factorial input export (per-cluster parquets)
#   6. Stage 8b — Incytr factorial R wrapper (gated on upstream Incytr exports)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SPINE="levy19"
NORM_FILE="outputs/reports/kinase_attribution/total_proteome_normalized.csv"
SKIP_NORMALIZE=0
SKIP_INCYTR=0

for arg in "$@"; do
  case "$arg" in
    --skip-normalize) SKIP_NORMALIZE=1 ;;
    --skip-incytr)    SKIP_INCYTR=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=== Step 1: bulk normalize ==="
if [[ -f "$NORM_FILE" ]] && [[ $SKIP_NORMALIZE -eq 1 ]]; then
  echo "  --skip-normalize and $NORM_FILE present; skipping"
elif [[ -f "$NORM_FILE" ]]; then
  echo "  $NORM_FILE present; skipping (pass --skip-normalize to silence)"
else
  pixi run normalize
fi

echo "=== Step 2: Stage 5 — snRNA per-cluster proportions ==="
pixi run python alz/snrna_proportions.py --spine "$SPINE"

echo "=== Step 3: Stage 6 — proportional decomposition (st + py) ==="
pixi run python alz/decomposition/build_celltype_decomposition.py --spine "$SPINE" --track both

echo "=== Step 4a: Stage 7 — per-cluster MEA (st) ==="
pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track st
echo "=== Step 4b: Stage 7 — per-cluster MEA (py) ==="
pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track py || \
  echo "  pY track skipped/failed (likely missing phospho_per_cluster_pY.parquet)"

echo "=== Step 5: Stage 8a — Incytr factorial input export ==="
pixi run python alz/integration/export_factorial_inputs.py

if [[ $SKIP_INCYTR -eq 1 ]]; then
  echo "=== Step 6: Stage 8b skipped (--skip-incytr) ==="
else
  echo "=== Step 6: Stage 8b — Incytr factorial R wrapper ==="
  pixi run incytr-factorial
fi

echo "=== Step 7: verification ==="
pixi run python alz/decomposition/verify_decomposition.py --spine "$SPINE"

echo "=== DONE ==="
