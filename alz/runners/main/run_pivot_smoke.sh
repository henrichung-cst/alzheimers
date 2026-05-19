#!/usr/bin/env bash
# End-to-end smoke run for the Levy-t5 per-cluster proportional decomposition pivot.
# See docs/incytr_deconvolution_pivot.md — Step 13.
#
# Sequential, no background polling. Fails loud on any stage error.
#
# Usage:
#   bash alz/runners/main/run_pivot_smoke.sh [--skip-normalize]
#
# Stages:
#   1. Stage 1 (bulk normalize) — only if total_proteome_normalized.csv missing
#   2. Stage 5 — snRNA per-cluster proportions
#   3. Stage 6 — proportional decomposition (st + py)
#   4. Stage 7 — per-cluster MEA enrichment (st + py)
#   5. Verification (alz/decomposition/verify_decomposition.py)
#
# Factorial Incytr was archived 2026-05-18 (see archive/incytr_factorial_2026-05-18/);
# pair-mode is the active Incytr path, driven by run_pair_mode_pipeline.sh.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SPINE="levy_t5"
NORM_FILE="outputs/reports/kinase_attribution/total_proteome_normalized.csv"
SKIP_NORMALIZE=0

for arg in "$@"; do
  case "$arg" in
    --skip-normalize) SKIP_NORMALIZE=1 ;;
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
pixi run python alz/snrna_proportions.py --run --spine "$SPINE"

echo "=== Step 3: Stage 6 — proportional decomposition (st + py) ==="
pixi run python alz/decomposition/build_celltype_decomposition.py --spine "$SPINE" --track both

echo "=== Step 4a: Stage 7 — per-cluster MEA (st) ==="
pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track st
echo "=== Step 4b: Stage 7 — per-cluster MEA (py) ==="
pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track py || \
  echo "  pY track skipped/failed (likely missing phospho_per_cluster_pY.parquet)"

echo "=== Step 5: verification ==="
pixi run python alz/decomposition/verify_decomposition.py --spine "$SPINE"

echo "=== DONE ==="
