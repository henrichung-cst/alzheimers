#!/usr/bin/env bash
# Per-cell sub-state AUROC end-to-end. The R extract step loads the multi-GB
# cell-level Seurat object (~5 GB compressed, ~15-30 GB in RAM), so it runs under
# a memory cap: if it exceeds the cap it is killed, protecting the shared box
# rather than OOM-ing it. Run in tmux/screen when >~28 GB RAM is free.
#
#   bash alz/analysis/run_tcell_percell_auroc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MEMMAX="${MEMMAX:-26G}"
LOG=outputs/reports/tcell_labeling/auroc/extract.log
mkdir -p outputs/reports/tcell_labeling/auroc

# 1. marker list (single source: SIGNATURES) for the R extractor
pixi run python alz/analysis/tcell_percell_auroc.py \
  --write-markers outputs/reports/tcell_labeling/auroc/marker_genes.txt

# 2. per-cell marker expression from the .rds, one donor at a time, memory-capped
for donor in donor1 donor2; do
  echo "=== extracting $donor (MemoryMax=$MEMMAX) ==="
  systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax=2G \
    pixi run Rscript alz/analysis/tcell_export_marker_cells.R "$donor" 2>&1 | tee -a "$LOG"
done

# 3. per-cell AUROC (light — Python only)
pixi run python alz/analysis/tcell_percell_auroc.py
