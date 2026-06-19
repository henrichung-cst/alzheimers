#!/usr/bin/env bash
# TODO #2: T/NK subset export + ProjecTILs projection for the 10x NSCLC reference.
#
# Two steps, both memory-bounded:
#   1. nsclc_subset_tnk.py  — stream the full 10x matrix, write the ~182 K-cell
#      T/NK subset to a native 10x h5 (peak RAM = one chunk's nnz).
#   2. nsclc_projectils_map.R — Read10X_h5 the subset, project onto CD8/CD4
#      human ProjecTILs refs (plan(sequential)). This is the heavy step.
#
# Runs inside a systemd --user scope with MemoryMax so an OOM is confined to
# this scope, never the shared box (pattern from run_hbca_download.sh).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MEM_CAP="${MEM_CAP:-14G}"
LOG_DIR="outputs/reports/nsclc_reference"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/projectils_$(date -u +%Y%m%dT%H%M%SZ).log"
SCOPE="nsclc_projectils_$(date -u +%s).scope"

echo "  scope=$SCOPE  cap=$MEM_CAP  log=$LOG"
echo "  starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

systemd-run --user --scope --unit "$SCOPE" \
  -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 \
  bash -c "
    set -euo pipefail
    echo '== [1/2] T/NK subset export =='
    pixi run python alz/ingest/nsclc_subset_tnk.py
    echo '== [2/2] ProjecTILs projection =='
    pixi run Rscript alz/ingest/nsclc_projectils_map.R
  " 2>&1 | tee "$LOG"

echo "  done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
