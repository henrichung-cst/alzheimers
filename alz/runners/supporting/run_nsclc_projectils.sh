#!/usr/bin/env bash
# TODO #2: full-cohort ProjecTILs projection for the 10x NSCLC reference.
#
# nsclc_projectils_map.R hyperslab-reads the full 10x CSC h5 in 25 K-cell column
# batches and projects every cell onto the CD8/CD4 human ProjecTILs refs; scGate
# (filter.cells=TRUE) gates T cells per cell. The full matrix is never loaded
# whole (1.3 B nnz). plan(sequential) — a future worker would duplicate the ref.
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
    pixi run Rscript alz/ingest/nsclc_projectils_map.R
  " 2>&1 | tee "$LOG"

echo "  done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
