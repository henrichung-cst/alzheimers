#!/usr/bin/env bash
# Mainline runner for data ingestion and characterization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/data_ingest/analysis.log"
mkdir -p outputs/reports/data_ingest

echo "=== Data ingestion started at $(date) ===" | tee "$LOG"

echo "--- Step 1/4: Sample Mapping ---" | tee -a "$LOG"
$PYTHON alz/data_ingest.py --mapping 2>&1 | tee -a "$LOG"

echo "--- Step 2/4: Phosphosite-to-Protein Matching ---" | tee -a "$LOG"
$PYTHON alz/data_ingest.py --phospho-match 2>&1 | tee -a "$LOG"

echo "--- Step 3/4: Marker Protein Assessment (WMB atlas) ---" | tee -a "$LOG"
$PYTHON alz/data_ingest.py --markers 2>&1 | tee -a "$LOG"

echo "--- Step 4/4: Data Quality Assessment ---" | tee -a "$LOG"
$PYTHON alz/data_ingest.py --quality 2>&1 | tee -a "$LOG"

echo "=== Data ingestion finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON alz/data_ingest.py --summary
