#!/usr/bin/env bash
# Mainline runner for attribution recovery: kinase and cell-type hypothesis tables
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/attribution_recovery/analysis.log"
mkdir -p outputs/reports/attribution_recovery

echo "=== Attribution recovery started at $(date) ===" | tee "$LOG"

echo "--- S3: Kinase activity matrix + hypothesis table ---" | tee -a "$LOG"
$PYTHON code/attribution_recovery.py --kinase-profiles 2>&1 | tee -a "$LOG"

echo "--- S4: Cell-type evidence table + kinase profiles ---" | tee -a "$LOG"
$PYTHON code/attribution_recovery.py --celltype-profiles 2>&1 | tee -a "$LOG"

echo "--- Building interactive viewer ---" | tee -a "$LOG"
$PYTHON code/build_viewer.py 2>&1 | tee -a "$LOG"

echo "=== Attribution recovery finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON code/attribution_recovery.py --summary
