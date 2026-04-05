#!/usr/bin/env bash
# Supporting runner for external atlas data acquisition
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/atlas_reference/acquisition.log"
mkdir -p outputs/reports/atlas_reference

echo "=== Atlas reference acquisition started at $(date) ===" | tee "$LOG"

echo "--- Step 1/5: Aging Mouse ---" | tee -a "$LOG"
$PYTHON code/atlas_reference.py --aging 2>&1 | tee -a "$LOG"

echo "--- Step 2/5: SEA-AD ---" | tee -a "$LOG"
$PYTHON code/atlas_reference.py --sea-ad 2>&1 | tee -a "$LOG"

echo "--- Step 3/5: WMB ---" | tee -a "$LOG"
$PYTHON code/atlas_reference.py --wmb 2>&1 | tee -a "$LOG"

echo "--- Step 4/5: Taxonomy mapping ---" | tee -a "$LOG"
$PYTHON code/atlas_reference.py --mapping 2>&1 | tee -a "$LOG"

echo "--- Step 5/5: Kinase gene coverage ---" | tee -a "$LOG"
$PYTHON code/atlas_reference.py --coverage 2>&1 | tee -a "$LOG"

echo "=== Atlas reference acquisition finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON code/atlas_reference.py --summary
