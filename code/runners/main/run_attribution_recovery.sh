#!/usr/bin/env bash
# Mainline runner for attribution recovery: cross-contrast analysis + final table
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/attribution_recovery/analysis.log"
mkdir -p outputs/reports/attribution_recovery

echo "=== Attribution recovery started at $(date) ===" | tee "$LOG"

echo "--- Stage 1/2: Cross-Contrast Consistency ---" | tee -a "$LOG"
$PYTHON code/attribution_recovery.py --cross-contrast 2>&1 | tee -a "$LOG"

echo "--- Stage 2/2: Final Attribution Table ---" | tee -a "$LOG"
$PYTHON code/attribution_recovery.py --comprehensive 2>&1 | tee -a "$LOG"

echo "=== Attribution recovery finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON code/attribution_recovery.py --summary
