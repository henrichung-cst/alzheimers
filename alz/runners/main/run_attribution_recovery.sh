#!/usr/bin/env bash
# Mainline runner for attribution recovery: kinase and cell-type hypothesis tables
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-pixi run --manifest-path "$REPO_ROOT/pixi.toml" python}"
LOG="outputs/reports/attribution_recovery/analysis.log"
mkdir -p outputs/reports/attribution_recovery

echo "=== Attribution recovery started at $(date) ===" | tee "$LOG"

# The Kedro `recovery` pipeline produces all three hypothesis tables atomically.
$PYTHON alz/attribution_recovery.py 2>&1 | tee -a "$LOG"

echo "=== Attribution recovery finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON alz/attribution_recovery.py --summary
