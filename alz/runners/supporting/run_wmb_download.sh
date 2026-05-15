#!/usr/bin/env bash
# Download all 13 WMB-10Xv3 log2 expression matrices (~95 GB total).
# Run this BEFORE run_wmb_expression.sh. Safe to re-run (idempotent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-pixi run --manifest-path "$REPO_ROOT/pixi.toml" python}"
LOG="outputs/reports/wmb_expression/wmb_download.log"
mkdir -p outputs/reports/wmb_expression

echo "=== WMB download started at $(date) ===" | tee "$LOG"
$PYTHON alz/atlas_reference.py --wmb-download 2>&1 | tee -a "$LOG"
echo "=== WMB download finished at $(date) ===" | tee -a "$LOG"
