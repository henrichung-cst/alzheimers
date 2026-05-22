#!/usr/bin/env bash
# Supporting runner for external atlas data acquisition (Allen WMB + SEA-AD)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-pixi run --manifest-path "$REPO_ROOT/pixi.toml" python}"
LOG_DIR="outputs/logs"
LOG="$LOG_DIR/atlas_reference.log"
mkdir -p "$LOG_DIR"

echo "=== Atlas reference acquisition started at $(date) ===" | tee "$LOG"

echo "--- Step 1/2: SEA-AD effect sizes ---" | tee -a "$LOG"
$PYTHON alz/reference/atlas.py --sea-ad 2>&1 | tee -a "$LOG"

echo "--- Step 2/2: WMB-10Xv3 log2 expression matrices ---" | tee -a "$LOG"
$PYTHON alz/reference/atlas.py --wmb-download 2>&1 | tee -a "$LOG"

echo "=== Atlas reference acquisition finished at $(date) ===" | tee -a "$LOG"
