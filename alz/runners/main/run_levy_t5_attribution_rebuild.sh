#!/usr/bin/env bash
# Levy_t5 attribution pivot rebuild: kinase_attribute → attribution_recovery → viewer.
# Run after switching config.CLUSTER_SPINE_FILE to the levy_t5 spine and extending
# cluster_to_wmb_class.csv + cluster_to_seaad_supertype.csv to 31 clusters.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

LOG_DIR="outputs/reports/levy_t5_rebuild/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/build.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)
echo "=== levy_t5 attribution rebuild start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo
echo "--- [1/3] kinase_attribute (regenerate unified_attribution.csv) ---"
pixi run python alz/kinase_attribute.py

echo
echo "--- [2/3] attribution_recovery (regenerate hypothesis tables) ---"
pixi run python alz/attribution_recovery.py

echo
echo "--- [3/3] build_unified_viewer (pair-mode levy_t5) ---"
pixi run python alz/build_unified_viewer.py

T1=$(date +%s)
echo
echo "=== levy_t5 attribution rebuild DONE in $((T1 - T0))s ==="
echo "log: $MAIN_LOG"
ls -lh outputs/reports/unified_viewer/index.html 2>/dev/null || true
