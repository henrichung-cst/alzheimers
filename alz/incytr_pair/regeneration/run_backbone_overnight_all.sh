#!/usr/bin/env bash
# Combined overnight Incytr regeneration runner.
#
# Runs the remaining cohorts sequentially in one operator-started job:
#   1. 5xFAD: KsG + PTM + phospho-only derive + filter; bridge deferred
#   2. T-cell: KsG pair-mode + per-donor filter; bridge skipped by design
#
# This does not run cohorts concurrently. If 5xFAD pair-mode/post-processing
# fails, the t-cell run is not started. The 5xFAD bridge is intentionally not a
# gate for t-cell.
#
# Run from the repo root in tmux:
#   bash alz/incytr_pair/regeneration/run_backbone_overnight_all.sh
#
# Combined log:
#   outputs/reports/incytr_pair_mode_regeneration/overnight_all_<timestamp>.log
#
# Per-cohort logs are still written by the leaf runners:
#   outputs/reports/incytr_pair_mode_5xfad/overnight_<timestamp>.log
#   outputs/reports/incytr_pair_mode_tcells/overnight_<timestamp>.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="outputs/reports/incytr_pair_mode_regeneration"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_all_${TS}.log"

exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) combined Incytr overnight run start ==="
echo "  REPO: $REPO_ROOT"
echo "  Log:  $LOG"
echo ""
echo "This runner is sequential: 5xFAD completes before t-cell starts."
echo "If any step fails, the script exits and later cohorts are not started."

echo ""
echo "=== $(date -Is) [1/2] 5xFAD ==="
SKIP_BRIDGE=yes bash "$REPO_ROOT/alz/incytr_pair/regeneration/run_backbone_overnight_5xfad.sh"
echo "=== $(date -Is) [1/2] 5xFAD complete ==="

echo ""
echo "=== $(date -Is) [2/2] t-cell ==="
bash "$REPO_ROOT/alz/incytr_pair/regeneration/run_backbone_overnight_tcells.sh"
echo "=== $(date -Is) [2/2] t-cell complete ==="

echo ""
echo "=== $(date -Is) combined Incytr overnight run complete ==="
echo "  Combined log: $LOG"
echo "  5xFAD logs:   outputs/reports/incytr_pair_mode_5xfad/overnight_*.log"
echo "  T-cell logs:  outputs/reports/incytr_pair_mode_tcells/overnight_*.log"
echo "  5xFAD bridge: deferred; run kinase-incytr-bridge per tissue after cohort jobs."
