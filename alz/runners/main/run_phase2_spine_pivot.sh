#!/usr/bin/env bash
# Phase 2 of docs/spine_pivot_rerun_plan.md
# Refresh bulk attribution for both modes after Song concordance refresh.
# Skips enrich/normalize (spine-agnostic).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="pixi run --manifest-path $REPO_ROOT/pixi.toml python"
LOG=outputs/reports/audits/phase2_spine_pivot_$(date +%Y%m%d_%H%M%S).log
mkdir -p outputs/reports/audits

{
  echo "=== Phase 2 spine-pivot rerun started at $(date) ==="

  echo "--- (1) Snapshot existing canonical (males_only, fresh) ---"
  rm -rf outputs/reports/kinase_attribution_males_only outputs/reports/attribution_recovery_males_only
  cp -r outputs/reports/kinase_attribution outputs/reports/kinase_attribution_males_only
  cp -r outputs/reports/attribution_recovery outputs/reports/attribution_recovery_males_only

  echo "--- (2) Full-cohort sensitivity: attribute + mechanism + recover ---"
  export KEDRO_ENV=full_cohort
  $PYTHON alz/bulk_mea/enrich.py
  $PYTHON alz/bulk_mea/attribute.py
  $PYTHON alz/bulk_mea/mechanism.py
  $PYTHON alz/bulk_mea/recover.py
  unset KEDRO_ENV

  echo "--- (3) Snapshot canonical → full_cohort ---"
  rm -rf outputs/reports/kinase_attribution_full_cohort outputs/reports/attribution_recovery_full_cohort
  cp -r outputs/reports/kinase_attribution outputs/reports/kinase_attribution_full_cohort
  cp -r outputs/reports/attribution_recovery outputs/reports/attribution_recovery_full_cohort

  echo "--- (4) Restore canonical as males_only primary ---"
  $PYTHON alz/bulk_mea/enrich.py
  $PYTHON alz/bulk_mea/attribute.py
  $PYTHON alz/bulk_mea/mechanism.py
  $PYTHON alz/bulk_mea/recover.py

  echo "=== Phase 2 finished at $(date) ==="
} 2>&1 | tee "$LOG"
