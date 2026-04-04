#!/usr/bin/env bash
# Bundled live runner for the ordered attribution pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="outputs/reports/live_pipeline"
LOG="$LOG_DIR/analysis.log"
mkdir -p "$LOG_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

note() {
  echo "[$(timestamp)] $1" | tee -a "$LOG"
}

: > "$LOG"
note "Live pipeline started"

WMB_FILE="outputs/reports/wmb_expression/wmb_kinase_expression.csv"
if [[ ! -f "$WMB_FILE" ]]; then
  note "Missing supporting input: $WMB_FILE"
  note "Run: bash code/runners/supporting/run_wmb_expression.sh"
  exit 1
fi

note "Running data ingestion"
bash code/runners/main/run_data_ingest.sh 2>&1 | tee -a "$LOG"

note "Running kinase attribution"
bash code/runners/main/run_kinase_attribution.sh 2>&1 | tee -a "$LOG"

note "Running attribution recovery"
bash code/runners/main/run_attribution_recovery.sh 2>&1 | tee -a "$LOG"

note "Live pipeline finished"
