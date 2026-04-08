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

# --- Prerequisite checks and auto-resolution ---

# 1. Atlas reference: SEA-AD data (needed for unified attribution)
SEA_AD_DIR="data/external/sea_ad"
if [[ ! -d "$SEA_AD_DIR" ]] || [[ -z "$(ls -A "$SEA_AD_DIR" 2>/dev/null)" ]]; then
  note "SEA-AD data not found — running atlas reference acquisition"
  bash code/runners/supporting/run_atlas_reference.sh 2>&1 | tee -a "$LOG"
fi

# 2. WMB h5ad files (needed for wmb_expression)
N_H5AD=$(find data/external/allen_abc/expression_matrices/WMB-10Xv3/ \
    -name "*-log2.h5ad" 2>/dev/null | wc -l)
if [[ "$N_H5AD" -lt 13 ]]; then
  note "Only $N_H5AD/13 WMB region h5ad files found — running WMB download"
  bash code/runners/supporting/run_wmb_download.sh 2>&1 | tee -a "$LOG"
fi

# 3. WMB expression matrices (kinase + proteome, direct pipeline inputs)
WMB_KINASE="outputs/reports/wmb_expression/wmb_kinase_expression.csv"
WMB_PROTEOME="outputs/reports/wmb_expression/wmb_proteome_expression.csv"
if [[ ! -f "$WMB_KINASE" ]] || [[ ! -f "$WMB_PROTEOME" ]]; then
  note "WMB expression matrix missing — running WMB expression export"
  bash code/runners/supporting/run_wmb_expression.sh 2>&1 | tee -a "$LOG"
fi

# 4. Song snRNA integration (within-cohort specificity + concordance)
SONG_SPEC="outputs/reports/snrna_integration/song_expression_specificity.csv"
SONG_CONC="outputs/reports/snrna_integration/song_concordance.csv"
if [[ ! -f "$SONG_SPEC" ]] || [[ ! -f "$SONG_CONC" ]]; then
  note "Song snRNA integration outputs missing — running snRNA integration"
  bash code/runners/supporting/run_snrna_integration.sh 2>&1 | tee -a "$LOG"
fi

note "Running data ingestion"
bash code/runners/main/run_data_ingest.sh 2>&1 | tee -a "$LOG"

note "Running kinase attribution"
bash code/runners/main/run_kinase_attribution.sh 2>&1 | tee -a "$LOG"

note "Running attribution recovery"
bash code/runners/main/run_attribution_recovery.sh 2>&1 | tee -a "$LOG"

note "Live pipeline finished"
