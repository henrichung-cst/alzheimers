#!/usr/bin/env bash
# Runs all supplementary reviewer-response diagnostics.
# Prerequisites: main pipeline must have been run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG_DIR="outputs/reports/supplementary"
LOG="$LOG_DIR/reviewer_diagnostics.log"
mkdir -p "$LOG_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

note() {
  echo "[$(timestamp)] $1" | tee -a "$LOG"
}

: > "$LOG"
note "Reviewer diagnostics started"

# --- Prerequisite: mechanism annotation (needed by parent_protein_qc) ---
MECH_FILE="outputs/reports/kinase_attribution/mechanism_annotation.csv"
if [[ ! -f "$MECH_FILE" ]]; then
  note "mechanism_annotation.csv not found — running kinase_mechanism.py"
  $PYTHON alz/kinase_mechanism.py 2>&1 | tee -a "$LOG"
fi

# --- Q4: Stringent FDR ---
note "Running Q4: Stringent FDR analysis"
$PYTHON alz/supplementary/fdr_stringent.py --run 2>&1 | tee -a "$LOG"

# --- Q1: Threshold sensitivity ---
note "Running Q1: Threshold sensitivity analysis"
$PYTHON alz/supplementary/threshold_sensitivity.py --run 2>&1 | tee -a "$LOG"

# --- Q2: Aggregation robustness ---
note "Running Q2: Aggregation robustness analysis"
$PYTHON alz/supplementary/aggregation_robustness.py --run 2>&1 | tee -a "$LOG"

# --- Q5: Parent protein QC ---
note "Running Q5: Parent protein QC"
$PYTHON alz/supplementary/parent_protein_qc.py --run 2>&1 | tee -a "$LOG"

note "Reviewer diagnostics finished"

echo ""
echo "=== Summaries ==="
$PYTHON alz/supplementary/fdr_stringent.py --summary
$PYTHON alz/supplementary/threshold_sensitivity.py --summary
$PYTHON alz/supplementary/aggregation_robustness.py --summary
$PYTHON alz/supplementary/parent_protein_qc.py --summary
