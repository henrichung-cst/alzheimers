#!/usr/bin/env bash
# Mainline runner for kinase attribution: stoichiometry MEA enrichment + unified attribution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python
LOG="outputs/reports/kinase_attribution/analysis.log"
mkdir -p outputs/reports/kinase_attribution

echo "=== Kinase attribution started at $(date) ===" | tee "$LOG"

echo "--- Stage 1/3: Cross-Plex Normalization + Stoichiometry ---" | tee -a "$LOG"
$PYTHON alz/kinase_normalize.py 2>&1 | tee -a "$LOG"

echo "--- Stage 2/3: OLS Models + MEA Kinase Enrichment ---" | tee -a "$LOG"
$PYTHON alz/kinase_enrich.py 2>&1 | tee -a "$LOG"

echo "--- Stage 3/3: Unified Cell-Type Attribution ---" | tee -a "$LOG"
$PYTHON alz/kinase_attribute.py 2>&1 | tee -a "$LOG"

echo "=== Kinase attribution finished at $(date) ===" | tee -a "$LOG"
echo "Summary:"
$PYTHON alz/kinase_summary.py
