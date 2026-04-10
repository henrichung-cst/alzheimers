#!/bin/bash
# Phase 1 proof of concept: Microglia-PVM -> L5 IT, WT vs App, 4mo males
#
# Runs Python adapters (alzheimers env) then R wrappers (incytr env).
# Outputs go to code/integration/intermediates/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
ADAPTERS_DIR="$INT_DIR/adapters"
WRAPPERS_DIR="$INT_DIR/wrappers"
OUT_DIR="$INT_DIR/intermediates"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Phase 1: Incytr Integration Proof of Concept"
echo "  Sender:    Microglia-PVM"
echo "  Receiver:  L5 IT"
echo "  Contrast:  WT vs App, 4mo, males only"
echo "  Output:    $OUT_DIR/"
echo "============================================================"
echo

# ---------------------------------------------------------------
# Python adapters (alzheimers conda env)
# ---------------------------------------------------------------
run_adapter() {
    local name="$1"
    echo "[$name]"
    micromamba run -n alzheimers python3 "$ADAPTERS_DIR/$name.py"
    echo
}

echo "=== Python Adapters ==="
run_adapter "export_expression"
run_adapter "export_kldata"
run_adapter "export_kl_output"
run_adapter "export_phospho"

# ---------------------------------------------------------------
# R wrappers (incytr conda env)
# ---------------------------------------------------------------
echo "=== R Pipeline ==="
echo "[run_incytr.R]"
micromamba run -n incytr Rscript "$WRAPPERS_DIR/run_incytr.R"
echo

echo "[postprocess.R]"
micromamba run -n incytr Rscript "$WRAPPERS_DIR/postprocess.R"
echo

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "============================================================"
echo "Phase 1 complete."
echo
echo "Key outputs:"
echo "  Expression-only:  $OUT_DIR/results_expronly.csv"
echo "  Full integration: $OUT_DIR/results_full.csv"
echo "  Concordant:       $OUT_DIR/results_concordant.csv"
echo "  Discordant A:     $OUT_DIR/results_discordant_A.csv"
echo "  Discordant B:     $OUT_DIR/results_discordant_B.csv"
echo "  Sensitivity:      $OUT_DIR/sensitivity_report.csv"
echo "  Ranking corr:     $OUT_DIR/ranking_correlation.json"
echo "============================================================"
