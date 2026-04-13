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

# Kinase-imputed gene expansion (depends on kldata + expression_genes)
if [ "${ENABLE_KINASE_IMPUTATION:-1}" = "1" ]; then
  run_adapter "export_kinase_imputed_genes"
fi

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
# Substrate-based kinase support (always runs, fast)
# ---------------------------------------------------------------
echo "=== Kinase Support Scoring ==="
echo "[compute_kinase_support.py]"
micromamba run -n alzheimers python3 "$ADAPTERS_DIR/compute_kinase_support.py"
echo

# Permutation tests (optional, slow)
if [ "${RUN_PERMUTATIONS:-0}" = "1" ]; then
  echo "[compute_kinase_support.py --permutations]"
  micromamba run -n alzheimers python3 "$ADAPTERS_DIR/compute_kinase_support.py" \
    --permutations
  echo
fi

# Bootstrap sensitivity (optional, slow)
if [ "${RUN_BOOTSTRAP:-0}" = "1" ]; then
  echo "=== Bootstrap Sensitivity ==="
  echo "[bootstrap_sensitivity.R]"
  micromamba run -n incytr Rscript "$WRAPPERS_DIR/bootstrap_sensitivity.R"
  echo
fi

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
echo "  Kinase support:   $OUT_DIR/kinase_support_scores.csv"
echo "  Adjusted ranks:   $OUT_DIR/adjusted_rankings.csv"
echo "  Reranking summary:$OUT_DIR/reranking_summary.json"
echo "============================================================"
