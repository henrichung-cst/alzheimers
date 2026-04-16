#!/bin/bash
# Factorial all-pairs Incytr pipeline: per-animal expression, OLS contrast
# estimation for 9 genotype x timepoint contrasts across 462 sender-receiver
# pairs.
#
# Runs Python adapters (alzheimers env) then R factorial pipeline (incytr env).
# The R pipeline runs under systemd-run with a 12GB memory cap.
#
# Environment variables (forwarded to R):
#   PAIR_FILTER       - Filter pairs, e.g. "Microglia-PVM:L5 IT", "*:L5 IT"
#   FORCE_RERUN       - Set to 1 to ignore checkpoints
#   MEMORY_LIMIT_GB   - R memory abort threshold (default 10)
#   --skip-adapters   - Skip Python adapter step (use existing intermediates)
#
# Usage:
#   bash code/integration/run_factorial_all_pairs.sh
#   bash code/integration/run_factorial_all_pairs.sh --skip-adapters
#   PAIR_FILTER="Microglia-PVM:L5 IT" bash code/integration/run_factorial_all_pairs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
ADAPTERS_DIR="$INT_DIR/adapters"
WRAPPERS_DIR="$INT_DIR/wrappers"
FAC_DIR="$INT_DIR/intermediates/factorial"

SKIP_ADAPTERS=0
for arg in "$@"; do
  case "$arg" in
    --skip-adapters) SKIP_ADAPTERS=1 ;;
  esac
done

mkdir -p "$FAC_DIR"

echo "============================================================"
echo "Factorial All-Pairs Incytr Integration Pipeline"
echo "  Cell types: 22 x 21 = 462 sender-receiver pairs"
echo "  Design:     10-parameter factorial OLS (males only)"
echo "  Contrasts:  9 (App/Tau/ApTt x 2mo/4mo/6mo)"
echo "  Output:     $FAC_DIR/all_pairs/"
if [ -n "${PAIR_FILTER:-}" ]; then
  echo "  Filter:     $PAIR_FILTER"
fi
echo "============================================================"
echo

# ---------------------------------------------------------------
# Python adapters (alzheimers conda env)
# ---------------------------------------------------------------
if [ "$SKIP_ADAPTERS" = "1" ]; then
  echo "=== Skipping Python Adapters (--skip-adapters) ==="
else
  run_adapter() {
      local name="$1"
      shift
      echo "[$name]"
      micromamba run -n alzheimers python3 "$ADAPTERS_DIR/$name.py" "$@"
      echo
  }

  echo "=== Python Adapters ==="
  run_adapter "export_expression_factorial"
  run_adapter "export_kldata"
  run_adapter "export_kl_output_factorial"
fi

# ---------------------------------------------------------------
# R factorial pipeline (incytr conda env)
# ---------------------------------------------------------------
echo "=== R Factorial All-Pairs Pipeline ==="
echo "[run_incytr_factorial_all_pairs.R]"

# Build env var forwarding for systemd-run
ENV_ARGS=()
for var in PAIR_FILTER FORCE_RERUN MEMORY_LIMIT_GB EXPR_DETECTION_THRESHOLD; do
  if [ -n "${!var:-}" ]; then
    ENV_ARGS+=(-E "$var=${!var}")
  fi
done

systemd-run --user --scope -p MemoryMax=12G \
  micromamba run -n incytr "${ENV_ARGS[@]}" \
  Rscript "$WRAPPERS_DIR/run_incytr_factorial_all_pairs.R"

echo
echo "============================================================"
echo "Factorial all-pairs pipeline complete."
echo
echo "Key outputs:"
echo "  Receiver Parquet: $FAC_DIR/all_pairs/recv_{receiver}.parquet"
echo "  Pair summary:     $FAC_DIR/all_pairs/pair_summary.csv"
echo
echo "Deferred to PR 2:"
echo "  - Kinase support scoring"
echo "  - Cross-pair aggregation"
echo "  - Backbone-level permutation tests"
echo "============================================================"
