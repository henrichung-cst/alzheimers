#!/bin/bash
# All-pairs Incytr pipeline: enumerate and run downstream for all 462
# sender-receiver pairs.
#
# Runs Python adapters (alzheimers env) then R all-pairs pipeline (incytr env).
# The R pipeline runs under systemd-run with a 12GB memory cap.
#
# Environment variables (forwarded to R):
#   PAIR_FILTER       - Filter pairs, e.g. "Microglia-PVM:L5 IT", "*:L5 IT"
#   FORCE_RERUN       - Set to 1 to ignore checkpoints
#   MEMORY_LIMIT_GB   - R memory abort threshold (default 10)
#   SKIP_EXPRONLY      - Set to 1 to skip expression-only evaluation
#   --skip-adapters    - Skip Python adapter step (use existing intermediates)
#
# Usage:
#   bash code/integration/run_all_pairs.sh
#   bash code/integration/run_all_pairs.sh --skip-adapters
#   PAIR_FILTER="Microglia-PVM:L5 IT" bash code/integration/run_all_pairs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
ADAPTERS_DIR="$INT_DIR/adapters"
WRAPPERS_DIR="$INT_DIR/wrappers"
OUT_DIR="$INT_DIR/intermediates"

SKIP_ADAPTERS=0
for arg in "$@"; do
  case "$arg" in
    --skip-adapters) SKIP_ADAPTERS=1 ;;
  esac
done

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "All-Pairs Incytr Integration Pipeline"
echo "  Cell types: 22 x 21 = 462 sender-receiver pairs"
echo "  Contrast:   WT vs App, 4mo, males only"
echo "  Output:     $OUT_DIR/all_pairs/"
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
  run_adapter "export_expression"
  run_adapter "export_kldata"
  run_adapter "export_kl_output" --all-pairs
  run_adapter "export_phospho"

  # Kinase-imputed gene expansion
  if [ "${ENABLE_KINASE_IMPUTATION:-1}" = "1" ]; then
    run_adapter "export_kinase_imputed_genes"
  fi
fi

# ---------------------------------------------------------------
# R all-pairs pipeline (incytr conda env)
# ---------------------------------------------------------------
echo "=== R All-Pairs Pipeline ==="
echo "[run_incytr_all_pairs.R]"

# Build env var forwarding for systemd-run (use array for safe word splitting)
ENV_ARGS=()
for var in PAIR_FILTER FORCE_RERUN MEMORY_LIMIT_GB SKIP_EXPRONLY EXPR_DETECTION_THRESHOLD; do
  if [ -n "${!var:-}" ]; then
    ENV_ARGS+=(-E "$var=${!var}")
  fi
done

systemd-run --user --scope -p MemoryMax=12G \
  micromamba run -n incytr "${ENV_ARGS[@]}" \
  Rscript "$WRAPPERS_DIR/run_incytr_all_pairs.R"

# ---------------------------------------------------------------
# Kinase support scoring (alzheimers conda env)
# ---------------------------------------------------------------
echo "=== Kinase Support Scoring (all pairs) ==="
echo "[compute_kinase_support_all_pairs.py]"
KSCORE_ARGS=()
if [ -n "${PAIR_FILTER:-}" ]; then
  # Convert colon-separated "Sender:Receiver" to glob "Sender__Receiver"
  KSCORE_ARGS+=(--pair-filter "$(echo "$PAIR_FILTER" | sed 's/ /_/g; s/\//-/g; s/:/__/')")
fi
if [ "${FORCE_RERUN:-}" = "1" ]; then
  KSCORE_ARGS+=(--force)
fi
micromamba run -n alzheimers python3 \
  "$ADAPTERS_DIR/compute_kinase_support_all_pairs.py" "${KSCORE_ARGS[@]}"

echo
echo "============================================================"
echo "All-pairs pipeline complete."
echo
echo "Key outputs:"
echo "  Per-pair results: $OUT_DIR/all_pairs/{sender}__{receiver}/results_full.csv"
echo "  Kinase support:   $OUT_DIR/all_pairs/{sender}__{receiver}/kinase_support_scores.csv"
echo "  Pair summary:     $OUT_DIR/all_pairs/pair_summary.csv"
echo "  Kinase summary:   $OUT_DIR/all_pairs/kinase_support_summary.csv"
echo "============================================================"
