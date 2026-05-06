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
#   ENABLE_CELLTYPE_MAPPING - Set to 1 to remap WMB classes to SEA-AD subclasses
#                             in multiomics evidence (default off; native-equivalent)
#
# INCYTR_* layer knobs (sourced from code/integration/incytr_runtime.sh; see
# docs/integrations/incytr_layer_inventory.md):
#   INCYTR_LAYER_KINASE_PACK   - 1 to enable kinase-imputed gene expansion +
#                                kinase support score sidecar (default 0).
#   INCYTR_LAYER_BACKBONE_PERMS - 1 to enable backbone permutation runner
#                                (default 0; pending design revisit).
#   INCYTR_CUTOFF_SIGPROB      - DuckDB pre-prune SigProb cutoff
#                                (default 0.0 = native-equivalent).
#
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

# shellcheck source=incytr_runtime.sh
source "$INT_DIR/incytr_runtime.sh"

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
      pixi run python3 "$ADAPTERS_DIR/$name.py" "$@"
      echo
  }

  echo "=== Python Adapters ==="
  run_adapter "export_expression_factorial"
  run_adapter "export_kldata"
  run_adapter "export_kl_output_factorial"

  # Kinase-imputed gene expansion (per-contrast, per-receiver) — kinase pack.
  if [ "$INCYTR_LAYER_KINASE_PACK" = "1" ]; then
    run_adapter "export_kinase_imputed_genes_factorial"
  fi
fi

# ---------------------------------------------------------------
# R factorial pipeline (incytr conda env)
# ---------------------------------------------------------------
echo "=== R Factorial All-Pairs Pipeline ==="
echo "[run_incytr_factorial_all_pairs.R]"

# Build env var forwarding for systemd-run
ENV_ARGS=()
for var in PAIR_FILTER FORCE_RERUN MEMORY_LIMIT_GB EXPR_DETECTION_THRESHOLD EXPR_IMPUTATION_FLOOR ENABLE_CELLTYPE_MAPPING "${INCYTR_FORWARDED_VARS[@]}"; do
  if [ -n "${!var:-}" ]; then
    ENV_ARGS+=(--setenv="$var=${!var}")
  fi
done

systemd-run --user --scope -p MemoryMax="${MEMORY_LIMIT_GB:-12}G" "${ENV_ARGS[@]}" \
  pixi run \
  Rscript "$WRAPPERS_DIR/run_incytr_factorial_all_pairs.R"

echo
echo "============================================================"
echo "Factorial all-pairs pipeline complete."
echo
echo "Key outputs:"
echo "  Receiver Parquet: $FAC_DIR/all_pairs/recv_{receiver}.parquet"
echo "  Pair summary:     $FAC_DIR/all_pairs/pair_summary.csv"
echo
echo "Factorial mode components:"
echo "  - Cross-pair aggregation (aggregate_factorial.py) — production"
echo "  - Kinase-imputed gene expansion — INCYTR_LAYER_KINASE_PACK sidecar"
echo "  - Kinase support scoring — sidecar/kinase_pack/compute_kinase_support_factorial.py"
echo "  - Backbone-level permutation tests — sidecar/backbone_perms/run_factorial_permutations.sh"
echo "    (set INCYTR_LAYER_BACKBONE_PERMS=1)"
echo "============================================================"
