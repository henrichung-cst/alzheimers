#!/usr/bin/env bash
# Sprint 2 D3: Promote verify_phase2.R from manual ritual to one-command runner.
#
# Wraps code/integration/wrappers/verify_phase2.R so that any Phase 1 / Phase 2
# regression check is a single bash invocation rather than a hand-typed Rscript
# call.
#
# Behavior:
#   - If Phase 1 per-pair CSVs and Phase 2 receiver Parquets both exist under
#     code/integration/intermediates/all_pairs/, run the comparison at the
#     plan-mandated tolerance (1e-10) and exit non-zero on mismatch.
#   - If neither side exists, exit 0 with a "no fixtures, skipped" message.
#     ALZ-15's vectorized receiver scoring is exercised whenever the all-pairs
#     pipeline is run; the auditor's claim is "this comparison passes when
#     fixtures exist," not "fixtures must exist for Sprint 2 sign-off."
#   - The committed Sprint-1 testthat suite + run_degenerate_2cond.sh already
#     anchor the package-level slot equivalence; this runner is the
#     wrapper-side complement for the receiver_scoring rewrite.
#
# Env overrides honored by verify_phase2.R: PHASE1_DIR, PHASE2_DIR
# CLI args passed through: --receivers a,b,c   --tol 1e-10
#
# Usage:
#   bash code/integration/tests/run_verify_phase2.sh
#   bash code/integration/tests/run_verify_phase2.sh --receivers VLMC,Astrocyte
#   bash code/integration/tests/run_verify_phase2.sh --tol 1e-8

set -euo pipefail

ALZHEIMERS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
WRAPPER="$ALZHEIMERS_DIR/code/integration/wrappers/verify_phase2.R"
INT_DIR="$ALZHEIMERS_DIR/code/integration/intermediates"

PHASE1_DIR="${PHASE1_DIR:-$INT_DIR/all_pairs}"
PHASE2_DIR="${PHASE2_DIR:-$PHASE1_DIR}"

echo "=== Sprint 2 D3: verify_phase2 (vectorized receiver scoring vs Phase 1 CSVs) ==="
echo "  PHASE1_DIR: $PHASE1_DIR"
echo "  PHASE2_DIR: $PHASE2_DIR"

have_phase2=0
if [[ -d "$PHASE2_DIR" ]] && \
   compgen -G "$PHASE2_DIR/recv_*.parquet" >/dev/null; then
  have_phase2=1
fi

have_phase1=0
if [[ -d "$PHASE1_DIR" ]] && \
   compgen -G "$PHASE1_DIR/*__*/results_full.csv" >/dev/null; then
  have_phase1=1
fi

if [[ "$have_phase1" -eq 0 || "$have_phase2" -eq 0 ]]; then
  echo
  echo "SKIP: Phase 1 and/or Phase 2 fixtures not present. Run the all-pairs"
  echo "      pipeline (bash code/integration/run_factorial_all_pairs.sh) to"
  echo "      produce them, then re-invoke this runner."
  echo "      have_phase1=$have_phase1 have_phase2=$have_phase2"
  exit 0
fi

if command -v pixi >/dev/null 2>&1 && [[ -f "$ALZHEIMERS_DIR/pixi.toml" ]]; then
  eval "$(cd "$ALZHEIMERS_DIR" && pixi shell-hook 2>/dev/null)"
fi

export PHASE1_DIR PHASE2_DIR

# Default tolerance: 1e-10 per audit-plan §5 Sprint 2.
TOL_DEFAULT="1e-10"
EXTRA_ARGS=("$@")
if ! printf '%s\n' "${EXTRA_ARGS[@]:-}" | grep -q -- '--tol'; then
  EXTRA_ARGS+=("--tol" "$TOL_DEFAULT")
fi

echo "  tolerance:  ${EXTRA_ARGS[*]}"
echo

Rscript "$WRAPPER" "${EXTRA_ARGS[@]}"

echo
echo "=== D3 PASS: vectorized receiver scoring matches Phase 1 within 1e-10. ==="
