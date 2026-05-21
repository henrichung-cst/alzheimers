#!/usr/bin/env bash
# Phase 0 of viewer_input_audit_remediation: re-run the mouse kinase pipeline
# in dependency order to clear stale enrich/mechanism/attribute/recover
# outputs left behind older than stoichiometry_matrix.csv.
#
# Run this BEFORE rebuilding the viewer. Does not touch decomposition,
# Incytr, or the human chain — those have their own runners.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT=outputs/reports/kinase_attribution
T0=$(date +%s)

echo "=== $(date -Is) [1/5] normalize (IRS + stoichiometry, all 72 samples) ==="
pixi run normalize

echo "=== $(date -Is) [2/5] enrich (OLS + MEA, males-only) ==="
pixi run enrich

echo "=== $(date -Is) [3/5] mechanism (raw phospho MEA + classification) ==="
pixi run python alz/kinase_mechanism.py

echo "=== $(date -Is) [4/5] attribute (unified cell-type attribution) ==="
pixi run attribute

echo "=== $(date -Is) [5/5] recover (hypothesis tables + bubble plots) ==="
pixi run recover

T1=$(date +%s)
echo
echo "=== done in $((T1 - T0))s ==="
echo "verifying mtime ordering:"
ls -la --time-style=long-iso \
  "$OUT/stoichiometry_matrix.csv" \
  "$OUT/site_level_ols.csv" \
  "$OUT/mea_stoichiometry.csv" \
  "$OUT/unified_attribution.csv" \
  outputs/reports/attribution_recovery/kinase_hypothesis_table.csv

stoich=$(stat -c %Y "$OUT/stoichiometry_matrix.csv")
ua=$(stat -c %Y "$OUT/unified_attribution.csv")
if [[ $ua -lt $stoich ]]; then
  echo "ERROR: unified_attribution.csv is older than stoichiometry_matrix.csv" >&2
  exit 1
fi
echo "mtime ordering OK"
