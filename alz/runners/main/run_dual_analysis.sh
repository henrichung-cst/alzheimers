#!/usr/bin/env bash
# Dual-track analysis: males-only (primary) + full-cohort (sensitivity)
#
# Runs the kinase attribution and recovery pipeline twice:
#   1. Males-only (after outlier exclusion) — primary analysis
#   2. Full cohort (after outlier exclusion, both sexes) — sensitivity analysis
#
# Prerequisites:
#   - data_ingest.py --run (including --outliers for sample_exclusions.csv)
#   - kinase_normalize.py (IRS normalization, uses all 72 samples)
#   - wmb_expression.py --run (WMB expression matrix)
#
# Outputs are archived to mode-specific directories after each track.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/hchung/.local/share/mamba/envs/alzheimers/bin/python

# Ensure normalization has been run (uses all 72 samples, mode-independent)
if [[ ! -f outputs/reports/kinase_attribution/stoichiometry_matrix.csv ]]; then
    echo "--- Normalization (all 72 samples) ---"
    $PYTHON alz/kinase_normalize.py
fi

# Ensure outlier detection has been run
if [[ ! -f outputs/reports/data_ingest/sample_exclusions.csv ]]; then
    echo "--- Outlier detection ---"
    $PYTHON alz/data_ingest.py --outliers
fi

# ── Track 1: Males-only (primary) ──────────────────────────────────────────
# Cohort defaults to males_only via conf/base/parameters.yml.
echo ""
echo "=========================================="
echo "  PRIMARY ANALYSIS: males-only"
echo "=========================================="

$PYTHON alz/kinase_enrich.py
$PYTHON alz/kinase_attribute.py
$PYTHON alz/kinase_mechanism.py
$PYTHON alz/attribution_recovery.py --run

# Archive primary outputs
rm -rf outputs/reports/kinase_attribution_males_only
rm -rf outputs/reports/attribution_recovery_males_only
cp -r outputs/reports/kinase_attribution outputs/reports/kinase_attribution_males_only
cp -r outputs/reports/attribution_recovery outputs/reports/attribution_recovery_males_only
echo "  Archived to outputs/reports/*_males_only/"

# ── Track 2: Full cohort (sensitivity) ────────────────────────────────────
# KEDRO_ENV=full_cohort overlays conf/full_cohort/parameters.yml.
echo ""
echo "=========================================="
echo "  SENSITIVITY ANALYSIS: full cohort"
echo "=========================================="

export KEDRO_ENV=full_cohort
$PYTHON alz/kinase_enrich.py
$PYTHON alz/kinase_attribute.py
$PYTHON alz/kinase_mechanism.py
$PYTHON alz/attribution_recovery.py --run
unset KEDRO_ENV

# Archive sensitivity outputs
rm -rf outputs/reports/kinase_attribution_full_cohort
rm -rf outputs/reports/attribution_recovery_full_cohort
cp -r outputs/reports/kinase_attribution outputs/reports/kinase_attribution_full_cohort
cp -r outputs/reports/attribution_recovery outputs/reports/attribution_recovery_full_cohort
echo "  Archived to outputs/reports/*_full_cohort/"

echo ""
echo "=== Dual analysis complete at $(date) ==="
