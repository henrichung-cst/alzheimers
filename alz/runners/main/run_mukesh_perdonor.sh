#!/usr/bin/env bash
# Per-donor MEA for the Mukesh / NBB human cohort.
#
# Runs two preprocessing tracks per residue class:
#   - stoichiometry (log2 phospho − log2 protein) — primary
#   - raw phospho (uncorrected, normalized intensity) — sensitivity check
#
# Mirrors the mouse `kinase_enrich + kinase_mechanism` pair so the unified
# viewer can render the same stoich-vs-raw cross-check per donor.
#
# Outputs under outputs/reports/kinase_attribution_human/perdonor/:
#   mea_perdonor{,_raw}{,_pY}.csv
#   kinase_donor_nes{,_raw}{,_pY}.csv
#   kinase_donor_fdr{,_raw}{,_pY}.csv
#   recurrence{,_raw}{,_pY}.csv
#   mea_global_shift{,_raw}{,_pY}.csv
#   winsorized_sites{,_raw}{,_pY}.csv

set -euo pipefail

cd "$(dirname "$0")/../../.."

LOG_DIR="outputs/reports/kinase_attribution_human/perdonor"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_mukesh_perdonor.log"

echo "[run_mukesh_perdonor] $(date -Is)" | tee "$LOG"
echo "[run_mukesh_perdonor] running python alz/cohorts/mukesh/mea.py --track both" | tee -a "$LOG"

pixi run python alz/cohorts/mukesh/mea.py --track both 2>&1 | tee -a "$LOG"

echo "[run_mukesh_perdonor] running python alz/cross_reference/seaad_human_agreement.py" | tee -a "$LOG"
pixi run python alz/cross_reference/seaad_human_agreement.py 2>&1 | tee -a "$LOG"

echo "[run_mukesh_perdonor] done $(date -Is)" | tee -a "$LOG"
