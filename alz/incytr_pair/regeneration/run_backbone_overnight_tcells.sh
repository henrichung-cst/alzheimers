#!/usr/bin/env bash
# T-cell state-MEA + pair-mode overnight runner, memory-capped via systemd-run.
#
# Run before bed (from repo root in a tmux session):
#   bash alz/incytr_pair/regeneration/run_backbone_overnight_tcells.sh
#
# Log: outputs/reports/incytr_pair_mode_tcells_percell_posneg/overnight_<timestamp>.log
#
# Sequence:
#   [1/2] donor1 projected-state kinase MEA on decomposed ST + pY inputs.
#   [2/2] pair-mode: donor2 ({d5,d7,d9,d11} vs d2) then donor1 ({d13,d17,d20}
#          vs d2). Resumable at contrast level (skips finished parquets).
#          Significance filter (SigProb>0.1 AND |PDS|>=0.2, uncapped) runs
#          per-donor inside run_pair_mode_tcells.sh immediately after each donor.
# No viewer build — Phase 2 deferred.
#
# Memory caps (systemd-run cgroup v2 hard limit):
#   MEM_MEA  (default 16G): Python projected-state kinase MEA
#   MEM_PAIR (default 24G): R pair-mode; donor1 (pr+py+ps, 3 contrasts) is heavier
# Override via env before calling this script.
#
# Tuning env (set before calling):
#   FULL_NBOOT=100    permutation count (set to 0 to skip permutations)
#   NPAIR_WORKERS=1   pair parallelism; 1 is safest on the shared box
#   N_CHUNK_MULT=8    pairs per subprocess chunk
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Locate pixi — handles tmux sessions opened outside the project directory
# where direnv has not activated the env.
PIXI="$(command -v pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"
if [[ ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found at $PIXI — activate pixi env or set PIXI=" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_DIR_OVERRIDE:-outputs/reports/incytr_pair_mode_tcells_percell_posneg}"
LOG_DIR="$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_${TS}.log"

# Redirect all subsequent output (stdout + stderr) through tee so the terminal
# and the log file both receive everything.
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) overnight t-cell run start ==="
echo "  REPO:    $REPO_ROOT"
echo "  Log:     $LOG"
echo "  pixi:    $PIXI"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  MEM_MEA=${MEM_MEA:-16G}  MEM_PAIR=${MEM_PAIR:-24G}"
echo "  FULL_NBOOT=${FULL_NBOOT:-100}  NPAIR_WORKERS=${NPAIR_WORKERS:-1}"

MEM_PAIR="${MEM_PAIR:-24G}"
MEM_MEA="${MEM_MEA:-16G}"

# ---------------------------------------------------------------------------
# [1/2] Donor1 projected-state kinase MEA (ST + pY)
# ---------------------------------------------------------------------------
echo ""
echo "=== $(date -Is) [1/2] donor1 projected-state kinase MEA (st + py) ==="
echo "=== $(date -Is) [1/2] kinase preflight start ==="
"$PIXI" run python -m alz.cohorts.tcells.state_mea \
    --donor donor1 --track both --dry-run
echo "=== $(date -Is) [1/2] kinase preflight done ==="
echo "=== $(date -Is) [1/2] kinase MEA start ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_MEA" -p MemorySwapMax=0 \
    --unit "state-mea-tcells-${TS}" \
    "$PIXI" run tcell-state-mea
echo "=== $(date -Is) [1/2] kinase MEA done ==="
echo "=== $(date -Is) [1/2] done ==="

# ---------------------------------------------------------------------------
# [2/2] Full T-cell pair-mode run (donor2 then donor1, all contrasts vs d2)
# ---------------------------------------------------------------------------
# run_pair_mode_tcells.sh handles contrast-level resumability: a contrast
# whose final parquet already exists is skipped.  Significance filtering
# (SigProb>0.1 AND |PDS|>=0.2, uncapped, no p_adj arm) runs per-donor
# inside that script after all its contrasts complete.
# Set FORCE_RERUN=1 inside run_pair_mode_tcells.sh to disable skipping.
echo ""
echo "=== $(date -Is) [2/2] pair-mode (donor2: 4 contrasts; donor1: 3 contrasts; nboot=${FULL_NBOOT:-100}) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PAIR" -p MemorySwapMax=0 \
    --unit "incytr-pair-tcells-${TS}" \
    env CONDA_OVERRIDE_CUDA="" OUTPUT_DIR_OVERRIDE="$OUTPUT_ROOT" \
        FULL_NBOOT="${FULL_NBOOT:-100}" \
    bash "$REPO_ROOT/alz/incytr_pair/run_pair_mode_tcells.sh"
echo "=== $(date -Is) [2/2] done ==="

echo ""
echo "=== $(date -Is) T-CELL OVERNIGHT RUN COMPLETE ==="
echo "  state MEA:      $REPO_ROOT/outputs/reports/kinase_attribution_tcells/donor1/state_mea/"
echo "  donor1 wide/:   $REPO_ROOT/$OUTPUT_ROOT/donor1/wide/"
echo "  donor2 wide/:   $REPO_ROOT/$OUTPUT_ROOT/donor2/wide/"
echo "  Log:           $LOG"
