#!/usr/bin/env bash
# 5xFAD pair-mode backbone overnight runner — PTM-inclusive (pr,ps,py,Ack,KGG).
#
# Run before bed (from repo root in a tmux session):
#   bash alz/incytr_pair/regeneration/run_backbone_overnight_5xfad.sh
#
# Log: outputs/reports/incytr_pair_mode_5xfad/overnight_<timestamp>.log
#
# Sequence:
#   [1/3] pair-mode: 8 contrasts (TG vs WT × 4 ages × 2 tissues),
#          channels=pr,ps,py,Ack,KGG; significance filter applied inline.
#          Writes: …/<tissue>/wide/ (paths) + …/<tissue>/backbone/ (grain shards),
#          both PTM-inclusive.
#   [2/3] kinase-incytr bridge — cortex: counts #Backbones/#Paths per kinase.
#   [3/3] kinase-incytr bridge — hippocampus.
#
# Phase 2 (viewer build) is OUT OF SCOPE — do not add it here.
#
# Memory caps (systemd-run cgroup v2 hard limit):
#   MEM_PAIR (default 24G): R pair-mode peaks ~13-15 GB at NPAIR_WORKERS=1
#   MEM_PY   (default 16G): DuckDB filter + bridge, lighter footprint
# Override either via env before calling this script.
#
# 5xFAD tuning (env before calling this script):
#   NPAIR_WORKERS=1    pair parallelism (default; 1 safest on shared box)
#   N_CHUNK_MULT=8     pairs per chunk (default; smaller than Song)
#   NPERM_WORKERS=1    permutation parallelism (default)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Locate pixi — handles tmux sessions opened outside the project directory.
PIXI="$(command -v pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"
if [[ ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found at $PIXI — activate pixi env or set PIXI=" >&2
    exit 1
fi

LOG_DIR="outputs/reports/incytr_pair_mode_5xfad"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_${TS}.log"

# Redirect all subsequent output (stdout + stderr) through tee so the terminal
# and the log file both receive everything.
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) 5xFAD overnight run start ==="
echo "  REPO:     $REPO_ROOT"
echo "  Log:      $LOG"
echo "  pixi:     $PIXI"
echo "  MEM_PAIR=${MEM_PAIR:-24G}  MEM_PY=${MEM_PY:-16G}"
echo "  NPAIR_WORKERS=${NPAIR_WORKERS:-1}"

MEM_PAIR="${MEM_PAIR:-24G}"
MEM_PY="${MEM_PY:-16G}"

# ---------------------------------------------------------------------------
# [1/3] Full pair-mode run (8 contrasts × 2 tissues, PTM-inclusive) + filter
# ---------------------------------------------------------------------------
# run_pair_mode_5xfad.sh handles contrast-level resumability (a contrast whose
# final parquet exists is skipped) and applies the canonical SigProb/PDS gate
# inline. Backbone grain shards are written during scoring, independent of the
# wide/ path filter.
echo ""
echo "=== $(date -Is) [1/3] pair-mode (8 contrasts × 2 tissues, channels=pr,ps,py,Ack,KGG) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PAIR" -p MemorySwapMax=0 \
    --unit "incytr5-pair-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    bash "$REPO_ROOT/alz/incytr_pair/run_pair_mode_5xfad.sh"
echo "=== $(date -Is) [1/3] done ==="

# ---------------------------------------------------------------------------
# [2/3] Kinase-Incytr bridge — cortex, then hippocampus
# ---------------------------------------------------------------------------
if [[ "${SKIP_BRIDGE:-no}" == "yes" ]]; then
    echo ""
    echo "=== $(date -Is) [2/3] kinase-incytr bridge SKIPPED (SKIP_BRIDGE=yes) ==="
    echo "Run bridge later with:"
    echo "  $PIXI run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex"
    echo "  $PIXI run kinase-incytr-bridge -- --cohort fivexfad --tissue hippocampus"
    echo ""
    echo "=== $(date -Is) 5xFAD OVERNIGHT RUN COMPLETE (bridge deferred) ==="
    echo "  Wide:      $REPO_ROOT/$LOG_DIR/cortex/wide/"
    echo "  Wide:      $REPO_ROOT/$LOG_DIR/hippocampus/wide/"
    echo "  Log:       $LOG"
    exit 0
fi

echo ""
echo "=== $(date -Is) [2/3] kinase-incytr bridge — cortex ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PY" -p MemorySwapMax=0 \
    --unit "incytr5-bridge-cortex-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    "$PIXI" run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex
echo "=== $(date -Is) [2/3] bridge cortex done ==="

echo ""
echo "=== $(date -Is) [3/3] kinase-incytr bridge — hippocampus ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PY" -p MemorySwapMax=0 \
    --unit "incytr5-bridge-hippo-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    "$PIXI" run kinase-incytr-bridge -- --cohort fivexfad --tissue hippocampus
echo "=== $(date -Is) [3/3] bridge hippocampus done ==="

echo ""
echo "=== $(date -Is) 5xFAD OVERNIGHT RUN COMPLETE ==="
echo "  Wide:      $REPO_ROOT/$LOG_DIR/cortex/wide/"
echo "  Wide:      $REPO_ROOT/$LOG_DIR/hippocampus/wide/"
echo "  Log:       $LOG"
