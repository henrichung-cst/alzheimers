#!/usr/bin/env bash
# Full AD pair-mode → payload → viewer build, chained and memory-capped.
#
# Run before bed (from repo root in a tmux session):
#   bash alz/incytr_pair/run_backbone_overnight.sh
#
# On completion: outputs/reports/unified_viewer/index.html is the full viewer.
# Log: outputs/reports/incytr_pair_mode/overnight_<timestamp>.log
#
# Sequence:
#   [1/4] pair-mode: 9 contrasts × 961 pairs, backbone-emitting via
#          Cal_pairwise_grid. Resumable at contrast level (skips finished contrasts).
#          Writes wide/ + backbone/ outputs.
#   [2/4] viewer build #1: full rebuild from new wide/ + backbone/ parquets.
#          B-3 heatmap tensors, B-4 entity payload (inline + sharded), B-5 score
#          col gating — all inside pixi run viewer.
#          Also emits edge_slices/incytr_pathways/gene_node_index.json.gz
#          which the bridge (step 3) requires.
#   [3/4] kinase-incytr bridge (B4): reads gene_node_index.json.gz + wide/ shards.
#          Emits kinase_participation.csv (#Backbones, #Paths per kinase).
#   [4/4] viewer build #2: cache-hit on incytr_pathways + backbone grains (fast);
#          rebuilds kinase section to incorporate kinase_participation.csv.
#
# Seamlessness proof: steps [2/4] and [4/4] invoke `pixi run viewer`
# (= python alz/build_unified_viewer.py), the SAME command used by the fixture
# (1-pair) run. No code path varies between fixture and full run; the only
# difference is the number of rows/pairs in wide/. The fixture is restricted
# by narrowing PAIR_SUBSET passed to the pair-mode driver (an input change);
# this script sets neither PAIR_SUBSET nor PAIR_LIMIT, so the driver runs the
# full 31×31 = 961-pair grid.
#
# Memory caps (systemd-run --user --scope cgroup v2 hard limit):
#   MEM_PAIR (default 24G): R pair-mode peaks ~13-15 GB at NPAIR_WORKERS=1
#   MEM_PY   (default 16G): Python/DuckDB viewer + bridge, lighter footprint
# Override either via env before calling this script.
#
# Tuning the pair-mode (env before calling this script):
#   FULL_NBOOT=100     permutation count (set to 0 to skip permutations)
#   NPAIR_WORKERS=1    pair parallelism; 1 is safest on the 30 GB shared box
#   N_CHUNK_MULT=48    pairs per subprocess chunk (~20 pairs, conservative RAM)
#   CHUNK_PARALLEL=2   chunk-level parallelism
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Locate pixi — handles the case where direnv has not been activated in the
# calling shell (tmux sessions opened outside the project directory).
PIXI="$(command -v pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"
if [[ ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found at $PIXI — activate pixi env or set PIXI=" >&2
    exit 1
fi

LOG_DIR="outputs/reports/incytr_pair_mode"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_${TS}.log"

# Redirect all subsequent output (stdout + stderr) through tee so the terminal
# and the log file both receive everything.
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) overnight run start ==="
echo "  REPO:    $REPO_ROOT"
echo "  Log:     $LOG"
echo "  pixi:    $PIXI"
echo "  MEM_PAIR=${MEM_PAIR:-24G}  MEM_PY=${MEM_PY:-16G}"
echo "  FULL_NBOOT=${FULL_NBOOT:-100}  NPAIR_WORKERS=${NPAIR_WORKERS:-1}"

MEM_PAIR="${MEM_PAIR:-24G}"
MEM_PY="${MEM_PY:-16G}"

# ---------------------------------------------------------------------------
# [1/4] Full pair-mode run (9 contrasts × 961 pairs, backbone-emitting)
# ---------------------------------------------------------------------------
# run_pair_mode.sh handles contrast-level resumability: a contrast whose
# final parquet already exists is skipped. Per-pair shards within an
# unfinished contrast are also reused. Set FORCE_RERUN=1 to disable skipping.
# Backbone emission is active by default (BACKBONE_OUT_DIR unset → canonical
# outputs/reports/incytr_pair_mode/backbone/<grain>/).
echo ""
echo "=== $(date -Is) [1/4] pair-mode (9 contrasts × 961 pairs, nboot=${FULL_NBOOT:-100}) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PAIR" -p MemorySwapMax=0 \
    --unit "incytr-pair-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    bash "$REPO_ROOT/alz/incytr_pair/run_pair_mode.sh"
echo "=== $(date -Is) [1/4] done ==="

# ---------------------------------------------------------------------------
# [2/4] Viewer build #1
# Full rebuild from new wide/ + backbone/ outputs:
#   B-3  heatmap count tensors per grain (R-EM, L-R-EM, R-EM-T)
#   B-4  entity payload: inline for R-EM/L-R-EM; sharded for R-EM-T
#   B-5  score-col gating: Ack/KGG/Rme1 skipped when all-zero (Song cohort)
# Also emits edge_slices/incytr_pathways/gene_node_index.json.gz, which
# the kinase-incytr bridge (step 3) requires as a prerequisite.
# Command is identical to the fixture (1-pair) viewer build.
# ---------------------------------------------------------------------------
echo ""
echo "=== $(date -Is) [2/4] viewer build #1 (backbone grains + pathways; generates gene_node_index) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PY" -p MemorySwapMax=0 \
    --unit "incytr-viewer1-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    "$PIXI" run viewer
echo "=== $(date -Is) [2/4] done ==="

# ---------------------------------------------------------------------------
# [3/4] Kinase-Incytr bridge (B4)
# Prerequisite: gene_node_index.json.gz from [2/4].
# Reads wide/ shards to count per-kinase backbone + path participation.
# Emits:
#   outputs/reports/kinase_incytr_bridge/song/kinase_participation.csv
# ---------------------------------------------------------------------------
echo ""
echo "=== $(date -Is) [3/4] kinase-incytr bridge (B4 — #Backbones/#Paths) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PY" -p MemorySwapMax=0 \
    --unit "incytr-bridge-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    "$PIXI" run kinase-incytr-bridge
echo "=== $(date -Is) [3/4] done ==="

# ---------------------------------------------------------------------------
# [4/4] Viewer build #2
# incytr_pathways block: cache hit (content unchanged since [2/4]) — fast.
# kinase section: rebuilds to incorporate kinase_participation.csv from [3/4],
# populating the #Backbones and #Paths columns in the kinase table.
# Command is identical to the fixture (1-pair) viewer build.
# ---------------------------------------------------------------------------
echo ""
echo "=== $(date -Is) [4/4] viewer build #2 (kinase section refresh with #Backbones/#Paths) ==="
systemd-run --user --scope \
    -p MemoryMax="$MEM_PY" -p MemorySwapMax=0 \
    --unit "incytr-viewer2-${TS}" \
    env CONDA_OVERRIDE_CUDA="" \
    "$PIXI" run viewer
echo "=== $(date -Is) [4/4] done ==="

echo ""
echo "=== $(date -Is) OVERNIGHT RUN COMPLETE ==="
echo "  Viewer:   $REPO_ROOT/outputs/reports/unified_viewer/index.html"
echo "  Wide/:    $REPO_ROOT/outputs/reports/incytr_pair_mode/wide/"
echo "  Backbone: $REPO_ROOT/outputs/reports/incytr_pair_mode/backbone/"
echo "  Log:      $LOG"
