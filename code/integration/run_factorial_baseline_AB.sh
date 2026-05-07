#!/bin/bash
# Pinned A+B baseline smoke test for the factorial Incytr pipeline.
#
# Runs ONE sender-receiver pair (default Microglia-PVM:L5 IT) so you can
# inspect the output schema, sanity-check column shape, and extrapolate the
# full-462-pair runtime before committing to the full job. To run all pairs,
# unset PAIR_FILTER explicitly: `PAIR_FILTER= bash <script>`.
#
# Configuration: original Incytr behavior + factorial extension (A) +
# performance refactors (B) + same-math PTM tracks (pY/pS as A under the
# 2026-05-06 tightened C definition). Every C-bucket layer is forced off:
#
#   INCYTR_LAYER_KINASE_PACK     = 0   (no kinase-imputed gene expansion)
#   INCYTR_LAYER_BACKBONE_PERMS  = 0   (no backbone permutation tests)
#   INCYTR_CUTOFF_SIGPROB        = 0.0 (DuckDB enumerator native-equivalent)
#   ENABLE_CELLTYPE_MAPPING      = 0   (identity WMB↔WMB mapping)
#   ENABLE_EM_PROMISCUITY_WEIGHT = 0   (no EM-degree weighting in SigProb)
#   EXPR_DETECTION_THRESHOLD     = 0.5 (native Find_highexp_gene 50%-percentile
#                                       parity, replacing the wrapper's
#                                       snRNA-tuned 10% rule — ALZ-22)
#
# Output is snapshotted to outputs/reports/incytr_baseline_AB/ so subsequent
# production runs do not overwrite the pinned baseline.
#
# Usage:
#   bash code/integration/run_factorial_baseline_AB.sh
#   PAIR_FILTER="Microglia-PVM:Endo NN" bash code/integration/run_factorial_baseline_AB.sh
#   PAIR_FILTER= bash code/integration/run_factorial_baseline_AB.sh   # full 462-pair run
#   FORCE_RERUN=1 bash code/integration/run_factorial_baseline_AB.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
FAC_DIR="$INT_DIR/intermediates/factorial"
SRC_DIR="$FAC_DIR/all_pairs"
PIN_DIR="$REPO_ROOT/outputs/reports/incytr_baseline_AB"
LOG_DIR="$PIN_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/run_${TS}.log"

mkdir -p "$LOG_DIR"

# Default to a single pair smoke test; user can unset PAIR_FILTER for full run.
: "${PAIR_FILTER:=Microglia-PVM:L5 IT}"
export PAIR_FILTER

# Total pairs in production scope (22 senders x 21 receivers, self-pairs excl).
TOTAL_PAIRS=462

echo "============================================================" | tee -a "$LOG"
echo "Pinned A+B baseline factorial Incytr run"                       | tee -a "$LOG"
echo "Started: $(date -Iseconds)"                                     | tee -a "$LOG"
echo "Log:     $LOG"                                                  | tee -a "$LOG"
echo "Pin:     $PIN_DIR"                                              | tee -a "$LOG"
if [ -n "$PAIR_FILTER" ]; then
  echo "Mode:    SMOKE TEST — single pair: $PAIR_FILTER"              | tee -a "$LOG"
else
  echo "Mode:    FULL RUN — all $TOTAL_PAIRS sender-receiver pairs"   | tee -a "$LOG"
fi
echo "============================================================"   | tee -a "$LOG"

export INCYTR_LAYER_KINASE_PACK=0
export INCYTR_LAYER_BACKBONE_PERMS=0
export INCYTR_CUTOFF_SIGPROB=0.0
export ENABLE_CELLTYPE_MAPPING=0
export ENABLE_EM_PROMISCUITY_WEIGHT=0
export EXPR_DETECTION_THRESHOLD=0.5
export ENABLE_MULTIOMICS_EVIDENCE=1

# ---------------------------------------------------------------
# Pass 1: full pipeline (adapters + 1 pair) on a clean slate.
# ---------------------------------------------------------------
echo                                                                  | tee -a "$LOG"
echo "=== Pass 1: adapters + 1 pair (cold) ==="                       | tee -a "$LOG"
T1_START=$(date +%s)
FORCE_RERUN=1 bash "$INT_DIR/run_factorial_all_pairs.sh" 2>&1 | tee -a "$LOG"
T1_END=$(date +%s)
PASS1_S=$((T1_END - T1_START))

# ---------------------------------------------------------------
# Pass 2: skip adapters, force receiver re-run on the same pair.
# Measures pure R per-pair cost in isolation.
# ---------------------------------------------------------------
if [ -n "$PAIR_FILTER" ]; then
  echo                                                                | tee -a "$LOG"
  echo "=== Pass 2: skip adapters, re-run same pair (R-only) ==="     | tee -a "$LOG"
  T2_START=$(date +%s)
  FORCE_RERUN=1 bash "$INT_DIR/run_factorial_all_pairs.sh" --skip-adapters 2>&1 | tee -a "$LOG"
  T2_END=$(date +%s)
  PASS2_S=$((T2_END - T2_START))

  ADAPTER_S=$((PASS1_S - PASS2_S))
  PER_PAIR_S=$PASS2_S
  HONEST_FULL_S=$((ADAPTER_S + PER_PAIR_S * TOTAL_PAIRS))
  HONEST_FULL_H=$(awk -v s="$HONEST_FULL_S" 'BEGIN { printf "%.2f", s/3600 }')
  NAIVE_FULL_S=$((PASS1_S * TOTAL_PAIRS))
  NAIVE_FULL_H=$(awk -v s="$NAIVE_FULL_S" 'BEGIN { printf "%.2f", s/3600 }')

  echo                                                                | tee -a "$LOG"
  echo "=== Timing decomposition ==="                                 | tee -a "$LOG"
  echo "Pass 1 (adapters + 1 pair): ${PASS1_S}s"                      | tee -a "$LOG"
  echo "Pass 2 (1 pair, no adapter): ${PASS2_S}s"                     | tee -a "$LOG"
  echo "Adapter cost (one-shot):     ${ADAPTER_S}s"                   | tee -a "$LOG"
  echo "Per-pair R cost:             ${PER_PAIR_S}s"                  | tee -a "$LOG"
  echo                                                                | tee -a "$LOG"
  echo "Naive  extrapolation (Pass1 x ${TOTAL_PAIRS}):  ${NAIVE_FULL_S}s  (${NAIVE_FULL_H} h)"  | tee -a "$LOG"
  echo "Honest extrapolation (adapter + ${TOTAL_PAIRS}*per-pair): ${HONEST_FULL_S}s (${HONEST_FULL_H} h)" | tee -a "$LOG"
else
  echo                                                                | tee -a "$LOG"
  echo "=== Timing ==="                                               | tee -a "$LOG"
  echo "Full run elapsed: ${PASS1_S}s"                                | tee -a "$LOG"
fi

echo                                                                  | tee -a "$LOG"
echo "=== Snapshotting pinned baseline ==="                            | tee -a "$LOG"
SNAP_DIR="$PIN_DIR/all_pairs_${TS}"
mkdir -p "$SNAP_DIR"
cp -a "$SRC_DIR"/. "$SNAP_DIR"/
ln -sfn "all_pairs_${TS}" "$PIN_DIR/all_pairs_latest"

cat > "$PIN_DIR/RUN_${TS}.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "label": "A+B baseline (pinned)",
  "config": {
    "INCYTR_LAYER_KINASE_PACK": "0",
    "INCYTR_LAYER_BACKBONE_PERMS": "0",
    "INCYTR_CUTOFF_SIGPROB": "0.0",
    "ENABLE_CELLTYPE_MAPPING": "0",
    "ENABLE_EM_PROMISCUITY_WEIGHT": "0",
    "EXPR_DETECTION_THRESHOLD": "0.5",
    "ENABLE_MULTIOMICS_EVIDENCE": "1"
  },
  "snapshot": "$SNAP_DIR",
  "log": "$LOG"
}
EOF

echo "Snapshot:  $SNAP_DIR"                                           | tee -a "$LOG"
echo "Manifest:  $PIN_DIR/RUN_${TS}.json"                             | tee -a "$LOG"
echo "Finished:  $(date -Iseconds)"                                   | tee -a "$LOG"
echo "============================================================"   | tee -a "$LOG"
