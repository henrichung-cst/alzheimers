#!/usr/bin/env bash
# Phase 4: per-receiver sweep over the Levy-19 spine, run in parallel.
#
# Each iteration restricts PAIR_FILTER to a single receiver (`*:<receiver>`),
# so `construct_factorial_paths` builds ~250K paths per sweep (vs 3.7M for the
# full 19x19 call) and the in-memory `scored` frame inside
# `score_factorial_paths_multimodel` stays at smoke-scale.
#
# Concurrency is the OUTER loop (xargs -P N), not the inner `mclapply`. Each
# sweep runs INCYTR_PARALLEL_NCORES=1 to avoid the COW-dirtying penalty that
# bit Plan A/B: when each fork worker touched the full sender/receiver columns
# of the 3.7M-row scored frame, COW page-sharing broke immediately. With one
# inner worker and a 250K-row frame per sweep, the artifact stays small.
#
# Measured per-sweep memory: 3.5 GB steady, ~5 GB construct peak (transient).
# At -P 3 the working set is ~10.5 GB steady, fits in 16 GB available.
#
# Pair parquets accumulate in outputs/reports/incytr_factorial/.staging/pair_parquets/
# and are checkpointed across sweeps by `score_pair` (file.exists short-circuit).
# Re-running is safe and cheap.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGING="outputs/reports/incytr_factorial/.staging/pair_parquets"
LOG_DIR="outputs/reports/audits"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/phase4_sweep_${RUN_TS}.log"
mkdir -p "$STAGING" "$LOG_DIR"

# Outer concurrency. 3 fits comfortably in 16 GB; raise to 4 only with care.
JOBS="${PHASE4_JOBS:-3}"

# Inner parallelism: 1 = no mclapply forking inside a sweep. This is the key
# knob that distinguishes this from Plan A/B.
export INCYTR_PARALLEL_NCORES="${INCYTR_PARALLEL_NCORES:-1}"
# Permutation p-values stay disabled.
export INCYTR_N_PERM="${INCYTR_N_PERM:-0}"

# Levy-19 receivers, from the active export's incytr_factorial_inputs.
# Newline-separated to survive whitespace in
# "Excitatory principal neurons in the hippocampal dentate gyrus".
RECEIVERS_FILE="$(mktemp)"
trap 'rm -f "$RECEIVERS_FILE"' EXIT
cat >"$RECEIVERS_FILE" <<'EOF'
Astrocytes
Endothelial-cell
Erbb4-inhibitory-neurons
Erbb4-VIP-inhibitory-neurons
Excitatory principal neurons in the hippocampal dentate gyrus
Excitatory-neurons
Excitatory-Pyramidal
Excitatory-Pyramidal-Satb2-Cux2
Excitatory-Rorb
Foxp2-Excitatory-Neurons-layers-6-and-2-3
glutamatergic-excitatory-neurons
Microglia
Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic
Oligodendrocytes
OPC
Pericyte
Reln-neurons
Striatal-medium-spiny-neuron
VIP-positive-interneuron
EOF
N_RCV=$(wc -l <"$RECEIVERS_FILE")

count_pairs() {
  find "$STAGING" -maxdepth 1 -name 'pair_*.parquet' -printf '.' 2>/dev/null | wc -c
}

BEFORE_TOTAL=$(count_pairs)
OVERALL_START=$SECONDS

{
  echo "Phase 4 receiver sweep (parallel) — $(date -Is)"
  echo "PHASE4_JOBS=$JOBS INCYTR_PARALLEL_NCORES=$INCYTR_PARALLEL_NCORES INCYTR_N_PERM=$INCYTR_N_PERM"
  echo "Staging dir: $STAGING"
  echo "Receivers: $N_RCV"
  echo "Pair parquets before: $BEFORE_TOTAL"
  echo ""
} | tee "$SUMMARY_LOG"

# Per-sweep worker. Reads receiver from $1, runs incytr-factorial, captures
# log + timing. Always exits 0 so xargs continues on a single-sweep failure;
# failures are surfaced in the summary log.
run_one_sweep() {
  local rcv="$1"
  local stem
  stem="$(printf '%s' "$rcv" | tr ' /' '__')"
  local rcv_log="$LOG_DIR/phase4_sweep_${RUN_TS}_${stem}.log"
  local before after new t0 dt
  before=$(count_pairs)
  t0=$SECONDS

  if PAIR_FILTER="*:$rcv" pixi run incytr-factorial >"$rcv_log" 2>&1; then
    after=$(count_pairs)
    new=$((after - before))
    dt=$((SECONDS - t0))
    printf '[OK]   %-72s new=%-3d wall=%ds log=%s\n' \
      "$rcv" "$new" "$dt" "$rcv_log" | tee -a "$SUMMARY_LOG"
  else
    local rc=$?
    dt=$((SECONDS - t0))
    printf '[FAIL] %-72s rc=%-3d wall=%ds log=%s\n' \
      "$rcv" "$rc" "$dt" "$rcv_log" | tee -a "$SUMMARY_LOG"
  fi
}

export -f run_one_sweep count_pairs
export STAGING LOG_DIR RUN_TS SUMMARY_LOG SECONDS

# xargs -d '\n' keeps newline-separated receivers intact (preserves spaces).
xargs -d '\n' -a "$RECEIVERS_FILE" -n 1 -P "$JOBS" -I {} \
  bash -c 'run_one_sweep "$@"' _ {}

AFTER_TOTAL=$(count_pairs)
NEW_TOTAL=$((AFTER_TOTAL - BEFORE_TOTAL))
WALL=$((SECONDS - OVERALL_START))

{
  echo ""
  echo "Sweep complete — $(date -Is)"
  echo "Pair parquets: $BEFORE_TOTAL -> $AFTER_TOTAL  (+$NEW_TOTAL)"
  echo "Wall: ${WALL}s ($(echo "scale=1; $WALL/60" | bc) min)"
  echo "Expected: 361 pair parquets (19x19); current: $AFTER_TOTAL"
  if [ "$AFTER_TOTAL" -lt 361 ]; then
    echo "MISSING: $((361 - AFTER_TOTAL)) — rerun to checkpoint-resume."
  fi
} | tee -a "$SUMMARY_LOG"
