#!/bin/bash
# Probe factorial peak RSS on the worst receiver (Chandelier), then auto-run
# the full 462-pair factorial sweep only if the projected peak fits the host.
#
# Usage:
#   bash code/integration/run_factorial_memory_gated.sh
#
# Env overrides:
#   HOST_MEMORY_CAP_GB   hard ceiling (default 30)
#   SAFETY_FACTOR        multiplier over observed probe peak (default 1.3)
#   PROBE_RECEIVER       receiver to probe (default Chandelier — known worst)
#   PROBE_MEMORY_GB      cgroup cap for probe (default 28)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
LOG_DIR="$INT_DIR/intermediates/verification_logs"
mkdir -p "$LOG_DIR"

HOST_MEMORY_CAP_GB="${HOST_MEMORY_CAP_GB:-30}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.3}"
PROBE_RECEIVER="${PROBE_RECEIVER:-Chandelier}"
PROBE_MEMORY_GB="${PROBE_MEMORY_GB:-28}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROBE_LOG="$LOG_DIR/factorial_probe_${STAMP}.log"
FULL_LOG="$LOG_DIR/factorial_full_${STAMP}.log"
SUMMARY="$LOG_DIR/factorial_gated_${STAMP}.txt"

log() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
hr()  { log "------------------------------------------------------------"; }

log "Factorial memory-gated run  ${STAMP}"
log "  Host cap:         ${HOST_MEMORY_CAP_GB} GB"
log "  Safety factor:    ${SAFETY_FACTOR}"
log "  Probe receiver:   ${PROBE_RECEIVER}"
log "  Probe cgroup cap: ${PROBE_MEMORY_GB} GB"
log "  Probe log:        ${PROBE_LOG}"
log "  Full log:         ${FULL_LOG}"
hr

# ---------------------------------------------------------------------------
# Step 1: probe Chandelier under /usr/bin/time -v
# ---------------------------------------------------------------------------
log "[probe] factorial Chandelier-only run starting"
t0=$(date +%s)
PAIR_FILTER="*:${PROBE_RECEIVER}" \
MEMORY_LIMIT_GB="${PROBE_MEMORY_GB}" \
FORCE_RERUN=1 \
  /usr/bin/time -v bash "$INT_DIR/run_factorial_all_pairs.sh" \
  >"$PROBE_LOG" 2>&1
probe_rc=$?
t1=$(date +%s)
probe_elapsed=$(( t1 - t0 ))
log "[probe] rc=$probe_rc  elapsed=$(( probe_elapsed / 60 ))m$(( probe_elapsed % 60 ))s"

if [ $probe_rc -ne 0 ]; then
  log "[probe] FAILED — will not run full sweep"
  log "  tail of probe log:"
  tail -20 "$PROBE_LOG" | sed 's/^/    /' | tee -a "$SUMMARY"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: parse peak RSS, project full-sweep requirement
# ---------------------------------------------------------------------------
peak_kb=$(grep "Maximum resident set size" "$PROBE_LOG" \
  | awk -F: '{gsub(/ /,"",$2); print $2}' | tail -1)

if [ -z "$peak_kb" ]; then
  log "ERROR: could not parse Maximum resident set size from $PROBE_LOG"
  exit 2
fi

peak_gb=$(awk -v kb="$peak_kb" 'BEGIN {printf "%.2f", kb/1024/1024}')
projected_gb=$(awk -v p="$peak_gb" -v sf="$SAFETY_FACTOR" \
  'BEGIN {printf "%.2f", p*sf}')
projected_ceil_gb=$(awk -v p="$projected_gb" \
  'BEGIN {x = p + 0; i = int(x); if (x > i) i = i + 1; print i}')

hr
log "Probe peak RSS:    ${peak_gb} GB"
log "Safety factor:     ${SAFETY_FACTOR}"
log "Projected need:    ${projected_gb} GB  (ceil ${projected_ceil_gb} GB)"
log "Host cap:          ${HOST_MEMORY_CAP_GB} GB"

# ---------------------------------------------------------------------------
# Step 3: decide
# ---------------------------------------------------------------------------
fits=$(awk -v need="$projected_gb" -v cap="$HOST_MEMORY_CAP_GB" \
  'BEGIN {print (need <= cap) ? 1 : 0}')

if [ "$fits" != "1" ]; then
  hr
  log "GATE FAILED — projected ${projected_gb} GB exceeds host cap ${HOST_MEMORY_CAP_GB} GB."
  log "Full 462-pair factorial run NOT started."
  log "Options:"
  log "  - rerun with HOST_MEMORY_CAP_GB=<larger> if you have more headroom"
  log "  - reduce scope (tighter imputation gates, smaller contrast set, pair batching)"
  exit 3
fi

hr
log "GATE PASSED — starting full 462-pair factorial run with MEMORY_LIMIT_GB=${projected_ceil_gb}"
hr

# ---------------------------------------------------------------------------
# Step 4: full run through the driver (skips single; we've already verified)
# ---------------------------------------------------------------------------
t0=$(date +%s)
MEMORY_LIMIT_GB="${projected_ceil_gb}" \
FORCE_RERUN=1 \
SKIP_SINGLE=1 \
SWEEP=full \
  /usr/bin/time -v bash "$INT_DIR/run_imputation_verification.sh" \
  >"$FULL_LOG" 2>&1
full_rc=$?
t1=$(date +%s)
full_elapsed=$(( t1 - t0 ))
log "[full] rc=$full_rc  elapsed=$(( full_elapsed / 60 ))m$(( full_elapsed % 60 ))s"

full_peak_kb=$(grep "Maximum resident set size" "$FULL_LOG" \
  | awk -F: '{gsub(/ /,"",$2); print $2}' | tail -1)
if [ -n "$full_peak_kb" ]; then
  full_peak_gb=$(awk -v kb="$full_peak_kb" 'BEGIN {printf "%.2f", kb/1024/1024}')
  log "Full-run peak RSS: ${full_peak_gb} GB"
fi

hr
log "Done. Summary: $SUMMARY"
exit $full_rc
