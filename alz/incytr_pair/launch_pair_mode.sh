#!/usr/bin/env bash
# Self-service launcher for the canonical nboot=100 pair-mode run -> wide/.
#
# Wraps run_pair_mode.sh (9 contrasts at nboot=100, then the significance
# filter) as a DETACHED systemd --user service so it survives closing the
# terminal / lid (linger enabled) and is suspend-inhibited so the laptop does
# not sleep mid-run. OOM-capped at 24G on the 30G shared box.
#
# Memory model: the driver runs mclapply(mc.preschedule=FALSE) -> one fork per
# pair, torn down after each pair so per-pair heap is reclaimed. That reclaim
# only happens at NPAIR_WORKERS>1 (W=1 is a single accumulating lapply). W=3:
# a measured W=2 contrast peaked 15.4G/24G with PSI memory pressure flat at 0.00,
# so the third ~4.5G fork (projected ~19G peak) stays clear of the cap with the
# permutation's transient dense matrices at nboot=100. ~33 min/contrast.
#
# Idempotent: refuses a second copy (two capped runs would breach RAM).
# Resumable: completed per-pair shards under wide/.shards/ are reused.
#
#   bash alz/incytr_pair/launch_pair_mode.sh            # start it, then close the lid
#   tail -f outputs/reports/incytr_pair_mode/pair_run.log
#   systemctl --user stop alz-pairmode-nboot100         # stop it
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
UNIT="alz-pairmode-nboot100"
LOG="$REPO_ROOT/outputs/reports/incytr_pair_mode/pair_run.log"

if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "$UNIT is already running — not launching a second capped run (OOM guard)."
  echo "  follow: tail -f $LOG"
  echo "  stop:   systemctl --user stop $UNIT"
  exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true

loginctl enable-linger "$USER" 2>/dev/null || \
  echo "note: could not enable linger — survives lid-close/screen-off; only a full logout would end it."
mkdir -p "$HOME/.cache/duckdb" "$(dirname "$LOG")"

systemd-run --user --unit="$UNIT" \
  -p MemoryMax=24G -p MemorySwapMax=0 \
  --setenv=PATH="$PATH" \
  --setenv=CONDA_OVERRIDE_CUDA="" \
  --setenv=DUCKDB_TEMP_DIR="$HOME/.cache/duckdb" \
  --setenv=NPAIR_WORKERS=3 \
  --setenv=NPERM_WORKERS=1 \
  --working-directory="$REPO_ROOT" \
  systemd-inhibit --what=handle-lid-switch:sleep:idle --mode=block \
    --why="pairmode nboot=100 full run" \
    bash alz/incytr_pair/run_pair_mode.sh

# Detached completion notifier (desktop popup + console wall + COMPLETE.txt).
systemctl --user reset-failed "$UNIT-notify.service" 2>/dev/null || true
systemd-run --user --unit="$UNIT-notify" \
  --setenv=PATH="$PATH" \
  --setenv=DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
  --setenv=WATCH_UNIT="$UNIT.service" \
  --setenv=WATCH_LOG="$LOG" \
  --working-directory="$REPO_ROOT" \
  bash bench/perf/notify_on_complete.sh >/dev/null 2>&1 \
  && echo "Armed $UNIT-notify (pings on completion)." \
  || echo "note: could not arm notifier (run continues regardless; check COMPLETE.txt / log)."

echo "Launched $UNIT — detached, suspend-inhibited, 24G cap, NPAIR_WORKERS=3."
echo "  follow:  tail -f $LOG"
echo "           journalctl --user -u $UNIT -f"
echo "  stop:    systemctl --user stop $UNIT"
