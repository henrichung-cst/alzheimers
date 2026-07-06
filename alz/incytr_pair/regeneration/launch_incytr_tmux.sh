#!/usr/bin/env bash
# Tmux harness for the 5xFAD + t-cell Incytr re-run (KsG + PTM + backbone).
# Song is done — its KsG output is canonical in production and is not regenerated.
#
#   bash alz/incytr_pair/regeneration/launch_incytr_tmux.sh
#   tmux attach -t incytr
#
# Creates a detached tmux session `incytr` with one combined run window plus a
# monitor window. The combined command is STAGED on the prompt (typed, NOT
# executed) — the operator reviews and presses Enter once.
#
# THIS SCRIPT LAUNCHES NOTHING ON ITS OWN. It only lays out the session.
#
# Gating:
#   * The combined runner is sequential. 5xFAD pair-mode/post-processing
#     completes before t-cell starts; 5xFAD bridge is deferred.
#   * Each leaf runner caps RAM at 24G via systemd-run and peaks ~13-15 GB.
#   * Review the combined log and per-cohort logs after the run.
#
# Windows:
#   0 all   run_backbone_overnight_all.sh  (5xFAD, then t-cell; bridge deferred; NO viewer)
#   1 mon   htop + log tail helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION="incytr"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not found on PATH." >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with:  tmux attach -t $SESSION" >&2
    echo "(kill it first with 'tmux kill-session -t $SESSION' to re-stage.)" >&2
    exit 1
fi

# Combined command. It is staged but not run. The leaf runners set
# CONDA_OVERRIDE_CUDA="" internally for their systemd scopes.
ALL_CMD='bash alz/incytr_pair/regeneration/run_backbone_overnight_all.sh'

# Stage a command onto a window's prompt WITHOUT executing it (no Enter), after
# printing a one-line banner the operator sees above the staged command.
stage() {
    local win="$1" banner="$2" cmd="$3"
    tmux send-keys -t "$SESSION:$win" "clear; echo '$banner'" C-m
    tmux send-keys -t "$SESSION:$win" "$cmd"   # no C-m → stays on prompt
}

tmux new-session -d -s "$SESSION" -n all -c "$REPO_ROOT"
tmux new-window  -t "$SESSION" -n mon -c "$REPO_ROOT"

stage all '[combined] 5xFAD pair-mode/post-processing, then t-cell. Bridge deferred. Press Enter once.' "$ALL_CMD"

# Monitor window: live RAM watch + tail of the newest log across all three trees.
tmux send-keys -t "$SESSION:mon" \
    "tail -F outputs/reports/incytr_pair_mode*/overnight_*.log 2>/dev/null" C-m
tmux split-window -t "$SESSION:mon" -v -c "$REPO_ROOT"
tmux send-keys -t "$SESSION:mon" "htop 2>/dev/null || top" C-m
tmux select-window -t "$SESSION:all"

cat <<EOF
tmux session '$SESSION' staged (nothing running yet).

  Attach:   tmux attach -t $SESSION
  Windows:  0 all   1 mon

Start the sequential run:
  all -> press Enter once

Logs:
  outputs/reports/incytr_pair_mode_regeneration/overnight_all_*.log
  outputs/reports/incytr_pair_mode_5xfad/overnight_*.log
  outputs/reports/incytr_pair_mode_tcells/overnight_*.log
EOF
