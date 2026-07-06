#!/usr/bin/env bash
# Full 5xFAD pair-mode scoring run. Launch inside tmux.
#
# Scores all 8 contrasts (cortex/hippocampus × 3/6/9/12mo) with the canonical
# PTM-inclusive channel set (pr,ps,py,Ack,KGG) and applies the SigProb/PDS gate.
#   → outputs/reports/incytr_pair_mode_5xfad/<tissue>/wide/   (filtered)
#
# Resumable: scoring skips contrasts whose parquet already exists; the gate is
# idempotent (a floor re-applied is a no-op).
#
# Memory: the whole script re-execs itself inside ONE systemd --user scope,
# MemoryMax=$CAP (default 20G), MemorySwapMax=0, so a runaway kills only itself,
# not the shared box. RUN THIS ALONE.
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#   tmux new -s incytr
#   bash alz/incytr_pair/run_5xfad.sh
#   # detach: Ctrl-b then d   ;   reattach: tmux attach -t incytr
# Linger is enabled, so closing your laptop (remote box) keeps it running.
# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

CAP="${MEM_CAP:-20G}"
BASE="outputs/reports/incytr_pair_mode_5xfad"
RUNNER="alz/incytr_pair/run_pair_mode_5xfad.sh"

# Re-exec the whole workflow inside one capped cgroup scope.
if [[ "${_CAPPED:-}" != "1" ]]; then
  if pgrep -f 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose' >/dev/null; then
    echo "[5xfad] ABORT: a heavy job is already running — run this alone."
    pgrep -af 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose'
    exit 1
  fi
  TS="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BASE"
  LOG="$BASE/run_${TS}.log"
  echo "[5xfad] start $(date -Is)  MemoryMax=$CAP  log=$LOG"
  exec systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    --unit "incytr-5xfad-${TS}" \
    env _CAPPED=1 LOG="$LOG" bash "$0" "$@" 2>&1 | tee "$LOG"
fi

# ── inside the scope ─────────────────────────────────────────────────────────
echo "[5xfad] === SCORE + FILTER (channels=pr,ps,py,Ack,KGG) $(date -Is) ==="
bash "$RUNNER" \
  || echo "[5xfad] scoring reported failures (see log)"

echo "[5xfad] parquet inventory (post-filter):"
find "$BASE" -name '*_incytr_output.parquet' -path '*/wide/*' \
  -printf '%s\t%p\n' 2>/dev/null | sort -k2
echo "[5xfad] DONE $(date -Is)"
