#!/usr/bin/env bash
# Full 5xFAD pair-mode Incytr — BOTH modes, back-to-back. Launch inside tmux.
#
# Runs sequentially (never concurrent — the box OOMs if two scoring jobs overlap):
#   1) phospho-only  CHANNELS=pr,ps,py          → outputs/.../<tissue>/wide/
#   2) PTM-extended  CHANNELS=pr,ps,py,Ack,KGG  → outputs/.../<tissue>/wide_ptm/
# Each mode = 8 contrasts (2 tissues x 4 ages, TG vs WT), nboot=100. Measured
# ~1.6-2.3 h/contrast → ~13-19 h per mode, ~26-38 h total.
#
# Resumable: the underlying runner skips any contrast whose parquet already
# exists, so killing and re-launching this script picks up where it left off
# (in EITHER mode — wide/ and wide_ptm/ are independent).
#
# Memory: the whole job runs in ONE systemd --user scope, MemoryMax=20G, so a
# runaway kills only itself, not the shared box. DO NOT start other heavy jobs
# alongside. Both modes run even if the first reports a failed contrast.
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#   tmux new -s incytr
#   bash alz/incytr_pair/run_5xfad_incytr_both.sh
#   # detach: Ctrl-b then d   ;   reattach: tmux attach -t incytr
# Linger is already enabled, so closing your laptop (remote box) keeps it running.
# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

CAP="${MEM_CAP:-20G}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/reports/incytr_pair_mode_5xfad"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/both_run_${TS}.log"
RUNNER="alz/incytr_pair/run_pair_mode_5xfad.sh"

echo "[both] start $(date -Is)  MemoryMax=$CAP  log=$LOG" | tee "$LOG"

# Refuse to start if another heavy job is already running.
if pgrep -f 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose' >/dev/null; then
  echo "[both] ABORT: a heavy job is already running — run this alone." | tee -a "$LOG"
  pgrep -af 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose' | tee -a "$LOG"
  exit 1
fi

# Both runner invocations inside one capped cgroup scope. `|| echo` keeps the
# second mode running even if the first has a failed contrast.
systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 \
  --unit "incytr-5xfad-both-${TS}" \
  bash -c "
    cd '$REPO'
    echo '[both] === MODE phospho-only \$(date -Is) ===' | tee -a '$LOG'
    bash '$RUNNER'       2>&1 | tee -a '$LOG' || echo '[both] phospho-only had failures' | tee -a '$LOG'
    echo '[both] === MODE ptm \$(date -Is) ===' | tee -a '$LOG'
    bash '$RUNNER' --ptm 2>&1 | tee -a '$LOG' || echo '[both] ptm had failures' | tee -a '$LOG'
  "

echo "[both] parquet inventory:" | tee -a "$LOG"
find "$LOG_DIR" -name '*_incytr_output.parquet' \( -path '*/wide/*' -o -path '*/wide_ptm/*' \) \
  -printf '%s\t%p\n' 2>/dev/null | sort -k2 | tee -a "$LOG"
echo "[both] DONE $(date -Is)" | tee -a "$LOG"
