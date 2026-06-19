#!/usr/bin/env bash
# Full 5xFAD pair-mode: ONE --ptm run, then DERIVE the phospho-only product from
# it. Launch inside tmux. Replaces the two-full-runs approach (run both modes
# back-to-back): the phospho-only result is recoverable from the --ptm superset
# to floating-point precision (derive_phospho_from_ptm.py; validated max|Δ|~9e-16
# on multimodel_score/PDS, every other column bit-identical), so we score once.
#
# Sequence (order is load-bearing — see the SKIP_FILTER note in run_pair_mode):
#   1) SCORE   bash run_pair_mode_5xfad.sh --ptm  with SKIP_FILTER=yes
#              → outputs/.../<tissue>/wide_ptm/   (8 contrasts, UNFILTERED supersets)
#   2) DERIVE  each wide_ptm/<contrast>.parquet → wide/<contrast>.parquet
#              (DuckDB-streamed; phospho-only = PTM minus the Ack/KGG layer)
#   3) FILTER  the canonical SigProb/PDS gate, in place, on BOTH wide_ptm/ and
#              wide/. Run LAST and only after every derive, because the gate is
#              in-place: filtering wide_ptm/ before derive would drop the rows
#              the derive reads (phospho PDS != PTM PDS).
#
# Why derive must read the UNFILTERED superset: the PTM and phospho PDS differ in
# ~99% of rows (589k sign flips at 3mo). The phospho gate keeps a DIFFERENT row
# set than the PTM gate, so a phospho product built from PTM-gated rows would
# silently miss phospho-significant / PTM-insignificant pathways.
#
# Resumable: scoring skips contrasts whose parquet exists; derive skips contrasts
# whose wide/ parquet exists; the gate is idempotent (a floor re-applied is a
# no-op). All derives finish before any gate runs, so a crash never leaves a
# derive reading already-gated wide_ptm.
#
# Memory: the whole script re-execs itself inside ONE systemd --user scope,
# MemoryMax=$CAP (default 20G), MemorySwapMax=0, so a runaway kills only itself,
# not the shared box. RUN THIS ALONE.
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#   tmux new -s incytr
#   bash alz/incytr_pair/run_5xfad_ptm_derive.sh
#   # detach: Ctrl-b then d   ;   reattach: tmux attach -t incytr
# Linger is enabled, so closing your laptop (remote box) keeps it running.
# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

CAP="${MEM_CAP:-20G}"
BASE="outputs/reports/incytr_pair_mode_5xfad"
RUNNER="alz/incytr_pair/run_pair_mode_5xfad.sh"
DERIVE="alz/incytr_pair/derive_phospho_from_ptm.py"
FILTER="alz/incytr_pair/filter_significant_paths.py"
TISSUES=(cortex hippocampus)

# Re-exec the whole workflow inside one capped cgroup scope.
if [[ "${_CAPPED:-}" != "1" ]]; then
  if pgrep -f 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose|derive_phospho_from_ptm' >/dev/null; then
    echo "[ptm+derive] ABORT: a heavy job is already running — run this alone."
    pgrep -af 'incytr_commandline|build_unified_viewer|build_5xfad_input_gene_list|fivexfad_decompose|derive_phospho_from_ptm'
    exit 1
  fi
  TS="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BASE"
  LOG="$BASE/ptm_derive_run_${TS}.log"
  echo "[ptm+derive] start $(date -Is)  MemoryMax=$CAP  log=$LOG"
  exec systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    --unit "incytr-5xfad-ptm-derive-${TS}" \
    env _CAPPED=1 LOG="$LOG" bash "$0" "$@" 2>&1 | tee "$LOG"
fi

# ── inside the scope ─────────────────────────────────────────────────────────
echo "[ptm+derive] === 1/3 SCORE (--ptm, unfiltered) $(date -Is) ==="
SKIP_FILTER=yes bash "$RUNNER" --ptm \
  || echo "[ptm+derive] scoring reported failures (continuing to derive what exists)"

echo "[ptm+derive] === 2/3 DERIVE phospho-only from wide_ptm/ $(date -Is) ==="
for t in "${TISSUES[@]}"; do
  for ptm in "$BASE/$t/wide_ptm/"*_incytr_output.parquet; do
    [[ -e "$ptm" ]] || continue
    out="${ptm/wide_ptm/wide}"
    if [[ -s "$out" ]]; then
      echo "[derive] resume (exists): $out"; continue
    fi
    echo "[derive] $ptm -> $out"
    pixi run python "$DERIVE" --ptm "$ptm" --out "$out" \
      || echo "[derive] FAIL $ptm"
  done
done

echo "[ptm+derive] === 3/3 FILTER both products (in place) $(date -Is) ==="
for t in "${TISSUES[@]}"; do
  for sub in wide_ptm wide; do
    d="$BASE/$t/$sub"
    [[ -d "$d" ]] || continue
    echo "[filter] $d"
    pixi run python "$FILTER" --dir "$d" || echo "[filter] FAIL $d"
  done
done

echo "[ptm+derive] parquet inventory (post-filter):"
find "$BASE" -name '*_incytr_output.parquet' \( -path '*/wide/*' -o -path '*/wide_ptm/*' \) \
  -printf '%s\t%p\n' 2>/dev/null | sort -k2
echo "[ptm+derive] DONE $(date -Is)"
