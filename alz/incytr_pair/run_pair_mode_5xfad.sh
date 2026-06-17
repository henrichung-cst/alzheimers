#!/usr/bin/env bash
# 5xFAD pair-mode runner — per-tissue, per-age TG-vs-WT.
#
# Contrasts: within each tissue, the 5xFAD transgene effect at each age:
#   cortex      = {TG_3mo, TG_6mo, TG_9mo, TG_12mo} vs the same-age WT
#   hippocampus = same  (4 contrasts/tissue, 8 total)
#
# Reuses alz/incytr_pair/incytr_commandline.R with env-parameterized inputs:
#   INPUTS_DIR_OVERRIDE  → per-tissue data/derived/5xfad_incytr_inputs/<tissue>
#   OUTPUT_DIR_OVERRIDE  → per-tissue outputs/reports/incytr_pair_mode_5xfad/<tissue>
#   CHANNELS             → "pr,ps,py" (total + IMAC/ST + pY, both tissues)
#   PR/PS/PY_FILE        → <track>_deconvoluted.csv ; gene key column = gene_symbol
#   USE_KLDATA=TRUE      → mouse kldata.csv symlinked per tissue to the Song
#                          kinase library (data/datasets/song/kinase/kldata_pspy.csv)
#   SPECIES=mouse
#   SCE4_GENEUSE_DIR     → UNSET. 5xFAD has no sce4 reference; gene.use is derived
#                          live as DEG ∪ prG (the t-cell path).
#
# Modes:
#   --smoke <tissue>   one age (3mo) at NBOOT=2 → scratch dir wide_smoke/
#   default            all contrasts at NBOOT=100, resumable, filter applied
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRIVER="alz/incytr_pair/incytr_commandline.R"
KLDATA_SRC="../../../datasets/song/kinase/kldata_pspy.csv"   # relative to INPUTS_DIR
AGES=(3 6 9 12)
TISSUES=(cortex hippocampus)

MODE="full"
SMOKE_TISSUE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; SMOKE_TISSUE="${2:-cortex}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="outputs/reports/incytr_pair_mode_5xfad"
mkdir -p "$LOG_DIR"

ensure_kldata() {
  # Idempotent: mouse kinase library symlink (5xFAD reuses the Song kldata).
  local indir="$1"
  if [[ ! -e "$indir/kldata.csv" ]]; then
    ln -sf "$KLDATA_SRC" "$indir/kldata.csv"
  fi
}

preflight() {
  local tissue="$1"
  local indir="data/derived/5xfad_incytr_inputs/$tissue"
  ensure_kldata "$indir"
  local need=( "$DRIVER" "$indir/incytr_obj.rds" "$indir/allmarkers.csv"
               "$indir/kldata.csv"
               "$indir/pr_deconvoluted.csv" "$indir/ps_deconvoluted.csv"
               "$indir/py_deconvoluted.csv" )
  for f in "${need[@]}"; do
    test -e "$f" || { echo "missing: $f"; return 1; }
  done
  echo "[$tissue] preflight OK"
}

run_one() {
  local tissue="$1" age="$2" out_subdir="$3" nboot="$4"
  local indir="data/derived/5xfad_incytr_inputs/$tissue"
  local c1="TG_${age}mo"
  local c2="WT_${age}mo"
  mkdir -p "$out_subdir"
  local out_parquet="$out_subdir/${c1}_${c2}_incytr_output.parquet"
  if [[ -s "$out_parquet" ]]; then
    echo "[$tissue $c1 vs $c2] resume (parquet exists)"
    return 0
  fi
  echo "=== $(date -Is) [$tissue] $c1 vs $c2 (nboot=$nboot) ==="
  local status_file="$out_subdir/.status_${c1}_${c2}.txt"
  echo "started $(date -Is)" > "$status_file"

  INPUTS_DIR_OVERRIDE="$indir" \
  OUTPUT_DIR_OVERRIDE="$out_subdir" \
  CHANNELS="pr,ps,py" \
  PR_FILE="pr_deconvoluted.csv" \
  PS_FILE="ps_deconvoluted.csv" \
  PY_FILE="py_deconvoluted.csv" \
  PR_GENE_COL="gene_symbol" \
  PS_GENE_COL="gene_symbol" \
  PY_GENE_COL="gene_symbol" \
  USE_KLDATA="TRUE" \
  SPECIES="mouse" \
  NBOOT="$nboot" \
  NPAIR_WORKERS="${NPAIR_WORKERS:-1}" \
  N_CHUNK_MULT="${N_CHUNK_MULT:-8}" \
  NPERM_WORKERS="${NPERM_WORKERS:-1}" \
    pixi run Rscript "$DRIVER" "$c1" "$c2" \
    || { echo "FAIL $(date -Is)" > "$status_file"; echo "  FAIL: [$tissue] $c1 vs $c2 (continuing)"; return 1; }
  echo "done $(date -Is)" > "$status_file"
}

if [[ "$MODE" == "smoke" ]]; then
  tissue="$SMOKE_TISSUE"
  LOG="$LOG_DIR/pair_run_smoke_${tissue}.log"
  SMOKE_DIR="$LOG_DIR/${tissue}/wide_smoke"
  mkdir -p "$SMOKE_DIR"
  {
    echo "=== Smoke run: $tissue, nboot=2, TG_3mo vs WT_3mo ==="
    preflight "$tissue"
    run_one "$tissue" 3 "$SMOKE_DIR" 2 || true
    echo "=== $(date -Is) Smoke done. Output: $SMOKE_DIR/ ==="
    ls -lh "$SMOKE_DIR/"
  } 2>&1 | tee "$LOG"
  exit 0
fi

LOG="$LOG_DIR/pair_run.log"
{
  failed=()
  for tissue in "${TISSUES[@]}"; do
    preflight "$tissue" || { failed+=("$tissue:preflight"); continue; }
    OUT_DIR="$LOG_DIR/${tissue}/wide"
    for age in "${AGES[@]}"; do
      run_one "$tissue" "$age" "$OUT_DIR" 100 \
        || failed+=("${tissue}:TG_${age}mo_vs_WT_${age}mo")
    done

    # Canonical significance floor (uncapped; no p_adj/FDR arm):
    # (SigProb_TG > 0.1 OR SigProb_WT > 0.1) AND |PDS| >= 0.2
    echo "=== $(date -Is) [$tissue] significance filter ==="
    pixi run python alz/incytr_pair/filter_significant_paths.py --dir "$OUT_DIR"
    ls -lh "$OUT_DIR/"
  done
  echo
  if (( ${#failed[@]} > 0 )); then
    echo "FAILED: ${failed[*]}"
    exit 1
  fi
  echo "All 5xFAD contrasts succeeded."
} 2>&1 | tee "$LOG"
