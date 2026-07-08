#!/usr/bin/env bash
# Run Incytr pair-mode over the levy_t5 cluster spine.
#
# 9 comparisons = 3 genotypes (AppP, Ttau, ApTt) × 3 timepoints (2mo, 4mo, 6mo)
# vs ma_<age>_WTyp at the same timepoint. Each call loops 31² = 961 sender ×
# receiver pairs internally.
#
# Modes:
#   --smoke   Run a single comparison (ma_2mo_AppP vs ma_2mo_WTyp) with NBOOT=2,
#             to verify driver + inputs end-to-end before the full burn.
#             Output goes to outputs/reports/incytr_pair_mode/wide_smoke/.
#   default   Run all 9 comparisons with NBOOT=100. Output to
#             outputs/reports/incytr_pair_mode/wide/. Skips comparisons whose
#             parquet already exists and is non-empty (resumable).
#
# Logs to outputs/reports/incytr_pair_mode/pair_run{,_smoke}.log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="full"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

DRIVER="alz/incytr_pair/incytr_commandline.R"
INPUTS_DIR="data/derived/incytr_inputs"
OUTPUT_DIR="outputs/reports/incytr_pair_mode/wide"
LOG_DIR="outputs/reports/incytr_pair_mode"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Pre-flight: confirm Phase 1 outputs.
echo "=== Pre-flight ($MODE) ==="
for f in \
  "$DRIVER" \
  "$INPUTS_DIR/incytr_obj.rds" \
  "$INPUTS_DIR/pr_yuyu_deconvoluted.csv" \
  "$INPUTS_DIR/ps_yuyu_deconvoluted.csv" \
  "$INPUTS_DIR/py_yuyu_deconvoluted.csv" \
  "$INPUTS_DIR/allmarkers.csv" \
  "$INPUTS_DIR/kldata.csv" \
  ; do
  test -e "$f" || { echo "missing: $f"; exit 1; }
done
echo "  Phase 1 inputs OK"

# AD gene.use IS sce4's own reconstructed per-PAIR node set (reproduction of
# sce4's Top300), not a re-derivation — see
# archive/sce4_reproduction_2026-06-08/README.md §6.7. The driver
# consumes it via SCE4_GENEUSE_DIR; the T-cell runner leaves this unset and
# derives. Build the 9 frozen artifacts if absent (idempotent).
GENEUSE_DIR="data/incytr_frozen/sce4_geneuse"
export SCE4_GENEUSE_DIR="$GENEUSE_DIR"
if [[ $(ls "$GENEUSE_DIR"/*.csv 2>/dev/null | wc -l) -lt 9 ]]; then
  echo "=== Building sce4 gene.use artifacts ($GENEUSE_DIR) ==="
  pixi run Rscript alz/incytr_pair/extract_sce4_geneuse.R \
    || { echo "sce4 gene.use extraction incomplete (see above) — aborting"; exit 1; }
fi
echo "  sce4 gene.use artifacts OK ($(ls "$GENEUSE_DIR"/*.csv 2>/dev/null | wc -l)/9)"

# Confirm upstream Incytr is loadable (catches a stale pixi env before R fires up
# the 27k-cell Seurat object).
pixi run Rscript -e 'suppressPackageStartupMessages(library(Incytr)); cat("Incytr", as.character(packageVersion("Incytr")), "loaded\n")' \
  || { echo "upstream Incytr not loadable — pixi run install-incytr?"; exit 1; }

run_one() {
  local geno="$1" age="$2" out_subdir="$3" nboot="$4"
  local c1="ma_${age}_${geno}"
  local c2="ma_${age}_WTyp"
  local out_parquet="$out_subdir/${c1}_${c2}_incytr_output.parquet"

  # KsG admission: map geno to contrast label when KsG is active (KSG_MEA_FILE set).
  # Smoke path never sets KSG_MEA_FILE, so this block never fires there.
  if [[ -n "${KSG_MEA_FILE:-}" ]]; then
    local contrast_prefix
    case "$geno" in
      AppP) contrast_prefix="App" ;;
      Ttau) contrast_prefix="Tau" ;;
      ApTt) contrast_prefix="ApTt" ;;
    esac
    export KSG_CONTRAST="${contrast_prefix}_${age}"
  fi

  # Resumable at the contrast level: a finished contrast's final parquet exists
  # (its per-pair shards were deleted after the concat), so re-run would redo it
  # from scratch. Skip it. Within an unfinished contrast the driver still
  # resumes from surviving per-pair shards.
  if [[ "${FORCE_RERUN:-0}" != "1" && -s "$out_parquet" ]]; then
    echo "=== $(date -Is) SKIP $c1 vs $c2 (exists, $(du -h "$out_parquet" | cut -f1)) ==="
    return 0
  fi

  echo "=== $(date -Is) $c1 vs $c2 (nboot=$nboot) ==="
  # Subprocess-per-chunk mode (driver default SUBPROCESS_CHUNKS=1):
  # the orchestrator spawns one Rscript per chunk so the OS reclaims the
  # dense `cond_mats` allocations from upstream Permutation_test between
  # chunks. The in-process mclapply path OOM'd within a single 90-pair
  # chunk on 2mo Ttau (RSS 18.8 GB at chunk 1/4) even with 1 worker
  # because R's gc does not return memory to the OS reliably. With
  # NPAIR_WORKERS=1 × N_CHUNK_MULT=8 = 8 chunks of ~120 pairs each.
  # On the 30 GB box main R RSS grows ~600 MB per pair within a chunk
  # (cond_mats heap-fragmentation), so heavy contrasts blow past 24G
  # before chunk end at this chunk size — set N_CHUNK_MULT=48 from the
  # caller (~20 pairs/chunk, peak ~19 GB). Caller may also override
  # NPAIR_WORKERS / NPERM_WORKERS to trade speed against headroom.
  NBOOT="$nboot" \
  NPAIR_WORKERS="${NPAIR_WORKERS:-1}" \
  N_CHUNK_MULT="${N_CHUNK_MULT:-8}" \
  NPERM_WORKERS="${NPERM_WORKERS:-1}" \
  CHUNK_PARALLEL="${CHUNK_PARALLEL:-2}" \
    pixi run Rscript "$DRIVER" "$c1" "$c2" \
    || { echo "  FAIL: $c1 vs $c2 (continuing)"; return 1; }

  # Driver writes to OUTPUT_DIR (resolved via REPO_ROOT inside the R script).
  # If smoke mode requested a different subdir, move the file.
  if [[ "$out_subdir" != "$OUTPUT_DIR" ]]; then
    mv "$OUTPUT_DIR/${c1}_${c2}_incytr_output.parquet" "$out_parquet"
  fi
}

if [[ "$MODE" == "smoke" ]]; then
  LOG="$LOG_DIR/pair_run_smoke.log"
  SMOKE_DIR="outputs/reports/incytr_pair_mode/wide_smoke"
  mkdir -p "$SMOKE_DIR"
  SMOKE_NBOOT="${SMOKE_NBOOT:-2}"
  {
    echo "=== Smoke run: nboot=$SMOKE_NBOOT, single comparison ==="
    run_one AppP 2mo "$SMOKE_DIR" "$SMOKE_NBOOT" || true
    echo "=== $(date -Is) Smoke done. Output: $SMOKE_DIR/ ==="
    ls -lh "$SMOKE_DIR/"
  } 2>&1 | tee "$LOG"
  exit 0
fi

LOG="$LOG_DIR/pair_run.log"
FULL_NBOOT="${FULL_NBOOT:-100}"

# KsG admission-only: widen gene.use for kinase-substrate genes on the
# production path only. Toggle = KSG_MEA_FILE being set; unset = byte-identical
# to pre-KsG. KSG_CONTRAST is computed per-iteration inside run_one.
export KSG_MEA_FILE="outputs/reports/kinase_attribution/mea_stoichiometry.csv"
export KSG_MEA_PY_FILE="outputs/reports/kinase_attribution/mea_stoichiometry_pY.csv"
export KSG_MOTIF_FILE="outputs/reports/kinase_attribution/stoichiometry_matrix.csv"
export KSG_ATTRIBUTION_FILE="data/derived/ksg/song_attribution_long.csv"

{
  echo "=== Full run: nboot=$FULL_NBOOT, 9 comparisons ==="
  failed=()
  for geno in AppP Ttau ApTt; do
    for age in 2mo 4mo 6mo; do
      run_one "$geno" "$age" "$OUTPUT_DIR" "$FULL_NBOOT" || failed+=("${age}_${geno}")
    done
  done
  echo
  echo "=== $(date -Is) Full run done ==="
  if (( ${#failed[@]} > 0 )); then
    echo "FAILED: ${failed[*]}"
  else
    echo "All 9 comparisons succeeded."
  fi

  # Apply the canonical significance floors to wide/ in place (the driver emits
  # all paths at cutoff=0; this is the downstream half): SigProb > 0.1 (either)
  # AND |PDS| >= 0.2. No p_adj arm. No Top300 cap by default; Top300 is too
  # rank-sensitive to PDS drift and is reserved for explicit sce4 diagnostics.
  echo "=== $(date -Is) significance filter (canonical floors, uncapped) ==="
  pixi run python alz/incytr_pair/filter_significant_paths.py --dir "$OUTPUT_DIR"

  ls -lh "$OUTPUT_DIR/"
} 2>&1 | tee "$LOG"
