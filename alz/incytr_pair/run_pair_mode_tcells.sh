#!/usr/bin/env bash
# T-cell pair-mode runner — per-donor, per-contrast.
#
# Contrasts (per meeting_notes_triage_2026-05-27.md): each later day vs the
# day-2 baseline, per donor independently.
#   donor1 = {d13, d17, d20} vs d2   (3 contrasts, per-cell labels, pr+py+ps)
#   donor2 = {d5, d7, d9, d11} vs d2 (4 contrasts, per-cell labels, pr+py)
#
# Reuses alz/incytr_pair/incytr_commandline.R with env-parameterized inputs:
#   INPUTS_DIR_OVERRIDE  → per-donor data/derived/tcells_incytr_inputs/<donor>
#   OUTPUT_DIR_OVERRIDE  → optional replacement for the canonical per-cell root
#   CHANNELS             → "pr,py,ps" (donor1) or "pr,py" (donor2)
#   PR_FILE/PY_FILE/PS_FILE → state-keyed deconvoluted CSVs
#   PR_GENE_COL/PY_GENE_COL/PS_GENE_COL → "gene_symbol"
#   USE_KLDATA=TRUE      → SiK kinase scoring on, human kldata.csv symlinked
#                          per donor (→ data/datasets/tcells/kinase/kldata_human.csv)
#
# Modes:
#   --smoke <donor>   one contrast at NBOOT=2 → scratch dir wide_smoke/
#   default           all contrasts at NBOOT=100, resumable, filter applied
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PIXI_BIN="${PIXI_BIN:-$(command -v pixi 2>/dev/null || true)}"
if [[ -z "$PIXI_BIN" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_BIN" ]]; then
  echo "pixi executable not found" >&2
  exit 1
fi

DRIVER="alz/incytr_pair/incytr_commandline.R"
MODE="full"
SMOKE_DONOR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; SMOKE_DONOR="${2:-donor1}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

declare -A DONOR_DAYS=( [donor1]="13 17 20" [donor2]="5 7 9 11" )
declare -A DONOR_CHANNELS=( [donor1]="pr,py,ps" [donor2]="pr,py" )
FULL_NBOOT="${FULL_NBOOT:-100}"

LOG_DIR="${OUTPUT_DIR_OVERRIDE:-outputs/reports/incytr_pair_mode_tcells}"
mkdir -p "$LOG_DIR"

preflight() {
  local donor="$1"
  local indir="data/derived/tcells_incytr_inputs/$donor"
  local need=( "$DRIVER" "$indir/incytr_obj.rds" "$indir/allmarkers.csv"
               "$indir/kldata.csv"
               "$indir/pr_deconvoluted.csv" "$indir/py_deconvoluted.csv" )
  if [[ "$donor" == "donor1" ]]; then need+=( "$indir/ps_deconvoluted.csv" ); fi
  for f in "${need[@]}"; do
    test -e "$f" || { echo "missing: $f"; return 1; }
  done
  echo "[$donor] preflight OK"
}

run_one() {
  local donor="$1" later="$2" out_subdir="$3" nboot="$4"
  local indir="data/derived/tcells_incytr_inputs/$donor"
  local c1="d${later}"
  local c2="d2"
  mkdir -p "$out_subdir"
  local out_parquet="$out_subdir/${c1}_${c2}_incytr_output.parquet"
  if [[ -s "$out_parquet" ]]; then
    echo "[$donor $c1 vs $c2] resume (parquet exists)"
    return 0
  fi
  echo "=== $(date -Is) [$donor] $c1 vs $c2 (nboot=$nboot) ==="
  local status_file="$out_subdir/.status_${c1}_${c2}.txt"
  echo "started $(date -Is)" > "$status_file"

  # KsG: admission-only, donor1 only (donor2 is pY-only, no ST, no attribution)
  local -a ksg_env=()
  if [[ "$donor" == "donor1" ]]; then
    mkdir -p data/derived/ksg
    ksg_env=(
      KSG_MEA_FILE="outputs/reports/kinase_attribution_tcells/donor1/mea/mea_timecourse.csv"
      KSG_MOTIF_FILE="outputs/reports/kinase_attribution_tcells/donor1/stoichiometry_matrix.csv"
      KSG_ATTRIBUTION_FILE="data/derived/ksg/tcells_donor1_attribution_long.csv"
      KSG_CONTRAST="D1_d${later}_vs_d2"
    )
  fi

  env \
    INPUTS_DIR_OVERRIDE="$indir" \
    OUTPUT_DIR_OVERRIDE="$out_subdir" \
    BACKBONE_OUT_DIR="${LOG_DIR}/${donor}/backbone" \
    CHANNELS="${DONOR_CHANNELS[$donor]}" \
    PR_FILE="pr_deconvoluted.csv" \
    PY_FILE="py_deconvoluted.csv" \
    PS_FILE="ps_deconvoluted.csv" \
    PR_GENE_COL="gene_symbol" \
    PY_GENE_COL="gene_symbol" \
    PS_GENE_COL="gene_symbol" \
    USE_KLDATA="TRUE" \
    SPECIES="human" \
    NBOOT="$nboot" \
    NPAIR_WORKERS="${NPAIR_WORKERS:-1}" \
    N_CHUNK_MULT="${N_CHUNK_MULT:-8}" \
    NPERM_WORKERS="${NPERM_WORKERS:-1}" \
    "${ksg_env[@]}" \
    "$PIXI_BIN" run Rscript "$DRIVER" "$c1" "$c2" \
    || { echo "FAIL $(date -Is)" > "$status_file"; echo "  FAIL: [$donor] $c1 vs $c2 (continuing)"; return 1; }
  echo "done $(date -Is)" > "$status_file"
}

if [[ "$MODE" == "smoke" ]]; then
  donor="$SMOKE_DONOR"
  LOG="$LOG_DIR/pair_run_smoke_${donor}.log"
  SMOKE_DIR="$LOG_DIR/${donor}/wide_smoke"
  mkdir -p "$SMOKE_DIR"
  {
    echo "=== Smoke run: $donor, nboot=2, first later-day vs d2 ==="
    preflight "$donor"
    read -r first _ <<< "${DONOR_DAYS[$donor]}"
    run_one "$donor" "$first" "$SMOKE_DIR" 2 || true
    echo "=== $(date -Is) Smoke done. Output: $SMOKE_DIR/ ==="
    ls -lh "$SMOKE_DIR/"
  } 2>&1 | tee "$LOG"
  exit 0
fi

LOG="$LOG_DIR/pair_run.log"
{
  failed=()
  # donor2 first: no ps channel — banks the simpler cohort
  # before the higher-RSS donor1 run.
  for donor in donor2 donor1; do
    preflight "$donor" || { failed+=("$donor:preflight"); continue; }
    OUT_DIR="$LOG_DIR/${donor}/wide"
    for later in ${DONOR_DAYS[$donor]}; do
      run_one "$donor" "$later" "$OUT_DIR" "$FULL_NBOOT" \
        || failed+=("${donor}:d${later}_vs_d2")
    done

    # Canonical significance floor (uncapped; no p_adj/FDR arm):
    # (SigProb_<later> > 0.1 OR SigProb_d2 > 0.1) AND |PDS| >= 0.2
    echo "=== $(date -Is) [$donor] significance filter ==="
    "$PIXI_BIN" run python alz/incytr_pair/filter_significant_paths.py --dir "$OUT_DIR"
    ls -lh "$OUT_DIR/"
  done
  echo
  if (( ${#failed[@]} > 0 )); then
    echo "FAILED: ${failed[*]}"
    exit 1
  fi
  echo "All T-cell contrasts succeeded."
} 2>&1 | tee "$LOG"
