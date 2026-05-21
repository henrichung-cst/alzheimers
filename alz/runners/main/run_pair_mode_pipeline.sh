#!/usr/bin/env bash
# Phase-2 clean rerun for change requests 01-04.
# See docs/plans/change_requests_sequencing.md for the model.
#
# Resumable: each step writes a sentinel under outputs/reports/change_requests/.state/.
# Re-running the wrapper skips steps whose sentinel exists.
#
# Usage:
#   bash alz/runners/main/run_pair_mode_pipeline.sh [--skip-atlas] [--skip-incytr]
#                                                [--workers N] [--force]
#                                                [--rerun STEP_KEY[,STEP_KEY...]]
#
# STEP_KEYs: A, B, C, D-st, D-py, E1, E2, F1, F2, G1, G2, H1, H2, I, V

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SPINE="levy_t5"
MIN_CELLS=5
WORKERS=3
LOG_DIR="outputs/reports/change_requests/$(date -u +%Y%m%dT%H%M%SZ)"
STATE_DIR="outputs/reports/change_requests/.state"
mkdir -p "$LOG_DIR" "$STATE_DIR"
MAIN_LOG="$LOG_DIR/build.log"

SKIP_ATLAS=0
SKIP_INCYTR=0
FORCE=0
RERUN_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-atlas)   SKIP_ATLAS=1; shift ;;
    --skip-incytr)  SKIP_INCYTR=1; shift ;;
    --workers)      WORKERS="$2"; shift 2 ;;
    --force)        FORCE=1; shift ;;
    --rerun)        RERUN_LIST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Clear sentinels listed in --rerun (comma-separated).
if [[ -n "$RERUN_LIST" ]]; then
  IFS=',' read -ra _keys <<< "$RERUN_LIST"
  for k in "${_keys[@]}"; do
    rm -f "$STATE_DIR/${k}.done"
    echo "cleared sentinel: $STATE_DIR/${k}.done"
  done
fi
if [[ $FORCE -eq 1 ]]; then
  rm -f "$STATE_DIR"/*.done
  echo "cleared all sentinels (--force)"
fi

exec > >(tee -a "$MAIN_LOG") 2>&1

# ---------- progress + timers ----------

T_TOTAL_START=$(date +%s)
STEP_IDX=0

# N_STEPS is the count of planned steps after skip-* flags. We compute it
# by listing the keys that will run.
PLANNED_KEYS=(A B C D-st D-py)
if [[ $SKIP_INCYTR -eq 0 ]]; then PLANNED_KEYS+=(E1 E2 E3); fi
PLANNED_KEYS+=(F1 F2)
if [[ $SKIP_ATLAS -eq 0 ]]; then PLANNED_KEYS+=(G1 G2); fi
PLANNED_KEYS+=(H1 H2 I V)
N_STEPS=${#PLANNED_KEYS[@]}

fmt_elapsed() {
  local s=$1
  printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# run_step <key> <label> -- <cmd...>
run_step() {
  local key="$1"; shift
  local label="$1"; shift
  local sentinel="$STATE_DIR/${key}.done"
  STEP_IDX=$((STEP_IDX + 1))
  local t0=$(date +%s)
  local total_elapsed=$((t0 - T_TOTAL_START))
  echo
  echo "================================================================"
  echo "[$STEP_IDX/$N_STEPS] ($key) $label"
  echo "  started:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  total elapsed:  $(fmt_elapsed $total_elapsed)"
  if [[ -f "$sentinel" ]]; then
    echo "  SKIP — cached: $(cat "$sentinel")"
    echo "================================================================"
    return 0
  fi
  echo "================================================================"
  "$@"
  local t1=$(date +%s)
  date -u +%Y-%m-%dT%H:%M:%SZ > "$sentinel"
  echo "----------------------------------------------------------------"
  echo "[$STEP_IDX/$N_STEPS] ($key) $label  DONE in $(fmt_elapsed $((t1 - t0)))"
}

run_step_softfail() {
  local key="$1"; shift
  local label="$1"; shift
  local sentinel="$STATE_DIR/${key}.done"
  STEP_IDX=$((STEP_IDX + 1))
  local t0=$(date +%s)
  local total_elapsed=$((t0 - T_TOTAL_START))
  echo
  echo "================================================================"
  echo "[$STEP_IDX/$N_STEPS] ($key) $label  (soft-fail)"
  echo "  started:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  total elapsed:  $(fmt_elapsed $total_elapsed)"
  if [[ -f "$sentinel" ]]; then
    echo "  SKIP — cached: $(cat "$sentinel")"
    echo "================================================================"
    return 0
  fi
  echo "================================================================"
  if "$@"; then
    local t1=$(date +%s)
    date -u +%Y-%m-%dT%H:%M:%SZ > "$sentinel"
    echo "[$STEP_IDX/$N_STEPS] ($key) $label  DONE in $(fmt_elapsed $((t1 - t0)))"
  else
    local t1=$(date +%s)
    echo "[$STEP_IDX/$N_STEPS] ($key) $label  WARNING: nonzero exit, continuing (elapsed $(fmt_elapsed $((t1 - t0))))"
  fi
}

# Pair-mode Incytr: alz/incytr/run_pair_mode.sh loops the 9 contrasts
# internally. Worker parallelism is set via CHUNK_PARALLEL (subprocess fan-out
# within each contrast's sender×receiver chunks); contrast-level parallel
# launches share output/ and would race, so we keep that loop sequential.
run_pair_incytr() {
  CHUNK_PARALLEL="$WORKERS" bash alz/incytr/run_pair_mode.sh
}

# ---------- pipeline ----------

echo "=== run_pair_mode_pipeline.sh starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "spine=$SPINE  min_cells=$MIN_CELLS  workers=$WORKERS  log_dir=$LOG_DIR"
echo "state_dir=$STATE_DIR  planned steps: $N_STEPS"

run_step A "cluster spine (CR02)" \
  pixi run python alz/integration/build_cluster_spine.py \
    --min-cells "$MIN_CELLS" --spine-name "$SPINE"

run_step B "snRNA proportions" \
  pixi run python alz/snrna_proportions.py --run --spine "$SPINE"

run_step C "cell-type decomposition (st + py)" \
  pixi run python alz/decomposition/build_celltype_decomposition.py \
    --spine "$SPINE" --track both

run_step D-st "per-cluster MEA (st)" \
  pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track st

run_step_softfail D-py "per-cluster MEA (py)" \
  pixi run python -m alz.decomposition.enrich_celltype --spine "$SPINE" --track py

if [[ $SKIP_INCYTR -eq 0 ]]; then
  incytr_dir="alz/incytr"
  inputs_dir="data/derived/incytr_inputs"
  # Mouse ligand-receptor DB layers ship as package data in the upstream
  # Incytr package (loaded via `Incytr::DB_Layer*_mouse_filtered`); no
  # bench-side symlinks needed.
  for f in incytr_commandline.R reconstruct_labels.R reconstruct_node_fc.R; do
    if [[ ! -e "$incytr_dir/$f" ]]; then
      echo "ERROR: required driver script $incytr_dir/$f missing" >&2
      echo "       driver scripts must live under alz/incytr/." >&2
      exit 1
    fi
  done
  if [[ ! -e "$inputs_dir/kldata.csv" ]]; then
    echo "ERROR: $inputs_dir/kldata.csv missing — run alz/incytr/build_pair_inputs.sh first." >&2
    exit 1
  fi
  run_step E1 "pair-mode inputs (alz/incytr/build_pair_inputs.sh)" \
    bash alz/incytr/build_pair_inputs.sh
  run_step E2 "pair-mode 9 contrasts (CHUNK_PARALLEL=$WORKERS)" \
    run_pair_incytr
  # Substrate for the unified viewer's transcript-trace panel: per-(cluster,
  # group) means of originalexp@data — the same matrix Cal_scFC reads. Run
  # once per pair-mode build (contrast-invariant), before the viewer rebuild
  # invalidates the transcript_trace shards on schema version bump.
  run_step E3 "emit_expr_bygroup parquet (transcript trace substrate)" \
    pixi run Rscript alz/incytr/emit_expr_bygroup.R
fi

run_step F1 "human ingest (reshape)" \
  pixi run python alz/ingest_mukesh.py --reshape

run_step F2 "human per-donor MEA + CTRL LOO (CR01)" \
  pixi run python alz/ingest_mukesh_perdonor.py --track both

if [[ $SKIP_ATLAS -eq 0 ]]; then
  run_step G1 "SEA-AD MTG expression download (CR03)" \
    pixi run python alz/atlas_reference.py --sea-ad-expression
  run_step G2 "Allen HBCA download (CR03)" \
    pixi run python alz/atlas_reference.py --hbca-download
fi

run_step H1 "human reference expression (CR03)" \
  pixi run python alz/human_reference_expression.py --ref both

run_step H2 "human cell-type attribution (CR03)" \
  pixi run python alz/human_celltype_attribution.py --force

run_step I "rebuild unified viewer" \
  pixi run python alz/build_unified_viewer.py

run_step_softfail V "verification" \
  pixi run python alz/decomposition/verify_decomposition.py --spine "$SPINE"

T_TOTAL_END=$(date +%s)
TOTAL=$((T_TOTAL_END - T_TOTAL_START))

cat > "$LOG_DIR/done.json" <<EOF
{
  "spine": "$SPINE",
  "min_cells": $MIN_CELLS,
  "workers": $WORKERS,
  "skip_atlas": $SKIP_ATLAS,
  "skip_incytr": $SKIP_INCYTR,
  "steps_planned": $N_STEPS,
  "total_seconds": $TOTAL,
  "total_elapsed": "$(fmt_elapsed $TOTAL)",
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo
echo "================================================================"
echo "ALL $N_STEPS STEPS PROCESSED  total: $(fmt_elapsed $TOTAL)"
echo "log:    $MAIN_LOG"
echo "state:  $STATE_DIR/  (delete a *.done file or pass --rerun KEY to redo a step)"
echo "done:   $LOG_DIR/done.json"
echo "================================================================"
