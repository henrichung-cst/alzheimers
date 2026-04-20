#!/bin/bash
# Verification driver for P0.2 (refined single-contrast adapter) and P0.1
# (factorial port). Times both pipelines and inventories outputs.
#
# Modes (select with SWEEP env var):
#   SWEEP=spot       (default) one pair, single + factorial, refined adapter
#   SWEEP=legacy     one pair, LEGACY=1 baseline regression against refined
#   SWEEP=full       all 462 pairs, single + factorial, refined adapter
#   SWEEP=grid-imp   tau x FDR sweep, imputation adapter only (cheap)
#   SWEEP=grid-full  tau x FDR sweep, full pipeline per cell (expensive)
#
# Usage examples:
#   bash code/integration/run_imputation_verification.sh
#   SWEEP=legacy bash code/integration/run_imputation_verification.sh
#   SWEEP=full bash code/integration/run_imputation_verification.sh
#   SWEEP=grid-imp TAU_GRID="0.1,0.2,0.4" FDR_GRID="0.05,0.10,0.25" \
#     bash code/integration/run_imputation_verification.sh
#   SWEEP=grid-full bash code/integration/run_imputation_verification.sh
#
# Common env overrides:
#   PAIR_FILTER                 default "Microglia-PVM:L5 IT" (ignored in full/grid-*)
#   FORCE_RERUN                 set to 1 to ignore checkpoints
#   EXPR_IMPUTATION_FLOOR       default 0.05 (R3)
#   SKIP_SINGLE / SKIP_FACTORIAL  skip one side (spot/legacy/full only)
#   TAU_GRID                    comma-separated taus for grid-* (default 0.1,0.2,0.4)
#   FDR_GRID                    comma-separated FDRs for grid-* (default 0.05,0.10,0.25)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT_DIR="$REPO_ROOT/code/integration"
ADAPTERS_DIR="$INT_DIR/adapters"
LOG_DIR="$INT_DIR/intermediates/verification_logs"
mkdir -p "$LOG_DIR"

SWEEP="${SWEEP:-spot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/summary_${SWEEP}_${STAMP}.txt"

print() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
hr()    { print "------------------------------------------------------------"; }

run_timed() {
  local label="$1"; shift
  local log="$1"; shift
  print "[$label] starting"
  print "  logging to: $log"
  local t0 t1
  t0=$(date +%s)
  ( "$@" ) >"$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))
  local mm=$(( elapsed / 60 ))
  local ss=$(( elapsed % 60 ))
  if [ $rc -eq 0 ]; then
    print "[$label] OK    elapsed ${mm}m${ss}s"
  else
    print "[$label] FAIL  rc=$rc  elapsed ${mm}m${ss}s (see log)"
  fi
  return $rc
}

check_file() {
  local label="$1" path="$2"
  if [ -s "$path" ]; then
    local rows size
    rows=$(wc -l < "$path" 2>/dev/null || echo "?")
    size=$(du -h "$path" 2>/dev/null | awk '{print $1}')
    print "  OK    $label  ($size, $rows lines)  $path"
  elif [ -e "$path" ]; then
    print "  EMPTY $label  $path"
  else
    print "  MISS  $label  $path"
  fi
}

# ---------------------------------------------------------------------------
# Standard output inventory (spot / legacy / full modes)
# ---------------------------------------------------------------------------
inventory_standard() {
  hr
  print "Output inventory"
  local per_recv_single per_recv_fac
  per_recv_single=$(find "$INT_DIR/intermediates" -maxdepth 1 \
    -name 'kinase_imputed_genes__*.csv' 2>/dev/null | wc -l)
  per_recv_fac=$(find "$INT_DIR/intermediates/factorial" -maxdepth 1 \
    -name 'kinase_imputed_genes__*.csv' 2>/dev/null | wc -l)
  print "  single-contrast per-receiver imputed files: $per_recv_single"
  print "  factorial       per-receiver imputed files: $per_recv_fac"
  check_file "single flat (legacy)"       "$INT_DIR/intermediates/kinase_imputed_genes.csv"
  check_file "single summary"             "$INT_DIR/intermediates/kinase_imputation_summary.csv"
  check_file "factorial summary"          "$INT_DIR/intermediates/factorial/kinase_imputation_summary_factorial.csv"
  check_file "single receiver parquet"    "$INT_DIR/intermediates/all_pairs/recv_L5_IT.parquet"
  check_file "factorial receiver parquet" "$INT_DIR/intermediates/factorial/all_pairs/recv_L5_IT.parquet"
}

# ---------------------------------------------------------------------------
# Run single + factorial pipelines. Honors SKIP_SINGLE / SKIP_FACTORIAL.
# Args: <tag>  (for log filename disambiguation)
# Exits with 1 if either leg fails.
# ---------------------------------------------------------------------------
run_both_pipelines() {
  local tag="$1"
  local single_rc=0 fac_rc=0
  if [ "${SKIP_SINGLE:-0}" != "1" ]; then
    hr
    print "SINGLE-CONTRAST run (run_all_pairs.sh) [$tag]"
    local log="$LOG_DIR/single_${tag}_${STAMP}.log"
    run_timed "single/$tag" "$log" bash "$INT_DIR/run_all_pairs.sh"
    single_rc=$?
  else
    print "SINGLE-CONTRAST skipped (SKIP_SINGLE=1)"
  fi

  if [ "${SKIP_FACTORIAL:-0}" != "1" ]; then
    hr
    print "FACTORIAL run (run_factorial_all_pairs.sh) [$tag]"
    local log="$LOG_DIR/factorial_${tag}_${STAMP}.log"
    run_timed "factorial/$tag" "$log" bash "$INT_DIR/run_factorial_all_pairs.sh"
    fac_rc=$?
  else
    print "FACTORIAL skipped (SKIP_FACTORIAL=1)"
  fi

  [ $single_rc -eq 0 ] && [ $fac_rc -eq 0 ]
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print "Imputation verification run ${STAMP}"
print "  SWEEP mode:             $SWEEP"
print "  PAIR_FILTER (incoming): ${PAIR_FILTER:-<unset>}"
print "  EXPR_IMPUTATION_FLOOR:  ${EXPR_IMPUTATION_FLOOR:-0.05}"
print "  Log dir:                $LOG_DIR"
hr

export EXPR_IMPUTATION_FLOOR="${EXPR_IMPUTATION_FLOOR:-0.05}"

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$SWEEP" in

  spot)
    export PAIR_FILTER="${PAIR_FILTER:-Microglia-PVM:L5 IT}"
    print "  Pair filter: $PAIR_FILTER"
    run_both_pipelines spot
    rc=$?
    inventory_standard
    hr
    print "Done. Summary written to $SUMMARY"
    exit $rc
    ;;

  legacy)
    export PAIR_FILTER="${PAIR_FILTER:-Microglia-PVM:L5 IT}"
    export KINASE_IMPUTATION_LEGACY_OVERRIDE=1
    print "  Pair filter: $PAIR_FILTER"
    print "  LEGACY mode: the adapter honors icfg.KINASE_IMPUTATION_LEGACY."
    print "  Flip it to True in config_integration.py before running this mode,"
    print "  or export KINASE_IMPUTATION_LEGACY_OVERRIDE if the adapter reads it."
    run_both_pipelines legacy
    rc=$?
    inventory_standard
    # Row-count diff against a stored baseline, if present.
    local_baseline="$INT_DIR/intermediates/kinase_imputed_genes.baseline.csv"
    if [ -f "$local_baseline" ]; then
      hr
      print "LEGACY row-count diff vs baseline"
      baseline_rows=$(wc -l < "$local_baseline")
      current_rows=$(wc -l < "$INT_DIR/intermediates/kinase_imputed_genes.csv")
      print "  baseline: $baseline_rows lines"
      print "  current:  $current_rows lines"
      if [ "$baseline_rows" = "$current_rows" ]; then
        print "  MATCH"
      else
        print "  DRIFT"
      fi
    else
      print "  (no baseline at $local_baseline; copy current to snapshot as baseline)"
    fi
    hr
    print "Done. Summary written to $SUMMARY"
    exit $rc
    ;;

  full)
    unset PAIR_FILTER
    print "  Pair filter: <unset> (all 462 pairs)"
    run_both_pipelines full
    rc=$?
    inventory_standard
    hr
    print "Done. Summary written to $SUMMARY"
    exit $rc
    ;;

  grid-imp|grid-full)
    export PAIR_FILTER="${PAIR_FILTER:-Microglia-PVM:L5 IT}"
    TAU_GRID="${TAU_GRID:-0.1,0.2,0.4}"
    FDR_GRID="${FDR_GRID:-0.05,0.10,0.25}"
    print "  Pair filter: $PAIR_FILTER"
    print "  TAU grid:    $TAU_GRID"
    print "  FDR grid:    $FDR_GRID"

    SENS_CSV="$LOG_DIR/sensitivity_${SWEEP}_${STAMP}.csv"
    echo "tau,fdr,total_gene_receiver_rows,pair_pathways,receiver_parquet_rows,elapsed_s,rc" > "$SENS_CSV"
    print "  results CSV: $SENS_CSV"

    IFS=',' read -ra TAUS <<< "$TAU_GRID"
    IFS=',' read -ra FDRS <<< "$FDR_GRID"

    for tau in "${TAUS[@]}"; do
      for fdr in "${FDRS[@]}"; do
        cell_tag="tau${tau}_fdr${fdr}"
        hr
        print "=== Grid cell: tau=$tau, fdr=$fdr ==="
        export KINASE_IMPUTATION_TAU_OVERRIDE="$tau"
        export KINASE_IMPUTATION_FDR_OVERRIDE="$fdr"

        cell_t0=$(date +%s)

        # Always re-run the adapter (cheap).
        adapter_log="$LOG_DIR/grid_adapter_${cell_tag}_${STAMP}.log"
        run_timed "adapter/$cell_tag" "$adapter_log" \
          pixi run python3 "$ADAPTERS_DIR/export_kinase_imputed_genes.py"
        adapter_rc=$?

        total_rows=0
        if [ $adapter_rc -eq 0 ]; then
          total_rows=$(find "$INT_DIR/intermediates" -maxdepth 1 \
            -name 'kinase_imputed_genes__*.csv' -exec cat {} + 2>/dev/null \
            | wc -l)
          # Subtract header lines (one per file).
          file_count=$(find "$INT_DIR/intermediates" -maxdepth 1 \
            -name 'kinase_imputed_genes__*.csv' 2>/dev/null | wc -l)
          total_rows=$(( total_rows - file_count ))
        fi

        pair_pathways="NA"
        parquet_rows="NA"
        cell_rc=$adapter_rc

        if [ "$SWEEP" = "grid-full" ] && [ $adapter_rc -eq 0 ]; then
          full_log="$LOG_DIR/grid_full_${cell_tag}_${STAMP}.log"
          # Force R re-run; adapter already wrote fresh imputed files.
          FORCE_RERUN=1 run_timed "pipeline/$cell_tag" "$full_log" \
            bash "$INT_DIR/run_all_pairs.sh"
          pipe_rc=$?
          if [ $pipe_rc -eq 0 ]; then
            parquet="$INT_DIR/intermediates/all_pairs/recv_L5_IT.parquet"
            if [ -s "$parquet" ]; then
              parquet_rows=$(wc -l < "$parquet" 2>/dev/null || echo NA)
            fi
            pair_sum="$INT_DIR/intermediates/all_pairs/pair_summary.csv"
            if [ -s "$pair_sum" ]; then
              pair_pathways=$(awk -F, 'NR>1 {s+=$NF} END {print s+0}' \
                "$pair_sum" 2>/dev/null || echo NA)
            fi
          fi
          cell_rc=$pipe_rc
        fi

        cell_t1=$(date +%s)
        cell_elapsed=$(( cell_t1 - cell_t0 ))
        echo "$tau,$fdr,$total_rows,$pair_pathways,$parquet_rows,$cell_elapsed,$cell_rc" >> "$SENS_CSV"
        print "  [$cell_tag] rows=$total_rows  pair_pathways=$pair_pathways  " \
              "parquet_rows=$parquet_rows  elapsed=${cell_elapsed}s  rc=$cell_rc"
      done
    done

    unset KINASE_IMPUTATION_TAU_OVERRIDE KINASE_IMPUTATION_FDR_OVERRIDE
    hr
    print "Sensitivity CSV: $SENS_CSV"
    print "Done. Summary written to $SUMMARY"
    exit 0
    ;;

  *)
    print "ERROR: unknown SWEEP=$SWEEP"
    print "  valid: spot | legacy | full | grid-imp | grid-full"
    exit 2
    ;;
esac
